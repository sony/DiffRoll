# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from task.diffusion import SpecRollDiffusion
import torchaudio

from math import sqrt


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d

def trim_spec_roll(roll, spectrogram):
    T_roll = roll.shape[-1]
    T_spec = spectrogram.shape[-1]
    
    # trimming extra time steps
    T_min = min(T_roll, T_spec)
    roll = roll[..., :T_min]
    spectrogram = spectrogram[..., :T_min]    

    return roll, spectrogram

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d(*args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer



@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        '''
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        if not uncond: # conditional model
            self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        else: # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None: # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip

class ResidualBlockv2(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        '''
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        '''
        super().__init__()
        self.dilated_conv = Conv2d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        if not uncond: # conditional model
            self.conditioner_projection = Conv2d(1, 2 * residual_channels, 1)
        else: # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv2d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        # x (B, 256, 88, T)
        # diffusion_step (B, 512)
        # conditioner (B, 1, 229, 640)
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None: # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip

class DiffWave(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_projection = Conv1d(1, params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
        if self.params.unconditional: # use unconditional model
            self.spectrogram_upsampler = None
        else:
            self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(params.n_mels, params.residual_channels, 2**(i % params.dilation_cycle_length), uncond=params.unconditional)
            for i in range(params.residual_layers)
        ])
        self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
        self.output_projection = Conv1d(params.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, spectrogram=None):
        assert (spectrogram is None and self.spectrogram_upsampler is None) or \
               (spectrogram is not None and self.spectrogram_upsampler is not None)
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        
        if self.spectrogram_upsampler: # use conditional model
            spectrogram = self.spectrogram_upsampler(spectrogram)
            
        skip = None
        
        index = 0
        for layer in self.residual_layers:
            index += 1
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x

    
class DiffRoll(SpecRollDiffusion):
    def __init__(self,
                 residual_channels,
                 unconditional,
                 n_mels,
                 residual_layers = 30,
                 dilation_cycle_length = 10,
                 spec_args = {},
                 **kwargs):
        super().__init__(**kwargs)
        self.input_projection = Conv1d(88, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(self.betas))

        self.residual_layers = nn.ModuleList([
            ResidualBlock(n_mels, residual_channels, 2**(i % dilation_cycle_length), uncond=unconditional)
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 88, 1)
        nn.init.zeros_(self.output_projection.weight)
        
        if unconditional:
            self.mel_layer = None
        else:
            self.mel_layer = torchaudio.transforms.MelSpectrogram(**spec_args)        

    def forward(self, roll, waveform, diffusion_step):
        # roll (B, 1, T, F)
        # waveform (B, L)
        roll = roll.squeeze(1).transpose(1,2)
        
        if self.mel_layer != None:
            spectrogram = self.mel_layer(waveform) # (B, n_mels, T)
            roll, spectrogram = trim_spec_roll(roll, spectrogram)
        else:
            spectrogram = None
        x = self.input_projection(roll)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
            
        skip = None
        
        index = 0
        for layer in self.residual_layers:
            index += 1
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            
            
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x) #(B, F, T)
        return x.transpose(1,2).unsqueeze(1), spectrogram #(B, T, F)

class DiffRollv2(SpecRollDiffusion):
    def __init__(self,
                 residual_channels,
                 unconditional,
                 n_mels,
                 residual_layers = 30,
                 dilation_cycle_length = 10,
                 spec_args = {},
                 **kwargs):
        super().__init__(**kwargs)
        self.input_projection = Conv2d(1, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(self.betas))
        self.spec_projection = Conv1d(n_mels, 88, 1)

        self.residual_layers = nn.ModuleList([
            ResidualBlockv2(n_mels, residual_channels, 2**(i % dilation_cycle_length), uncond=unconditional)
            for i in range(residual_layers)
        ])
        self.skip_projection = Conv2d(residual_channels, residual_channels, 1)
        self.output_projection = Conv2d(residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)
        
        if unconditional:
            self.mel_layer = None
        else:
            self.mel_layer = torchaudio.transforms.MelSpectrogram(**spec_args)        

    def forward(self, roll, waveform, diffusion_step):
        # roll (B, 1, T, 88)
        # waveform (B, L)
        roll = roll.transpose(-1,-2)

        if self.mel_layer != None:
            spectrogram = self.mel_layer(waveform) # (B, n_mels, T)
            roll, spectrogram = trim_spec_roll(roll, spectrogram)
            spectrogram = self.spec_projection(spectrogram).unsqueeze(1) # (B, 1, 88, T)
        else:
            spectrogram = None
        
        x = self.input_projection(roll)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        skip = None
        
        index = 0
        for layer in self.residual_layers:
            index += 1
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            
            
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x) #(B, 1, F, T)
        
        return x.transpose(-2,-1), spectrogram #(B, T, F)