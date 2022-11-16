import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from .utils import extract_notes_wo_velocity
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torchaudio
MIN_MIDI = 21
MAX_MIDI = 108
HOP_LENGTH = 160
SAMPLE_RATE = 16000

from mir_eval.util import midi_to_hz
import os
from mido import Message, MidiFile, MidiTrack
from mir_eval.util import hz_to_midi

import sys


class DiscreteDiffusion(pl.LightningModule):
    def __init__(self,
                 lr,
                 timesteps,
                 loss_type,
                 loss_keys,
                 schedule_args,                
                 frame_threshold,
                 training,
                 sampling,
                 debug=False,
                 generation_filter=0.0
                ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # define beta schedule
        # beta is variance
        at, bt, ct, att, btt, ctt = alpha_schedule(
            time_step=self.hparams.timesteps,
            N=2,
            **schedule_args
        )
        
        self.register_buffer('at', torch.tensor(at))
        self.register_buffer('bt', torch.tensor(bt))
        self.register_buffer('ct', torch.tensor(ct))
        self.register_buffer('att', torch.tensor(att))
        self.register_buffer('btt', torch.tensor(btt))
        self.register_buffer('ctt', torch.tensor(ctt))
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
#         self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
#         self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.inner_loop = tqdm(range(self.hparams.timesteps), desc='sampling loop time step')
        
        self.reverse_diffusion = getattr(self, sampling.type)   

    def training_step(self, batch, batch_idx):
        losses, tensors = self.step(batch)

        # self.log("Train/amt_loss", losses['amt_loss'])
        
        # calculating total loss based on keys give
        total_loss = 0
        for k in self.hparams.loss_keys:
            total_loss += losses[k]
            self.log(f"Train/{k}", losses[k])            
            
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        losses, tensors = self.step(batch)
        total_loss = 0
        for k in self.hparams.loss_keys:
            total_loss += losses[k]
            self.log(f"Val/{k}", losses[k])           
        # self.log("Val/amt_loss", losses['amt_loss'])
        
        if batch_idx == 0:
            self.visualize_figure(tensors['pred_roll'], 'Val/pred_roll', batch_idx)
            
            if hasattr(self.hparams, 'condition'): # the condition for classifier free
                if self.hparams.condition == 'trainable_spec':
                    fig, ax = plt.subplots(1,1)
                    im = ax.imshow(self.trainable_parameters.detach().cpu(), aspect='auto', origin='lower', cmap='jet')
                    fig.colorbar(im, orientation='vertical')
                    self.logger.experiment.add_figure(f"Val/trainable_uncon", fig, global_step=self.current_epoch)
                    plt.close()
                                      
                
                # if self.hparams.condition == 'trainable_z':
                #     for idx, res_layer in enumerate(self.residual_layers):
                #         fig, ax = plt.subplots(1,1)
                #         im = ax.imshow(res_layer.uncon_z.detach().cpu(), aspect='auto', origin='lower', cmap='jet')
                #         fig.colorbar(im, orientation='vertical')
                #         self.logger.experiment.add_figure(f"Val/trainable_z{idx}", fig, global_step=self.current_epoch)
                #         plt.close()                        
            
            if self.current_epoch == 0: 
                self.visualize_figure(tensors['label_roll'], 'Val/label_roll', batch_idx)
                if self.hparams.unconditional==False and tensors['spec']!=None:
                    self.visualize_figure(tensors['spec'].transpose(-1,-2).unsqueeze(1),
                                          'Val/spec',
                                          batch_idx)
                    
                if isinstance(batch, list):
                    self.visualize_figure(tensors['spec2'].transpose(-1,-2).unsqueeze(1),
                                          'Val/spec2',
                                          batch_idx)
                    self.visualize_figure(tensors['pred_roll2'], 'Val/pred_roll2', batch_idx)
                    self.visualize_figure(tensors['label_roll2'], 'Val/label_roll2', batch_idx)
    def test_step(self, batch, batch_idx):
        noise_list, spec = self.sampling(batch, batch_idx)
    
        
        # noise_list is a list of tuple (pred_t, t), ..., (pred_0, 0)
        roll_pred = noise_list[-1][0] # (B, 1, T, F)        
        roll_label = batch["frame"].unsqueeze(1).cpu()
        
        if batch_idx==0:
            torch.save(spec, 'spec.pt')
            self.visualize_figure(spec.transpose(-1,-2).unsqueeze(1),
                                  'Test/spec',
                                  batch_idx)                
            for noise_npy, t_index in noise_list:
                if (t_index+1)%10==0: 
                    fig, ax = plt.subplots(2,2)
                    for idx, j in enumerate(noise_npy):
                        # j (1, T, F)
                        ax.flatten()[idx].imshow(j[0].T, aspect='auto', origin='lower')
                        self.logger.experiment.add_figure(
                            f"Test/pred",
                            fig,
                            global_step=self.hparams.timesteps-t_index)
                        plt.close()

            fig1, ax1 = plt.subplots(2,2)
            fig2, ax2 = plt.subplots(2,2)
            for idx in range(4):

                ax1.flatten()[idx].imshow(roll_label[idx][0].T, aspect='auto', origin='lower')
                self.logger.experiment.add_figure(
                    f"Test/label",
                    fig1,
                    global_step=0)

                ax2.flatten()[idx].imshow((roll_pred[idx][0]>self.hparams.frame_threshold).T, aspect='auto', origin='lower')
                self.logger.experiment.add_figure(
                    f"Test/pred_roll",
                    fig2,
                    global_step=0)  
                plt.close()            

            torch.save(noise_list, 'noise_list.pt')
            
            #======== Begins animation ===========
            t_list = torch.arange(1, self.hparams.timesteps, 5).flip(0)
            if t_list[-1] != self.hparams.timesteps:
                t_list = torch.cat((t_list, torch.tensor([self.hparams.timesteps])), 0)
            ims = []
            fig, axes = plt.subplots(2,4, figsize=(16, 5))

            title = axes.flatten()[0].set_title(None, fontsize=15)
            ax_flat = axes.flatten()
            caxs = []
            for ax in axes.flatten():
                div = make_axes_locatable(ax)
                caxs.append(div.append_axes('right', '5%', '5%'))

            ani = animation.FuncAnimation(fig,
                                          self.animate_sampling,
                                          frames=tqdm(t_list, desc='Animating'),
                                          fargs=(fig, ax_flat, caxs, noise_list, ),                                          
                                          interval=500,                                          
                                          blit=False,
                                          repeat_delay=1000)
            ani.save('algo2.gif', dpi=80, writer='imagemagick')
            #======== Animation saved ===========
              
            
        frame_p, frame_r, frame_f1, _ = precision_recall_fscore_support(roll_label.flatten(),
                                                                        roll_pred.flatten()>self.hparams.frame_threshold,
                                                                        average='binary')
        
        for sample_idx, (roll_pred_i, roll_label_i) in enumerate(zip(roll_pred, roll_label.numpy())):
            # roll_pred (B, 1, T, F)
            p_est, i_est = extract_notes_wo_velocity(roll_pred_i[0],
                                                     roll_pred_i[0],
                                                     onset_threshold=self.hparams.frame_threshold,
                                                     frame_threshold=self.hparams.frame_threshold,
                                                     rule='rule1'
                                                    )
            
            p_ref, i_ref = extract_notes_wo_velocity(roll_label_i[0],
                                                     roll_label_i[0],
                                                     onset_threshold=self.hparams.frame_threshold,
                                                     frame_threshold=self.hparams.frame_threshold,
                                                     rule='rule1'
                                                    )            
            
            scaling = self.hparams.spec_args.hop_length / self.hparams.spec_args.sample_rate
            # scaling = HOP_LENGTH / SAMPLE_RATE

            # Converting time steps to seconds and midi number to frequency
            i_ref = (i_ref * scaling).reshape(-1, 2)
            p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
            i_est = (i_est * scaling).reshape(-1, 2)
            p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

            p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)
            
            if batch_idx==0:
                torchaudio.save(f'audio_{sample_idx}.mp3',
                                batch['audio'][sample_idx].unsqueeze(0).cpu(),
                                sample_rate=self.hparams.spec_args.sample_rate)     
                clean_notes = (i_est[:,1]-i_est[:,0])>self.hparams.generation_filter

                save_midi(os.path.join('./', f'clean_midi_{sample_idx}.mid'),
                          p_est[clean_notes],
                          i_est[clean_notes],
                          [127]*len(p_est))
                save_midi(os.path.join('./', f'raw_midi_{sample_idx}.mid'),
                          p_est,
                          i_est,
                          [127]*len(p_est))            

                self.log("Test/Note_F1", f)         
        self.log("Test/Frame_F1", frame_f1)  
        
    def predict_step(self, batch, batch_idx):
        noise = batch[0]
        waveform = batch[1]
        # if self.hparams.inpainting_f or self.hparams.inpainting_t:
        #     roll_label = batch[2]
        
        device = noise.device
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        
        self.inner_loop.refresh()
        self.inner_loop.reset()
        
        noise_list = []
        noise_list.append((noise, self.hparams.timesteps))

        for t_index in reversed(range(0, self.hparams.timesteps)):
            noise, spec = self.reverse_diffusion(noise, waveform, t_index)
            noise_npy = noise.detach().cpu().numpy()
                    # self.hparams.timesteps-i is used because slide bar won't show
                    # if global step starts from self.hparams.timesteps
            noise_list.append((noise_npy, t_index))                       
            self.inner_loop.update()
            #======== Animation saved ===========      
            
        # noise_list is a list of tuple (pred_t, t), ..., (pred_0, 0)
        roll_pred = noise_list[-1][0] # (B, 1, T, F)        

        if batch_idx==0:
            self.visualize_figure(spec.transpose(-1,-2).unsqueeze(1),
                                  'Test/spec',
                                  batch_idx)
            for noise_npy, t_index in noise_list:
                if (t_index+1)%10==0: 
                    fig, ax = plt.subplots(2,2)
                    for idx, j in enumerate(noise_npy):
                        if idx<4:
                            # j (1, T, F)
                            ax.flatten()[idx].imshow(j[0].T, aspect='auto', origin='lower')
                            self.logger.experiment.add_figure(
                                f"Test/pred",
                                fig,
                                global_step=self.hparams.timesteps-t_index)
                            plt.close()
                        else:
                            break

            fig1, ax1 = plt.subplots(2,2)
            fig2, ax2 = plt.subplots(2,2)
            for idx, roll_pred_i in enumerate(roll_pred):
                
                ax2.flatten()[idx].imshow((roll_pred_i[0]>self.hparams.frame_threshold).T, aspect='auto', origin='lower')
                self.logger.experiment.add_figure(
                    f"Test/pred_roll",
                    fig2,
                    global_step=0)  
                plt.close()            

            torch.save(noise_list, 'noise_list.pt')
            # torch.save(spec, 'spec.pt')
            # torch.save(roll_label, 'roll_label.pt')
            
            #======== Begins animation ===========
            t_list = torch.arange(1, self.hparams.timesteps, 5).flip(0)
            if t_list[-1] != self.hparams.timesteps:
                t_list = torch.cat((t_list, torch.tensor([self.hparams.timesteps])), 0)
            ims = []
            fig, axes = plt.subplots(2,4, figsize=(16, 5))

            title = axes.flatten()[0].set_title(None, fontsize=15)
            ax_flat = axes.flatten()
            caxs = []
            for ax in axes.flatten():
                div = make_axes_locatable(ax)
                caxs.append(div.append_axes('right', '5%', '5%'))

            ani = animation.FuncAnimation(fig,
                                          self.animate_sampling,
                                          frames=tqdm(t_list, desc='Animating'),
                                          fargs=(fig, ax_flat, caxs, noise_list, ),                                          
                                          interval=500,                                          
                                          blit=False,
                                          repeat_delay=1000)
            ani.save('algo2.gif', dpi=80, writer='imagemagick')         
            #======== Animation saved ===========
            
        # export as midi
        for roll_idx, np_frame in enumerate(noise_list[-1][0]):
            # np_frame = (1, T, 88)
            np_frame = np_frame[0]
            p_est, i_est = extract_notes_wo_velocity(np_frame, np_frame)

            scaling = HOP_LENGTH / SAMPLE_RATE
            # Converting time steps to seconds and midi number to frequency
            i_est = (i_est * scaling).reshape(-1, 2)
            p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

            clean_notes = (i_est[:,1]-i_est[:,0])>self.hparams.generation_filter

            save_midi(os.path.join('./', f'clean_midi_e{batch_idx}_{roll_idx}.mid'),
                      p_est[clean_notes],
                      i_est[clean_notes],
                      [127]*len(p_est))
            save_midi(os.path.join('./', f'raw_midi_{batch_idx}_{roll_idx}.mid'),
                      p_est,
                      i_est,
                      [127]*len(p_est))
        


            
    def visualize_figure(self, tensors, tag, batch_idx):
        fig, ax = plt.subplots(2,2)
        for idx, tensor in enumerate(tensors): # visualize only 4 piano rolls
            if idx<4:
                # roll_pred (1, T, F)
                ax.flatten()[idx].imshow(tensor[0].T.cpu(), aspect='auto', origin='lower')
            else:
                break
        self.logger.experiment.add_figure(f"{tag}", fig, global_step=self.current_epoch)
        plt.close()
        
    def step(self, batch):
        # batch["frame"] (B, 1, 640, 88)
        # batch["audio"] (B, L)
        if isinstance(batch, list):
            batch_size = batch[0]["frame"].shape[0]
            roll = self.normalize(batch[0]["frame"]).unsqueeze(1) 
            waveform = batch[0]["audio"]
            roll2 = self.normalize(batch[1]["frame"]).unsqueeze(1) 
            waveform2 = batch[1]["audio"]
            device = roll.device            
        else:
            batch_size = batch["frame"].shape[0]
            roll = self.normalize(batch["frame"]).unsqueeze(1) 
            waveform = batch["audio"]
            device = roll.device
        
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ## sampling the same t within each batch, might not work well
        # t = torch.randint(0, self.hparams.timesteps, (1,), device=device)[0].long() # [0] to remove dimension
        # t_tensor = t.repeat(batch_size).to(roll.device)
        
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long() # more diverse sampling
        

        noise = torch.randn_like(roll) # creating label noise
        B, _, roll_T, roll_F = roll.shape
        x_start = roll_to_log_onehot(roll, 3) #(B, 3, 640*88)
        
        x_t = self.q_sample(x_start, t)
        
        
        # When debugging model is use, change waveform into roll
        if self.hparams.training.mode == 'epsilon':
            if self.hparams.debug==True:
                epsilon_pred, spec = self(x_t, roll, t) # predict the noise N(0, 1)
            else:
                epsilon_pred, spec = self(x_t, waveform, t) # predict the noise N(0, 1)
            diffusion_loss = self.p_losses(noise, epsilon_pred, loss_type=self.hparams.loss_type)

            pred_roll = extract_x0(
                x_t,
                epsilon_pred,
                t,
                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod)
            
        elif self.hparams.training.mode == 'x_0':
            pred_roll, spec = self(x_t.view(-1,3,roll_T,roll_F), waveform, t) # predict the noise N(0, 1)
            diffusion_loss = self.p_losses(x_start.view(B, 3, roll_T, roll_F), pred_roll, loss_type=self.hparams.loss_type)
            if isinstance(batch, list): # when using multiple dataset do one more feedforward

                x_t2 = q_sample( # sampling noise at time t
                    x_start=roll2,
                    t=t,
                    sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                    sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
                    noise=noise)                
                pred_roll2, spec2 = self(x_t2, waveform2, t, sampling=True) # sampling = True
                # line 656 of diffwav.py will be activated and the second dataset would be always p=-1
                # i.e. the spectrograms are always -1
                unconditional_diffusion_loss = self.p_losses(roll2, pred_roll2, loss_type=self.hparams.loss_type)
            
        elif self.hparams.training.mode == 'ex_0':
            epsilon_pred, spec = self(x_t, waveform, t) # predict the noise N(0, 1)
            pred_roll = extract_x0(
                x_t,
                epsilon_pred,
                t,
                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod)            
            diffusion_loss = self.p_losses(roll, pred_roll, loss_type=self.hparams.loss_type)   
            
        
        else:
            raise ValueError(f"training mode {self.training.mode} is not supported. Please either use 'x_0' or 'epsilon'.")
        
        # pred_roll = torch.sigmoid(pred_roll) # to convert logit into probability
        # amt_loss = F.binary_cross_entropy(pred_roll, roll)
        
        
        pred_roll = torch.exp(pred_roll[:,1]).unsqueeze(1) # convert log pro back to prob (B, 1, 640, 88)
        
        if isinstance(batch, list):
            tensors = {
                "pred_roll": pred_roll,
                "label_roll": roll,
                "spec": spec,
                "spec2": spec2,
                "label_roll2": roll2,
                "pred_roll2": pred_roll2,
            }            
            
            losses = {
                "diffusion_loss": diffusion_loss,
                'unconditional_diffusion_loss': unconditional_diffusion_loss
                # "amt_loss": amt_loss
            }            
        else:
            tensors = {
                "pred_roll": pred_roll,
                "label_roll": roll,
                "spec": spec
            }
            
            losses = {
                "diffusion_loss": diffusion_loss,
                # "amt_loss": amt_loss
            }                   
        
        return losses, tensors
    
    def sampling(self, batch, batch_idx):
        batch_size = batch["frame"].shape[0]
        roll = self.normalize(batch["frame"]).unsqueeze(1)
        waveform = batch["audio"]
        device = roll.device
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        
        self.inner_loop.refresh()
        self.inner_loop.reset()
        
        noise = torch.randn_like(roll)
        noise_list = []
        noise_list.append((noise, self.hparams.timesteps))

        for t_index in reversed(range(0, self.hparams.timesteps)):
            if self.hparams.debug==True:
                noise, spec = self.reverse_diffusion(noise, roll, t_index)
            else:
                noise, spec = self.reverse_diffusion(noise, waveform, t_index)
            noise_npy = noise.detach().cpu().numpy()
                    # self.hparams.timesteps-i is used because slide bar won't show
                    # if global step starts from self.hparams.timesteps
            noise_list.append((noise_npy, t_index))                       
            self.inner_loop.update()
        
        return noise_list, spec
        
    def p_losses(self, label, prediction, loss_type="l1"):
        if loss_type == 'l1':
            loss = F.l1_loss(label, prediction)
        elif loss_type == 'l2':
            loss = F.mse_loss(label, prediction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(label, prediction)
        elif loss_type == "kl":
            loss = F.kl_div(torch.log_softmax(prediction, 1), label[:,:-1], log_target=True)
        else:
            raise NotImplementedError()

        return loss
    

        
    def cfdg_ddpm_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred_c, spec = self(x, waveform, t_tensor)
        x0_pred_0, _ = self(x, torch.zeros_like(waveform), t_tensor, sampling=True) # if sampling = True, the input condition will be overwritten
        x0_pred = (1+self.hparams.sampling.w)*x0_pred_c - self.hparams.sampling.w*x0_pred_0
#         x0_pred = x0_pred_c
        # x0_pred = x0_pred_0

        if t_index == 0:
            sigma = (1/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))            
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index-1]/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))                    
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec
    
    def generation_ddpm_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred_0, _ = self(x, torch.zeros_like(waveform), t_tensor, sampling=True) # if sampling = True, the input condition will be overwritten
        x0_pred = x0_pred_0
        
#         x0_pred = x0_pred_c
        # x0_pred = x0_pred_0

        if t_index == 0:
            sigma = (1/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))            
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index-1]/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))                    
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, _
    
    def inpainting_ddpm_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred_c, spec = self(x, waveform, t_tensor, inpainting_t=self.hparams.inpainting_t, inpainting_f=self.hparams.inpainting_f)
        x0_pred_0, _ = self(x, torch.zeros_like(waveform), t_tensor, sampling=True) # if sampling = True, the input condition will be overwritten
        x0_pred = (1+self.hparams.sampling.w)*x0_pred_c - self.hparams.sampling.w*x0_pred_0
#         x0_pred = x0_pred_c
        # x0_pred = x0_pred_0

        if t_index == 0:
            sigma = (1/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))            
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index-1]/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))                    
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec
    
    def cfdg_ddim_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred, spec = self(x, waveform, t_tensor)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred_c, spec = self(x, waveform, t_tensor)
        x0_pred_0, _ = self(x, torch.zeros_like(waveform), t_tensor)
        x0_pred = (1+self.hparams.sampling.w)*x0_pred_c - self.hparams.sampling.w*x0_pred_0
#         x0_pred = x0_pred_c
        # x0_pred = x0_pred_0

        if t_index == 0:
            sigma = 0 
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = 0          
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec            

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#         scheduler = TriStageLRSchedule(optimizer,
#                                        [1e-8, self.hparams.lr, 1e-8],
#                                        [0.2,0.6,0.2],
#                                        max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
#         scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

#         return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]

    def animate_sampling(self, t_idx, fig, ax_flat, caxs, noise_list):
        # Tuple of (x_t, t), (x_t-1, t-1), ... (x_0, 0)
        # x_t (B, 1, T, F)
        # clearing figures to prevent slow down in each iteration.d
        fig.canvas.draw()
        for idx in range(len(noise_list[0][0])): # visualize only 4 piano rolls
            ax_flat[idx].cla()
            ax_flat[4+idx].cla()
            caxs[idx].cla()
            caxs[4+idx].cla()     

            # roll_pred (1, T, F)
            im1 = ax_flat[idx].imshow(noise_list[0][0][idx][0].detach().T.cpu(), aspect='auto', origin='lower')
            im2 = ax_flat[4+idx].imshow(noise_list[1+self.hparams.timesteps-t_idx][0][idx][0].T, aspect='auto', origin='lower')
            fig.colorbar(im1, cax=caxs[idx])
            fig.colorbar(im2, cax=caxs[4+idx])

        fig.suptitle(f't={t_idx}')
        row1_txt = ax_flat[0].text(-400,45,f'Gaussian N(0,1)')
        row2_txt = ax_flat[4].text(-300,45,'x_{t-1}')
        
    def q_sample(self, log_x_start, t): # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t)
        log_sample = log_sample_categorical(log_EV_qxt_x0)
        return log_sample
    
    def q_pred(self, log_x_start, t):           # q(xt|x0)
        # log_x_start: (B, classes+1, L)

        # This part is manually added
        log_at = torch.log(self.at)
        log_bt = torch.log(self.bt)
        log_ct = torch.log(self.ct)

        log_cumprod_at = torch.log(self.att)
        log_cumprod_bt = torch.log(self.btt)
        log_cumprod_ct = torch.log(self.ctt)
        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)
        # end of manually added

    #     log_x_start is can be onehot or not
        t = (t + (self.hparams.timesteps + 1))%(self.hparams.timesteps + 1) # When t>timesteps, it restarts from 0
    #     print(f"{log_cumprod_at.shape=}")
        log_cumprod_at = extract(log_cumprod_at, t, log_x_start.shape)         # at~
    #     print(f"{log_cumprod_at.shape=}")
        log_cumprod_bt = extract(log_cumprod_bt, t, log_x_start.shape)         # bt~
        log_cumprod_ct = extract(log_cumprod_ct, t, log_x_start.shape)         # ct~
        log_1_min_cumprod_ct = extract(log_1_min_cumprod_ct, t, log_x_start.shape)       # 1-ct~

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:,:-1,:]+log_cumprod_at, log_cumprod_bt), # probability of same class
                log_add_exp(log_x_start[:,-1:,:]+log_1_min_cumprod_ct, log_cumprod_ct) # probability of being mask
            ],
            dim=1
        )

        return log_probs
        
        
def roll_to_log_onehot(x, num_classes):
    # x.shape (B, 88, T)
    x = x.flatten(1) # (B, 88*T)
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x.long(), num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.size())))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x

def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.linspace(att_1, att_T, time_step)
    ctt = np.linspace(ctt_1, ctt_T, time_step)

    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]

    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct
    bt = (1-at-ct)/N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1-att-ctt)/N
    return at, bt, ct, att, btt, ctt
    
    
def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)

def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))



def log_sample_categorical(logits):           # use gumbel to sample onehot vector from log probability
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample = (gumbel_noise + logits).argmax(dim=1)
    log_sample = roll_to_log_onehot(sample, 3) 
    return log_sample