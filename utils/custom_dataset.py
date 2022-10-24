from torch.utils.data import Dataset
from torchaudio.functional import resample
import torchaudio
import pathlib
import torch

class Custom(Dataset):
    def __init__(
        self,
        audio_path,
        audio_ext,
        max_segment_samples=327680,
        sample_rate=16000
    ):
        r"""Instrument classification dataset takes the meta of an audio
        segment as input, and return the waveform, onset_roll, and targets of
        the audio segment. Dataset is used by DataLoader.
        Args:
            audio_path: str
            audio_ext: str, e.g. mp3, wav, flac
            segment_samples: int, how long you want to cut the audio. If set to None, get the full audio
        """
        audiofolder = pathlib.Path(audio_path)
        self.audio_name_list = list(audiofolder.glob(f'*.{audio_ext}'))

        self.sample_rate = sample_rate
        self.segment_samples = max_segment_samples
        

    def __len__(self):
        return len(self.audio_name_list)

    def __getitem__(self, idx):
        r"""Get input and target of a segment for training.
        Args:
            idx for the audio list
        Returns:
          data_dict: {
            'waveform': (samples_num,)
            'onset_roll': (frames_num, classes_num),
            'offset_roll': (frames_num, classes_num),
            'reg_onset_roll': (frames_num, classes_num),
            'reg_offset_roll': (frames_num, classes_num),
            'frame_roll': (frames_num, classes_num),
            'velocity_roll': (frames_num, classes_num),
            'mask_roll':  (frames_num, classes_num),
            'pedal_onset_roll': (frames_num,),
            'pedal_offset_roll': (frames_num,),
            'reg_pedal_onset_roll': (frames_num,),
            'reg_pedal_offset_roll': (frames_num,),
            'pedal_frame_roll': (frames_num,)}
        """
        
        # try:
        waveform, rate = torchaudio.load(self.audio_name_list[idx])
        if waveform.shape[0]==2: # if the audio file is stereo take mean
            waveform = waveform.mean(0) # no need keepdim (audio length), dataloader will generate the B dim
        else:
            waveform = waveform[0] # remove the first dim

        if rate!=self.sample_rate:
            waveform = resample(waveform, rate, self.sample_rate)            
        # except:    
        #     waveform = torch.tensor([[]])
        #     rate = 0
        #     print(f"{self.audio_name_list[idx].name} is corrupted")
            

        data_dict = {}

        # Load segment waveform.
        # with h5py.File(waveform_hdf5_path, 'r') as hf:
        audio_length = len(waveform)
            

        start_sample = 0
        start_time = 0
        end_sample = audio_length
        
        if waveform.shape[0]>=self.segment_samples:
            waveform_seg = waveform[:self.segment_samples]
        else:
            pad = self.segment_samples - waveform.shape[0]
            waveform_seg = torch.nn.functional.pad(waveform, [0,pad], value=0)
        # (segment_samples,), e.g., (160000,)            

        x = torch.randn(1, 640, 88)
        data_dict['waveform'] = waveform_seg
        data_dict['file_name'] = self.audio_name_list[idx].name

        return x, waveform_seg