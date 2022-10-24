from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import model as Model
import AudioLoader.music.amt as MusicDataset
from utils.custom_dataset import Custom

from AudioLoader.music.amt import MAPS, MAESTRO
from omegaconf import OmegaConf
import warnings

@hydra.main(config_path="config", config_name="sampling")
def main(cfg):       
    cfg.data_root = to_absolute_path(cfg.data_root)
    cfg.dataset.args.audio_path = to_absolute_path(cfg.dataset.args.audio_path)
    S = cfg.dataset.num_samples # choose the number of samples to generate
    x = torch.randn(S, 1, 640, 88)    
    
    if cfg.task.sampling.type=='inpainting_ddpm_x0':
        if cfg.dataset.name in ['MAESTRO', 'MAPS']:
            dataset = getattr(MusicDataset, cfg.dataset.name)(**OmegaConf.to_container(cfg.dataset.args, resolve=True))
            waveform = torch.empty(S, cfg.dataset.args.sequence_length)
            roll_labels = torch.empty(S, 640, 88)
            for i in range(S):
                sample = dataset[i]
                waveform[i] = sample['audio']
                roll_labels[i] = sample['frame']
            dataset = TensorDataset(x, waveform, roll_labels)
        elif cfg.dataset.name in ['Custom']:
            dataset = Custom(**cfg.dataset.args)
        else:
            pass
        
    elif cfg.task.sampling.type=='generation_ddpm_x0':
        waveform = torch.randn(S, 327680)
        dataset = TensorDataset(x, waveform)

    if len(dataset) < cfg.dataloader.batch_size:
        warnings.warn(f"Batch size is larger than total number of audio clips. Forcing batch size to {len(dataset)}")
    loader = DataLoader(dataset, **cfg.dataloader)

    # Model
    if cfg.task.frame_threshold!=None:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(to_absolute_path(cfg.checkpoint_path),
                                                                    sampling=cfg.task.sampling,
                                                                    frame_threshold=cfg.task.frame_threshold,
                                                                    generation_filter=cfg.task.generation_filter,
                                                                    inpainting_t=cfg.task.inpainting_t,
                                                                    inpainting_f=cfg.task.inpainting_f)
    else:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(to_absolute_path(cfg.checkpoint_path),
                                                                    sampling=cfg.task.sampling,
                                                                    generation_filter=cfg.task.generation_filter,
                                                                    inpainting_t=cfg.task.inpainting_t,
                                                                    inpainting_f=cfg.task.inpainting_f)
    
    name = f"Generation-{cfg.model.name}-k={cfg.model.args.kernel_size}"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)    

    trainer = pl.Trainer(**cfg.trainer,
                         logger=logger)
    
    trainer.predict(model, loader)
    
if __name__ == "__main__":
    main()