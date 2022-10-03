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

from AudioLoader.music.amt import MAPS, MAESTRO
from omegaconf import OmegaConf

@hydra.main(config_path="config", config_name="sampling")
def main(cfg):       
    cfg.data_root = to_absolute_path(cfg.data_root)
    S = cfg.dataset.num_samples # choose the number of samples to generate
    x = torch.randn(S, 1, 640, 88)    
    
    if cfg.task.sampling.type=='inpainting_ddpm_x0':
        dataset = getattr(MusicDataset, cfg.dataset.name)(**OmegaConf.to_container(cfg.dataset.args, resolve=True))
        waveform = torch.empty(S, cfg.dataset.args.sequence_length)
        for i in range(S):
            waveform[i] = dataset[i]['audio']
        dataset = TensorDataset(x, waveform)
        
    elif cfg.task.sampling.type=='generation_ddpm_x0':
        waveform = torch.randn(S, 327680)
        dataset = TensorDataset(x, waveform)

    loader = DataLoader(dataset, **cfg.dataloader)

    # Model
    if cfg.task.frame_threshold!=None:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(to_absolute_path(cfg.checkpoint_path),
                                                                    sampling=cfg.task.sampling,
                                                                    frame_threshold=cfg.task.frame_threshold,
                                                                    generation_filter=cfg.task.generation_filter,
                                                                    inpainting=cfg.task.inpainting)
    else:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(to_absolute_path(cfg.checkpoint_path),
                                                                    sampling=cfg.task.sampling,
                                                                    generation_filter=cfg.task.generation_filter,
                                                                    inpainting=cfg.task.inpainting)
    
    name = f"Generation-{cfg.model.name}-k={cfg.model.args.kernel_size}"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)    

    trainer = pl.Trainer(**cfg.trainer,
                         logger=logger)
    
    trainer.predict(model, loader)
    
if __name__ == "__main__":
    main()