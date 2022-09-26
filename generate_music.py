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

from AudioLoader.music.amt import MAPS, MAESTRO

@hydra.main(config_path="config", config_name="generate")
def main(cfg):       
    cfg.data_root = to_absolute_path(cfg.data_root)
    
    S = cfg.dataset.num_samples # choose the number of samples to generate
    
    x = torch.randn(S, 1, 640, 88)
    waveform = torch.randn(S, 327680)
    dataset = TensorDataset(x, waveform)
        
    loader = DataLoader(dataset, **cfg.dataloader)

    # Model
    if cfg.task.frame_threshold!=None:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(to_absolute_path(cfg.checkpoint_path), frame_threshold=cfg.task.frame_threshold)
    else:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(to_absolute_path(cfg.checkpoint_path))
    
    name = f"Generation-{cfg.model.name}-k={cfg.args.kernel_size}" \
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)    

    trainer = pl.Trainer(**cfg.trainer,
                         logger=logger)
    
    trainer.predict(model, loader)
    
if __name__ == "__main__":
    main()