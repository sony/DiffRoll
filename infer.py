from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path

from model.unet import Unet

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from AudioLoader.music.amt import MAPS, MAESTRO

@hydra.main(config_path="config", config_name="infer")
def main(cfg):
    infer_samples = 8
    infer_set = torch.utils.data.TensorDataset(
        torch.randn(
            (infer_samples,
             1,
             *cfg.shape
            )
        )
    )
        
    infer_loader =  DataLoader(infer_set, batch_size=infer_samples)

    # Model
    model = Unet.load_from_checkpoint(to_absolute_path(cfg.checkpoint_path))

      
    
    name = f"Infer-diffusion_dim={model.hparams.dim}-" \
           f"channels={model.hparams.channels}-MAESTRO'"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)    

    trainer = pl.Trainer(**cfg.trainer,
                         logger=logger)
    
    trainer.predict(model, infer_loader)
    
if __name__ == "__main__":
    main()