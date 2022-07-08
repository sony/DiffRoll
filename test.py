
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

@hydra.main(config_path="config", config_name="test")
def main(cfg):       
    cfg.data_root = to_absolute_path(cfg.data_root)

    test_set = MAESTRO(**cfg.dataset.test)
        
    test_loader = DataLoader(test_set, batch_size=4)

    # Model
    model = Unet.load_from_checkpoint(to_absolute_path(cfg.checkpoint_path))
    
    name = f"Test-" \
           f"channels={model.hparams.channels}-" \
           f"{model.hparams.loss_type}-" \
           f"MAESTRO"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)    

    trainer = pl.Trainer(**cfg.trainer,
                         logger=logger)
    
    trainer.test(model, test_loader)
    
if __name__ == "__main__":
    main()