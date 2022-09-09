from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import model as Model

# from AudioLoader.music.amt import MAPS, MAESTRO
import AudioLoader.music.amt as MusicDataset

@hydra.main(config_path="config", config_name="test")
def main(cfg):       
    cfg.data_root = to_absolute_path(cfg.data_root)

#     test_set = MAESTRO(**cfg.dataset.test)
    test_set = getattr(MusicDataset, cfg.dataset.name)(**cfg.dataset.test)
        
    test_loader = DataLoader(test_set, batch_size=4)
    

    # Model
    if cfg.task.frame_threshold!=None and cfg.task.sampling.type!=None:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(to_absolute_path(cfg.checkpoint_path), frame_threshold=cfg.task.frame_threshold, sampling=cfg.task.sampling)
    elif cfg.task.frame_threshold==None and cfg.task.sampling.type!=None:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(to_absolute_path(cfg.checkpoint_path), sampling=cfg.task.sampling)
    elif cfg.task.frame_threshold!=None and cfg.task.sampling.type==None:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(to_absolute_path(cfg.checkpoint_path), frame_threshold=cfg.task.frame_threshold)              
    else:
        model = getattr(Model, cfg.model.name).load_from_checkpoint(to_absolute_path(cfg.checkpoint_path))
    
    if cfg.model.name=='ClassifierFreeDiffRoll':
        name = f"Test-x0_pred_0-{cfg.model.name}-" \
               f"{cfg.task.sampling.type}-w{cfg.task.sampling.w}-{cfg.dataset.name}"
        logger = TensorBoardLogger(save_dir=".", version=1, name=name)        
    else:
        name = f"Test-x0_pred_0-{cfg.model.name}-" \
               f"{cfg.task.sampling.type}-{cfg.dataset.name}"
        logger = TensorBoardLogger(save_dir=".", version=1, name=name)

    trainer = pl.Trainer(**cfg.trainer,
                         logger=logger)
    
    trainer.test(model, test_loader)
    
if __name__ == "__main__":
    main()