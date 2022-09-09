from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path
import model as Model
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import AudioLoader.music.amt as MusicDataset
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="unsupervised_pretrained")
def main(cfg):
    cfg.data_root = to_absolute_path(cfg.data_root)
    cfg.pretrained_path = to_absolute_path(cfg.pretrained_path)
    cfg.dataset.train = OmegaConf.to_container(cfg.dataset.train, resolve=True) # convert Omega list into python list
    # Otherwise groups argument won't work in MAPS dataset
    
    train_set = getattr(MusicDataset, cfg.dataset.name)(**OmegaConf.to_container(cfg.dataset.train, resolve=True))
    val_set = getattr(MusicDataset, cfg.dataset.name)(**cfg.dataset.val)
    test_set = getattr(MusicDataset, cfg.dataset.name)(**cfg.dataset.test)
        
    train_loader = DataLoader(train_set, **cfg.dataloader.train)
    val_loader = DataLoader(val_set, **cfg.dataloader.val)
    test_loader = DataLoader(test_set, **cfg.dataloader.test)
    
    

    # Model
    model = getattr(Model, cfg.model.name).load_from_checkpoint(checkpoint_path=cfg.pretrained_path,\
                                                                **cfg.model.args, spec_args=cfg.spec.args, **cfg.task)
    
    if cfg.model.name == 'DiffRollBaseline':
        name = f"Pretrained-{cfg.model.name}-L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
               f"t={cfg.task.time_mode}-x_t={cfg.task.x_t}-{cfg.dataset.name}"
    elif cfg.model.name == 'ClassifierFreeDiffRoll':
        name = f"Pretrained-{cfg.model.name}-L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
               f"beta{cfg.task.beta_end}-{cfg.task.training.mode}-" + \
               f"{cfg.task.sampling.type}-w={cfg.task.sampling.w}-" + \
               f"p={cfg.model.args.spec_dropout}-{cfg.dataset.name}"             
    else:
        name = f"Pretrained-{cfg.model.name}-{cfg.task.sampling.type}-L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
               f"beta{cfg.task.beta_end}-{cfg.task.training.mode}-" + \
               f"dilation{cfg.model.args.dilation_base}-{cfg.task.loss_type}-{cfg.dataset.name}"    

    optimizer = Adam(model.parameters(), lr=1e-3)
    
    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint)    
    
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)    

    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=[checkpoint_callback,],
                         logger=logger)
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
if __name__ == "__main__":
    main()