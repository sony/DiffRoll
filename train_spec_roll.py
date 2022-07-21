
from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path

from model.unet import SpecUnet

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from AudioLoader.music.amt import MAPS, MAESTRO

@hydra.main(config_path="config", config_name="spec_roll")
def main(cfg):       
    cfg.data_root = to_absolute_path(cfg.data_root)
    
    train_set = MAESTRO(**cfg.dataset.train)
    val_set = MAESTRO(**cfg.dataset.val)
    test_set = MAESTRO(**cfg.dataset.test)
    
    infer_samples = 8
    infer_set = torch.utils.data.TensorDataset(
        torch.randn(
            (infer_samples,
             cfg.model.args.channels,
             *train_set[0]['frame'].shape
            )
        )
    )
        
    train_loader = DataLoader(train_set, **cfg.dataloader.train)
    val_loader = DataLoader(val_set, **cfg.dataloader.val)
    test_loader = DataLoader(test_set, **cfg.dataloader.test)
    infer_loader =  DataLoader(infer_set, batch_size=infer_samples)

    # Model
    model = SpecUnet(     
        **cfg.model.args,
        spec_args=cfg.spec.args,
        **cfg.task,
    )

    optimizer = Adam(model.parameters(), lr=1e-3)
    
    checkpoint_callback = ModelCheckpoint(**cfg.modelcheckpoint)    
    
    name = f"diffusion_dim={cfg.model.args.dim}-" \
           f"channels={cfg.model.args.channels}-" \
           f"{model.hparams.loss_type}-" \
           f"MAESTRO"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)    

    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=[checkpoint_callback,],
                         logger=logger)
    
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
if __name__ == "__main__":
    main()