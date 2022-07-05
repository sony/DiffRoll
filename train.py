
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




# forward diffusion (using the nice property)


@hydra.main(config_path="config", config_name="pianoroll")
def main(cfg):       
    cfg.data_root = to_absolute_path(cfg.data_root)
    
    train_set = MAESTRO(**cfg.dataset.train)
    val_set = MAESTRO(**cfg.dataset.val)
    test_set = MAESTRO(**cfg.dataset.test)

    train_loader = DataLoader(train_set, batch_size=4, num_workers=16)
    val_loader = DataLoader(val_set, batch_size=4)
    test_loader = DataLoader(test_set, batch_size=4)

    # Model
    model = Unet(
        cfg.pl,        
        **cfg.model,
        dim_mults=(1, 2, 4,)
    )

    optimizer = Adam(model.parameters(), lr=1e-3)
    
    checkpoint_callback = ModelCheckpoint(monitor="Train/loss",
                                          filename="{epoch:02d}-{Train/loss:.2f}",
                                          save_top_k=2,
                                          mode="min",
                                          auto_insert_metric_name=False)    
    
    name = f"diffusion_dim={cfg.model.dim}-" \
           f"channels={cfg.model.channels}-MAESTRO'"
    logger = TensorBoardLogger(save_dir=".", version=1, name=name)    

    trainer = pl.Trainer(**cfg.trainer,
                         callbacks=[checkpoint_callback,],
                         logger=logger)
    
    trainer.fit(model, train_loader)


    
    # inference code
    # sample 64 images
    samples = sample(model, image_size=(312,88), batch_size=64, channels=1)

    import matplotlib.pyplot as plt 

    samples[-1].shape

    batch.shape

    samples[-1].shape

    # show a random one
    random_index = 5
    plt.imshow(samples[-1][random_index].squeeze(0).transpose(), aspect='auto')

    # show a random one
    random_index = 0
    plt.imshow(samples[-1][random_index].squeeze(0).transpose(), aspect='auto')
    
if __name__ == "__main__":
    main()