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

class DoubleDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

def train_dataloader(self):
    concat_dataset = ConcatDataset(
        datasets.ImageFolder(traindir_A),
        datasets.ImageFolder(traindir_B)
    )

    loader = torch.utils.data.DataLoader(
        concat_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    return loader

@hydra.main(config_path="config", config_name="unsupervised_pretrained")
def main(cfg):
    # force it to train with two losses
    
    if len(cfg.task.loss_keys)==1:
        cfg.task.loss_keys = ['diffusion_loss', 'unconditional_diffusion_loss']
    
    cfg.data_root = to_absolute_path(cfg.data_root)
    cfg.pretrained_path = to_absolute_path(cfg.pretrained_path)
    
    # convert Omega list into python list
    # Otherwise groups argument won't work in MAPS dataset
    train_set1 = getattr(MusicDataset, cfg.dataset.name1)(**OmegaConf.to_container(cfg.dataset.train1, resolve=True))
    train_set2 = getattr(MusicDataset, cfg.dataset.name2)(**cfg.dataset.train2)
    concat_trainset = DoubleDataset(train_set1, train_set2)
    
    val_set1 = getattr(MusicDataset, cfg.dataset.name1)(**cfg.dataset.val1)
    val_set2 = getattr(MusicDataset, cfg.dataset.name2)(**cfg.dataset.val2)
    concat_valset = DoubleDataset(val_set1, val_set2)
    
    test_set = getattr(MusicDataset, cfg.dataset.name1)(**cfg.dataset.test)
    
        
    train_loader = DataLoader(concat_trainset, **cfg.dataloader.train)
    val_loader = DataLoader(concat_valset, **cfg.dataloader.val)
    test_loader = DataLoader(test_set, **cfg.dataloader.test)
    
    

    # Model
    model = getattr(Model, cfg.model.name).load_from_checkpoint(checkpoint_path=cfg.pretrained_path,\
                                                                **cfg.model.args, spec_args=cfg.spec.args, **cfg.task)
    
    if cfg.model.name == 'DiffRollBaseline':
        name = f"BothPretrained-{cfg.model.name}-L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
               f"t={cfg.task.time_mode}-x_t={cfg.task.x_t}-{cfg.dataset.name1}"
    elif cfg.model.name == 'ClassifierFreeDiffRoll':
        name = f"BothPretrained-{cfg.model.name}-L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
               f"beta{cfg.task.beta_end}-{cfg.task.training.mode}-" + \
               f"{cfg.task.sampling.type}-w={cfg.task.sampling.w}-" + \
               f"p={cfg.model.args.spec_dropout}-k={cfg.model.args.kernel_size}-" + \
               f"dia={cfg.model.args.dilation_base}-{cfg.model.args.dilation_bound}-"
    else:
        name = f"BothPretrained-{cfg.model.name}-{cfg.task.sampling.type}-L{cfg.model.args.residual_layers}-C{cfg.model.args.residual_channels}-" + \
               f"beta{cfg.task.beta_end}-{cfg.task.training.mode}-" + \
               f"dilation{cfg.model.args.dilation_base}-{cfg.task.loss_type}-{cfg.dataset.name1}"    

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