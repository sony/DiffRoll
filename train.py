
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


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def p_losses(denoise_model, x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


def extract(a, t, x_shape):
    # extract alpha at timestep=t
    # t should be an index
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion (using the nice property)
def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    # sqrt_alphas is mean of the Gaussian N()    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape) # extract the value of \bar{\alpha} at time=t
    # sqrt_alphas is variance of the Gaussian N()
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )


    # scale down the input, and scale up the noise as time increases?
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Unet(
        dim=28,
        channels=1,
        dim_mults=(1, 2, 4,)
    )
    model.to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)


    timesteps = 200

    # define beta schedule
    betas = linear_beta_schedule(timesteps=timesteps)

    # define alphas 
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

    # calculations for diffusion q(x_t | x_{t-1}) and others
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    epochs = 5

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()

            batch_size = batch["frame"].shape[0]
            batch = batch["frame"].to(device).unsqueeze(1)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, loss_type="huber")

            if step % 100 == 0:
                print("Loss:", loss.item())

            loss.backward()
            optimizer.step()


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