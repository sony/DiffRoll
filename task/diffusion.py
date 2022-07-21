import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """
    x_start: x0
    t: timestep information
    """    
    
    # sqrt_alphas is mean of the Gaussian N()    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape) # extract the value of \bar{\alpha} at time=t
    # sqrt_alphas is variance of the Gaussian N()
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )


    # scale down the input, and scale up the noise as time increases?
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def extract_x0(x_t, epsilon, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    x_t: The output from q_sample
    epsilon: The noise predicted from the model
    t: timestep information
    """
    
    # sqrt_alphas is mean of the Gaussian N()    
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_t.shape) # extract the value of \bar{\alpha} at time=t
    # sqrt_alphas is variance of the Gaussian N()
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_t.shape
    )

    # obtaining x0 based on the inverse of eq.4 of DDPM paper
    return (x_t - sqrt_one_minus_alphas_cumprod_t * epsilon) / sqrt_alphas_cumprod_t


def extract(a, t, x_shape):
    # extract alpha at timestep=t
    # t should be an index
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)




# Algorithm 2 (including returning all images)
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, *image_size))


class RollDiffusion(pl.LightningModule):
    def __init__(self,
                 lr,
                 timesteps,
                 loss_type
                ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # define beta schedule
        # beta is variance
        self.betas = linear_beta_schedule(timesteps=timesteps)

        # define alphas 
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    def training_step(self, batch, batch_idx):
        batch_size = batch["frame"].shape[0]
        batch = batch["frame"].unsqueeze(1)  
        device = batch.device
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long()

        loss = self.p_losses(batch, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, loss_type=self.hparams.loss_type)
        self.log("Train/loss", loss)            
      
        return loss
    
    def validation_step(self, batch, batch_idx):
        batch_size = batch["frame"].shape[0]
        batch = batch["frame"].unsqueeze(1)  
        device = batch.device
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long()

        loss = self.p_losses(batch, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, loss_type=self.hparams.loss_type)
        self.log("Val/loss", loss)
        
    def test_step(self, batch, batch_idx):
        batch_size = batch["frame"].shape[0]
        batch = batch["frame"].unsqueeze(1)
        device = batch.device
        
        
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long()

        loss = self.p_losses(batch, t, self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, loss_type=self.hparams.loss_type)
        self.log("Test/loss", loss)        
        
    def predict_step(self, batch, batch_idx):
        # inference code
        # Unwrapping TensorDataset (list)
        # It is a pure noise
        img = batch[0]
        b = img.shape[0] # extracting batchsize
        device=img.device
        imgs = []
        for i in tqdm(reversed(range(0, self.hparams.timesteps)), desc='sampling loop time step', total=self.hparams.timesteps):
            img = self.p_sample(
                           img,
                           torch.full(
                               (b,),
                               i,
                               device=device,
                               dtype=torch.long),
                           i)
            img_npy = img.cpu().numpy()
            
            if (i+1)%10==0:
                for idx, j in enumerate(img_npy):
                    # j (1, T, F)
                    fig, ax = plt.subplots(1,1)
                    ax.imshow(j[0].T, aspect='auto', origin='lower')
                    self.logger.experiment.add_figure(
                        f"sample_{idx}",
                        fig,
                        global_step=self.hparams.timesteps-i)
                    # self.hparams.timesteps-i is used because slide bar won't show
                    # if global step starts from self.hparams.timesteps
            imgs.append(img_npy)
        torch.save(imgs, 'imgs.pt')
    
    def p_losses(self, x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = q_sample(x_start=x_start, t=t, sqrt_alphas_cumprod=sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod, noise=noise)
        predicted_noise = self(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss
    
    def p_sample(self, x, t, t_index):
        # x is Guassian noise
        
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise             
 
    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#         scheduler = TriStageLRSchedule(optimizer,
#                                        [1e-8, self.hparams.lr, 1e-8],
#                                        [0.2,0.6,0.2],
#                                        max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
#         scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

#         return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]







class SpecRollDiffusion(pl.LightningModule):
    def __init__(self,
                 lr,
                 timesteps,
                 loss_type,
                 loss_keys
                ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # define beta schedule
        # beta is variance
        self.betas = linear_beta_schedule(timesteps=timesteps)

        # define alphas 
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1- alphas_cumprod)

    def training_step(self, batch, batch_idx):
        losses, tensors = self.step(batch)
        self.log("Train/diffusion_loss", losses['diffusion_loss'])
        self.log("Train/amt_loss", losses['amt_loss'])
        
        # calculating total loss based on keys give
        total_loss = 0
        for k in self.hparams.loss_keys:
            total_loss += losses[k]
            
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        losses, tensors = self.step(batch)
        self.log("Val/diffusion_loss", losses['diffusion_loss'])
        self.log("Val/amt_loss", losses['amt_loss'])
        
        if batch_idx == 0:
            self.visualize_figure(tensors['pred_roll'], 'Val/pred_roll', batch_idx)
            self.visualize_figure(tensors['label_roll'], 'Val/label_roll', batch_idx)
            
    def test_step(self, batch, batch_idx):
        losses, tensors = self.step(batch)
        self.log("Test/diffusion_loss", losses['diffusion_loss'])
        self.log("Test/amt_loss", losses['amt_loss'])
        
        if batch_idx == 0:
            self.visualize_figure(tensors['pred_roll'], 'Test/pred_roll', batch_idx)
            self.visualize_figure(tensors['pred_roll']>0.6, 'Test/binary_roll', batch_idx)
            self.visualize_figure(tensors['label_roll'], 'Test/label_roll', batch_idx)
        
    def visualize_figure(self, tensors, tag, batch_idx):
        fig, ax = plt.subplots(2,2)
        for idx in range(4): # visualize only 4 piano rolls
            # roll_pred (1, T, F)
            ax.flatten()[idx].imshow(tensors[idx][0].T.cpu(), aspect='auto', origin='lower')
        self.logger.experiment.add_figure(f"{tag}{idx}", fig, global_step=self.current_epoch)
        plt.close()
        
    def step(self, batch):
        batch_size = batch["frame"].shape[0]
        roll = batch["frame"].unsqueeze(1)
        waveform = batch["audio"]
        device = roll.device
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long()

        noise = torch.randn_like(roll) # creating label noise
        
        x_t = q_sample( # sampling noise at time t
            x_start=roll,
            t=t,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            noise=noise)
        
        epsilon_pred = self(x_t, waveform, t) # predict the noise at previous step (t-1)
        diffusion_loss = self.p_losses(noise, epsilon_pred, loss_type=self.hparams.loss_type)
        
        pred_roll = extract_x0(
            x_t,
            epsilon_pred,
            t,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod)
        
        pred_roll = torch.sigmoid(pred_roll) # to convert logit into probability
        amt_loss = F.binary_cross_entropy(pred_roll, roll)
        
        losses = {
            "diffusion_loss": diffusion_loss,
            "amt_loss": amt_loss
        }
        
        tensors = {
            "pred_roll": pred_roll,
            "label_roll": roll
        }
        
        return losses, tensors    
        
    def p_losses(self, label, prediction, loss_type="l1"):
        if loss_type == 'l1':
            loss = F.l1_loss(label, prediction)
        elif loss_type == 'l2':
            loss = F.mse_loss(label, prediction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(label, prediction)
        else:
            raise NotImplementedError()

        return loss
    

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#         scheduler = TriStageLRSchedule(optimizer,
#                                        [1e-8, self.hparams.lr, 1e-8],
#                                        [0.2,0.6,0.2],
#                                        max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
#         scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

#         return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]
