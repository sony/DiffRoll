import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from .utils import extract_notes_wo_velocity
from mir_eval.transcription import precision_recall_f1_overlap as evaluate_notes
from mir_eval.util import midi_to_hz
import numpy as np
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
MIN_MIDI = 21
# from model.utils import Normalization
def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)

def q_sample(x_start, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, noise=None):
    """
    x_start: x0 (B, 1, T, F)
    t: timestep information (B,)
    """    
    # sqrt_alphas is mean of the Gaussian N()    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t]# extract the value of \bar{\alpha} at time=t
    # sqrt_alphas is variance of the Gaussian N()
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    
    # boardcasting into correct shape
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None].to(x_start.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None].to(x_start.device)

    # scale down the input, and scale up the noise as time increases?
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def extract_x0(x_t, epsilon, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """
    x_t: The output from q_sample
    epsilon: The noise predicted from the model
    t: timestep information
    """
    # sqrt_alphas is mean of the Gaussian N()    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t] # extract the value of \bar{\alpha} at time=t
    # sqrt_alphas is variance of the Gaussian N()
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t]
    
    sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t[:, None, None, None].to(x_t.device)
    sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t[:, None, None, None].to(x_t.device)    

    # obtaining x0 based on the inverse of eq.4 of DDPM paper
    return (x_t - sqrt_one_minus_alphas_cumprod_t * epsilon) / sqrt_alphas_cumprod_t


class RollDiffusion(pl.LightningModule):
    def __init__(self,
                 lr,
                 timesteps,
                 loss_type,
                 beta_start,
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
    
    def p_sample(self, x, t_index):
        # x is Guassian noise
        
        # extracting coefficients at time t
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index]

        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self(x, t_tensor) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            posterior_variance_t = self.posterior_variance[t_index]
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
                 loss_keys,
                 beta_start,
                 beta_end,                 
                 frame_threshold,
                 training,
                 sampling,
                 debug=False
                ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # define beta schedule
        # beta is variance
        self.betas = linear_beta_schedule(beta_start, beta_end, timesteps=timesteps)

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
        self.inner_loop = tqdm(range(self.hparams.timesteps), desc='sampling loop time step')
        
        self.reverse_diffusion = getattr(self, sampling.type)
        self.alphas = alphas

    def training_step(self, batch, batch_idx):
        losses, tensors = self.step(batch)
        self.log("Train/diffusion_loss", losses['diffusion_loss'])
        # self.log("Train/amt_loss", losses['amt_loss'])
        
        # calculating total loss based on keys give
        total_loss = 0
        for k in self.hparams.loss_keys:
            total_loss += losses[k]
            
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        losses, tensors = self.step(batch)
        self.log("Val/diffusion_loss", losses['diffusion_loss'])
        # self.log("Val/amt_loss", losses['amt_loss'])
        
        if batch_idx == 0:
            self.visualize_figure(tensors['pred_roll'], 'Val/pred_roll', batch_idx)
            
            if hasattr(self.hparams, 'condition'): # the condition for classifier free
                if self.hparams.condition == 'trainable_spec':
                    fig, ax = plt.subplots(1,1)
                    im = ax.imshow(self.trainable_parameters.detach().cpu(), aspect='auto', origin='lower', cmap='jet')
                    fig.colorbar(im, orientation='vertical')
                    self.logger.experiment.add_figure(f"Val/trainable_uncon", fig, global_step=self.current_epoch)
                    plt.close()
                
                # if self.hparams.condition == 'trainable_z':
                #     for idx, res_layer in enumerate(self.residual_layers):
                #         fig, ax = plt.subplots(1,1)
                #         im = ax.imshow(res_layer.uncon_z.detach().cpu(), aspect='auto', origin='lower', cmap='jet')
                #         fig.colorbar(im, orientation='vertical')
                #         self.logger.experiment.add_figure(f"Val/trainable_z{idx}", fig, global_step=self.current_epoch)
                #         plt.close()                        
            
            if self.current_epoch == 0: 
                self.visualize_figure(tensors['label_roll'], 'Val/label_roll', batch_idx)
                if self.hparams.unconditional==False and tensors['spec']!=None:
                    self.visualize_figure(tensors['spec'].transpose(-1,-2).unsqueeze(1),
                                          'Val/spec',
                                          batch_idx)
    def test_step(self, batch, batch_idx):
        noise_list, spec = self.sampling(batch, batch_idx)
    
        
        # noise_list is a list of tuple (pred_t, t), ..., (pred_0, 0)
        roll_pred = noise_list[-1][0] # (B, 1, T, F)        
        roll_label = batch["frame"].unsqueeze(1).cpu()
        
        if batch_idx==0:
            self.visualize_figure(spec.transpose(-1,-2).unsqueeze(1),
                                  'Test/spec',
                                  batch_idx)                
            for noise_npy, t_index in noise_list:
                if (t_index+1)%10==0: 
                    fig, ax = plt.subplots(2,2)
                    for idx, j in enumerate(noise_npy):
                        # j (1, T, F)
                        ax.flatten()[idx].imshow(j[0].T, aspect='auto', origin='lower')
                        self.logger.experiment.add_figure(
                            f"Test/pred",
                            fig,
                            global_step=self.hparams.timesteps-t_index)
                        plt.close()

            fig1, ax1 = plt.subplots(2,2)
            fig2, ax2 = plt.subplots(2,2)
            for idx in range(4):

                ax1.flatten()[idx].imshow(roll_label[idx][0].T, aspect='auto', origin='lower')
                self.logger.experiment.add_figure(
                    f"Test/label",
                    fig1,
                    global_step=0)

                ax2.flatten()[idx].imshow((roll_pred[idx][0]>self.hparams.frame_threshold).T, aspect='auto', origin='lower')
                self.logger.experiment.add_figure(
                    f"Test/pred_roll",
                    fig2,
                    global_step=0)  
                plt.close()            

            torch.save(noise_list, 'noise_list.pt')
            
            #======== Begins animation ===========
            t_list = torch.arange(1, self.hparams.timesteps, 5).flip(0)
            if t_list[-1] != self.hparams.timesteps:
                t_list = torch.cat((t_list, torch.tensor([self.hparams.timesteps])), 0)
            ims = []
            fig, axes = plt.subplots(2,4, figsize=(16, 5))

            title = axes.flatten()[0].set_title(None, fontsize=15)
            ax_flat = axes.flatten()
            caxs = []
            for ax in axes.flatten():
                div = make_axes_locatable(ax)
                caxs.append(div.append_axes('right', '5%', '5%'))

            ani = animation.FuncAnimation(fig,
                                          self.animate_sampling,
                                          frames=tqdm(t_list, desc='Animating'),
                                          fargs=(fig, ax_flat, caxs, noise_list, ),                                          
                                          interval=500,                                          
                                          blit=False,
                                          repeat_delay=1000)
            ani.save('algo2.gif', dpi=80, writer='imagemagick')
            #======== Animation saved ===========
            
            
        frame_p, frame_r, frame_f1, _ = precision_recall_fscore_support(roll_label.flatten(),
                                                                        roll_pred.flatten()>self.hparams.frame_threshold,
                                                                        average='binary')
        
        for roll_pred_i, roll_label_i in zip(roll_pred, roll_label.numpy()):
            # roll_pred (B, 1, T, F)
            p_est, i_est = extract_notes_wo_velocity(roll_pred_i[0],
                                                     roll_pred_i[0],
                                                     onset_threshold=self.hparams.frame_threshold,
                                                     frame_threshold=self.hparams.frame_threshold,
                                                     rule='rule1'
                                                    )
            
            p_ref, i_ref = extract_notes_wo_velocity(roll_label_i[0],
                                                     roll_label_i[0],
                                                     onset_threshold=self.hparams.frame_threshold,
                                                     frame_threshold=self.hparams.frame_threshold,
                                                     rule='rule1'
                                                    )            
            
            scaling = self.hparams.spec_args.hop_length / self.hparams.spec_args.sample_rate
            # scaling = HOP_LENGTH / SAMPLE_RATE

            # Converting time steps to seconds and midi number to frequency
            i_ref = (i_ref * scaling).reshape(-1, 2)
            p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
            i_est = (i_est * scaling).reshape(-1, 2)
            p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

            p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)            
        
            self.log("Test/Note_F1", f)         
        self.log("Test/Frame_F1", frame_f1)
        
        
#     def test_step(self, batch, batch_idx):
#         """This is for directly predicting x0"""
        
#         batch_size = batch["frame"].shape[0]
#         roll_label = self.normalize(batch["frame"]).unsqueeze(1)
#         waveform = batch["audio"]
#         device = roll_label.device
#         # Algorithm 1 line 3: sample t uniformally for every example in the batch

#         t_index=199
#         t_tensor = torch.tensor(t_index).repeat(batch_size).to(device)
        
#         # Equation 11 in the paper
#         noise = torch.randn_like(roll_label)
#         roll_pred, spec = self(noise, waveform, t_tensor)
        
#         roll_pred = roll_pred.cpu()
#         roll_label = roll_label.cpu()
        
#         if batch_idx==0:
#             self.visualize_figure(spec.transpose(-1,-2).unsqueeze(1),
#                                   'Test/spec',
#                                   batch_idx)                

#             fig, ax = plt.subplots(2,2)
#             fig1, ax1 = plt.subplots(2,2)
#             fig2, ax2 = plt.subplots(2,2)
#             for idx in range(4):
#                 ax.flatten()[idx].imshow(roll_pred[idx][0].T.detach(), aspect='auto', origin='lower')
#                 self.logger.experiment.add_figure(
#                     f"Test/pred",
#                     fig,
#                     global_step=self.hparams.timesteps-t_index)                

#                 ax1.flatten()[idx].imshow(roll_label[idx][0].T, aspect='auto', origin='lower')
#                 self.logger.experiment.add_figure(
#                     f"Test/label",
#                     fig1,
#                     global_step=0)

#                 ax2.flatten()[idx].imshow((roll_pred[idx][0]>self.hparams.frame_threshold).T, aspect='auto', origin='lower')
#                 self.logger.experiment.add_figure(
#                     f"Test/pred_roll",
#                     fig2,
#                     global_step=0)  
#                 plt.close()            
            
            
#         frame_p, frame_r, frame_f1, _ = precision_recall_fscore_support(roll_label.flatten(),
#                                                                         roll_pred.flatten()>self.hparams.frame_threshold,
#                                                                         average='binary')
        
#         for roll_pred_i, roll_label_i in zip(roll_pred.numpy(), roll_label.numpy()):
#             # roll_pred (B, 1, T, F)
#             p_est, i_est = extract_notes_wo_velocity(roll_pred_i[0],
#                                                      roll_pred_i[0],
#                                                      onset_threshold=self.hparams.frame_threshold,
#                                                      frame_threshold=self.hparams.frame_threshold,
#                                                      rule='rule1'
#                                                     )
            
#             p_ref, i_ref = extract_notes_wo_velocity(roll_label_i[0],
#                                                      roll_label_i[0],
#                                                      onset_threshold=self.hparams.frame_threshold,
#                                                      frame_threshold=self.hparams.frame_threshold,
#                                                      rule='rule1'
#                                                     )            
            
#             scaling = self.hparams.spec_args.hop_length / self.hparams.spec_args.sample_rate
#             # scaling = HOP_LENGTH / SAMPLE_RATE

#             # Converting time steps to seconds and midi number to frequency
#             i_ref = (i_ref * scaling).reshape(-1, 2)
#             p_ref = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_ref])
#             i_est = (i_est * scaling).reshape(-1, 2)
#             p_est = np.array([midi_to_hz(MIN_MIDI + midi) for midi in p_est])

#             p, r, f, o = evaluate_notes(i_ref, p_ref, i_est, p_est, offset_ratio=None)            
        
#             self.log("Test/Note_F1", f)         
#         self.log("Test/Frame_F1", frame_f1)        
        
    def predict_step(self, batch, batch_idx): 
        if batch_idx==0:
            noise_list, spec = self.sampling(batch, batch_idx)
            # noise_list is a list of tuple (pred_t, t), ..., (pred_0, 0)
            torch.save(noise_list, 'noise_list.pt')
            


            #======== Begins animation ===========
            t_list = torch.arange(1, self.hparams.timesteps, 5).flip(0)
            if t_list[-1] != self.hparams.timesteps:
                t_list = torch.cat((t_list, torch.tensor([self.hparams.timesteps])), 0)
            ims = []
            fig, axes = plt.subplots(2,4, figsize=(16, 5))

            title = axes.flatten()[0].set_title(None, fontsize=15)
            ax_flat = axes.flatten()
            caxs = []
            for ax in axes.flatten():
                div = make_axes_locatable(ax)
                caxs.append(div.append_axes('right', '5%', '5%'))

            ani = animation.FuncAnimation(fig,
                                          self.animate_sampling,
                                          frames=tqdm(t_list, desc='Animating'),
                                          fargs=(fig, ax_flat, caxs, noise_list, ),                                          
                                          interval=500,                                          
                                          blit=False,
                                          repeat_delay=1000)
            ani.save('algo2.gif', dpi=80, writer='imagemagick')
            #======== Animation saved ===========              
        


            
    def visualize_figure(self, tensors, tag, batch_idx):
        fig, ax = plt.subplots(2,2)
        for idx in range(4): # visualize only 4 piano rolls
            # roll_pred (1, T, F)
            ax.flatten()[idx].imshow(tensors[idx][0].T.cpu(), aspect='auto', origin='lower')
        self.logger.experiment.add_figure(f"{tag}{idx}", fig, global_step=self.current_epoch)
        plt.close()
        
    def step(self, batch):
        # batch["frame"] (B, 640, 88)
        # batch["audio"] (B, L)
        
        batch_size = batch["frame"].shape[0]
        roll = self.normalize(batch["frame"]).unsqueeze(1) 
        waveform = batch["audio"]
        device = roll.device
        
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ## sampling the same t within each batch, might not work well
        # t = torch.randint(0, self.hparams.timesteps, (1,), device=device)[0].long() # [0] to remove dimension
        # t_tensor = t.repeat(batch_size).to(roll.device)
        
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=device).long() # more diverse sampling
        

        noise = torch.randn_like(roll) # creating label noise
        
        x_t = q_sample( # sampling noise at time t
            x_start=roll,
            t=t,
            sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod,
            noise=noise)
        
        
        
        # When debugging model is use, change waveform into roll
        if self.hparams.training.mode == 'epsilon':
            if self.hparams.debug==True:
                epsilon_pred, spec = self(x_t, roll, t) # predict the noise N(0, 1)
            else:
                epsilon_pred, spec = self(x_t, waveform, t) # predict the noise N(0, 1)
            diffusion_loss = self.p_losses(noise, epsilon_pred, loss_type=self.hparams.loss_type)

            pred_roll = extract_x0(
                x_t,
                epsilon_pred,
                t,
                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod)
            
        elif self.hparams.training.mode == 'x_0':
            pred_roll, spec = self(x_t, waveform, t) # predict the noise N(0, 1)
            diffusion_loss = self.p_losses(roll, pred_roll, loss_type=self.hparams.loss_type)
            
        elif self.hparams.training.mode == 'ex_0':
            epsilon_pred, spec = self(x_t, waveform, t) # predict the noise N(0, 1)
            pred_roll = extract_x0(
                x_t,
                epsilon_pred,
                t,
                sqrt_alphas_cumprod=self.sqrt_alphas_cumprod,
                sqrt_one_minus_alphas_cumprod=self.sqrt_one_minus_alphas_cumprod)            
            diffusion_loss = self.p_losses(roll, pred_roll, loss_type=self.hparams.loss_type)   
            
        
        else:
            raise ValueError(f"training mode {self.training.mode} is not supported. Please either use 'x_0' or 'epsilon'.")
        
        # pred_roll = torch.sigmoid(pred_roll) # to convert logit into probability
        # amt_loss = F.binary_cross_entropy(pred_roll, roll)
        
        losses = {
            "diffusion_loss": diffusion_loss,
            # "amt_loss": amt_loss
        }
        
        tensors = {
            "pred_roll": pred_roll,
            "label_roll": roll,
            "spec": spec
        }
        
        return losses, tensors
    
    def sampling(self, batch, batch_idx):
        batch_size = batch["frame"].shape[0]
        roll = self.normalize(batch["frame"]).unsqueeze(1)
        waveform = batch["audio"]
        device = roll.device
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        
        self.inner_loop.refresh()
        self.inner_loop.reset()
        
        noise = torch.randn_like(roll)
        noise_list = []
        noise_list.append((noise, self.hparams.timesteps))

        for t_index in reversed(range(0, self.hparams.timesteps)):
            if self.hparams.debug==True:
                noise, spec = self.reverse_diffusion(noise, roll, t_index)
            else:
                noise, spec = self.reverse_diffusion(noise, waveform, t_index)
            noise_npy = noise.detach().cpu().numpy()
                    # self.hparams.timesteps-i is used because slide bar won't show
                    # if global step starts from self.hparams.timesteps
            noise_list.append((noise_npy, t_index))                       
            self.inner_loop.update()
        
        return noise_list, spec
        
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
    
    def ddpm(self, x, waveform, t_index):
        # x is Guassian noise
        
        # extracting coefficients at time t
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index]

        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        epsilon, spec = self(x, waveform, t_tensor)
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * epsilon / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0:
            return model_mean, spec
        else:
            # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return (model_mean + torch.sqrt(posterior_variance_t) * noise), spec
        
    def ddpm_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred, spec = self(x, waveform, t_tensor)

        if t_index == 0:
            sigma = (1/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))            
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index-1]/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))                    
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec
    
    def ddim_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred, spec = self(x, waveform, t_tensor)

        if t_index == 0:
            sigma = 0
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = 0                 
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec               
        
    def ddim(self, x, waveform, t_index):
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        epsilon, spec = self(x, waveform, t_tensor)
        
        if t_index == 0:
            model_mean = (x - self.sqrt_one_minus_alphas_cumprod[t_index] * epsilon) / self.sqrt_alphas_cumprod[t_index] 
        else:
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * (
                (x - self.sqrt_one_minus_alphas_cumprod[t_index] * epsilon) / self.sqrt_alphas_cumprod[t_index]) + (
                self.sqrt_one_minus_alphas_cumprod[t_index-1] * epsilon)
            
        return model_mean, spec

    def ddim2ddpm(self, x, waveform, t_index):
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        epsilon, spec = self(x, waveform, t_tensor)

        if t_index == 0:   
            model_mean = (x - self.sqrt_one_minus_alphas_cumprod[t_index] * epsilon) / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index-1]/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))                    
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * (
                (x - self.sqrt_one_minus_alphas_cumprod[t_index] * epsilon) / self.sqrt_alphas_cumprod[t_index]) + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * epsilon) + sigma * torch.randn_like(x)

        return model_mean, spec           
        
#     def cfdg_ddpm(self, x, waveform, t_index):
#         # x is Guassian noise
        
#         # extracting coefficients at time t
#         betas_t = self.betas[t_index]
#         sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
#         sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index]

#         # boardcasting t_index into a tensor
#         t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
#         # Equation 11 in the paper
#         # Use our model (noise predictor) to predict the mean 
#         epsilon_c, spec = self(x, waveform, t_tensor)
#         epsilon_0, _ = self(x, torch.zeros_like(waveform), t_tensor)
#         epsilon = (1+self.hparams.sampling.w)*epsilon_c - self.hparams.sampling.w*epsilon_0
        
#         model_mean = sqrt_recip_alphas_t * (
#             x - betas_t * epsilon / sqrt_one_minus_alphas_cumprod_t
#         )
#         if t_index == 0:
#             return model_mean, spec
#         else:
#             # posterior_variance_t = extract(self.posterior_variance, t, x.shape)
#             posterior_variance_t = self.posterior_variance[t_index]
#             noise = torch.randn_like(x)
#             # Algorithm 2 line 4:
#             return (model_mean + torch.sqrt(posterior_variance_t) * noise), spec     
        
        
    def cfdg_ddpm_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred_c, spec = self(x, waveform, t_tensor)
        x0_pred_0, _ = self(x, torch.zeros_like(waveform), t_tensor, sampling=True) # if sampling = True, the input condition will be overwritten
        x0_pred = (1+self.hparams.sampling.w)*x0_pred_c - self.hparams.sampling.w*x0_pred_0
#         x0_pred = x0_pred_c
        # x0_pred = x0_pred_0

        if t_index == 0:
            sigma = (1/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))            
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = (self.sqrt_one_minus_alphas_cumprod[t_index-1]/self.sqrt_one_minus_alphas_cumprod[t_index]) * (
                torch.sqrt(1-self.alphas[t_index]))                    
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec           
    
    
    def cfdg_ddim_x0(self, x, waveform, t_index):
        # x is x_t, when t=T it is pure Gaussian
        
        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred, spec = self(x, waveform, t_tensor)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred_c, spec = self(x, waveform, t_tensor)
        x0_pred_0, _ = self(x, torch.zeros_like(waveform), t_tensor)
        x0_pred = (1+self.hparams.sampling.w)*x0_pred_c - self.hparams.sampling.w*x0_pred_0
#         x0_pred = x0_pred_c
        # x0_pred = x0_pred_0

        if t_index == 0:
            sigma = 0 
            model_mean = x0_pred / self.sqrt_alphas_cumprod[t_index] 
        else:
            sigma = 0          
            model_mean = (self.sqrt_alphas_cumprod[t_index-1]) * x0_pred + (
                torch.sqrt(1 - self.sqrt_alphas_cumprod[t_index-1]**2 - sigma**2) * (
                    x-self.sqrt_alphas_cumprod[t_index]* x0_pred)/self.sqrt_one_minus_alphas_cumprod[t_index]) + (
                sigma * torch.randn_like(x))

        return model_mean, spec            

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#         scheduler = TriStageLRSchedule(optimizer,
#                                        [1e-8, self.hparams.lr, 1e-8],
#                                        [0.2,0.6,0.2],
#                                        max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
#         scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

#         return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]

    def animate_sampling(self, t_idx, fig, ax_flat, caxs, noise_list):
        # Tuple of (x_t, t), (x_t-1, t-1), ... (x_0, 0)
        # x_t (B, 1, T, F)
        # clearing figures to prevent slow down in each iteration.d
        fig.canvas.draw()
        for idx in range(4): # visualize only 4 piano rolls
            ax_flat[idx].cla()
            ax_flat[4+idx].cla()
            caxs[idx].cla()
            caxs[4+idx].cla()     

            # roll_pred (1, T, F)
            im1 = ax_flat[idx].imshow(noise_list[0][0][idx][0].detach().T.cpu(), aspect='auto', origin='lower')
            im2 = ax_flat[4+idx].imshow(noise_list[1+self.hparams.timesteps-t_idx][0][idx][0].T, aspect='auto', origin='lower')
            fig.colorbar(im1, cax=caxs[idx])
            fig.colorbar(im2, cax=caxs[4+idx])

        fig.suptitle(f't={t_idx}')
        row1_txt = ax_flat[0].text(-400,45,f'Gaussian N(0,1)')
        row2_txt = ax_flat[4].text(-300,45,'x_{t-1}')    