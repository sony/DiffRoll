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
    output = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


class SpecRollBaseline(pl.LightningModule):
    def __init__(self,
                 lr,
                 timesteps,
                 loss_keys,
                 beta_start,
                 beta_end,                 
                 frame_threshold,
                 norm_args,
                 time_mode,
                 x_t,
                 debug=False
                ):
        super().__init__()
        
        # === This part is for debugging====
        # define beta schedule
        # beta is variance
        self.timesteps = 200
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
        self.alphas = alphas
        
        # === End of debugging====
        
        
        
        self.save_hyperparameters()
        

    def training_step(self, batch, batch_idx):
        losses, tensors = self.step(batch)

        self.log("Train/amt_loss", losses['amt_loss'])
        
        # calculating total loss based on keys give
        total_loss = 0
        for k in self.hparams.loss_keys:
            total_loss += losses[k]
            
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        losses, tensors = self.step(batch)
        self.log("Val/amt_loss", losses['amt_loss'])
        
        if batch_idx == 0:
            self.visualize_figure(tensors['pred_roll'], 'Val/pred_roll', batch_idx)
            if self.current_epoch == 0: 
                self.visualize_figure(tensors['label_roll'], 'Val/label_roll', batch_idx)
                if self.hparams.unconditional==False and tensors['spec']!=None:
                    self.visualize_figure(tensors['spec'].transpose(-1,-2).unsqueeze(1),
                                          'Val/spec',
                                          batch_idx)
#     def test_step(self, batch, batch_idx):
#         losses, tensors = self.step(batch)
    
#         roll_pred = tensors['pred_roll'].detach().cpu() # (B, 1, T, F)
#         roll_label = batch["frame"].unsqueeze(1).cpu()
        
#         if batch_idx==0:
#             self.visualize_figure(tensors['spec'].cpu().transpose(-1,-2).unsqueeze(1),
#                                   'Test/spec',
#                                   batch_idx)                
#             fig, ax = plt.subplots(2,2)
#             for idx, j in enumerate(roll_pred):
#                 # j (1, T, F)
#                 ax.flatten()[idx].imshow(j[0].T, aspect='auto', origin='lower')
#                 self.logger.experiment.add_figure(
#                     f"Test/pred",
#                     fig,
#                     global_step=self.hparams.timesteps)
#                 plt.close()

#             fig1, ax1 = plt.subplots(2,2)
#             fig2, ax2 = plt.subplots(2,2)
#             for idx in range(4):

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
        
    def test_step(self, batch, batch_idx):
        """This is for ddpm"""
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
        
    def sampling(self, batch, batch_idx):
        batch_size = batch["frame"].shape[0]
        waveform = batch["audio"]
        roll = batch["frame"].unsqueeze(1)
        device = roll.device
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        
        noise = torch.randn_like(roll)
        noise_list = []
        noise_list.append((noise, self.hparams.timesteps))

        for t_index in reversed(range(0, self.hparams.timesteps)):
            noise, spec = self.reverse_diffusion(noise, waveform, t_index)
            noise_npy = noise.detach().cpu().numpy()
                    # self.hparams.timesteps-i is used because slide bar won't show
                    # if global step starts from self.hparams.timesteps
            noise_list.append((noise_npy, t_index))                       
        
        return noise_list, spec
    
    def reverse_diffusion(self, x, waveform, t_index):
        # x is Guassian noise
        
        # extracting coefficients at time t
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t_index]

        # boardcasting t_index into a tensor
        t_tensor = torch.tensor(t_index).repeat(x.shape[0]).to(x.device)
        
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean 
        x0_pred, spec = self(x, waveform, t_tensor)
        
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
        
        
    def predict_step(self, batch, batch_idx):
        def animate_sampling(t_idx):
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
            row1_txt = axes.flatten()[0].text(-400,45,f'Gaussian N(0,1)')
            row2_txt = axes.flatten()[4].text(-300,45,'x_{t-1}')            
            # row1_txt.set_text(f'x_t')
            # row2_txt.set_text(f'extracted x_0')        
        
        if batch_idx==0:
            noise_list, spec = self.sampling(batch, batch_idx)
            # noise_list is a list of tuple (pred_t, t), ..., (pred_0, 0)
            torch.save(noise_list, 'noise_list.pt')
            


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
                                          animate_sampling,
                                          frames=tqdm(t_list, desc='Animating'),
                                          interval=500,
                                          blit=False,
                                          repeat_delay=1000)

            ani.save('algo2.gif', dpi=80, writer='imagemagick')               
        


            
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
        roll = (batch["frame"]).unsqueeze(1) 
        waveform = batch["audio"]
        device = roll.device
        
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        ## sampling the same t within each batch, might not work well
        # t = torch.randint(0, self.hparams.timesteps, (1,), device=device)[0].long() # [0] to remove dimension
        # t_tensor = t.repeat(batch_size).to(roll.device)
        
        if self.hparams.time_mode == 'constant':
            t = torch.ones((batch_size,), device=device).long() # more diverse sampling
        if self.hparams.time_mode == 'constant_maxT':
            t = torch.full((batch_size,), self.hparams.timesteps-1, device=device).long() # more diverse sampling            
        elif self.hparams.time_mode == 'random':
            t = torch.randint(low=0,high=100,size=(batch_size,)) # more diverse sampling
        else:
            raise ValueError(f'{self.hparams.time_mode=} is not recognized')            
        
        if self.hparams.x_t == 'zeros':
            x_t = torch.zeros_like(roll) # dummy noise
        elif self.hparams.x_t == 'gaussian':
            x_t = torch.rand_like(roll)
        else:
            raise ValueError(f'{self.hparams.x_t=} is not recognized')
        
        pred_roll, spec = self(x_t, waveform, t) # predict the noise N(0, 1)
        amt_loss = torch.nn.functional.mse_loss(pred_roll, roll)
        
        # pred_roll = torch.sigmoid(pred_roll) # to convert logit into probability
        # amt_loss = F.binary_cross_entropy(pred_roll, roll)
        
        losses = {
            # "diffusion_loss": diffusion_loss,
            "amt_loss": amt_loss
        }
        
        tensors = {
            "pred_roll": pred_roll,
            "label_roll": roll,
            "spec": spec
        }
        
        return losses, tensors
        

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
#         scheduler = TriStageLRSchedule(optimizer,
#                                        [1e-8, self.hparams.lr, 1e-8],
#                                        [0.2,0.6,0.2],
#                                        max_update=len(self.train_dataloader.dataloader)*self.trainer.max_epochs)   
#         scheduler = MultiStepLR(optimizer, [1,3,5,7,9], gamma=0.1, last_epoch=-1, verbose=False)

#         return [optimizer], [{"scheduler":scheduler, "interval": "step"}]
        return [optimizer]
