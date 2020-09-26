"""
    Model and Training boilerplate code for Generative and VAE Models
"""
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from RekogNizer.ganutils import tensor2img, get_sample_image, save_checkpoint, generate_image_sample, create_graphs_and_samples
from torch.optim.lr_scheduler import StepLR, OneCycleLR, MultiStepLR, CyclicLR
from matplotlib.pyplot import imshow, imsave
import pandas as pd

import os
import sys
import glob
from tqdm import tqdm_notebook as tqdm
import datetime
import numpy as np



################# Used Exclusively for GANs ##################
class R1GANResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None, groups=1):
        super(R1GANResidualBlock, self).__init__()
        p = kernel_size//2
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=p),
            #nn.LeakyReLU(0.2)
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size, padding=p),
            #nn.LeakyReLU(0.2)
            nn.ReLU()
        )
        self.proj = nn.Conv2d(inplanes, planes, 1) if inplanes != planes else None
    
    def forward(self, x):
        identity = x
        
        y = self.conv1(x)
        y = self.conv2(y)
        
        identity = identity if self.proj is None else self.proj(identity)
        y = y + identity
        return y


class R1GANDiscriminator(nn.Module):
    """
        Convolutional Discriminator
    """
    def __init__(self, in_channel=1,img_size=64):
        super(R1GANDiscriminator, self).__init__()
        self.dim_size = img_size//(2**4)
        self.D = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1), # (N, 64, 64, 64)
            R1GANResidualBlock(64, 128),
            nn.AvgPool2d(3, 2, padding=1), # (N, 128, 32, 32)
            R1GANResidualBlock(128, 256),
            nn.AvgPool2d(3, 2, padding=1), # (N, 256, 16, 16)
            R1GANResidualBlock(256, 512),
            nn.AvgPool2d(3, 2, padding=1), # (N, 512, 8, 8)
            R1GANResidualBlock(512, 1024),
            nn.AvgPool2d(3, 2, padding=1) # (N, 1024, 4, 4)
        )
        self.fc = nn.Linear(1024*self.dim_size*self.dim_size, 1) # (N, 1)
        
    def forward(self, x):
        B = x.size(0)
        h = self.D(x)
        h = h.view(B, -1)
        y = self.fc(h)
        return y

class R1GANGenerator(nn.Module):
    """
        Convolutional Generator
    """
    def __init__(self, out_channel=1, n_filters=128, n_noise=512,ngpu=1,img_size=64):
        super(R1GANGenerator, self).__init__()
        self.ngpu = ngpu
        self.dim_size = img_size//(2**4)
        self.fc = nn.Linear(n_noise, 1024*self.dim_size*self.dim_size) # (NewN, 1024, 8, 8)
        self.G = nn.Sequential(
            R1GANResidualBlock(1024, 512),#(NewN, 512, 8, 8)
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 512, 8, 8) (NewN, 512, 16, 16)
            R1GANResidualBlock(512, 256),#(NewN, 256, 16, 16)
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 256, 16, 16) (NewN, 256, 32, 32)
            R1GANResidualBlock(256, 128),#(NewN, 128, 32, 32)
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 128, 32, 32) (NewN, 128, 64, 64)
            R1GANResidualBlock(128, 64),#(NewN, 64, 64, 64)
            nn.Upsample(scale_factor=2, mode='bilinear'), # (N, 64, 64, 64) (NewN, 64, 128, 128)
            R1GANResidualBlock(64, 64),#(NewN, 64, 128, 128)
            nn.Conv2d(64, out_channel, 3, padding=1) # (N, 3, 64, 64) (NewN, 3, 128, 128)
        )
        
    def forward(self, z):
        B = z.size(0)
        h = self.fc(z)
        h = h.view(B, 1024, self.dim_size,self.dim_size)
        x = self.G(h)
        return x

class R1GAN():
    ##### Basic weights initializer #############
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    #### Loss function specific to R1 GANs ####
    def r1loss(self,inputs, label=None):
        # non-saturating loss with R1 regularization
        l = -1 if label else 1
        return F.softplus(l*inputs).mean()

    def train(self, 
              data_loader, 
              device,
              log_term = 10,
              save_term = 200,
              max_epochs=200,
              r1_gamma=10,
              G_losses = [],
              D_losses = [],
              img_list = [],
              step=0,
              lr_schedule=False,
              model_save_prefix=None):
        
        
        self.log_term = log_term
        self.save_term = save_term
        self.max_epochs = max_epochs
        self.r1_gamma = r1_gamma

        #step = 0
        self.D.to(device)
        self.G.to(device)
        if(model_save_prefix is not None):
            self.model_path_prefix = os.path.join(self.save_path,model_save_prefix)

        for epoch in range(self.max_epochs):
            for idx, images in enumerate(tqdm(data_loader, total=len(data_loader)),0):
                #self.G.zero_grad()
                self.D.zero_grad()
                # Training Discriminator
                
                x = images.to(device)
                b_size = x.size(0)
                labels = torch.full((b_size,), 1., dtype=torch.float, device=device)

                x.requires_grad = True
                x_outputs = self.D(x)
                d_real_loss = self.r1loss(x_outputs, True)
                # Reference >> https://github.com/rosinality/style-based-gan-pytorch/blob/a3d000e707b70d1a5fc277912dc9d7432d6e6069/train.py
                # little different with original DiracGAN
                grad_real = grad(outputs=x_outputs.sum(), inputs=x, create_graph=True)[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
                grad_penalty = 0.5*self.r1_gamma*grad_penalty
                D_x_loss = d_real_loss + grad_penalty

                z = (torch.rand(size=[b_size, self.n_noise])*2-1).to(device)
                x_fake = self.G(z)
                z_outputs = self.D(x_fake.detach())
                D_z_loss = self.r1loss(z_outputs, False)
                D_loss = D_x_loss + D_z_loss
                
                #self.D.zero_grad()
                D_loss.backward()
                self.D_opt.step()

                # Training Generator
                self.G.zero_grad()
                z = (torch.rand(size=[b_size, self.n_noise])*2-1).to(device)
                x_fake = self.G(z)
                z_outputs = self.D(x_fake)
                G_loss = self.r1loss(z_outputs, True)
                
                
                G_loss.backward()
                self.G_opt.step()
                
                G_losses.append(G_loss.item())
                D_losses.append(D_loss.item())

                if step % self.save_term == 0:
                    save_checkpoint({'global_step': step,
                        'D':self.D.state_dict(),
                        'G':self.G.state_dict(),
                        'd_optim': self.D_opt.state_dict(),
                        'g_optim' : self.G_opt.state_dict()},
                        '{}{:06d}.pth.tar'.format(self.model_path_prefix,step))
                
                if step % self.log_term == 0:
                    dt = datetime.datetime.now().strftime('%H:%M:%S')
                    print('Epoch: {}/{}, Step: {}, D Loss: {:.4f}, DReal Loss: {:.4f}, DFake Loss:{:.4f} G Loss: {:.4f}, gp: {:.4f}, Time:{}'.format(epoch, 
                                                            self.max_epochs, 
                                                            step, 
                                                            D_loss.item(),
                                                            d_real_loss.item(),
                                                            D_z_loss.item(), 
                                                            G_loss.item(), 
                                                            grad_penalty.item(), 
                                                            dt))
                                                            
                    # with torch.no_grad():
                    #     fake = G(n_noise).detach().cpu()
                    # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    #columns=["Step", "G_loss", "D_loss", "DReal_loss", "DFake_loss","Grad_Penalty"]
                    vals_array = [step,G_loss.item(), D_loss.item(), d_real_loss.item(),D_z_loss.item(),grad_penalty.item()]
                    self.df_loss = self.df_loss.append({self.df_loss.columns.values[idx]:val for idx,val in enumerate(vals_array)}, ignore_index=True)
                    self.G.eval()
                    img,rand_z_none = get_sample_image(self.G, device, self.n_noise, n_samples=64, )
                    img_data_np = img#.eval()
                    min_val = np.min(img_data_np)
                    max_val = np.max(img_data_np)
                    img_data_clamped = (img_data_np - min_val) / (max_val - min_val)
                    #img_list.append(vutils.make_grid(img, padding=2, normalize=True))
                    #img_list.append(img_data_clamped)
                    if(step % (self.log_term*2) == 0):
                        imsave('samples/{}_step{:06d}.jpg'.format(self.model_prefix, step), img_data_clamped)
                    self.G.train()
                if(lr_schedule == True):
                    self.D_scheduler.step()
                    self.G_scheduler.step()
                
                step += 1
        
        #### Save the last state as well 
        save_checkpoint({'global_step': step,
            'D':self.D.state_dict(),
            'G':self.G.state_dict(),
            'd_optim': self.D_opt.state_dict(),
            'g_optim' : self.G_opt.state_dict()},
            '{}{:06d}.pth.tar'.format(self.model_path_prefix,step))
        
        return G_losses, D_losses, img_list
        
    def load_model(self, weights_path):
        checkpoint = torch.load(weights_path)
        self.D.load_state_dict(checkpoint['D'])
        self.G.load_state_dict(checkpoint['G'])
        self.D_opt.load_state_dict(checkpoint['d_optim'])
        self.G_opt.load_state_dict(checkpoint['g_optim'])

    def __init__(self, 
                 img_size = 64,
                 d_in_channel=1,
                 g_out_channel=1, 
                 g_n_filters=128, 
                 g_n_noise=512,
                 g_opt=None, 
                 d_opt=None,
                 init_weights=True,
                 log_term = 10,
                 save_term = 200,
                 max_epochs=200,
                 r1_gamma=10,
                 save_path='ckpt',
                 samples_path='samples',
                 model_prefix='r1gan_adam'):
        self.G = R1GANGenerator(out_channel=g_out_channel, 
                                n_filters=g_n_filters, 
                                n_noise=g_n_noise,img_size=img_size)
        self.D = R1GANDiscriminator(in_channel=d_in_channel,img_size=img_size)
        if(init_weights == True ):
            self.G.apply(self.weights_init)
            self.D.apply(self.weights_init)
        
        self.D_opt = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.G_opt = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        #### Setting to user-provided optimizers if not NULL
        if(g_opt is not None):
            self.G_opt = g_opt
        if(d_opt is not None ):
            self.D_opt = d_opt
        self.D_scheduler = StepLR(self.D_opt, step_size=2, gamma=0.5)
        self.G_scheduler = StepLR(self.G_opt, step_size=2, gamma=0.5)
        self.df_loss =pd.DataFrame(columns=["Step", "G_loss", "D_loss", "DReal_loss", "DFake_loss","Grad_Penalty"])

        #### Training essentials #####
        self.log_term = log_term
        self.save_term = save_term
        self.max_epochs = max_epochs
        self.r1_gamma = r1_gamma
        self.n_noise = g_n_noise
        self.save_path = save_path
        self.model_prefix = model_prefix

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(samples_path):
            os.makedirs(samples_path)

        self.model_path_prefix = os.path.join(self.save_path,self.model_prefix)
