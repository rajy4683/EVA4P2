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
import torchvision.utils as vutils


import os
import sys
import glob
from tqdm import tqdm_notebook as tqdm
import datetime
import matplotlib.pyplot as plt
import numpy as np

def tensor2img(tensor):
    img = (np.transpose(tensor.detach().cpu().numpy(), [1,2,0])+1)/2.
    return img

def get_sample_image(G, device,n_noise=100, n_samples=64):
    """
        save sample 100 images
    """
    n_rows = int(np.sqrt(n_samples))
    z = (torch.rand(size=[n_samples, n_noise])*2-1).to(device) # U[-1, 1]
    x_fake = G(z)
    x_fake = torch.cat([torch.cat([x_fake[n_rows*j+i] for i in range(n_rows)], dim=1) for j in range(n_rows)], dim=2)
    result = tensor2img(x_fake)
    return result,z

def save_checkpoint(state, file_name='checkpoint.pth.tar'):
    torch.save(state, file_name)

def create_graphs_and_samples(G_losses,D_losses, img_list, data_loader,device, size_tuple=(30,30)):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="G")
    plt.plot(D_losses,label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


    real_batch = next(iter(data_loader))

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    # plt.subplot(1,2,2)
    # plt.axis("off")
    # plt.title("Fake Images")
    # plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    # plt.show()

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.array(img_list[-1]))#np.transpose(np.array(img_list[-1]),(1,2,0)))
    plt.show()

    #return generate_image_sample(size_tuple)

def generate_image_sample(G, device, n_noise=256,size_tuple=(30,30)):
    G.to(device)
    output_images,rand_z = get_sample_image(G, device, n_noise=n_noise, n_samples=64)
    #output_images_list = vutils.make_grid(output_images, padding=2, normalize=True)

    # img_data_np = output_images#.eval()
    # min_val = np.min(img_data_np)
    # max_val = np.max(img_data_np)
    # img_data_clamped = (img_data_np - min_val) / (max_val - min_val)
    plt.figure(figsize=size_tuple)

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    #plt.imshow(np.transpose(output_images_list,(1,2,0)))
    plt.imshow(np.array(output_images))
    plt.show()
    return rand_z

