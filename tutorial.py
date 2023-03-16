


#

import torch 
torch.manual_seed(42)
import numpy as np 
import matplotlib.pyplot as plt 

from tqdm.notebook import tqdm



#=============================Configurations


'''
In order to use GPU in python
we have to transfer the tensors
to GPU - cuda would be used for that
cuda - will hel in the transfer of the image to GPU device


noise-vector-dimension  - shape of the random noise
we would pass in the generator


lr = learning rate
beta_1, beta_2 = going to be useful in adam optimizer


'''

device = 'cuda'  # e.g image = image.to(device)
batch_size = 128  # going to be used in trainloader, training loop
noise_dim = 64  # going to be used in creating generator model


# optimizer parameters

lr = 0.0002
beta_1 = 0.5
beta_2 = 0.99


# Training variables

epochs = 20  # the number of time we want to run our training loop















