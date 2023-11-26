


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






#=============================== Load MNIST Dataset


'''
load the dataset and decalre  some of the augmentations
'''


from torchvision import datasets, transforms as T

'''
declaring the transforms

there are many thing you can apply 
in the T.compose([]) 
- like horizontal flip, vertical flip, random rotations
        with random rotation - you pass in the degree e.g (-20, +20)
- the second thing is ToTensor - this is because your 
        images will be in numpy but note that  - 
        All transformations accept PIL(python imaging library) Image, 
        Tensor Image or batch of Tensor Images as input
        - to covert the images to torch tensor
        - it will shift your c (channel) to the zero axis - 
        e.g (h, w, c) -> shifted (c, h, w)
         (c, h, w) =  this is the convention pytorch uses (the image shape is c,h,w)



'''


train_augs = T.Compose([
    T.RandomRotation((-20,+20)),
    T.ToTensor()
])



'''
load the mnist train set
 - there are many datasets 
   - determine the path the dataset would be saved
   - the download = true
    - we wnat to train so train=  true
    - then the augmentation, so transform=your train augs

'''

trainset = datasets.MNIST('MNIST/',download=True, train=True, transform=train_augs)



'''
take a look at the dataset
- plot some of the images


'''

image, label = trainset[900] 
plt.imshow(image.squeeze(), cmap='gray')


'''
we can see the length of the dataset

'''

print("total images in trainset = ", len(trainset))

# total images in trainset =  60000








#

























