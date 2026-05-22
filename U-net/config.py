""" Config file containing all the hyeperparameters of
    the CNN and the GP
"""
import torch
from torch.cuda.amp import GradScaler
import numpy as np

class Config:

    #gpu
    gpu_device = 0

    #CNN
    challenge = 'MnM2' #mscmr19 #MnM2
    patience = 1000
    lr_cnn = 3e-4
    filters = 48 #filters image encoder
    depth = 5 # depth network (fixed)
    dim = 128 # cropping dimension
    pixel_size = 1.375 #pixel dimension
    in_size = (filters)*2**(depth-1)
    b_s = 2 # batch size
    n_epochs = 1500
    nr_labels = 4 # labels to segment
    dropout = 0.5
    weights_loss = torch.tensor([0.1,0.3,0.3,0.3])
    # scaler to perform half precision operations
    scaler = GradScaler()
    # do you need half precision?
    half_precision = False
    #inference
    dataset_path = f'...' #update for your setup
    test_folder = f'...' #update for your setup
    weights = '...pth'#update with weights file name
