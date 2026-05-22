import torch
torch.set_printoptions(edgeitems=4, linewidth=117)
import torch.nn as nn
import blocks_GP as blocks
from dataset import build_testset
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from config import Config
import pickle
import os
from config import Config
from helpers import *

seed = 11131313
queries = build_testset(1,Config.dim, Config.pixel_size,Config.challenge)

smx = []
labels = []
imgs = []

# initializing the model
model = blocks.Conformal_UNet(
    image_encoder = blocks.MRIEncoder(Config.depth, Config.filters),
    upsampler     = blocks.Decoder(Config.in_size, Config.nr_labels),
    loss = nn.CrossEntropyLoss(Config.weights_loss),
    reason = 'test',
)

model.to('cuda')

model_checkpoint = torch.load(Config.weights, map_location='cuda')
model.load_state_dict(model_checkpoint['net'])
tmp = model.state_dict()

smx = []
imgs = []
labels = []

Ns = 30 #number of samples
for q,query in enumerate(queries):
    query = recursive_to(query, 'cuda')
    q_image, q_seg = query
    pred = model(query,False)

    ##montecarlo dropout softmax
    pred_stack = np.zeros((Ns,4,q_image.shape[-1],q_image.shape[-1]))
    for n in range(Ns):
        pred_stack[n,:]=model(query,False)[0,:].detach().cpu().numpy()
    smx.append(torch.softmax(torch.tensor(pred_stack), dim=1).detach().cpu().numpy())

    gt = q_seg[0,0,:,:].detach().cpu()
    labels.append(gt.numpy())
    imgs.append(q_image.detach().cpu().numpy()[0,0,:,:])

    # if you are testign MnM2 and want to have comparable dataset size as mscmr19, break when q=900
os.makedirs('../softmax', exist_ok=True)
with open(f'../softmax/{Config.challenge}/smx_{Config.challenge}.pkl', 'wb') as file:
    pickle.dump(np.array(smx), file)
with open(f'../softmax/{Config.challenge}/labels_{Config.challenge}.pkl', 'wb') as file:
    pickle.dump(np.array(labels), file)
with open(f'../softmax/{Config.challenge}/imgs.pkl', 'wb') as file:
    pickle.dump(np.array(imgs), file)


del model
torch.cuda.empty_cache()
