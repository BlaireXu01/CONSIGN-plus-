import os
import torch
torch.set_printoptions(edgeitems=4, linewidth=117)
import torch.nn as nn
import torch.optim as optim
import blocks_GP as blocks
from dataset import build_dataset
import trainer
from helpers import count_parameters
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
import matplotlib.pyplot as plt
from config import Config

def torch_init(gpu_index):
    torch.cuda.set_device(gpu_index)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    return device

def main():

    CUDA_LAUNCH_BLOCKING="1"
    print("Started script: {}, with pytorch {}".format(os.path.basename(__file__), torch.__version__))
    print(f"Number of available GPUs: {torch.cuda.device_count()}")
    # select gpu
    device = torch_init(Config.gpu_device)
    print(f"GPU Nr.{torch.cuda.current_device()} has been selected")


    # initializing the model
    model = blocks.Conformal_UNet(
        image_encoder = blocks.MRIEncoder(Config.depth, Config.filters),
        upsampler     = blocks.Decoder(Config.in_size, Config.nr_labels),
        loss = nn.CrossEntropyLoss(Config.weights_loss),
        reason = 'train',
    )

    model.to(device)

    print("Nr. parameters model:", '{:,}\n'.format(count_parameters(model)))

    # start training
    train(model, device)

def train(model, device):

    optimizer_cnn = optim.Adam(model.parameters(), lr=Config.lr_cnn)
    lr_sched_cnn = optim.lr_scheduler.LambdaLR(optimizer_cnn, (lambda n: 1.0 if n <= 50 else (0.1 if n <= 700 else 0.01)))

    # construct the dataset
    training_istances, val_istances = build_dataset(Config.b_s, Config.dim, Config.pixel_size, Config.challenge, Config.dataset_path)

    tr = trainer.FSSTrainer(
        model                = model,
        optimizer            = optimizer_cnn,
        scaler               = Config.scaler,
        lr_sched             = lr_sched_cnn,
        training_loader      = training_istances,
        val_loader           = val_istances,
        device               = device,
        name                 = f'{Config.challenge}_{Config.filters}_{Config.dropout}_{Config.gpu_device}_{Config.b_s}',
        patience             = Config.patience,
        half_precision       = Config.half_precision
    )

    tr.train(Config.n_epochs)

#Run main
main()
