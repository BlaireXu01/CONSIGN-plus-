import numpy as np
import torch
import torchvision
from skimage import transform
import random
import matplotlib.pyplot as plt
import cv2 as cv
import SimpleITK as sitk
from config import Config
import glob
import os
import nibabel as nib


def dice_coeff(gt,pred):
    """
    Arguments:
    pred: prediction of segmentation
    gt: ground truth of segmentation
    label: current label (1,2,3), or 0 if you want binary segmentation
    Returns:
    dice: dice coefficient for the label
    """

    b_s = gt.shape[0]
    dice = 0


    for b in range(b_s):
        pred_b = torch.softmax(pred[b,:,:,:], dim=0)
        pred_b = torch.argmax(pred_b, dim=0)
        labels = np.unique(gt.cpu())

        for c in labels:
            # Copy the gt image to not alterate the input
            gt_c_i = np.copy(gt[b,0,:,:].cpu().detach().numpy())
            gt_c_i[gt_c_i != c] = 0
            # Copy the pred image to not alterate the input
            pred_c_i = np.copy(pred_b.cpu().detach().numpy())
            pred_c_i[pred_c_i != c] = 0

            # Compute the Dice
            dice += dc(gt_c_i, pred_c_i)

    return dice/(b_s*len(labels))

def dc(result, reference):

    result = np.atleast_1d(result.astype(bool))
    reference = np.atleast_1d(reference.astype(bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    if size_i1 == 0 and size_i2 == 0:
        return 1
    elif (size_i1 == 0 and size_i2 >0) or (size_i1 > 0 and size_i2 == 0):
        return 0
    else:
        dc = 2. * intersection / float(size_i1 + size_i2)
    return dc


def rescale_(img_data1, gt_data1, scale_vector):
    """
    Arguments:
    img_data1: image
    gt_data1: segmentation
    scale_vector: vector containing the scaling for the rasampling
    crop_size: dimension of the new image [lx,ly]
    Returns:
    cropped_image: resampled and rescaled image
    cropped_mask: resampled and rescaled gt
    """
    img_rescaled = transform.rescale(img_data1,
                                     scale_vector,
                                     order=1,
                                     preserve_range=True,
                                     mode = 'constant')

    mask_rescaled = np.round(transform.rescale(gt_data1,
                                      scale_vector,
                                      order=0,
                                      preserve_range=True,
                                      mode = 'constant'))

    img_rescaled = (img_rescaled*(255/img_rescaled.max())).astype('uint8')
    mask_rescaled = mask_rescaled.astype('uint8')

    return img_rescaled, mask_rescaled


def plot_losses(val_loss,train_loss,stringa):
    """
    Arguments:
    val_loss: validation losses
    train_loss: train losses
    stringa: name of the file
    Returns:
    query_img: list of query images
    query_labels: list of query segmenation
    support_imgs: list of support images
    support_labels: list of support segmentations
    """
    indices = list(range(len(val_loss)))
    max_val_loss = max(val_loss)
    # Plotting
    plt.plot(indices, val_loss, label='Validation Loss', color='blue')
    plt.plot(indices, train_loss, label='Train Loss', color='orange')
    plt.axhline(y=max_val_loss, color='red', linestyle='--', label=f'Max Val Loss ({max_val_loss:.4f})', alpha=0.7)

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title('Validation and Train Loss')
    plt.legend()  # Show legend to differentiate the lines
    # Set the x-axis grid to be every 0.1
    plt.xticks(range(0, len(val_loss), 100))
    plt.yticks([i * 0.1 for i in range(int(1 / 0.1) + 1)])
    plt.grid(which='both', axis='both', linestyle='--', color='gray', alpha=0.5)

    plt.savefig(stringa+'losses_behaviour.png', dpi=600)
    plt.clf()


def count_parameters(model):
     return sum(p.numel() for p in model.parameters() if p.requires_grad)

def recursive_to(data, device):
    """Recursively goes through a structure of lists and dicts and moves all tensors to requested device
    """
    if isinstance(data, dict):
        return {key: recursive_to(val, device) for key, val in data.items()}
    elif isinstance(data, (list, tuple)):
        return [recursive_to(elem, device) for elem in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

def crop_informations(mask, is_train, dim):
        crop_size = [dim,dim]
        gt = np.asarray(mask)
        gt[gt>0] = 1
        gt = (gt*255).astype('uint8')[0,:,:]
        contours, _ = cv.findContours(gt, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            x,y,w,h = cv.boundingRect(contours[0])
            com = [y + h/2, x + w/2]
        else:
            com = [mask.shape[1]/2,mask.shape[2]/2]

        x_min = int(com[0] - crop_size[0] // 2)
        y_min = int(com[1] - crop_size[0] // 2)

        return x_min, y_min, crop_size[0], crop_size[1]

def print_loading_bar(iterations, total):
    progress = int(iterations / total * 100)
    bar_length = 20
    num_blocks = int(bar_length * progress / 100)
    bar = "[" + "=" * num_blocks + " " * (bar_length - num_blocks) + "]"
    print(f"\r{bar} {progress}%", end="", flush=True)

def print_loading_spinner(iterations):
    spinners = ["|", "/", "-", "\\"]
    spinner_index = iterations % len(spinners)
    spinner = spinners[spinner_index]
    print(f"\r{spinner} Loading...", end="", flush=True)
