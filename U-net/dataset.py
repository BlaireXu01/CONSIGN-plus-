import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import transform
import random
from helpers import rescale_, crop_informations, print_loading_bar, print_loading_spinner
import matplotlib.pyplot as plt
import glob
import os
import time
from PIL import Image
from config import Config

class MnM2_dataset(Dataset):
    """Heart  segmentation dataset."""

    def __init__(self, set, dim, is_train = True):
        """
        Arguments:
        set: list containing couples of images and their segmentation
        dim: cropped image will be dim X dim
        is_train: boolean to enable augmentation
        """
        self.set = set
        self.is_train = is_train
        self.dim = dim

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):

        image = self.set[idx][0]
        mask = self.set[idx][1]

        x, y = self.transform(image, mask)
        return x, y

    def transform(self, image, mask):
        # Convert format to PIL image
        image = TF.to_pil_image(image)
        mask = TF.to_pil_image(mask)

        if self.is_train:
            # Random rotation
            if random.random() > 0.5:
                angle = 15*(random.sample([i for i in range(-4, 4 + 1)],1))[0]
                image = TF.rotate(image,angle)
                mask = TF.rotate(mask,angle)

            if random.random() > 0.5:
                image = TF.adjust_brightness(image, random.random() + 0.5)

            if random.random() > 0.5:
                image = TF.adjust_contrast(image, random.random() + 0.5)

        x,y,w,h = crop_informations(TF.to_tensor(mask), self.is_train, self.dim)

        # center crop
        image = TF.crop(image,x,y,h,w)
        mask = TF.crop(mask,x,y,h,w)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        image = TF.normalize(image, mean=[0.5], std=[0.5])

        return image, mask


def build_dataset(batch, dim, pixels_size, challenge, path):
    """
    Arguments:
    batch: batch size
    dim: cropped image will be dim X dim
    challenge: name of the dataset
    n_subsets: number of subsets
    Returns:
    training_istances: Dataloader with training queries and training supports
    val_istances: Dataloader with validation queries and validation supports
    """
    print("## PRE-PROCESSING ##\n")
    device = torch.device('cuda:0')

    print(f'Pixel size = {pixels_size}')
    print(f'Cropping dimensions = {dim} X {dim}\n')

    ratio = 0.8
    train_set = []
    val_set = []

    if challenge == 'MnM2':
        SA, LA, file = extract_images_from_MnM2(path, 'training', dim, pixels_size)
        random.Random(141234).shuffle(SA)
        train_set = (SA[:int(len(SA)*ratio)])
        val_set = (SA[int(len(SA)*ratio):])
    else:
        C0, T2, LGE, file = extract_images_from_MSCMR(path, 'training', dim, pixels_size)
        random.Random(141234).shuffle(C0)
        random.Random(141234).shuffle(T2)
        random.Random(141234).shuffle(LGE)
        train_set = C0[:int(len(C0)*ratio)] + T2[:int(len(T2)*ratio)] + LGE[:int(len(LGE)*ratio)]
        val_set = C0[int(len(C0)*ratio):] + T2[int(len(T2)*ratio):] + LGE[int(len(LGE)*ratio):]

    Z_trains = MnM2_dataset(train_set, dim,  True)
    Z_vals = MnM2_dataset(val_set, dim, False)

    # batch size
    b_s = batch
    print('Constructing training set...')
    # construct query and support istances
    query_img = []
    query_labels = []

    query_i, query_l = build_support_query(Z_trains)
    query_img += query_i
    query_labels += query_l

    print('\nDone!\n')
    print('Constructing validation set...')

    val_query_img = []
    val_query_labels = []

    val_query_img_i, val_query_labels_i = build_support_query(Z_vals)
    val_query_img += val_query_img_i
    val_query_labels += val_query_labels_i

    print('\nDone!\n')

    training_queries = list(zip(query_img, query_labels))
    training_istances = DataLoader(training_queries, shuffle=True, batch_size=b_s)

    val_queries = list(zip(val_query_img, val_query_labels))
    val_istances = DataLoader(val_queries, shuffle=False, batch_size=b_s)

    print(f'Number of training istances: {len(training_queries)}')
    print(f'Number of validation istances: {len(val_queries)}')

    return training_istances, val_istances

def build_testset(batch, dim, pixel_size, challenge):
    """
    Arguments:

    """

    device = torch.device('cuda:0')

    print('Extracting images...')
    # test images from test set
    if challenge == 'MnM2':
        SA,LA, _  = extract_images_from_MnM2(Config.dataset_path, 'testing', dim, pixel_size)
    else:
        _, _, LGE, _ = extract_images_from_MSCMR(Config.dataset_path, 'testing', dim, pixel_size)

    if challenge == 'MnM2':
        dataset = MnM2_dataset(SA, dim, False)
    else:
        dataset = MnM2_dataset(LGE, dim, False)

    print('\nDone!\n')

    print('Constructing test set...')

    query_img, query_labels = build_support_query(dataset)

    print('\nDone!\n')
    # bacth size
    b_s = batch

    # set up DataLoader for training set
    query_loader = DataLoader(list(zip(query_img, query_labels)), shuffle=False, batch_size=b_s)
    return query_loader

def extract_images_from_MnM2(path, phase, dim, pixel_size):
    """
    Arguments:
    path: path of the dataset's folder
    """
    print('Extracting images...')
    SA = []
    LA = []

    target_resolution = [pixel_size,pixel_size]
    crop_size = [dim,dim]

    if phase == 'training':
        init_patient=1
        end_patient=201
    else:
        init_patient=201
        end_patient=361
    count = 0

    for i in range(init_patient,end_patient):
        print_loading_bar(i + 1 - init_patient, end_patient-init_patient)
        time.sleep(0.01)

        path_new = path+phase+'/'+str(i).zfill(3)
        filepath_gt = glob.glob(os.path.join(path_new,'*{0}*.gz'.format('gt*')))
        filepath_tot = [file for file in glob.glob(os.path.join(path_new, '*{0}*.gz'.format('*'))) if '_gt' not in file and 'CINE' not in file]

        filepath_gt = sorted(filepath_gt)
        filepath_tot = sorted(filepath_tot)

        time_frames = ['ED','ES']

        for time_frame in time_frames:
            mri_files = [file for file in filepath_tot if time_frame in file]
            gt_files = [file for file in filepath_gt if time_frame in file]

            for mri_file, gt_file in zip(mri_files,gt_files):

                img = nib.load(mri_file)
                img_data = img.get_fdata()

                gt = nib.load(gt_file)
                gt_data = gt.get_fdata()

                spacing = img.header.get_zooms()
                scale_vector = [spacing[0] / target_resolution[0], spacing[1] / target_resolution[1]]

                for k in range(0,img_data.shape[2]):

                    r_img, r_mask = rescale_(img_data[:,:,k],gt_data[:,:,k],scale_vector)
                    elem = [r_img, r_mask]
                    rv_label = 3
                    lv_label = 1
                    my_label = 2

                    if 'LA' in mri_file:
                        LA.append(elem)
                    else:
                        SA.append(elem)

    print('\nDone!\n')

    return SA,LA, 'mnm2'
def extract_images_from_MSCMR(path, phase, dim, pixel_size):
    """

    """
    print('Extracting images...')
    # mri without heart
    no_heart = []
    C0 = []
    T2 = []
    LGE = []

    target_resolution = [pixel_size,pixel_size]
    crop_size = [dim,dim]

    filepath_gt = glob.glob(path+f'{phase}/label/*')
    filepath_mri = glob.glob(path+f'{phase}/img/*')
    filepath_gt = sorted(filepath_gt)
    filepath_mri = sorted(filepath_mri)

    for mri_file, gt_file in zip(filepath_mri,filepath_gt):

        img = nib.load(mri_file)
        img_data = img.get_fdata()

        gt = nib.load(gt_file)
        gt_data = gt.get_fdata()

        spacing = img.header.get_zooms()
        scale_vector = [spacing[0] / target_resolution[0], spacing[1] / target_resolution[1]]
        for k in range(0,img_data.shape[2]):

            r_img, r_mask = rescale_(img_data[:,:,k],gt_data[:,:,k],scale_vector)

            elem = [r_img, r_mask//80]

            rv_label = 1
            lv_label = 3
            my_label = 2

            mask_1 = elem[1] == 1
            mask_3 = elem[1] == 3

            # Swap the values using the boolean masks
            elem[1][mask_1] = 3
            elem[1][mask_3] = 1

            if 'C0' in mri_file:
                C0.append(elem)
            elif 'T2' in mri_file:
                T2.append(elem)
            elif 'LGE' in mri_file:
                LGE.append(elem)


    print('\nDone!\n')
    return C0, T2,LGE ,'mscmr19'

def build_support_query(Z):
    """
    Arguments:

    """
    query_img = []
    query_labels = []

    # construct training set
    for i in range(len(Z)):

        if i%25==0:
            print_loading_spinner(i)
            time.sleep(0.1)

        # extract image and its segmentation
        img, label = (Z[i])

        # append them in two lists
        query_img.append(img)
        query_labels.append(torch.mul(label,255).type(torch.LongTensor))

    return query_img, query_labels
