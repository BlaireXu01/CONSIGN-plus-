import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import softmax
from config_cp import Config
import pickle
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import atexit

baseline_passing_recalls = {0: [], 1: [], 2: [], 3: []}     
baseline_passing_recalls_LIDC = {0: [], 1: []}              


# def print_auto_data_driven_limits():
#     challenge = getattr(Config, 'challenge', 'Unknown')
    
#     if 'LIDC' in challenge:
#         if len(baseline_passing_recalls_LIDC[1]) > 50: 
#             bg_2_5 = np.percentile(baseline_passing_recalls_LIDC[0], 2.5)
#             nodule_2_5 = np.percentile(baseline_passing_recalls_LIDC[1], 2.5)
            
#             print("\n" + "★"*65)
#             print(f"📊 [LIDC Data-Driven Benchmark] Baseline 的真实能力底线 (2.5th Percentile)")
#             print(f"   背景 (BG): {bg_2_5:.3f}")
#             print(f"   肺结节 (Nodule): {nodule_2_5:.3f}")
#             print(f"\n👉 请直接复制此数组到 config_cp.py 的 betas 中:")
#             print(f"   [{bg_2_5:.2f}, {nodule_2_5:.2f}]")
#             print("★"*65 + "\n")
#         else:
#             print("\n⚠️ 收集到的 LIDC 样本不足，无法计算可靠的 Data-Driven Benchmark。")
            
#     else:
#         if len(baseline_passing_recalls[2]) > 50: 
#             bg_10 = np.percentile(baseline_passing_recalls[0], 1)
#             lv_10 = np.percentile(baseline_passing_recalls[1], 1)
#             myo_10 = np.percentile(baseline_passing_recalls[2], 1)
#             rv_10 = np.percentile(baseline_passing_recalls[3], 1)
            
#             # bg_10 = np.mean(baseline_passing_recalls[0])
#             # lv_10 = np.mean(baseline_passing_recalls[1])
#             # myo_10 = np.mean(baseline_passing_recalls[2])
#             # rv_10 = np.mean(baseline_passing_recalls[3])
            
#             # bg_10 = np.median(baseline_passing_recalls[0])
#             # lv_10 = np.median(baseline_passing_recalls[1])
#             # myo_10 = np.median(baseline_passing_recalls[2])
#             # rv_10 = np.median(baseline_passing_recalls[3])
            
            
#             print("\n" + "★"*65)
#             print(f"📊 [{challenge} Data-Driven Benchmark] Baseline 的真实能力底线 (2.5th Percentile)")
#             print(f"   背景(BG): {bg_10:.3f}")
#             print(f"   左心室(LV): {lv_10:.3f}")
#             print(f"   心肌(MYO): {myo_10:.3f}")
#             print(f"   右心室(RV): {rv_10:.3f}")
#             print(f"\n👉 请直接复制此数组到 config_cp.py 的 betas 中:")
#             print(f"   [{bg_10:.2f}, {lv_10:.2f}, {myo_10:.2f}, {rv_10:.2f}]")
#             print("★"*65 + "\n")

# # 注册这个全新的双核钩子
# atexit.register(print_auto_data_driven_limits)

def print_loading_bar(iterations, total):
    progress = int(iterations / total * 100)
    bar_length = 20
    num_blocks = int(bar_length * progress / 100)
    bar = "[" + "=" * num_blocks + " " * (bar_length - num_blocks) + "]"
    print(f"\r{bar} {progress}%", end="", flush=True)

def recon(c,U_cal,mean_cal):
    proj = U_cal @ c
    return mean_cal + proj.view(Config.labels,Config.dim,Config.dim)

def recon_batch(c,U_cal,mean_cal):
    proj = U_cal @ c.permute(1,0)
    return mean_cal[None,:] + proj.permute(1,0).view(-1,Config.labels,Config.dim,Config.dim)

##############################################
# Clinical Score functions 
##############################################
def cal_score_SVD_clinical(sigma, Y, diff_between_lab):
    predictions = np.argmax(sigma, axis=0)
    
    if not diff_between_lab:
        return 1.0 if np.sum(predictions==Y)/(Config.dim**2) > np.mean(Config.current_betas) else 0.0
    
    labels = np.unique(Y)

    if getattr(Config, 'use_strict_scoring', False):
        for l in labels:
            recall_l = np.sum((predictions==Y)*(Y==l)) / np.sum(Y==l)
            if recall_l < Config.current_betas[l]:
                return 0.0
        return 1.0      
    else:
        recalls = []
        for l in labels:
            recalls.append(np.sum((predictions==Y)*(Y==l)) / np.sum(Y==l))
            
        if getattr(Config, 'use_weighted', False):
            valid_weights = [Config.current_weights[l] for l in labels]
            weighted_mean = np.average(recalls, weights=valid_weights)
            return 1.0 if weighted_mean > np.mean(Config.current_betas) else 0.0
            
        else:
            is_passing = np.mean(recalls) > np.mean(Config.current_betas)
            if is_passing and not getattr(Config, 'use_tversky', False) and not getattr(Config, 'use_weighted', False): 
                challenge = getattr(Config, 'challenge', '')
                
                if 'LIDC' in challenge:
                    if 1 in labels and len(recalls) == 2:
                        baseline_passing_recalls_LIDC[0].append(recalls[0]) 
                        baseline_passing_recalls_LIDC[1].append(recalls[1])
                else:
                    if len(recalls) == 4:
                        for i in range(4):
                            baseline_passing_recalls[i].append(recalls[i])
                            
            return 1.0 if is_passing else 0.0
        
def cal_score_PW_clinical(gt_is_in_set, Y, diff_between_lab):
    if not diff_between_lab:
        return 1.0 if np.sum(gt_is_in_set)/(Config.dim**2) > np.mean(Config.current_betas) else 0.0
    
    labels = np.unique(Y)
    
    if getattr(Config, 'use_strict_scoring', False):
        for l in labels:
            recall_l = np.sum(gt_is_in_set*(Y==l)) / (np.sum(Y==l))
            if recall_l < Config.current_betas[l]:
                return 0.0
        return 1.0
    else:
        recalls = []
        for l in labels:
            recalls.append(np.sum(gt_is_in_set*(Y==l)) / (np.sum(Y==l)))
            
        if getattr(Config, 'use_weighted', False):
            valid_weights = [Config.current_weights[l] for l in labels]
            weighted_mean = np.average(recalls, weights=valid_weights)
            return 1.0 if weighted_mean > np.mean(Config.current_betas) else 0.0
            
        else:
            return 1.0 if np.mean(recalls) > np.mean(Config.current_betas) else 0.0

def val_score(predictions, Y):
    labels = np.unique(Y)
    score = []
    for l in labels:
        score.append(np.sum((predictions==Y[None,:,:])*(Y==l)[None,:,:],axis=(1,2))/np.sum(Y==l))
    return np.array(score) # shape: (num_labels_present, N_samples)

#######################
# Extract and Sample Utils 
#######################
def extract_softmax():
    print('Extracting softmax...\n')
    challenge = Config.challenge
    file_softmax = f'./softmax/{challenge}/smx_{challenge}.pkl'
    file_gt = f'./softmax/{challenge}/labels_{challenge}.pkl'
    file_img = f'./softmax/{challenge}/imgs.pkl'

    with open(file_softmax, 'rb') as file:
        smx_MC = np.array(pickle.load(file)) 
    with open(file_gt, 'rb') as file:
        labels = np.array(pickle.load(file)) 
    with open(file_img, 'rb') as file:
        imgs = np.array(pickle.load(file)) 

    print("Done!\n")

    if Config.challenge in ['MnM2','mscmr19']:
        index_n0 = [i for i,lbl in enumerate(labels[:len(smx_MC)]) if lbl.max()>0]
        labels = labels[:len(smx_MC)][index_n0]
        imgs = imgs[:len(smx_MC)][index_n0]
        smx_MC = smx_MC[index_n0]

    n = Config.n 
    np.random.seed(111111) 
    cal_splits = []
    val_splits = []
    for i in range(5):
        cal_splits.append(np.random.choice(np.arange(smx_MC.shape[0]),n,replace=False))
        val_splits.append(np.array([v for v in range(smx_MC.shape[0]) if v not in cal_splits[i]]))

    return smx_MC, labels, imgs, cal_splits, val_splits

def uq_samples(smx,n_samples):
    samples = []
    if n_samples==0:
        n_samples=smx.shape[1]

    num_rows, num_cols = smx.shape
    sampled_indices = np.full((n_samples, num_cols), 0, dtype=int) 

    for i in range(num_cols):
        true_indices = np.where(smx[:, i])[0]  
        if true_indices.size > 0:
            sampled_indices[:, i] = np.random.choice(true_indices, size=n_samples, replace=True)

    return sampled_indices

##############################################################
# Ablation Loss Function 
##############################################################
class AblationLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(AblationLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets, parallel=False):
        num_classes = preds.shape[1]
        targets_one_hot = F.one_hot(targets.to(torch.long), num_classes).permute(0, 3, 1, 2).float()
        softargmax = F.softmax(preds/0.0001, dim=1)
        # softargmax = preds

        TP = (softargmax * targets_one_hot).sum(dim=(2, 3)) 
        FP = (softargmax * (1 - targets_one_hot)).sum(dim=(2, 3))
        FN = ((1 - softargmax) * targets_one_hot).sum(dim=(2, 3))
        
        if Config.use_tversky:
            score_per_class = (TP + self.smooth) / (TP + Config.alpha_fn * FN + Config.beta_fp * FP + self.smooth)
        else:
            score_per_class = (TP + self.smooth) / (TP + FN + self.smooth)
            
        valid_classes = targets_one_hot.sum(dim=(2, 3)) > 0
        per_class_error = 1.0 - score_per_class

        gamma = 1.5
        per_class_error = torch.pow(1.0 - score_per_class + self.smooth, gamma)
        
        device = preds.device
        weights = torch.tensor(Config.current_weights, dtype=torch.float32, device=device)
        batch_weights = weights.unsqueeze(0).expand(per_class_error.shape[0], -1)

        valid_mask = valid_classes.repeat(per_class_error.shape[0], 1) if parallel else valid_classes
        
        if Config.use_weighted:
            masked_weighted_errors = batch_weights * per_class_error * valid_mask.float()
            masked_weights = batch_weights * valid_mask.float()
            
            sum_weighted_errors = torch.sum(masked_weighted_errors, dim=1)
            sum_weights = torch.sum(masked_weights, dim=1)
        
            loss_batch = sum_weighted_errors / (sum_weights + 1e-8)
            
        else:
            masked_errors = per_class_error * valid_mask.float()
            sum_errors = torch.sum(masked_errors, dim=1)
            num_valid = torch.sum(valid_mask.float(), dim=1)
            
            loss_batch = sum_errors / (num_valid + 1e-8)
            
        if not parallel:
            return loss_batch[0]
        else:
            return loss_batch

def extract_lambdas(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        lambdas = []
        for i in [32, 36, 40, 44, 48]:
            line = lines[i].strip()
            parts = line.split()
            lambdas.append(np.array([float(parts[0]), float(parts[1]), float(parts[2])]))
        return np.array(lambdas)

def extract_metrics(file_path):
    cov, chao, corr = [], [], []
    with open(file_path, "r") as file:
        for idx, line in enumerate(file):
            if idx in [1,2,4,5,7,8]:
                numbers = [float(x) for x in line.replace(line.split('=')[0], '').replace('=', '').strip().split()]
                cov.append(numbers)
            elif idx in [11,12,14,15,17,18]:
                numbers = [float(x) for x in line.replace(line.split('=')[0], '').replace('=', '').strip().split()]
                chao.append(numbers)
            elif idx in [21,22,24,25,27,28]:
                numbers = [float(x) for x in line.replace(line.split('=')[0], '').replace('=', '').strip().split()]
                corr.append(numbers)
    return np.array(cov),np.array(chao),np.array(corr)

##############################################################
############### helpers function for plots ##############################################################
def colorized(img,label_to_color):
    img = np.array(np.vectorize(lambda x: label_to_color.get(x, (1, 1, 1)))(img))
    return np.transpose(img[:3,:,:],(1,2,0))

def heart_visual(image,PW_pred,SVD_pred,SACP_pred,gt):
    label_names = {0: "Background", 1: "Left Ventricle", 2: "Myocardium",3: "Right Ventricle"}

    cmap = plt.get_cmap("tab20", len(label_names))
    label_to_color = {label: cmap(idx) for idx, label in enumerate(label_names.keys())}
    patches = []
    for label, idx in label_names.items():
        if isinstance(label, int):  # ensure the label is an integer
            patch = mpatches.Patch(color=label_to_color[label], label=f"{label} {idx}")
            patches.append(patch)
        else:
            print(f"Warning: Label {label} is not an integer. Skipping.")

    fig, ax = plt.subplots(1,Config.n_samples_4_visualization*3+2,figsize=(30,8))
    image = (image - image.min()) / (image.max() - image.min())
    image = np.clip(image + 0.5, 0, 1)
    ax[0].imshow(image,cmap='gray')
    ax[1].imshow(colorized(gt,label_to_color),cmap=cmap, interpolation='none')

    for i in range(Config.n_samples_4_visualization):
        ax[i+2].imshow(colorized(SVD_pred[i],label_to_color),cmap=cmap, interpolation='none')
        ax[i+2+Config.n_samples_4_visualization].imshow(colorized(PW_pred[i],label_to_color),cmap=cmap, interpolation='none')
        ax[i+4+Config.n_samples_4_visualization].imshow(colorized(SACP_pred[i],label_to_color),cmap=cmap, interpolation='none')
    for i in range(Config.n_samples_4_visualization*3+2):
        ax[i].set_yticks([])
        ax[i].set_xticks([])

    fig.legend(handles=patches, loc="lower center", title="",fontsize=30,ncol=4)
    plt.tight_layout()
    plt.savefig(f'{Config.challenge}.png')
    plt.show()
    return 0

def COCO_visual(image,PW_pred,SVD_pred,SACP_pred,gt):
    if Config.challenge == 'COCO_animals':
        label_names = {0: "Background", 1: "Bird", 2: "Cat", 3: "Cow", 4: "Dog", 5: "Horse", 6: "Person", 7: "Sheep"}
    else:
        label_names = {0: "Background", 1: "Plane", 2: "Bycicle", 3: "Bus", 4: "Car", 5: "Motorbike", 6: "Person", 7: "Train"}

    cmap = plt.cm.get_cmap("tab20", len(label_names))
    label_to_color = {label: cmap(idx) for idx, label in enumerate(label_names.keys())}
    patches = []
    for label, idx in label_names.items():
        if isinstance(label, int):  # ensure the label is an integer
            patch = mpatches.Patch(color=label_to_color[label], label=f"{label} {idx}")
            patches.append(patch)
        else:
            print(f"Warning: Label {label} is not an integer. Skipping.")

    fig, ax = plt.subplots(1,Config.n_samples_4_visualization*3+2,figsize=(30,8))
    image = (image - image.min()) / (image.max() - image.min())
    ax[0].imshow(np.transpose(image,(1,2,0))[128:384,128:384,:])
    ax[1].imshow(colorized(gt,label_to_color),cmap=cmap, interpolation='none')

    for i in range(Config.n_samples_4_visualization):
        ax[i+2].imshow(colorized(SVD_pred[i],label_to_color),cmap=cmap, interpolation='none')
        ax[i+2+Config.n_samples_4_visualization].imshow(colorized(PW_pred[i],label_to_color),cmap=cmap, interpolation='none')
        ax[i+4+Config.n_samples_4_visualization].imshow(colorized(SACP_pred[i],label_to_color),cmap=cmap, interpolation='none')
    for i in range(Config.n_samples_4_visualization*3+2):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.tight_layout()
    fig.legend(handles=patches, loc="lower center", title="",fontsize=30,ncol=8)
    plt.savefig(f'{Config.challenge}.png')
    plt.show()
    return 0

def LIDC_visual(image,PW_pred,SVD_pred,SACP_pred,gt):
    label_names = {0: "Background", 1: "Cancer"}

    cmap = plt.cm.get_cmap("tab20", len(label_names))
    label_to_color = {label: cmap(idx) for idx, label in enumerate(label_names.keys())}
    patches = []
    for label, idx in label_names.items():
        if isinstance(label, int):  # ensure the label is an integer
            patch = mpatches.Patch(color=label_to_color[label], label=f"{label} {idx}")
            patches.append(patch)
        else:
            print(f"Warning: Label {label} is not an integer. Skipping.")

    fig, ax = plt.subplots(1,Config.n_samples_4_visualization*3+2,figsize=(30,8))
    image = (image - image.min()) / (image.max() - image.min())
    ax[0].imshow(image,cmap='gray')
    ax[1].imshow(colorized(gt,label_to_color),cmap=cmap, interpolation='none')

    for i in range(Config.n_samples_4_visualization):
        ax[i+2].imshow(colorized(SVD_pred[i],label_to_color),cmap=cmap, interpolation='none')
        ax[i+2+Config.n_samples_4_visualization].imshow(colorized(PW_pred[i],label_to_color),cmap=cmap, interpolation='none')
        ax[i+4+Config.n_samples_4_visualization].imshow(colorized(SACP_pred[i],label_to_color),cmap=cmap, interpolation='none')
    for i in range(Config.n_samples_4_visualization*3+2):
        ax[i].set_xticks([])
        ax[i].set_yticks([])

    plt.tight_layout()
    fig.legend(handles=patches, loc="lower center", title="",fontsize=30,ncol=2)
    plt.savefig(f'{Config.challenge}.png')
    plt.show()
    return 0
