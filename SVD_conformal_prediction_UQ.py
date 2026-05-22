import numpy as np
import matplotlib.pyplot as plt
from helpers_cp import *
import torch
from config_cp import Config
from scipy.stats.qmc import LatinHypercube
import torch.nn as nn
import torch.optim as optim
import os
import time 

def SVD_calibration(cal_smx,cal_labels):
    print(f'SVD Calibration ({Config.ablation_mode}):')
    n = Config.n
    a_bound,b_bound,U,Sig,mean_sample = SVD_preprocess(cal_smx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lam = torch.tensor(Config.lam).to(device)
    
    a_bound = torch.tensor(a_bound).to(device)
    b_bound = torch.tensor(b_bound).to(device)

    converged = Config.converged
    cal_already_checked = [] 

    while not converged:
        for i in range(n):
            print_loading_bar(i, n)
            if i not in cal_already_checked:
                Y_cal = cal_labels[i,:] 
                mean_cal = torch.tensor(mean_sample[i,:]).to(device) 
                U_cal = torch.tensor(U[i,:,:Config.K_max]).double().to(device) 
                Sig_cal = torch.tensor(Sig[i,:Config.K_max]).to(device) 

                mid_point = (a_bound[i,:]+b_bound[i,:])/2
                length = Sig_cal*lam*(b_bound[i,:]-a_bound[i,:])
                l_b = mid_point - length/2 
                up_b = mid_point + length/2 

                criterion = AblationLoss().to(device)
                
                sampler = LatinHypercube(Config.K_max)
                random_coeff = torch.tensor(sampler.random(Config.init_trials*100)).to(device)
                random_coeff = random_coeff*(length) + l_b
                try:
                    random_losses = criterion(recon_batch(random_coeff,U_cal,mean_cal), torch.tensor(Y_cal)[None,:].to(device),True)
                except:
                    random_losses = []
                    for chunk in range(20):
                        random_losses.append(criterion(recon_batch(random_coeff[chunk*50:(chunk+1)*50,:],U_cal,mean_cal), torch.tensor(Y_cal)[None,:].to(device),True))
                    random_losses = torch.stack(random_losses).view(-1)

                min_losses = torch.argsort(random_losses)
                random_coeff = random_coeff[min_losses[:Config.init_trials]]
                
                for trial in range(Config.init_trials):
                    coeff = random_coeff[trial].requires_grad_(True)
                    optimizer = optim.Adam([coeff], lr=Config.lr)
                    
                    early_count = 0
                    best_loss = float('inf')
                    patience = 5 
                    min_delta = 1e-4 
                    
                    for epoch in range(Config.epochs):
                        optimizer.zero_grad()
                        smx = recon(coeff,U_cal,mean_cal) 
                        loss = criterion(smx[None,:], torch.tensor(Y_cal)[None,:].to(device))
                            
                        loss.backward() 
                        optimizer.step() 
                        coeff.data.clamp_(min=l_b, max=up_b) 

                        if best_loss - loss.item() > min_delta:
                            best_loss = loss.item()
                            early_count = 0
                        else:
                            early_count += 1
                            
                        if early_count >= patience:
                            break 

                    sigma = recon(coeff,U_cal,mean_cal)
                    
                    score = cal_score_SVD_clinical(sigma.detach().cpu().numpy(), Y_cal, Config.diff_between_lab)

                    if score > 0.5: 
                        cal_already_checked.append(i)
                        break

        R_hat = 1 - len(cal_already_checked)/n
        print(f"New risk = {R_hat:.4f}")
        if R_hat <= (Config.alpha - (Config.B-Config.alpha)/n):
            converged = True
        elif lam>=Config.lam_max:
            print('Lambda has reached lambda_max, consider relaxing the parameters alpha and beta.')
            exit()
        else:
            error_gap = R_hat - Config.alpha
            lam += Config.step_lam

    print(f"\nLambda SVD: {lam:.2f}\n")
    return lam.detach().cpu().numpy()

def SVD_preprocess(smx):
    n = smx.shape[0]
    dim = smx.shape[-1]
    mean_sample = np.mean(smx,axis=1)
    distances = smx-mean_sample[:,None,:]
    reshaped_distances = np.transpose(distances.reshape(-1,smx.shape[1], (dim**2)*Config.labels),(0,2,1))
    reshaped_distances = torch.tensor(reshaped_distances)
    U, Sig, Vt = torch.linalg.svd(reshaped_distances, full_matrices=False)
    U = np.array(U)
    Sig = np.array(Sig)
    Vt = np.array(Vt)
    a_bound = np.zeros((n,Config.K_max))
    b_bound = np.zeros((n,Config.K_max))
    coefficients = Sig[..., None] * Vt

    for k in range(Config.K_max):
        inner_product = coefficients[:,k]
        a_bound[:,k] = np.quantile(inner_product,Config.alpha/2,method = 'linear', axis = 1)
        b_bound[:,k] = np.quantile(inner_product,1-(Config.alpha/2),method = 'linear', axis = 1)

    return a_bound,b_bound,U,Sig,mean_sample

def SVD_sample(lam_SVD, a_bound, b_bound, U, X, Sig, N_samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_cuda = torch.cuda.is_available()
    
    if is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    else:
        start_time = time.time()

    c = np.zeros((N_samples, Config.K_max))
    for k in range(Config.K_max):
        l_b = (a_bound[k]+b_bound[k])/2 - Sig[k]*lam_SVD*(b_bound[k]-a_bound[k])/2
        c[:,k] = np.random.rand(N_samples)*(Sig[k]*lam_SVD*(b_bound[k]-a_bound[k])) + l_b

    U_gpu = torch.tensor(U[:, :Config.K_max], device=device, dtype=torch.float32) 
    X_gpu = torch.tensor(X, device=device, dtype=torch.float32).view(-1, 1)

    pred_samples_list = []
    chunks = Config.chunks

    for start_idx in range(0, N_samples, chunks):
        end_idx = min(start_idx + chunks, N_samples)
        curr_chunk_size = end_idx - start_idx

        c_gpu = torch.tensor(c[start_idx:end_idx, :], device=device, dtype=torch.float32).t()
        proj_gpu = torch.mm(U_gpu, c_gpu)

        sigma_gpu = X_gpu + proj_gpu
        sigma_gpu = sigma_gpu.view(Config.labels, -1, curr_chunk_size)

        preds_gpu = torch.argmax(sigma_gpu, dim=0) 
        pred_samples_list.append(preds_gpu.cpu().numpy().T) 

    pred_samples = np.vstack(pred_samples_list) 
    unc_pixels = np.where(np.ptp(pred_samples, axis=0) > 0)[0]
    unc_pixels_samples = pred_samples[:, unc_pixels]
    
    if is_cuda:
        end_event.record()
        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        print(f"Sampling {N_samples} samples take: {elapsed_ms:.2f} ms")
    else:
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"[Sampling {N_samples} samples take: {elapsed_ms:.2f} ms")

    return unc_pixels, unc_pixels_samples, pred_samples

def SVD_metrics(lam_SVD,a_bound,b_bound,U,X,Sig,Y,N_samples):
    unc_pixels,unc_pixels_samples,pred_samples = SVD_sample(lam_SVD,a_bound,b_bound,U,X,Sig,N_samples)

    score = val_score(pred_samples.reshape(N_samples,Config.dim,Config.dim),Y) # (num_labels, N_samples)
    
    valid_betas = np.array([Config.current_betas[l] for l in np.unique(Y)])
    
    valid_samples_strict = np.ones(N_samples, dtype=bool)
    for idx, l in enumerate(np.unique(Y)):
        valid_samples_strict &= (score[idx] > Config.current_betas[l])
    flag_sEC = np.any(valid_samples_strict)    
    flag_EC = np.any(np.mean(score, axis=0) > np.mean(valid_betas))

    if unc_pixels.shape[0]>0:
        col,count = np.unique(unc_pixels_samples.T,axis=1,return_counts=True)
        chao = count.shape[0] + (col.T[count == 1].shape[0]*(col.T[count == 1].shape[0]-1))/(2*(col.T[count == 2].shape[0]+1))
    else:
        chao = 1

    if N_samples <= 1000:
        if unc_pixels_samples.shape[1]>1:
            corr_mean = torch.mean(torch.nan_to_num(torch.corrcoef(torch.abs(torch.tensor(unc_pixels_samples))), nan=0.0))
        else:
            corr_mean = 1
    else:
        corr_mean = 0

    return np.array([flag_sEC, flag_EC, chao, corr_mean])

def plot_svd_components(U, mean_sample, img_idx=0, num_pcs=3):
    """
    U: [n, Pixels*Labels, K]
    mean_sample: [n, Pixels*Labels]
    """
    print(f"Extracting Principal Component of {img_idx} ...")
    
    U_img = U[img_idx] 
    
    num_pcs = min(num_pcs, U_img.shape[1])
    labels = Config.labels
    dim = Config.dim

    fig, axes = plt.subplots(num_pcs, labels, figsize=(labels * 3, num_pcs * 3))
    
    if num_pcs == 1: axes = np.expand_dims(axes, axis=0)
    if labels == 1: axes = np.expand_dims(axes, axis=-1)

    for k in range(num_pcs):
        pc_k = U_img[:, k].reshape(labels, dim, dim)
        
        for l in range(labels):
            ax = axes[k, l]
            v_max = np.max(np.abs(pc_k[l]))
            im = ax.imshow(pc_k[l], cmap='RdBu', vmin=-v_max, vmax=v_max)
            
            if k == 0:
                ax.set_title(f"Label {l}", fontsize=12)
            if l == 0:
                ax.set_ylabel(f"PC {k+1}", rotation=0, labelpad=20, fontsize=12)
            ax.axis('off')
            
    plt.tight_layout()
    save_path = f"svd_visual_img_{img_idx}.png"
    plt.savefig(save_path)
    print(f"The visualizations are saved at: {save_path}")
    plt.show()

def SVD_preprocess(smx):
    n = smx.shape[0]
    dim = smx.shape[-1]
    mean_sample = np.mean(smx,axis=1)
    distances = smx-mean_sample[:,None,:]
    reshaped_distances = np.transpose(distances.reshape(-1,smx.shape[1], (dim**2)*Config.labels),(0,2,1))
    reshaped_distances = torch.tensor(reshaped_distances)
    U, Sig, Vt = torch.linalg.svd(reshaped_distances, full_matrices=False)
    U = np.array(U)
    Sig = np.array(Sig)
    Vt = np.array(Vt)

    if getattr(Config, 'visualize_svd', False):
        plot_svd_components(U, mean_sample, img_idx=0, num_pcs=3)
    
    a_bound = np.zeros((n,Config.K_max))
    b_bound = np.zeros((n,Config.K_max))
    coefficients = Sig[..., None] * Vt

    for k in range(Config.K_max):
        inner_product = coefficients[:,k]
        a_bound[:,k] = np.quantile(inner_product,Config.alpha/2,method = 'linear', axis = 1)
        b_bound[:,k] = np.quantile(inner_product,1-(Config.alpha/2),method = 'linear', axis = 1)


    return a_bound,b_bound,U,Sig,mean_sample


def SVD_plot(lam_SVD,a_bound,b_bound,U,X,Sig):

    _,_,pred_samples = SVD_sample(lam_SVD,a_bound,b_bound,U,X,Sig,Config.n_samples_4_visualization)

    ############################ PLOT STUFF ##############################################
    samples = []
    for i in range(Config.n_samples_4_visualization):
        samples.append(pred_samples[i,:].reshape(Config.dim,Config.dim))

    return np.array(samples)