import numpy as np
import matplotlib.pyplot as plt
from helpers_cp import *
from pytictoc import TicToc
import torch
from config_cp import Config
from scipy.ndimage import uniform_filter

def PW_calibration(cal_smx,cal_labels,strategy,spatial='NO'):
    print('Pixel Wise Calibration:')
    n = Config.n 

    dim = cal_smx.shape[-1] 
    mean_sample = np.mean(cal_smx,axis=1) 

    lam = 0.0 
    l = 0.01 

    converged = Config.converged
    cal_already_checked = [] 
    
    while not converged:
        for i in range(n):
            print_loading_bar(i, n)
            if i not in cal_already_checked:
                Y_cal = cal_labels[i,:] 
                X_cal = mean_sample[i,:] 

                if spatial=='NO':
                    X_flat = X_cal.reshape(Config.labels,-1)
                    sorted_idx = np.argsort(-X_flat, axis=0) 
                    sorted_vals = np.take_along_axis(X_flat, sorted_idx, axis=0)
                    cumsum = np.cumsum(sorted_vals, axis=0)
                    if strategy == 'APS':
                        num_needed = (cumsum > lam).argmax(axis=0) + 1 
                    elif strategy == 'RAPS':
                        o_ranks = np.array([1+k for k in range(Config.labels)])
                        reg = Config.theta*(o_ranks-Config.k_reg)
                        reg[reg<0]=0
                        num_needed = (cumsum + reg[:,None] > lam).argmax(axis=0) + 1 

                elif spatial=='SACP':
                    if strategy == 'APS':
                        X_avg = uniform_filter(X_cal,size=Config.neigh_size,mode='reflect')
                        X_blend = (1. - Config.neigh_weight)*X_cal + Config.neigh_weight*X_avg
                        X_flat = X_blend.reshape(Config.labels,-1)
                        sorted_idx = np.argsort(-X_flat, axis=0)
                        sorted_vals = np.take_along_axis(X_flat, sorted_idx, axis=0)
                        cum_p_avg = np.cumsum(sorted_vals, axis=0)
                        ge_mask = (cum_p_avg > lam)
                        num_needed = (np.where(ge_mask.any(axis=0),ge_mask.argmax(axis=0) + 1, Config.labels)).reshape(-1)
                    else:
                        X_avg = uniform_filter(X_cal,size=Config.neigh_size,mode='reflect')
                        X_blend = (1. - Config.neigh_weight)*X_cal + Config.neigh_weight*X_avg
                        X_flat = X_blend.reshape(Config.labels,-1)
                        sorted_idx = np.argsort(-X_flat, axis=0)
                        
                        order = np.argsort(-X_cal, axis=0) 
                        order_rows = order.transpose(1,2,0).reshape(-1,Config.labels)
                        unique_orders, inv_flat = np.unique(order_rows,axis=0,return_inverse=True) 
                        inv2d = inv_flat.reshape(Config.dim,Config.dim)
                        n_perms = unique_orders.shape[0]
                        ranks = np.arange(1, Config.labels+1)
                        reg = np.clip(Config.theta * (ranks - Config.k_reg), 0, None)
                        final_cum_ordered = np.zeros((Config.labels,Config.dim,Config.dim),dtype=X_cal.dtype)
                        
                        for pid,p_row in enumerate(unique_orders):
                            p=p_row.astype(int).tolist()
                            X_p = X_cal[p,:,:] 
                            cum_p = np.cumsum(X_p,axis=0)
                            cum_p += reg[:,None,None]
                            cum_p_neigh = uniform_filter(cum_p,size=Config.neigh_size,mode='reflect')
                            cum_p_blend = (1. - Config.neigh_weight)*cum_p + Config.neigh_weight*cum_p_neigh
                            mask = (inv2d==pid)
                            if not mask.any():
                                continue
                            final_cum_ordered[:,mask] = cum_p_blend[:,mask]

                        ge_mask = (final_cum_ordered > lam)
                        num_needed = (np.where(ge_mask.any(axis=0),ge_mask.argmax(axis=0) + 1, Config.labels)).reshape(-1)

                prediction_set = np.zeros_like(X_flat, dtype=bool)
                cols = np.arange(X_flat.shape[1])
                prediction_set[sorted_idx[:Config.labels, cols], cols] = np.less.outer(np.arange(Config.labels), num_needed).astype(bool)
                prediction_set = prediction_set.reshape(X_cal.shape)

                gt_is_in_set = prediction_set[Y_cal,np.arange(dim)[:,None],np.arange(dim)[None,:]] 
                
                score = cal_score_PW_clinical(gt_is_in_set, Y_cal, Config.diff_between_lab) 

                if score > 0.5: # True
                    cal_already_checked.append(i)

        R_hat = 1 - len(cal_already_checked)/n
        if R_hat <= (Config.alpha - (Config.B-Config.alpha)/n):
            converged = True
        elif lam>=Config.lam_max:
            print('Lambda has reached lambda_max, consider relaxing.')
            exit()
        else:
            lam += l
            if lam >= 1:
                print(f"\nLambda PW: {1}\n")
                return 1-1e-9
    print(f"\nLambda PW: {lam:.4f}\n")
    return lam

def PW_sample(lam_PW,X,N_samples,strategy,spatial):
    if spatial=='NO':
        X_flat = X.reshape(Config.labels,-1)
        sorted_idx = np.argsort(-X_flat, axis=0)
        sorted_vals = np.take_along_axis(X_flat, sorted_idx, axis=0)
        cumsum = np.cumsum(sorted_vals, axis=0)

        if strategy == 'APS':
            num_needed = (cumsum > lam_PW).argmax(axis=0) + 1
        elif strategy == 'RAPS':
            o_ranks = np.array([1+i for i in range(Config.labels)])
            reg = Config.theta*(o_ranks-Config.k_reg)
            reg[reg<0]=0
            shifted_cumsum = np.vstack([np.zeros((1, cumsum.shape[-1])), cumsum[:-1, :]])
            num_needed = (cumsum  > lam_PW).argmax(axis=0) + 1

    elif spatial=='SACP':
        if strategy == 'APS':
            X_avg = uniform_filter(X,size=Config.neigh_size,mode='reflect')
            X_blend = (1. - Config.neigh_weight)*X + Config.neigh_weight*X_avg
            X_flat = X_blend.reshape(Config.labels,-1)
            sorted_idx = np.argsort(-X_flat, axis=0)
            sorted_vals = np.take_along_axis(X_flat, sorted_idx, axis=0)
            cum_p_avg = np.cumsum(sorted_vals, axis=0)
            ge_mask = (cum_p_avg > lam_PW)
            num_needed = (np.where(ge_mask.any(axis=0),ge_mask.argmax(axis=0) + 1, Config.labels)).reshape(-1)
        else:
            X_avg = uniform_filter(X,size=Config.neigh_size,mode='reflect')
            X_blend = (1. - Config.neigh_weight)*X + Config.neigh_weight*X_avg
            X_flat = X_blend.reshape(Config.labels,-1)
            sorted_idx = np.argsort(-X_flat, axis=0)
            order = np.argsort(-X, axis=0)
            order_rows = order.transpose(1,2,0).reshape(-1,Config.labels)
            unique_orders, inv_flat = np.unique(order_rows,axis=0,return_inverse=True)
            inv2d = inv_flat.reshape(Config.dim,Config.dim)
            n_perms = unique_orders.shape[0]
            ranks = np.arange(1, Config.labels+1)
            reg = np.clip(Config.theta * (ranks - Config.k_reg), 0, None)
            final_cum_ordered = np.zeros((Config.labels,Config.dim,Config.dim),dtype=X.dtype)
            for pid,p_row in enumerate(unique_orders):
                p=p_row.astype(int).tolist()
                X_p = X[p,:,:]
                cum_p = np.cumsum(X_p,axis=0)
                cum_p += reg[:,None,None]
                cum_p_neigh = uniform_filter(cum_p,size=Config.neigh_size,mode='reflect')
                cum_p_blend = (1. - Config.neigh_weight)*cum_p + Config.neigh_weight*cum_p_neigh
                mask = (inv2d==pid)
                if not mask.any(): continue
                final_cum_ordered[:,mask] = cum_p_blend[:,mask]

            ge_mask = (final_cum_ordered > lam_PW)
            num_needed = (np.where(ge_mask.any(axis=0),ge_mask.argmax(axis=0) + 1, Config.labels)).reshape(-1)

    thresh_prediction_set = np.zeros_like(X_flat, dtype=bool)
    cols = np.arange(X_flat.shape[1])
    thresh_prediction_set[sorted_idx[:Config.labels, cols], cols] = np.less.outer(np.arange(Config.labels), num_needed).astype(bool)
    thresh_prediction_set = thresh_prediction_set.reshape(X.shape)

    prediction_set = np.sum(thresh_prediction_set,axis=0)
    vec_prediction_set = prediction_set.flatten()
    vec_thresh_prediction_set = thresh_prediction_set.reshape(Config.labels,-1)

    unc_pixels = (np.where(vec_prediction_set>1))[0]
    unc_pixels_smx = vec_thresh_prediction_set[:,unc_pixels]
    unc_pixels_samples = uq_samples(unc_pixels_smx,N_samples)

    return unc_pixels, unc_pixels_samples, vec_thresh_prediction_set

def PW_metrics(lam_PW,X,Y,N_samples,strategy,spatial='NO'):
    """
    input: calibrated lambda, test image X, gt Y, number of samples to sample
    output: strict empirical coverage (sEC), global coverage (EC), Chao, correlation
    """
    unc_pixels, unc_pixels_samples, vec_thresh_prediction_set = PW_sample(lam_PW,X,N_samples,strategy,spatial)

    ################################## Evaluate Proj  #####################################
    pred_samples = uq_samples(vec_thresh_prediction_set,N_samples)
    
    score = val_score(pred_samples.reshape(N_samples,Config.dim,Config.dim),Y)

    valid_betas = np.array([Config.current_betas[l] for l in np.unique(Y)])

    valid_samples_strict = np.ones(N_samples, dtype=bool)
    for idx, l in enumerate(np.unique(Y)):
        valid_samples_strict &= (score[idx] > Config.current_betas[l]) 
    flag_sEC = np.any(valid_samples_strict)
    flag_EC = np.any(np.mean(score, axis=0) > np.mean(valid_betas))

    if unc_pixels.shape[0]>0:
        #evaluate chao estimator
        col,count = np.unique(unc_pixels_samples.T,axis=1,return_counts=True)
        chao = count.shape[0] + (col.T[count == 1].shape[0]*(col.T[count == 1].shape[0]-1))/(2*(col.T[count == 2].shape[0]+1))
    else:
        chao = 1.

    #evaluate correlation
    if N_samples <= 1000:
        if unc_pixels_samples.shape[1]>1:
            corr_mean = torch.mean(torch.nan_to_num(torch.corrcoef(torch.abs(torch.tensor(unc_pixels_samples))), nan=0.0))
        else:
            corr_mean = 1.
    else:
        corr_mean = 0.

    n_unc_pixels = unc_pixels.shape[0]
    return np.array([flag_sEC, flag_EC, chao, corr_mean])

def PW_plot(lam_PW,X,strategy,spatial='NO'):

    _, _, vec_thresh_prediction_set = PW_sample(lam_PW,X,Config.n_samples_4_visualization,strategy,spatial)
    samples = []
    ############################ PLOT STUFF ###############################################
    pred_samples = uq_samples(vec_thresh_prediction_set,Config.n_samples_4_visualization)
    for i in range(Config.n_samples_4_visualization):

        samples.append(pred_samples[i].reshape(Config.dim,Config.dim))

    return np.array(samples)
