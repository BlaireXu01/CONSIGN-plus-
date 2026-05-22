import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap
matplotlib.use('Agg') 
from helpers_cp import *
from config_cp import Config
from PW_conformal_prediction import PW_calibration, PW_metrics
from SVD_conformal_prediction_UQ import SVD_calibration, SVD_preprocess, SVD_metrics, SVD_sample

def visualize_stress_test(idx, X_noisy, Y, pred_samples_svd, lam_PW, noise_level, raw_img):
    noise_for_raw = np.random.normal(0, noise_level, raw_img.shape)
    noisy_raw = np.clip(raw_img + noise_for_raw, 0, 1)
    
    svd_sample = pred_samples_svd[0] 
    
    dim = Config.dim
    mask = X_noisy >= (1 - lam_PW)
    random_scores = mask.astype(float) * np.random.rand(*mask.shape)
    pw_sample = np.argmax(random_scores, axis=0) 

    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    colors[0, -1] = 0.0  
    my_cmap = ListedColormap(colors)

    fig, axes = plt.subplots(1, 4, figsize=(22, 7))
    title_size = 18 

    axes[0].imshow(noisy_raw, cmap='gray')
    axes[0].set_title(f"Noisy Input MRI\n(Noise Level {noise_level})", fontsize=title_size)

    axes[1].imshow(noisy_raw, cmap='gray') 
    axes[1].imshow(Y, cmap=my_cmap, alpha=0.6) 
    axes[1].set_title("Ground Truth Overlay", fontsize=title_size)

    axes[2].imshow(noisy_raw, cmap='gray') 
    axes[2].imshow(svd_sample, cmap=my_cmap, alpha=0.6) 
    axes[2].set_title("CONSIGN (SVD) Overlay", fontsize=title_size)

    axes[3].imshow(noisy_raw, cmap='gray') 
    axes[3].imshow(pw_sample, cmap=my_cmap, alpha=0.6) 
    axes[3].set_title("PW (RAPS) Overlay", fontsize=title_size)

    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout(pad=2.0)
    save_path = f"overlay_stress_test_noise{noise_level}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

parser = argparse.ArgumentParser(description="Run MIA Course Project Stress Test")
parser.add_argument('--mode', type=str, required=True, help="Ablation mode (e.g. baseline, all_combined)")
parser.add_argument('--split', type=int, default=-1, help="Split to run (0-4). -1 runs all.")
args = parser.parse_args()
mode = args.mode
target_split = args.split

print(f'\nStarting calibration for {Config.challenge} dataset')
smx, labels, imgs, c_splits, v_splits = extract_softmax()
Config.update_mode(mode)

cross_val_metrics_SVD = []
cross_val_metrics_PW = []
cross_val_metrics_SACP = []
lambdas = []

for cal_idx, (c_split, v_split) in enumerate(zip(c_splits, v_splits)):
    if target_split != -1 and cal_idx != target_split:
        continue
    
    print(f'\n########## Split {cal_idx+1} #############\n')
    cal_smx = smx[c_split, :]
    cal_labels = labels[c_split, :]
    
    val_smx = smx[v_split, :]
    val_labels = labels[v_split, :]
    val_imgs = imgs[v_split, :] 

    if not Config.converged:
        lam_PW = PW_calibration(cal_smx, cal_labels, Config.pw_strategy)
        lam_SVD = SVD_calibration(cal_smx, cal_labels)
        lam_SACP = PW_calibration(cal_smx, cal_labels, Config.pw_strategy, 'SACP')
    else:
        try:
            lams = extract_lambdas(f"metrics/{Config.challenge}_{mode}_{Config.K_max}.txt")
            lam_SVD, lam_PW, lam_SACP = lams[cal_idx]
        except:
            print("Missing lambda file, falling back to calibration...")
            lam_PW = PW_calibration(cal_smx, cal_labels, Config.pw_strategy)
            lam_SVD = SVD_calibration(cal_smx, cal_labels)
            lam_SACP = PW_calibration(cal_smx, cal_labels, Config.pw_strategy, 'SACP')

    lambdas.append(np.array([lam_SVD, lam_PW, lam_SACP]))

    n_test = val_smx.shape[0]
    softmax_mean_raw = np.mean(val_smx, axis=1) 
    a_bound, b_bound, U, Sig, mean_sample_raw = SVD_preprocess(val_smx)


    if getattr(Config, 'add_noise_test', False):
        i_vis = 0 
        X_vis = mean_sample_raw[i_vis, :].copy()
        X_vis = np.clip(X_vis + np.random.normal(0, Config.noise_level, X_vis.shape), 0, 1)
        _, _, pred_samples_vis = SVD_sample(lam_SVD, a_bound[i_vis,:], b_bound[i_vis,:], U[i_vis], X_vis, Sig[i_vis], 10)
        visualize_stress_test(i_vis, X_vis.reshape(Config.labels, Config.dim, Config.dim), 
                              val_labels[i_vis,:], pred_samples_vis.reshape(10, Config.dim, Config.dim), 
                              lam_PW, Config.noise_level, val_imgs[i_vis])

    metrics_PW, metrics_SVD, metrics_SACP = [], [], []

    for N_samples in Config.n_samples_4_metrics:
        print(f'Evaluating metrics with {N_samples} samples...')
        avg_metrics_PW, avg_metrics_SVD, avg_metrics_SACP = 0.0, 0.0, 0.0

        for i in range(n_test):
            X_pw = softmax_mean_raw[i, :].copy()
            X_svd = mean_sample_raw[i, :].copy()
            Y = val_labels[i, :]
            
            if getattr(Config, 'add_noise_test', False):
                noise_level = Config.noise_level
                X_pw = np.clip(X_pw + np.random.normal(0, noise_level, X_pw.shape), 0, 1)
                X_svd = np.clip(X_svd + np.random.normal(0, noise_level, X_svd.shape), 0, 1)

            avg_metrics_PW += PW_metrics(lam_PW, X_pw, Y, N_samples, Config.pw_strategy)
            avg_metrics_SACP += PW_metrics(lam_SACP, X_pw, Y, N_samples, Config.pw_strategy, 'SACP')
            avg_metrics_SVD += SVD_metrics(lam_SVD, a_bound[i, :], b_bound[i, :], U[i], X_svd, Sig[i], Y, N_samples)

        metrics_PW.append(avg_metrics_PW / n_test)
        metrics_SVD.append(avg_metrics_SVD / n_test)
        metrics_SACP.append(avg_metrics_SACP / n_test)

    cross_val_metrics_PW.append(metrics_PW)
    cross_val_metrics_SVD.append(metrics_SVD)
    cross_val_metrics_SACP.append(metrics_SACP)

os.makedirs('metrics_noise', exist_ok=True)
filename = f"metrics_noise/{Config.challenge}_{mode}_{Config.K_max}.txt"

cv_svd = np.array(cross_val_metrics_SVD)
cv_pw = np.array(cross_val_metrics_PW)
cv_sacp = np.array(cross_val_metrics_SACP)

m_svd, s_svd = np.mean(cv_svd, axis=0), np.std(cv_svd, axis=0)
m_pw, s_pw = np.mean(cv_pw, axis=0), np.std(cv_pw, axis=0)
m_sacp, s_sacp = np.mean(cv_sacp, axis=0), np.std(cv_sacp, axis=0)

titles = ['sEC (Strict Coverage)', 'EC (Global Coverage)', 'Chao estimator', 'Correlation']

with open(filename, "w") as f:
    for i in range(m_svd.shape[1]):
        f.write(f"{titles[i]}:\n")
        f.write(f"mean SVD = {' '.join(f'{v:.3f}' for v in m_svd[:, i])}\n")
        f.write(f"std SVD = {' '.join(f'{v:.3f}' for v in s_svd[:, i])}\n\n")
        f.write(f"mean PW = {' '.join(f'{v:.3f}' for v in m_pw[:, i])}\n")
        f.write(f"std PW = {' '.join(f'{v:.3f}' for v in s_pw[:, i])}\n\n")
        f.write(f"mean SACP = {' '.join(f'{v:.3f}' for v in m_sacp[:, i])}\n")
        f.write(f"std SACP = {' '.join(f'{v:.3f}' for v in s_sacp[:, i])}\n\n")
    for i, arr in enumerate(lambdas):
        f.write(f"# lambdas split {i+1}\n")
        np.savetxt(f, arr[None, :], fmt="%.3f", header="lambda", comments='')
        f.write("\n")

print("\n" + "="*50)
print(f"sEC (Strict Coverage)  - noise_level: {getattr(Config, 'noise_level', 0)}")
print("-" * 50)
print(f"{'Samples':<10} | {'SVD (CONSIGN)':<15} | {'PW (RAPS)':<15} | {'SACP':<10}")
print("-" * 50)
for idx, n_samp in enumerate(Config.n_samples_4_metrics):
    print(f"{n_samp:<10} | {m_svd[idx, 0]:<15.4f} | {m_pw[idx, 0]:<15.4f} | {m_sacp[idx, 0]:<10.4f}")
print("="*50 + "\n")

print(f"Results are saved at: {filename}")