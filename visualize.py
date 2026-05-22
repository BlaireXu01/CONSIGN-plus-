import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from helpers_cp import *
from config_cp import Config
from PW_conformal_prediction import PW_plot
from SVD_conformal_prediction_UQ import SVD_preprocess, SVD_plot


def get_robust_lambdas(file_path):
    lambdas = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find metrics file: {file_path}")
        
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if "# lambdas split" in line:
                val_line = lines[idx+2].strip()
                parts = val_line.split()
                if len(parts) >= 3:
                    lambdas.append(np.array([float(parts[0]), float(parts[1]), float(parts[2])]))
    return np.array(lambdas)


CURRENT_DATASET = "MnM2"  

TARGET_SPLIT = 0                 
FOLDER_PATH = "metrics_figure"   
K_VALUES = [2, 5]


DATASET_CONFIGS = {
    "MnM2": {
        "colors": ['#1f77b4', '#d62728', '#f7b6d2', '#9edae5'],
        "labels": ['0 Background', '1 Left Ventricle', '2 Myocardium', '3 Right Ventricle'],
        "vmax": 3
    },
    "LIDC": {
        "colors": ['#1f77b4', '#9edae5'],
        "labels": ['0 Background', '1 Cancer'],
        "vmax": 1
    }
}

if CURRENT_DATASET not in DATASET_CONFIGS:
    raise ValueError(f"Unsupported dataset: {CURRENT_DATASET}")

curr_config = DATASET_CONFIGS[CURRENT_DATASET]
custom_cmap = ListedColormap(curr_config["colors"])
VMAX = curr_config["vmax"]
LEGEND_COLORS = curr_config["colors"]
LEGEND_LABELS = curr_config["labels"]

FILE_TEMPLATES = {
    "CONSIGN+-Full": f"{CURRENT_DATASET}_all_combined_{{}}.txt",
    "CONSIGN+-Beta": f"{CURRENT_DATASET}_class_beta_only_{{}}.txt",
    "CONSIGN+-Loss": f"{CURRENT_DATASET}_weighted_tversky_{{}}.txt",
    "CONSIGN+-Base": f"{CURRENT_DATASET}_baseline_{{}}.txt"
}

MODEL_NAMES = [
    "CONSIGN+-Full", 
    "CONSIGN+-Beta", 
    "CONSIGN+-Loss", 
    "CONSIGN+-Base", 
    "PW", 
    "SACP"
]


def draw_subplot(ax, img, prediction=None, title=""):
    if prediction is None:
        ax.imshow(img, cmap='gray')
    else:
        if prediction.ndim == 3:
            mask = prediction[0]
        else:
            mask = prediction
            
        ax.imshow(mask, cmap=custom_cmap, vmin=0, vmax=VMAX, interpolation='nearest')
        
    ax.set_title(title, fontsize=20, pad=15)
    ax.axis('off')


print(f"\n=======================================================")
print(f"        Starting Visualization - Dataset: {CURRENT_DATASET}")
print(f"        Target Split: {TARGET_SPLIT}")
print(f"=======================================================\n")

save_dir = f"saved_visualizations/{CURRENT_DATASET}_Combined/split_{TARGET_SPLIT}"
os.makedirs(save_dir, exist_ok=True)

smx, labels, imgs, c_splits, v_splits = extract_softmax()
v_split = v_splits[TARGET_SPLIT]
val_smx = smx[v_split, :]
val_labels = labels[v_split, :]
val_imgs = imgs[v_split, :]
n_test = val_smx.shape[0]

for i in range(n_test):
    fig = plt.figure(figsize=(30, 9))
    gs = GridSpec(2, 9, figure=fig, width_ratios=[0.4] + [2]*8, wspace=0.05, hspace=0.25, bottom=0.12)
    
    ax_label = fig.add_subplot(gs[:, 0]) 
    ax_label.text(0.5, 0.5, CURRENT_DATASET, fontsize=22, fontweight='bold', 
                  ha='center', va='center', transform=ax_label.transAxes)
    ax_label.axis('off')
        
    axes = np.empty((2, 8), dtype=object)
    for r in range(2):
        for c in range(8):
            axes[r, c] = fig.add_subplot(gs[r, c+1]) 

    X = np.mean(val_smx, axis=1)[i, :]
    Y = val_labels[i, :]
    img = val_imgs[i]
    
    for row_idx, k in enumerate(K_VALUES):
        Config.K = k 
        
        a_bound, b_bound, U, Sig, mean_sample = SVD_preprocess(val_smx)
        U_test = U[i, :]
        Sig_test = Sig[i, :]
        
        baseline_file = os.path.join(FOLDER_PATH, FILE_TEMPLATES["CONSIGN+-Base"].format(k))
        base_lambdas = get_robust_lambdas(baseline_file)
        lam_PW = base_lambdas[TARGET_SPLIT, 1]
        lam_SACP = base_lambdas[TARGET_SPLIT, 2]
        
        draw_subplot(axes[row_idx, 0], img, title=f"Image (K={k})" if row_idx==0 else f"(K={k})")
        draw_subplot(axes[row_idx, 1], img, prediction=Y, title="Ground Truth" if row_idx==0 else "")
        
        for col_offset, model_name in enumerate(MODEL_NAMES):
            ax = axes[row_idx, col_offset + 2] 
            
            if model_name == "PW":
                samples = PW_plot(lam_PW, X, Config.pw_strategy)
            elif model_name == "SACP":
                samples = PW_plot(lam_SACP, X, Config.pw_strategy, 'SACP')
            else:
                file_path = os.path.join(FOLDER_PATH, FILE_TEMPLATES[model_name].format(k))
                lambdas = get_robust_lambdas(file_path)
                lam_SVD = lambdas[TARGET_SPLIT, 0]
                samples = SVD_plot(lam_SVD, a_bound[i, :], b_bound[i, :], U_test, mean_sample[i, :], Sig_test)
            
            if row_idx == 0:
                display_title = model_name.replace("CONSIGN+-", r"CONSIGN$^+$-")
            else:
                display_title = ""
                
            draw_subplot(ax, img, prediction=samples, title=display_title)

    legend_patches = [mpatches.Patch(color=LEGEND_COLORS[j], label=LEGEND_LABELS[j]) for j in range(len(LEGEND_LABELS))]
    fig.legend(handles=legend_patches, loc='lower center', ncol=len(LEGEND_LABELS), 
               fontsize=20, bbox_to_anchor=(0.5, 0.02), frameon=True)

    img_name = f"combined_compare_{i+1}.png"
    save_path = os.path.join(save_dir, img_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=250, facecolor='white')
    plt.close('all') 
    
    print(f"Saved {CURRENT_DATASET} plot: {img_name} ({i+1}/{n_test})")

print(f"\nAll plots for {CURRENT_DATASET} Split {TARGET_SPLIT} saved to: {save_dir}/")