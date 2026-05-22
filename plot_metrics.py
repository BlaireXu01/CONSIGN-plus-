import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ==========================================
# 1. Shared Data Parsing Functions
# ==========================================

def parse_txt_gpu(filepath, is_baseline):
    """Parsing logic specific to GPU performance comparison"""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None
        
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    text = text.replace("'", "")
    lines = text.split('\n')
    merged_lines = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('lambda'): 
            continue
        if line.startswith('mean') or line.startswith('std') or line.endswith(':'):
            merged_lines.append(line)
        else:
            if len(merged_lines) > 0 and '=' in merged_lines[-1] and re.match(r'^[0-9\.\-]', line):
                merged_lines[-1] = merged_lines[-1] + ' ' + line
            else:
                merged_lines.append(line)
                
    data = {}
    current_metric = None
    for line in merged_lines:
        if "sEC (Strict Coverage):" in line and is_baseline:
            current_metric = "skip"
        elif "EC (Global Coverage):" in line and is_baseline:
            current_metric = "cov"
        elif "Coverage:" in line and not is_baseline:
            current_metric = "cov"
        elif "Chao estimator:" in line:
            current_metric = "chao"
        elif "Correlation:" in line:
            current_metric = "corr"
        elif "=" in line and current_metric != "skip" and current_metric is not None:
            parts = line.split("=")
            key_parts = parts[0].strip().split()
            stat = key_parts[0] 
            method = key_parts[1] 
            vals = np.array([float(x) for x in parts[1].split()])
            
            if current_metric not in data: 
                data[current_metric] = {}
            if method not in data[current_metric]: 
                data[current_metric][method] = {}
            data[current_metric][method][stat] = vals
            
    return data

def parse_txt_metrics(filepath):
    """Robust parsing logic for the generalized metrics comparison plots"""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return None

    with open(filepath, 'r') as f:
        content = f.read()
        
    content = content.replace("'", "")
    data = {'sEC': {}, 'EC': {}, 'Chao': {}, 'Corr': {}}
    
    blocks = {
        'sEC': r'sEC \(Strict Coverage\):(.*?)(?=EC \(Global Coverage\):|$)',
        'EC': r'EC \(Global Coverage\):(.*?)(?=Chao estimator:|$)',
        'Chao': r'Chao estimator:(.*?)(?=Correlation:|$)',
        'Corr': r'Correlation:(.*?)(?=# lambdas|$)'
    }
    
    for key_sec, pattern in blocks.items():
        match_block = re.search(pattern, content, re.DOTALL)
        if match_block:
            block_text = match_block.group(1)
            matches = re.finditer(r'(mean [A-Za-z]+|std [A-Za-z]+)\s*=\s*([\d\.\s]+)', block_text)
            for m in matches:
                key = m.group(1).strip()
                vals = [float(x) for x in m.group(2).split()]
                data[key_sec][key] = np.array(vals)
                
    return data

def load_and_clean_metrics_data(datasets, K_values, configs):
    """Helper function to load data and handle LIDC array truncation"""
    all_data = {}
    for dataset in datasets:
        all_data[dataset] = {}
        for K in K_values:
            all_data[dataset][K] = []
            for config in configs:
                filepath = f"{dataset}_{config}_{K}.txt"
                data = parse_txt_metrics(filepath)
                
                # Truncate redundant data for LIDC (indices 5 and 6)
                if data is not None and dataset == 'LIDC':
                    for sec_key, sec_val in data.items():
                        for k, arr in sec_val.items():
                            if len(arr) == 9:
                                sec_val[k] = np.delete(arr, [5, 6])
                                
                all_data[dataset][K].append(data)
    return all_data

# ==========================================
# 2. Individual Plotting Functions
# ==========================================

def plot_gpu_performance():
    M_b2 = parse_txt_gpu("MnM2_baseline_2.txt", True)
    M_b5 = parse_txt_gpu("MnM2_baseline_5.txt", True)
    M_c2 = parse_txt_gpu("MnM2_0.1_0.9_2.txt", False)
    M_c5 = parse_txt_gpu("MnM2_0.1_0.9_5.txt", False)

    L_b2 = parse_txt_gpu("LIDC_baseline_2.txt", True)
    L_b5 = parse_txt_gpu("LIDC_baseline_5.txt", True)
    L_c2 = parse_txt_gpu("LIDC_0.2_0.8_2.txt", False)
    L_c5 = parse_txt_gpu("LIDC_0.2_0.8_5.txt", False)

    datasets_info = [
        {"name": "MnM2", "b2": M_b2, "b5": M_b5, "c2": M_c2, "c5": M_c5, "alpha": 0.1},
        {"name": "LIDC", "b2": L_b2, "b5": L_b5, "c2": L_c2, "c5": L_c5, "alpha": 0.2}
    ]

    sns.set_style("whitegrid")
    colors = sns.color_palette("tab10", 4)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(32, 18))
    
    n_samples_cov = [10, 5000, 10000]
    columns_cov = [0, 5, 6]
    bar_width = 0.18
    x_indices_cov = np.arange(len(n_samples_cov))
    
    n_samples_corr = [10, 50, 100, 500, 1000]
    columns_corr = [0, 1, 2, 3, 4]
    n_samples_chao = np.array([10, 50, 100, 500, 1000, 5000, 10000])
    
    for row, ds in enumerate(datasets_info):
        b2, b5, c2, c5 = ds['b2'], ds['b5'], ds['c2'], ds['c5']
        if None in [b2, b5, c2, c5]: 
            continue 

        # Column 0: Coverage Bar Chart
        ax_cov = axes[row, 0]
        ax_cov.bar(x_indices_cov - 1.5*bar_width, c2['cov']['SVD']['mean'][columns_cov], yerr=c2['cov']['SVD']['std'][columns_cov], width=bar_width, color=colors[0], alpha=0.5, hatch='//', edgecolor='white', capsize=5)
        ax_cov.bar(x_indices_cov + 0.5*bar_width, c5['cov']['SVD']['mean'][columns_cov], yerr=c5['cov']['SVD']['std'][columns_cov], width=bar_width, color=colors[2], alpha=0.5, hatch='//', edgecolor='white', capsize=5)
        
        ax_cov.bar(x_indices_cov - 0.5*bar_width, b2['cov']['SVD']['mean'][columns_cov], yerr=b2['cov']['SVD']['std'][columns_cov], width=bar_width, color=colors[0], capsize=5)
        ax_cov.bar(x_indices_cov + 1.5*bar_width, b5['cov']['SVD']['mean'][columns_cov], yerr=b5['cov']['SVD']['std'][columns_cov], width=bar_width, color=colors[2], capsize=5)

        ax_cov.axhline(y=1-ds['alpha'], color='black', linestyle='--', linewidth=2.5)
        ax_cov.set_xticks(x_indices_cov)
        ax_cov.set_xticklabels([str(i) for i in n_samples_cov])
        ax_cov.set_xlim(-0.5, len(n_samples_cov) - 0.5)
        ax_cov.set_ylim(0.05, 1.05)
        ax_cov.tick_params(axis='both', labelsize=20)
        ax_cov.set_title(f"{ds['name']} - Average Coverage (sEC) ($\\alpha={ds['alpha']}$)", fontsize=26, fontweight='bold')
        ax_cov.set_ylabel('Coverage', fontsize=24)
        ax_cov.set_xlabel('# samples', fontsize=24)
        ax_cov.text(0.95, 0.95, "ns", transform=ax_cov.transAxes, ha='right', va='top', fontsize=25, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
        ax_cov.grid(True, axis='y', linestyle='--', alpha=0.7)

        # Column 1: Spatial Correlation
        ax_corr = axes[row, 1]
        ax_corr.plot(n_samples_corr, b2['corr']['SVD']['mean'][columns_corr], color=colors[0], marker='^', linestyle='-', linewidth=3, markersize=14, markeredgecolor="black", markeredgewidth=1.2)
        ax_corr.fill_between(n_samples_corr, b2['corr']['SVD']['mean'][columns_corr] - b2['corr']['SVD']['std'][columns_corr], b2['corr']['SVD']['mean'][columns_corr] + b2['corr']['SVD']['std'][columns_corr], color=colors[0], alpha=0.2)
        ax_corr.plot(n_samples_corr, b5['corr']['SVD']['mean'][columns_corr], color=colors[2], marker='v', linestyle='-', linewidth=3, markersize=14, markeredgecolor="black", markeredgewidth=1.2)
        ax_corr.fill_between(n_samples_corr, b5['corr']['SVD']['mean'][columns_corr] - b5['corr']['SVD']['std'][columns_corr], b5['corr']['SVD']['mean'][columns_corr] + b5['corr']['SVD']['std'][columns_corr], color=colors[2], alpha=0.2)

        ax_corr.plot(n_samples_corr, c2['corr']['SVD']['mean'][columns_corr], color=colors[0], marker='^', markerfacecolor='none', linestyle='--', linewidth=3, markersize=14, markeredgecolor=colors[0], markeredgewidth=2)
        ax_corr.plot(n_samples_corr, c5['corr']['SVD']['mean'][columns_corr], color=colors[2], marker='v', markerfacecolor='none', linestyle='--', linewidth=3, markersize=14, markeredgecolor=colors[2], markeredgewidth=2)

        ax_corr.set_ylim(-0.05, 1.05)
        ax_corr.tick_params(axis='both', labelsize=20)
        ax_corr.set_title(f"{ds['name']} - Spatial Corr. ($\\alpha={ds['alpha']}$)", fontsize=26, fontweight='bold')
        ax_corr.set_ylabel('Spatial Corr.', fontsize=24)
        ax_corr.set_xlabel('# samples', fontsize=24)
        ax_corr.text(0.95, 0.95, "ns", transform=ax_corr.transAxes, ha='right', va='top', fontsize=25, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
        ax_corr.grid(True, axis='both', linestyle='--', alpha=0.7)

        # Column 2: Chao Estimator
        ax_chao = axes[row, 2]
        ax_chao.plot(n_samples_chao, b2['chao']['SVD']['mean'], color=colors[0], marker='^', linestyle='-', linewidth=3, markersize=14, markeredgecolor="black", markeredgewidth=1.2)
        ax_chao.fill_between(n_samples_chao, b2['chao']['SVD']['mean'] - b2['chao']['SVD']['std'], b2['chao']['SVD']['mean'] + b2['chao']['SVD']['std'], color=colors[0], alpha=0.2)
        ax_chao.plot(n_samples_chao, b5['chao']['SVD']['mean'], color=colors[2], marker='v', linestyle='-', linewidth=3, markersize=14, markeredgecolor="black", markeredgewidth=1.2)
        ax_chao.fill_between(n_samples_chao, b5['chao']['SVD']['mean'] - b5['chao']['SVD']['std'], b5['chao']['SVD']['mean'] + b5['chao']['SVD']['std'], color=colors[2], alpha=0.2)

        ax_chao.plot(n_samples_chao, c2['chao']['SVD']['mean'], color=colors[0], marker='^', markerfacecolor='none', linestyle='--', linewidth=3, markersize=14, markeredgecolor=colors[0], markeredgewidth=2)
        ax_chao.plot(n_samples_chao, c5['chao']['SVD']['mean'], color=colors[2], marker='v', markerfacecolor='none', linestyle='--', linewidth=3, markersize=14, markeredgecolor=colors[2], markeredgewidth=2)

        ax_chao.set_yscale("log")
        ax_chao.tick_params(axis='both', labelsize=20)
        ax_chao.set_title(f"{ds['name']} - Chao Estimator ($\\alpha={ds['alpha']}$)", fontsize=26, fontweight='bold')
        ax_chao.set_ylabel('Chao Estimator', fontsize=24)
        ax_chao.set_xlabel('# samples', fontsize=24)
        ax_chao.text(0.95, 0.95, "ns", transform=ax_chao.transAxes, ha='right', va='top', fontsize=25, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
        ax_chao.grid(True, axis='both', linestyle='--', alpha=0.7)

    g2 = mlines.Line2D([], [], color=colors[0], marker='^', linestyle='-', linewidth=3, markersize=14, markeredgecolor="black")
    c_Vanilla2 = mlines.Line2D([], [], color=colors[0], marker='^', markerfacecolor='none', linestyle='--', linewidth=3, markersize=14, markeredgecolor=colors[0])
    g5 = mlines.Line2D([], [], color=colors[2], marker='v', linestyle='-', linewidth=3, markersize=14, markeredgecolor="black")
    c_Vanilla5 = mlines.Line2D([], [], color=colors[2], marker='v', markerfacecolor='none', linestyle='--', linewidth=3, markersize=14, markeredgecolor=colors[2])
    
    fig.legend(
        [g2, c_Vanilla2, g5, c_Vanilla5], 
        [r"$CONSIGN_2^+(Ours)$", r"$CONSIGN_2$", r"$CONSIGN_5^+(Ours)$", r"$CONSIGN_5$"],
        loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=28
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('GPU.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("GPU.png generated successfully.")

def plot_class_beta():
    sns.set_style("whitegrid")
    colors = sns.color_palette("tab10", 4)
    datasets = ['MnM2', 'LIDC']
    configs = ['baseline', 'class_beta_only']
    K_values = [2, 5]
    alphas = {'MnM2': 0.1, 'LIDC': 0.2}

    n_samples_all = [10, 50, 100, 500, 1000, 5000, 10000]
    n_samples_cov = [10, 5000, 10000]
    columns_cov = [0, 5, 6]  
    n_samples_corr = [10, 50, 100, 500, 1000]
    columns_corr = [0, 1, 2, 3, 4]  

    all_data = load_and_clean_metrics_data(datasets, K_values, configs)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(34, 16))

    method_labels = ['Baseline', 'Class Beta', 'PW (RAPS)', 'SACP']
    metrics = ['EC','sEC', 'Corr', 'Chao']
    metric_titles = ['Average Coverage (sEC)','Strict Coverage (strict sEC)', 'Correlation', 'Chao Estimator']

    legend_handles = []
    legend_labels = []

    for row, dataset in enumerate(datasets):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            
            if metric in ['sEC', 'EC']:
                bar_width = 0.1
                offsets = np.linspace(-3.5, 3.5, 8)
                x_indices = np.arange(len(n_samples_cov))
                idx_bar = 0
                
                for K in K_values:
                    ds_data = all_data[dataset][K]
                    if not ds_data or None in ds_data:
                        idx_bar += 4
                        continue
                        
                    y_arrays = [
                        ds_data[0][metric]['mean SVD'][columns_cov],
                        ds_data[1][metric]['mean SVD'][columns_cov],
                        ds_data[0][metric]['mean PW'][columns_cov],
                        ds_data[0][metric]['mean SACP'][columns_cov]
                    ]
                    yerr_arrays = [
                        ds_data[0][metric]['std SVD'][columns_cov],
                        ds_data[1][metric]['std SVD'][columns_cov],
                        ds_data[0][metric]['std PW'][columns_cov],
                        ds_data[0][metric]['std SACP'][columns_cov]
                    ]
                    
                    hatch = '' if K == 2 else '//'
                    for m_idx in range(4):
                        label = f"$CONSIGN^+_{{{K}}}$ ({method_labels[m_idx]})" if m_idx < 2 else f"{method_labels[m_idx]} (K={K})"
                        ax.bar(x_indices + offsets[idx_bar] * bar_width, y_arrays[m_idx], yerr=yerr_arrays[m_idx], 
                               width=bar_width, color=colors[m_idx], hatch=hatch,
                               capsize=4, edgecolor='black', zorder=3, label=label)
                        idx_bar += 1
                        
                ax.set_xticks(x_indices)
                ax.set_xticklabels([str(n) for n in n_samples_cov])
                target = 1.0 - alphas[dataset]
                ax.axhline(y=target, color='black', linestyle='--', linewidth=2.5, zorder=4)
                ax.set_ylim(0.0, 1.05)
                ax.set_ylabel('Coverage', fontsize=22)
                
            elif metric == 'Corr':
                for K in K_values:
                    ds_data = all_data[dataset][K]
                    if not ds_data or None in ds_data: continue
                    
                    y_arrays = [
                        ds_data[0]['Corr']['mean SVD'][columns_corr],
                        ds_data[1]['Corr']['mean SVD'][columns_corr],
                        ds_data[0]['Corr']['mean PW'][columns_corr],
                        ds_data[0]['Corr']['mean SACP'][columns_corr]
                    ]
                    yerr_arrays = [
                        ds_data[0]['Corr']['std SVD'][columns_corr],
                        ds_data[1]['Corr']['std SVD'][columns_corr],
                        ds_data[0]['Corr']['std PW'][columns_corr],
                        ds_data[0]['Corr']['std SACP'][columns_corr]
                    ]
                    
                    linestyle = '-' if K == 2 else '--'
                    markers = ['o', 's', '^', 'D'] if K == 2 else ['v', 'p', '*', 'X']
                    
                    for m_idx in range(4):
                        label = f"$CONSIGN^+_{{{K}}}$ ({method_labels[m_idx]})" if m_idx < 2 else f"{method_labels[m_idx]} (K={K})"
                        line = ax.plot(n_samples_corr, y_arrays[m_idx], label=label, color=colors[m_idx], 
                                       marker=markers[m_idx], linestyle=linestyle, linewidth=3, 
                                       markersize=12, markeredgecolor='black')[0]
                        ax.fill_between(n_samples_corr, y_arrays[m_idx] - yerr_arrays[m_idx], 
                                        y_arrays[m_idx] + yerr_arrays[m_idx], color=colors[m_idx], alpha=0.15)
                        
                        if row == 0 and col == 2:
                            legend_handles.append(line)
                            legend_labels.append(label)
                            
                ax.set_ylabel('Correlation', fontsize=22)
                
            elif metric == 'Chao':
                for K in K_values:
                    ds_data = all_data[dataset][K]
                    if not ds_data or None in ds_data: continue
                    
                    y_arrays = [
                        ds_data[0]['Chao']['mean SVD'],
                        ds_data[1]['Chao']['mean SVD'],
                        ds_data[0]['Chao']['mean PW'],
                        ds_data[0]['Chao']['mean SACP']
                    ]
                    yerr_arrays = [
                        ds_data[0]['Chao']['std SVD'],
                        ds_data[1]['Chao']['std SVD'],
                        ds_data[0]['Chao']['std PW'],
                        ds_data[0]['Chao']['std SACP']
                    ]
                    
                    linestyle = '-' if K == 2 else '--'
                    markers = ['o', 's', '^', 'D'] if K == 2 else ['v', 'p', '*', 'X']
                    
                    for m_idx in range(4):
                        label = f"$CONSIGN^+_{{{K}}}$ ({method_labels[m_idx]})" if m_idx < 2 else f"{method_labels[m_idx]} (K={K})"
                        ax.plot(n_samples_all, y_arrays[m_idx], label=label, color=colors[m_idx], 
                                marker=markers[m_idx], linestyle=linestyle, linewidth=3, 
                                markersize=12, markeredgecolor='black')
                        ax.fill_between(n_samples_all, y_arrays[m_idx] - yerr_arrays[m_idx], 
                                        y_arrays[m_idx] + yerr_arrays[m_idx], color=colors[m_idx], alpha=0.15)
                
                ax.set_yscale('log')
                ax.set_ylabel('Chao Estimator', fontsize=22)
                
            ax.set_xlabel('# samples', fontsize=20)
            ax.tick_params(axis='both', labelsize=16)
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            
            title_text = f"{dataset} - {metric_titles[col]}"
            if col in [0, 1]:
                title_text += f" ($\\alpha={alphas[dataset]}$)"
            ax.set_title(title_text, fontsize=24, fontweight='bold')

    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=4, fontsize=22, bbox_to_anchor=(0.5, 1.08))
    plt.tight_layout()
    plt.savefig('Class_Beta.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Class_Beta.png generated successfully.")

def plot_weighted_tversky():
    sns.set_style("whitegrid")
    colors = sns.color_palette("tab10", 4)
    datasets = ['MnM2', 'LIDC']
    configs = ['baseline', 'weighted_tversky']
    K_values = [2, 5]
    alphas = {'MnM2': 0.1, 'LIDC': 0.2}

    n_samples_all = [10, 50, 100, 500, 1000, 5000, 10000]
    n_samples_cov = [10, 5000, 10000]
    columns_cov = [0, 5, 6]  
    n_samples_corr = [10, 50, 100, 500, 1000]
    columns_corr = [0, 1, 2, 3, 4]  

    all_data = load_and_clean_metrics_data(datasets, K_values, configs)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(34, 16))

    method_labels = ['Baseline', 'Tversky', 'PW (RAPS)', 'SACP']
    metrics = [ 'EC','sEC', 'Corr', 'Chao']
    metric_titles = ['Average Coverage (sEC)','Strict Coverage (strict sEC)',  'Correlation', 'Chao Estimator']

    legend_handles = []
    legend_labels = []

    for row, dataset in enumerate(datasets):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            
            if metric in ['sEC', 'EC']:
                bar_width = 0.1
                offsets = np.linspace(-3.5, 3.5, 8)
                x_indices = np.arange(len(n_samples_cov))
                idx_bar = 0
                
                for K in K_values:
                    ds_data = all_data[dataset][K]
                    if not ds_data or None in ds_data:
                        idx_bar += 4
                        continue
                        
                    y_arrays = [
                        ds_data[0][metric]['mean SVD'][columns_cov],
                        ds_data[1][metric]['mean SVD'][columns_cov],
                        ds_data[0][metric]['mean PW'][columns_cov],
                        ds_data[0][metric]['mean SACP'][columns_cov]
                    ]
                    yerr_arrays = [
                        ds_data[0][metric]['std SVD'][columns_cov],
                        ds_data[1][metric]['std SVD'][columns_cov],
                        ds_data[0][metric]['std PW'][columns_cov],
                        ds_data[0][metric]['std SACP'][columns_cov]
                    ]
                    
                    hatch = '' if K == 2 else '//'
                    for m_idx in range(4):
                        label = f"$CONSIGN^+_{{{K}}}$ ({method_labels[m_idx]})" if m_idx < 2 else f"{method_labels[m_idx]} (K={K})"
                        ax.bar(x_indices + offsets[idx_bar] * bar_width, y_arrays[m_idx], yerr=yerr_arrays[m_idx], 
                               width=bar_width, color=colors[m_idx], hatch=hatch,
                               capsize=4, edgecolor='black', zorder=3, label=label)
                        idx_bar += 1
                        
                ax.set_xticks(x_indices)
                ax.set_xticklabels([str(n) for n in n_samples_cov])
                target = 1.0 - alphas[dataset]
                ax.axhline(y=target, color='black', linestyle='--', linewidth=2.5, zorder=4)
                ax.set_ylim(0.0, 1.05)
                ax.set_ylabel('Coverage', fontsize=22)
                
            elif metric == 'Corr':
                for K in K_values:
                    ds_data = all_data[dataset][K]
                    if not ds_data or None in ds_data: continue
                    
                    y_arrays = [
                        ds_data[0]['Corr']['mean SVD'][columns_corr],
                        ds_data[1]['Corr']['mean SVD'][columns_corr],
                        ds_data[0]['Corr']['mean PW'][columns_corr],
                        ds_data[0]['Corr']['mean SACP'][columns_corr]
                    ]
                    yerr_arrays = [
                        ds_data[0]['Corr']['std SVD'][columns_corr],
                        ds_data[1]['Corr']['std SVD'][columns_corr],
                        ds_data[0]['Corr']['std PW'][columns_corr],
                        ds_data[0]['Corr']['std SACP'][columns_corr]
                    ]
                    
                    linestyle = '-' if K == 2 else '--'
                    markers = ['o', 's', '^', 'D'] if K == 2 else ['v', 'p', '*', 'X']
                    
                    for m_idx in range(4):
                        label = f"$CONSIGN^+_{{{K}}}$ ({method_labels[m_idx]})" if m_idx < 2 else f"{method_labels[m_idx]} (K={K})"
                        line = ax.plot(n_samples_corr, y_arrays[m_idx], label=label, color=colors[m_idx], 
                                       marker=markers[m_idx], linestyle=linestyle, linewidth=3, 
                                       markersize=12, markeredgecolor='black')[0]
                        ax.fill_between(n_samples_corr, y_arrays[m_idx] - yerr_arrays[m_idx], 
                                        y_arrays[m_idx] + yerr_arrays[m_idx], color=colors[m_idx], alpha=0.15)
                        
                        if row == 0 and col == 2:
                            legend_handles.append(line)
                            legend_labels.append(label)
                            
                ax.set_ylabel('Correlation', fontsize=22)
                
            elif metric == 'Chao':
                for K in K_values:
                    ds_data = all_data[dataset][K]
                    if not ds_data or None in ds_data: continue
                    
                    y_arrays = [
                        ds_data[0]['Chao']['mean SVD'],
                        ds_data[1]['Chao']['mean SVD'],
                        ds_data[0]['Chao']['mean PW'],
                        ds_data[0]['Chao']['mean SACP']
                    ]
                    yerr_arrays = [
                        ds_data[0]['Chao']['std SVD'],
                        ds_data[1]['Chao']['std SVD'],
                        ds_data[0]['Chao']['std PW'],
                        ds_data[0]['Chao']['std SACP']
                    ]
                    
                    linestyle = '-' if K == 2 else '--'
                    markers = ['o', 's', '^', 'D'] if K == 2 else ['v', 'p', '*', 'X']
                    
                    for m_idx in range(4):
                        label = f"$CONSIGN^+_{{{K}}}$ ({method_labels[m_idx]})" if m_idx < 2 else f"{method_labels[m_idx]} (K={K})"
                        ax.plot(n_samples_all, y_arrays[m_idx], label=label, color=colors[m_idx], 
                                marker=markers[m_idx], linestyle=linestyle, linewidth=3, 
                                markersize=12, markeredgecolor='black')
                        ax.fill_between(n_samples_all, y_arrays[m_idx] - yerr_arrays[m_idx], 
                                        y_arrays[m_idx] + yerr_arrays[m_idx], color=colors[m_idx], alpha=0.15)
                
                ax.set_yscale('log')
                ax.set_ylabel('Chao Estimator', fontsize=22)
                
            ax.set_xlabel('# samples', fontsize=20)
            ax.tick_params(axis='both', labelsize=16)
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            
            title_text = f"{dataset} - {metric_titles[col]}"
            if col in [0, 1]:
                title_text += f" ($\\alpha={alphas[dataset]}$)"
            ax.set_title(title_text, fontsize=24, fontweight='bold')

    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=4, fontsize=22, bbox_to_anchor=(0.5, 1.08))
    plt.tight_layout()
    plt.savefig('WeightedTversky.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("WeightedTversky.png generated successfully.")

def plot_all_combined():
    sns.set_style("whitegrid")
    colors = sns.color_palette("tab10", 5)
    datasets = ['MnM2', 'LIDC']
    configs = ['baseline', 'class_beta_only', 'all_combined']
    K_values = [2, 5]
    alphas = {'MnM2': 0.1, 'LIDC': 0.2}

    n_samples_all = [10, 50, 100, 500, 1000, 5000, 10000]
    n_samples_cov = [10, 5000, 10000]
    columns_cov = [0, 5, 6]  
    n_samples_corr = [10, 50, 100, 500, 1000]
    columns_corr = [0, 1, 2, 3, 4]  

    all_data = load_and_clean_metrics_data(datasets, K_values, configs)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(36, 17))

    method_labels = ['Baseline', 'Class Beta', 'Class Beta + Tversky', 'PW (RAPS)', 'SACP']
    metrics = [ 'EC', 'sEC','Corr', 'Chao']
    metric_titles = ['Average Coverage (sEC)', 'Strict Coverage (strict sEC)', 'Correlation', 'Chao Estimator']

    legend_handles = []
    legend_labels = []

    for row, dataset in enumerate(datasets):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            
            if metric in ['sEC', 'EC']:
                bar_width = 0.08
                offsets = np.linspace(-4.5, 4.5, 10)
                x_indices = np.arange(len(n_samples_cov))
                idx_bar = 0
                
                for K in K_values:
                    ds_data = all_data[dataset][K]
                    if not ds_data or None in ds_data:
                        idx_bar += 5
                        continue
                        
                    y_arrays = [
                        ds_data[0][metric]['mean SVD'][columns_cov],
                        ds_data[1][metric]['mean SVD'][columns_cov],
                        ds_data[2][metric]['mean SVD'][columns_cov],
                        ds_data[0][metric]['mean PW'][columns_cov],
                        ds_data[0][metric]['mean SACP'][columns_cov]
                    ]
                    yerr_arrays = [
                        ds_data[0][metric]['std SVD'][columns_cov],
                        ds_data[1][metric]['std SVD'][columns_cov],
                        ds_data[2][metric]['std SVD'][columns_cov],
                        ds_data[0][metric]['std PW'][columns_cov],
                        ds_data[0][metric]['std SACP'][columns_cov]
                    ]
                    
                    hatch = '' if K == 2 else '//'
                    for m_idx in range(5):
                        label = f"$CONSIGN^+_{{{K}}}$ ({method_labels[m_idx]})" if m_idx < 3 else f"{method_labels[m_idx]} (K={K})"
                        ax.bar(x_indices + offsets[idx_bar] * bar_width, y_arrays[m_idx], yerr=yerr_arrays[m_idx], 
                               width=bar_width, color=colors[m_idx], hatch=hatch,
                               capsize=3, edgecolor='black', zorder=3, label=label)
                        idx_bar += 1
                        
                ax.set_xticks(x_indices)
                ax.set_xticklabels([str(n) for n in n_samples_cov])
                target = 1.0 - alphas[dataset]
                ax.axhline(y=target, color='black', linestyle='--', linewidth=2.5, zorder=4)
                ax.set_ylim(0.0, 1.05)
                ax.set_ylabel('Coverage', fontsize=22)
                
            elif metric == 'Corr':
                for K in K_values:
                    ds_data = all_data[dataset][K]
                    if not ds_data or None in ds_data: continue
                    
                    y_arrays = [
                        ds_data[0]['Corr']['mean SVD'][columns_corr],
                        ds_data[1]['Corr']['mean SVD'][columns_corr],
                        ds_data[2]['Corr']['mean SVD'][columns_corr],
                        ds_data[0]['Corr']['mean PW'][columns_corr],
                        ds_data[0]['Corr']['mean SACP'][columns_corr]
                    ]
                    yerr_arrays = [
                        ds_data[0]['Corr']['std SVD'][columns_corr],
                        ds_data[1]['Corr']['std SVD'][columns_corr],
                        ds_data[2]['Corr']['std SVD'][columns_corr],
                        ds_data[0]['Corr']['std PW'][columns_corr],
                        ds_data[0]['Corr']['std SACP'][columns_corr]
                    ]
                    
                    linestyle = '-' if K == 2 else '--'
                    markers = ['o', 's', '^', 'D', 'v'] if K == 2 else ['X', 'p', '*', 'h', '>']
                    
                    for m_idx in range(5):
                        label = f"$CONSIGN^+_{{{K}}}$ ({method_labels[m_idx]})" if m_idx < 3 else f"{method_labels[m_idx]} (K={K})"
                        line = ax.plot(n_samples_corr, y_arrays[m_idx], label=label, color=colors[m_idx], 
                                       marker=markers[m_idx], linestyle=linestyle, linewidth=3, 
                                       markersize=12, markeredgecolor='black')[0]
                        ax.fill_between(n_samples_corr, y_arrays[m_idx] - yerr_arrays[m_idx], 
                                        y_arrays[m_idx] + yerr_arrays[m_idx], color=colors[m_idx], alpha=0.15)
                        
                        if row == 0 and col == 2:
                            legend_handles.append(line)
                            legend_labels.append(label)
                            
                ax.set_ylabel('Correlation', fontsize=22)
                
            elif metric == 'Chao':
                for K in K_values:
                    ds_data = all_data[dataset][K]
                    if not ds_data or None in ds_data: continue
                    
                    y_arrays = [
                        ds_data[0]['Chao']['mean SVD'],
                        ds_data[1]['Chao']['mean SVD'],
                        ds_data[2]['Chao']['mean SVD'],
                        ds_data[0]['Chao']['mean PW'],
                        ds_data[0]['Chao']['mean SACP']
                    ]
                    yerr_arrays = [
                        ds_data[0]['Chao']['std SVD'],
                        ds_data[1]['Chao']['std SVD'],
                        ds_data[2]['Chao']['std SVD'],
                        ds_data[0]['Chao']['std PW'],
                        ds_data[0]['Chao']['std SACP']
                    ]
                    
                    linestyle = '-' if K == 2 else '--'
                    markers = ['o', 's', '^', 'D', 'v'] if K == 2 else ['X', 'p', '*', 'h', '>']
                    
                    for m_idx in range(5):
                        label = f"$CONSIGN^+_{{{K}}}$ ({method_labels[m_idx]})" if m_idx < 3 else f"{method_labels[m_idx]} (K={K})"
                        ax.plot(n_samples_all, y_arrays[m_idx], label=label, color=colors[m_idx], 
                                marker=markers[m_idx], linestyle=linestyle, linewidth=3, 
                                markersize=12, markeredgecolor='black')
                        ax.fill_between(n_samples_all, y_arrays[m_idx] - yerr_arrays[m_idx], 
                                        y_arrays[m_idx] + yerr_arrays[m_idx], color=colors[m_idx], alpha=0.15)
                
                ax.set_yscale('log')
                ax.set_ylabel('Chao Estimator', fontsize=22)
                
            ax.set_xlabel('# samples', fontsize=20)
            ax.tick_params(axis='both', labelsize=16)
            ax.grid(True, which='both', linestyle='--', alpha=0.7)
            
            title_text = f"{dataset} - {metric_titles[col]}"
            if col in [0, 1]: 
                title_text += f" ($\\alpha={alphas[dataset]}$)"
            ax.set_title(title_text, fontsize=24, fontweight='bold')

    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=5, fontsize=20, bbox_to_anchor=(0.5, 1.08))
    plt.tight_layout()
    plt.savefig('AllCombined.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("AllCombined.png generated successfully.")

# ==========================================
# 3. Main Execution Block
# ==========================================

if __name__ == "__main__":
    print("Executing plot scripts...")
    plot_gpu_performance()
    plot_class_beta()
    plot_weighted_tversky()
    plot_all_combined()
    print("All metric comparison figures generated successfully!")