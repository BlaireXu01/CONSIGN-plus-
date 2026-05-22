# CONSIGN$^+$: Efficient and Risk-Aware Conformal Medical Image Segmentation via Topological Subspace Calibration

This repository contains the implementation for evaluating and optimizing uncertainty quantification (UQ) in medical image segmentation using Conformal Prediction (CP). It compares standard pixel-wise calibration methods against subspace frameworks that utilize Singular Value Decomposition (SVD) spatial priors. 

The project addresses critical deployment bottlenecks in medical imaging, such as extreme class imbalance and spatial fragmentation, by introducing targeted optimizations like Class-Specific Targets (Class Beta) and Tversky Loss engineering.

## Key Features

* **Subspace vs. Pixel-Wise Calibration:** Direct comparison between SVD-based spatial priors (CONSIGN variants) and standard pixel-wise methods (RAPS, SACP).
* **Comprehensive Ablation Studies:** Built-in support to isolate the effects of class-specific target thresholds (`class_beta_only`) and surrogate loss functions (`tversky_only`, `weighted_only`).
* **Robustness Evaluation:** Automated stress testing against Additive White Gaussian Noise (AWGN) to evaluate the topological coherence of the prediction sets.
* **Clinical Safety Metrics:** Evaluates models using standard Empirical Coverage (EC), Strict Sampled Empirical Coverage (sEC), Spatial Correlation, and the Chao Estimator for uncertainty volume.

## Repository Structure

### Core Logic
* `compare_SVD_PW.py`: The main entry point for running the calibration pipeline and ablation studies.
* `SVD_conformal_prediction_UQ.py`: Implementation of the SVD-based conformal prediction framework, including preprocessing, truncated subspace expansion, and calibration.
* `PW_conformal_prediction.py`: Implementation of baseline pixel-wise CP strategies (APS, RAPS, and locally smoothed SACP).
* `config_cp.py`: Centralized configuration class managing all hyperparameters, dataset selections (e.g., MnM2, LIDC), and ablation modes.
* `helpers_cp.py`: Utility module containing functions for stochastic sampling, metric calculations (sEC, Chao, Correlation), and data extraction.

### Evaluation & Visualization
* `noise.py`: Script to test algorithmic resilience by injecting varying levels of Gaussian noise into the mean softmax probability maps.
* `plot_metrics.py`: Generates publication-ready quantitative charts (line and bar plots) comparing metrics across different sample sizes and ablation modes.
* `visualize.py`: Generates qualitative visualizations comparing the segmentation boundaries and prediction sets of the different methods against the Ground Truth.

## Installation

Ensure you have a Python environment with the following dependencies installed:

```bash
pip install numpy scipy torch torchvision matplotlib seaborn

