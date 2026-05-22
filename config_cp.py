""" Config file containing all the hyeperparameters of
    the CNN and the GP
"""
import torch
import numpy as np

class Config:
    """Hyperparameters config file with Ablation Support"""
    challenge = 'MnM2'  #LIDC
    ablation_mode = 'all_combined'
    
    alpha = 0.1 # confidence 
    beta = 0.9  # global beta
    
    lam = 0.01
    step_lam =  0.01 
    lam_max = 500 
    K_max = 5 
    converged = False 
    neigh_size = (1, 7, 7) 
    neigh_weight = 0.3 
    diff_between_lab = True 
    B = 1 

    theta = 0.05
    k_reg = 0 
    pw_strategy = 'APS' 

    epochs = 10000
    init_trials = 10 
    lr = 1

    n_samples_4_metrics = [10 , 50, 100, 500, 1000, 5000, 10000] 
    n_samples_4_visualization = 2 
    chunks = 1000 
    use_weighted = False
    use_tversky = False
    current_weights = []
    current_betas = []
    alpha_fn = 1.0
    beta_fp = 0.0

    @classmethod
    def setup_dataset(cls):
        if cls.challenge in ['MnM2']:
            cls.labels = 4
            cls.n = 500
            cls.dim = 128
            cls.chunks = 10000
            cls.lr = 1
        elif cls.challenge == 'LIDC':
            cls.labels = 2
            cls.n = 700
            cls.dim = 128
            cls.chunks = 10000
            cls.lr = 1
        
        cls.k_reg = cls.labels // 2

    @classmethod
    def update_mode(cls, mode):

        cls.ablation_mode = mode
        cls.setup_dataset()
        cls.use_weighted = False
        cls.use_tversky = False
        cls.use_strict_scoring = False 
        cls.alpha_fn = 1.0
        cls.beta_fp = 0.0
        cls.current_weights = [1.0] * cls.labels
        cls.current_betas = [cls.beta] * cls.labels

        dataset_priors = {
            'MnM2': {
                'weights': [0.1, 0.3, 0.3, 0.3],    
                'betas': [0.93, 0.76, 0.71, 0.73] , 
                'alpha_fn': 0.8, 'beta_fp': 0.2
            },
            'LIDC': {
                'weights': [0.1, 0.9],               # BG, Cancer
                'betas': [0.60, 0.95],
                'alpha_fn': 0.9, 'beta_fp': 0.1
            },
        }
        
        priors = dataset_priors[cls.challenge]
                

        if mode == 'weighted_only':
            cls.use_weighted = True
            cls.current_weights = priors['weights']
        elif mode == 'tversky_only':
            cls.use_tversky = True
            cls.alpha_fn = priors['alpha_fn']
            cls.beta_fp = priors['beta_fp']
        elif mode == 'class_beta_only':
            cls.current_betas = priors['betas']
            cls.use_strict_scoring = True 
        elif mode == 'loss_combined': 
            cls.use_weighted = True
            cls.use_tversky = True
            cls.current_weights = priors['weights']
            cls.alpha_fn = priors['alpha_fn']
            cls.beta_fp = priors['beta_fp']         
        elif mode == 'tversky_beta_combined':
            cls.use_weighted = False
            cls.use_tversky = True
            cls.alpha_fn = priors['alpha_fn']
            cls.beta_fp = priors['beta_fp']
            cls.use_strict_scoring = True
            cls.current_betas = priors['betas']
        elif mode == 'weighted_beta_combined':
            cls.use_weighted = True         
            cls.use_tversky = False          
            cls.current_weights = priors['weights']
            cls.use_strict_scoring = True  
            cls.current_betas = priors['betas']            
        elif mode == 'all_combined':
            cls.use_weighted = True
            cls.use_tversky = True
            cls.current_weights = priors['weights']
            cls.current_betas = priors['betas']
            cls.alpha_fn = priors['alpha_fn']
            cls.beta_fp = priors['beta_fp']
            cls.use_strict_scoring = True

Config.setup_dataset()