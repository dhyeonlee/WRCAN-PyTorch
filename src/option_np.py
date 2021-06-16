import argparse
import template
import os.path
import easydict
### See option.py for detailed information.

args = easydict.EasyDict({
    'debug': False,
    'template': '.',
    'degradation': False,
    # Hardware specifications
    'n_threads': 0,
    'cpu': False,
    'n_GPUs': 1,
    'seed': 1,
    
    # Data specifications
    'dir_data': '~/Datasets/',
    'dir_demo': '../test',
    'data_train': 'DIV2K',
    'data_test': 'DIV2K',
    'data_range': '1-800/801-810',
    'ext': 'sep',
    'scale': '4',
    'patch_size': 192,
    'rgb_range': 255,
    'n_colors': 3,
    'chop': False,
    'ch_shuffle': False,
    'lr_noise_sigma': 0.0,
    'no_augment': False,
    
    # Model specifications
    'model' : 'EDSR',
    'normalization': 'None',
    'act': 'relu',
    'pre_train': '',
    'extend': '',
    'n_depth': 2,
    'n_resgroups': 4,
    'n_resblocks': 16,
    'n_feats': 64,
    'res_scale': 1.0,
    'shift_mean': True,
    'dilation': False,
    'precision': 'single',
    
    # Training Speficiations
    'reset': False,
    'test_every': 1000,
    'epochs': 300,
    'batch_size': 16,
    'split_batch': 1,
    'self_ensemble': False,
    'test_only': False,
    'gan_k': 1,
    'gan_arch': 'patch',
    
    # Optimization specifications
    'lr': 1e-4,
    'decay': '200',
    'gamma': 0.5,
    'optimizer': 'ADAM',
    'momentum': 0.9,
    'betas': (0.9, 0.999),
    'epsilon': 1e-8,
    'weight_decay': 0,
    'gclip': 0,
    
    # Loss specifications
    'loss': '1*L1',
    'skip_threshold': 1e8,
    
    # Log specifications
    'save': 'test',
    'load': '',
    'resume': 0,
    'save_models': False,
    'print_every': 100,
    'save_results': False,
    'save_gt': False
    
})


