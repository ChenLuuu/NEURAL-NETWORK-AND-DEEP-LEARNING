# you may do your own hyperparameter search job here.
import numpy as np

# Model Architecture Parameters
MLP_ARCHITECTURE = {
    'hidden_dims': [600, 10],  # 784 -> 600 -> 10
    'activation': 'ReLU',
    'weight_decay': [1e-4, 1e-4],
    'l2_reg': 1e-4
}

CNN_ARCHITECTURE = {
    'conv1': {
        'in_channels': 1,
        'out_channels': 8,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    'conv2': {
        'in_channels': 8,
        'out_channels': 16,
        'kernel_size': 3,
        'stride': 1,
        'padding': 1
    },
    'linear1': {
        'in_dim': 784,
        'out_dim': 100
    },
    'linear2': {
        'in_dim': 100,
        'out_dim': 10
    }
}

# Training Parameters
TRAINING_PARAMS = {
    'batch_size': 32,
    'num_epochs': 5,
    'log_iters': 100,
    'random_seed': 309
}

# Optimizer Parameters
OPTIMIZER_PARAMS = {
    'optimizer_type': 'MomentGD',  # or 'SGD'
    'init_lr': 0.1,
    'momentum': 0.9  # only for MomentGD
}

# Learning Rate Scheduler Parameters
SCHEDULER_PARAMS = {
    'scheduler_type': 'MultiStepLR',  # or 'ExponentialLR'
    'milestones': [800, 2400, 4000],  # for MultiStepLR
    'gamma': 0.5,  # for MultiStepLR
    # 'gamma': 0.95  # for ExponentialLR
}