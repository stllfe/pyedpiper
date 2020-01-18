from pathlib import Path
from src.optimizers.radam import RAdam
from torch.optim import SGD, Adam
from torch.nn import SmoothL1Loss, CrossEntropyLoss

from ignite.metrics import (
    MeanSquaredError,
    MeanAbsoluteError,
    Precision,
    Recall,
    Accuracy,
)

from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    CosineAnnealingLR
)

# project structure
CONFIGS_DIR = Path('configs')
RUNS_DIR = Path('runs')
SRC_DIR = Path('src')
MODELS_DIR = SRC_DIR / 'models'

# implemented modules
SCHEDULERS = {
    'step': StepLR,
    'cosine': CosineAnnealingLR,
    'multistep': MultiStepLR,
}

OPTIMIZERS = {
    'radam': RAdam,
    'adam': Adam,
    'sgd': SGD,
    'default': RAdam
}

METRICS = {
    'mse': MeanSquaredError,
    'mae': MeanAbsoluteError,
    'accuracy': Accuracy,
    'precision': Precision,
    'recall': Recall,
}

LOSSES = {
    'classification':
        {
            'default': CrossEntropyLoss
        },
    'regression':
        {
            'default': SmoothL1Loss
        },
}
