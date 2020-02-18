from pathlib import Path

from torch.nn import SmoothL1Loss, CrossEntropyLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    CosineAnnealingLR
)

from src.common.types import TaskTypes
from src.optimizers.radam import RAdam

# project structure
CONFIGS_DIR = Path('configs')
RUNS_DIR = Path('runs')
SRC_DIR = Path('src')

MODELS_DIR = SRC_DIR / 'models'

# default modules
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

LOSSES = {
    TaskTypes.Classification: CrossEntropyLoss,
    TaskTypes.Regression: SmoothL1Loss,
    **{task_type: None for task_type in TaskTypes},
}

IMG_EXTENSIONS = ('jpg', 'jpeg', 'png', 'ppm', 'bmp', 'pgm', 'tif', 'tiff', 'webp')
NP_EXTENSIONS = ('npy',)
PICKLE_EXTENSIONS = ('pickle',)
TORCH_EXTENSIONS = ('pth', 'pt')
