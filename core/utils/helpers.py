import logging
import pickle
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from numpy import ndarray
from torch import Tensor

from core.common.consts import TIMESTAMP_FORMAT
from core.common.decorators import file_loader


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_project_root() -> Path:
    file_dir = Path(__file__)
    parents = file_dir.parents

    core_dir = next((parent for parent in parents if parent.name == 'core'), None)
    main_py_dir = next((parent for parent in parents if list(parent.glob('main.py'))), None)

    if core_dir:
        return core_dir.parent

    if main_py_dir:
        return main_py_dir

    error = "Can't locate the root folder. Make sure you have a `core` folder or a `main.py` in your project root!"
    logging.error(error)
    raise Exception(error)


def save_checkpoint(directory, model, optimizer, epoch, loss, name=None):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    file_name = '{}/{}.pt'.format(directory, name) if name else '{}/{}_epoch.pt'.format(directory, epoch)
    torch.save(checkpoint, file_name)


def tensor_to_numpy(tensor: Tensor) -> ndarray:
    data = tensor.detach().cpu().numpy()
    return data


@file_loader
def numpy_loader(path):
    return np.load(path)


@file_loader
def pickle_loader(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


@file_loader
def torch_loader(path, location='cpu'):
    return torch.load(path, map_location=location)


@file_loader
def pil_loader(path):
    # Open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as file:
        image = Image.open(file)
        return image.convert('RGB')


def snake_to_camel(string: str):
    return ''.join(word.title() for word in string.split('_'))


def camel_to_snake(string: str):
    string = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', string).lower()


def timestamp(format=TIMESTAMP_FORMAT):
    return datetime.now().strftime(format)


def get_absolute_path(path: str) -> Path:
    """
    Returns the full path from posix one
    :param path: str or Path-like object
    :return: Path object
    """
    return Path(path).resolve()


def get_relative_path(path: str) -> Path:
    """
    Returns the path combined with the project root
    :param path: str or Path-like object
    :return: Path object
    """
    return (get_project_root() / Path(path)).resolve()
