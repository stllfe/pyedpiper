import logging
import pickle
import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from numpy import ndarray
from torch import Tensor


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


def numpy_loader(path):
    return np.load(path)


def pickle_loader(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def torch_loader(path, location='cpu'):
    return torch.load(path, map_location=location)


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
