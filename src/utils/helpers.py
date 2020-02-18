import logging
import os
import pickle
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from src.common.config import Config
from src.common.consts import CONFIGS_DIR
from src.common.decorators import file_loader


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def get_project_root() -> Path:
    file_dir = Path(__file__)
    parents = file_dir.parents

    src_dir = next((parent for parent in parents if parent.name == 'src'), None)
    main_py_dir = next((parent for parent in parents if list(parent.glob('main.py'))), None)

    if src_dir:
        return src_dir.parent

    if main_py_dir:
        return main_py_dir

    error = "Can't locate the root folder. Make sure you have a `src` folder or a `main.py` in your project root!"
    logging.error(error)
    raise Exception(error)


def order_by_date_modified(files, latest_first=False) -> List:
    """
    Orders files by the last time modified.
    :param files:
        list of PosixPath objects or string paths to sort.
    :param latest_first:
        whether or not to start from the latest file
    :return:
        list with the same objects as in files but with correct order
    """
    files.sort(key=os.path.getmtime, reverse=latest_first)
    return files


def save_checkpoint(directory, model, optimizer, epoch, loss, name=None):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss
    }
    file_name = '{}/{}.pt'.format(directory, name) if name else '{}/{}_epoch.pt'.format(directory, epoch)
    torch.save(checkpoint, file_name)


def load_configuration() -> Config:
    root = get_project_root()
    config_paths = (root / CONFIGS_DIR).glob('**/*.json')
    config_paths = list(config_paths)

    if not config_paths:
        error = "Can't locate configuration files. Place `file.json` in `{}` first!"
        error = error.format(CONFIGS_DIR)
        logging.error(error)
        raise FileNotFoundError(error)

    config = Config()

    for config_path in config_paths:
        config.add_file(config_path)

    config.load()
    return config


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
