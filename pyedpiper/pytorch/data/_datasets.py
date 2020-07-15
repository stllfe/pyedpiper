import logging
from abc import abstractmethod, ABCMeta
from copy import deepcopy
from os.path import basename
from pathlib import Path
from typing import Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from albumentations import Compose
from torch.utils.data import Dataset

log = logging.getLogger(__name__)


def file_loader(function: Callable) -> Callable:
    """
    Decorator function to add `try` block and logging for loaders.
    :return: wrapped function
    """
    from functools import wraps

    @wraps(function)
    def wrapper(path, **kwargs):

        try:
            return function(path, **kwargs)

        except Exception as e:
            filename = basename(path)
            log.error(f"Error reading file `{filename}`: {e}")
            raise e

    return wrapper


@file_loader
def numpy_loader(path):
    return np.load(str(path))


@file_loader
def plt_loader(path):
    return plt.imread(str(path))


@file_loader
def pil_loader(path):
    return Image.open(str(path))


def _clean_path(path: Union[str, Path]) -> str:
    return (str(path)).split('/')[-1].split('.')[0]


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self,
                 root: Union[str, Path],
                 labels: Union[pd.DataFrame, dict, int, str, Path],
                 key: str = 'label',
                 index: str = 'image',
                 extension: str = 'png',
                 loader: Callable = plt_loader,
                 transform: Compose = None,
                 extract_filename: Callable = None, ):

        super().__init__()

        self.root = Path(root)
        self.labels = deepcopy(labels)
        self.transform = transform
        self.loader = loader
        self.extension = extension[1:] if extension.startswith('.') else extension
        self.index = index
        self.key = key

        self._prepare_labels()

        self.files = list(self.root.glob(f'*.{self.extension}'))

        assert self.files, "Files with the specified extension can't be found!"

        self._extract_filename = self._setup_filename_extractor(user_callback=extract_filename)
        self._get_target = self._setup_target_getter()

        self.targets = [self._get_target(self._extract_filename(file)) for file in self.files]
        self.samples = list(zip(self.files, self.targets))

    def _setup_filename_extractor(self, user_callback):
        class AmbiguousFileExtensionError(Exception):
            pass

        if callable(user_callback):
            return user_callback

        if user_callback is None:
            # Validate that files in the directory have only one extension
            for path in self.files:
                path = Path(path)

                if len(path.suffixes) > 2:
                    error = ("Files have more than one extension suffix. "
                             "Please provide your own `extract_filename` function.")
                    log.error(error)
                    raise AmbiguousFileExtensionError(error)

            if self._with_extension(self.index):
                return lambda filepath: _clean_path(filepath) + f'.{self.extension}'

            return lambda filepath: _clean_path(filepath)

        else:
            error = f"`extract_filename` type `{type(user_callback)}` is not callable!"
            log.error(error)
            raise TypeError(error)

    def _with_extension(self, index: str):

        def _check(name):
            return len(Path(name).suffixes) > 0

        if isinstance(self.labels, pd.DataFrame):
            flags = self.labels[index].apply(_check)
        elif isinstance(self.labels, dict):
            flags = list(_check(key) for key in self.labels.keys())
        else:
            return False

        all_with_extension = all(flags)
        any_with_extension = any(flags)

        if any_with_extension and not all_with_extension:
            raise Exception("Some filenames in provided labels have extensions and some don't."
                            "Please provide your own `extract_filename` callback or clean up labels.")
        else:
            return all_with_extension

    def _setup_target_getter(self):

        if isinstance(self.labels, pd.DataFrame):
            # Try to setup indexing
            if isinstance(self.labels.index, pd.RangeIndex):
                try:
                    self.labels.set_index(self.index, inplace=True)
                except KeyError as e:
                    raise ValueError(f"Labels DataFrame should contain '{self.index}' column. \n{e}")

            getter = lambda idx: self.labels.loc[idx][self.key]

        elif isinstance(self.labels, dict):
            # Then it's a dict
            if self.key:
                getter = lambda idx: self.labels[idx][self.key]
            else:
                getter = lambda idx: self.labels[idx]

        elif isinstance(self.labels, int):
            getter = lambda idx: self.labels
        else:
            error = (f"Labels of type: `{type(self.labels)}` are not allowed."
                     "Use one of: `int`, `dict`, `pandas.DataFrame`")

            log.error(error)
            raise TypeError(error)

        return getter

    def __len__(self):
        return len(self.samples)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def _prepare_labels(self):
        if isinstance(self.labels, (Path, str)):
            self.labels = pd.read_csv(self.labels)


class ImageDataset(BaseDataset):

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = self.loader(path)

        if image.dtype == np.float32:
            image *= 255
            image = image.clip(0, 255).astype(np.uint8)

        if self.transform:
            image = self.transform(image=image)['image']

        return image, target
