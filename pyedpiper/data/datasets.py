import logging
import numpy as np
import pandas as pd

from abc import ABCMeta
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import (
    Callable,
    Iterable,
    Optional,
    Union,
)

from torch.utils.data import Dataset

from .utils import plt_loader

log = logging.getLogger(__name__)


def _clean_path(path: Union[str, Path]) -> str:
    return (str(path)).split('/')[-1].split('.')[0]


class BaseDataset(Dataset, metaclass=ABCMeta):

    def __init__(self,
                 root: Union[str, Path],
                 labels: Union[pd.DataFrame, dict, int, str, Path],
                 key: str,
                 index: str,
                 extensions: Iterable[str],
                 loader: Callable,
                 transform: Optional[Callable] = None,
                 extract_filename: Optional[Callable] = None):

        super().__init__()

        self.root = Path(root)
        self.labels = deepcopy(labels)
        self.transform = transform
        self.loader = loader
        self.extensions = extensions
        self.index = index
        self.key = key
        self.files = list()

        self._prepare_extensions()
        self._prepare_labels()
        self._prepare_files()

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

            if self._with_extension(self.index):
                # Validate that files in the directory have only one extension
                for path in self.files:

                    if len(path.suffixes) > 1:
                        error = ("Files have more than one extension. "
                                 "Please provide your own `extract_filename` function.")
                        log.error(error)
                        raise AmbiguousFileExtensionError(error)

                return lambda filepath: filepath

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
                    raise ValueError(f"Labels DataFrame should contain the '{self.index}' column. \n{e}")

            getter = lambda idx: self.labels.loc[idx][self.key]

        elif isinstance(self.labels, dict):
            # Then it's a dict
            if self.key:
                item = next(iter(self.labels.values()))
                assert isinstance(item, dict), (
                    f"Using `key` attribute with dict typed labels implies nested dicts. Got `{type(item)}` instead."
                )
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

    def _prepare_extensions(self):
        self.extensions = tuple(map(lambda ext: ext[1:] if ext.startswith('.') else ext, self.extensions))

    def _prepare_files(self):
        files = list()
        for ext in self.extensions:
            files += list(self.root.glob(f'*.{ext}'))

        self.files = files
        assert self.files, "Files with the specified extensions can't be found!"


def _get_unified_transform(t: Callable):
    import importlib.util

    if importlib.util.find_spec("albumentations") is not None:
        import albumentations as alb
        if isinstance(t, alb.Compose):
            def wrapper(image):
                return t(image=image)['image']

            return wrapper

    return t


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class ImageDataset(BaseDataset):

    def __init__(self,
                 root: Union[str, Path],
                 labels: Union[pd.DataFrame, dict, int, str, Path],
                 key: str,
                 index: str,
                 extensions: Iterable[str] = IMG_EXTENSIONS,
                 loader: Callable = plt_loader,
                 transform: Optional[Callable] = None,
                 extract_filename: Optional[Callable] = None):

        super().__init__(
            root=root,
            labels=labels,
            key=key,
            index=index,
            extensions=extensions,
            loader=loader,
            transform=transform,
            extract_filename=extract_filename,
        )
        self.transform = _get_unified_transform(self.transform)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = self.loader(path)

        if image.dtype == np.float32:
            image *= 255
            image = image.clip(0, 255).astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        return image, target
