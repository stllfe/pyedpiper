import logging
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler, DataLoader
from torchvision.datasets.folder import (
    ImageFolder,
    DatasetFolder,
    pil_loader
)

from src.common.config import Config
from src.common.consts import (
    IMG_EXTENSIONS,
    NP_EXTENSIONS,
    PICKLE_EXTENSIONS,
)
from src.common.types import (
    Modes,
    DatasetTypes,
)
from src.utils.helpers import (
    numpy_loader,
    pickle_loader,
)


class BaseDataConfigurator(ABC):
    def __init__(self, config: Config):
        self._config = config
        self.datasets = {}
        self.dataloaders = {}

        self.datasets = {mode: self._build_dataset(mode) for mode in Modes}
        self.dataloaders = {mode: self._build_dataloader(mode) for mode in Modes}

    @abstractmethod
    def _build_dataset(self, mode):
        pass

    @abstractmethod
    def _build_dataloader(self, mode):
        pass

    def get_dataset(self, mode: Modes):
        try:
            return self.datasets[mode]
        except KeyError:
            return self._build_dataset(mode)

    def get_dataloader(self, mode: Modes):
        try:
            return self.dataloaders[mode]
        except KeyError:
            return self._build_dataloader(mode)


class DataConfigurator(BaseDataConfigurator):
    def _build_dataset(self, mode):
        if self._config.data_type == DatasetTypes.Images:
            dataset = ImageFolder(
                root=self._config.data_root,
                transform=self._get_transforms(mode),
            )

        elif self._config.data_type == DatasetTypes.ImagesAndMasks:
            # todo: implement me
            pass

        elif self._config.data_type == DatasetTypes.Custom:
            if self._config.data_extensions:
                extensions = tuple(self._config.data_extensions)
            else:
                extensions = '*'

            dataset = DatasetFolder(
                root=self._config.data_root,
                transform=self._get_transforms(mode),
                extensions=extensions,
                loader=self._get_file_loader(extensions)
            )
        else:
            error = '`data_type` type: {}'.format(self._config.data_type)
            logging.error(error)
            raise NotImplementedError(error)
        return dataset

    def _build_dataloader(self, mode):
        dataset = self.get_dataset(mode)
        is_weighted = self._config.data_weighted
        sampler = self._get_weighted_sampler(dataset) if is_weighted else None
        shuffle = self._config.data_shuffle if not sampler else False

        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            shuffle=shuffle,

            batch_size=self._config.batch_size,
            num_workers=self._config.num_workers,
            pin_memory=self._config.pin_memory,

            drop_last=True
        )

    def _get_transforms(self, mode: Modes):
        # todo: implement me
        pass

    @staticmethod
    def _get_class_weights(dataset):
        targets = np.array(dataset.targets)
        class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
        weights = 1. / class_sample_count
        return weights

    @staticmethod
    def _get_file_loader(extensions):
        extensions = list(map(str.lower, extensions))
        extensions_map = {
            NP_EXTENSIONS: numpy_loader,
            IMG_EXTENSIONS: pil_loader,
            PICKLE_EXTENSIONS: pickle_loader,
        }
        for supported_extensions, handler in extensions_map.items():
            if extensions in supported_extensions:
                return handler

        error = "There is no default file loader found for extensions: {} ".format(extensions)
        message = "Make sure extensions are supported, otherwise use your custom `DataConfigurator`"
        logging.error(error + message)
        raise NotImplementedError(error + message)

    def _get_weighted_sampler(self, dataset):
        targets = np.array(dataset.targets)
        weights = self._get_class_weights(dataset)

        samples_weight = np.array([weights[t] for t in targets])
        samples_weight = torch.from_numpy(samples_weight)

        return WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'),
                                     num_samples=len(targets),
                                     replacement=True)
