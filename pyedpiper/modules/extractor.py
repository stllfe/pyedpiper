from abc import ABCMeta, abstractmethod
from torch import nn as nn


class Extractor(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def _get_out_channels(self) -> int:
        pass

    @property
    def out_channels(self) -> int:
        return self._get_out_channels()
