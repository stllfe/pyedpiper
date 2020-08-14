from abc import ABCMeta, abstractmethod

from efficientnet_pytorch import EfficientNet
from pretrainedmodels import se_resnext50_32x4d
from torch import nn as nn


class Extractor(nn.Module, metaclass=ABCMeta):

    @abstractmethod
    def _get_out_channels(self) -> int:
        pass

    @property
    def out_channels(self) -> int:
        return self._get_out_channels()


class EfficientNetExtractor(Extractor):

    def __init__(self,
                 model_no: int,
                 pretrained=True,
                 advprop=True,
                 overrides: dict = None):

        from functools import partial

        super().__init__()
        assert 0 <= model_no <= 7, "Wrong model number!"
        assert not (pretrained ^ advprop), "Argument `advprop` can be used with the pretrained version only!"

        if pretrained:
            factory_fn = partial(EfficientNet.from_pretrained, advprop=advprop)
        else:
            factory_fn = partial(EfficientNet.from_name, override_params=overrides)

        self.net = factory_fn(f"efficientnet-b{model_no}")

    def _get_out_channels(self):
        return self.net._conv_head.out_channels

    def forward(self, x):
        return self.net.extract_features(x)


class SEResNeXt50Extractor(Extractor):

    def __init__(self, pretrained=True):
        super().__init__()
        self.net = se_resnext50_32x4d(pretrained='imagenet' if pretrained else None)

    def _get_out_channels(self):
        layer4_conv = list(filter(lambda m: isinstance(m, nn.Conv2d), self.net.layer4.modules()))[-1]
        return layer4_conv.out_channels

    def forward(self, x):
        x = self.net.features(x)
        return x
