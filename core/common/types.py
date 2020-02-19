from enum import Enum


class StrEnum(str, Enum):
    def __str__(self):
        return str(self.value)


class Modes(StrEnum):
    Train = 'train'
    Val = 'val'
    Test = 'test'


class DataTypes(StrEnum):
    Images = 'images'
    ImagesAndMasks = 'images_and_masks'
    Custom = 'custom'


class TaskTypes(StrEnum):
    Regression = 'regression'
    Classification = 'classification'
    Segmentation = 'segmentation'
    Generation = 'generation'
    ObjectDetection = 'object_detection'
    Custom = 'custom'


class Devices(StrEnum):
    CPU = 'cpu'
    GPU = 'cuda'
    GPU0 = 'cuda:0'
    GPU1 = 'cuda:1'
    GPU2 = 'cuda:2'
    GPU3 = 'cuda:3'
