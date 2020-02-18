from enum import Enum


class StrEnum(str, Enum):
    def __str__(self):
        return str(self.value)


class Modes(StrEnum):
    Train = 'train'
    Val = 'val'
    Test = 'test'


class DatasetTypes(StrEnum):
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
