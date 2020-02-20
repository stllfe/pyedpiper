import json
import logging
import os

from datetime import datetime
from pathlib import Path
from typing import Mapping

from core.common.consts import CONFIGS_DIR, TIMESTAMP_FORMAT
from core.common.types import (
    Devices,
    TaskTypes,
    DataTypes,
)
from core.utils.helpers import get_project_root, snake_to_camel
from core.utils.validations import is_implemented_type
from attrdict import AttrDict


# class AttrDict(dict):
#     """ Nested Attribute Dictionary
#
#     A class to convert a nested Dictionary into an object with key-values
#     accessible using attribute notation (AttrDict.attribute) in addition to
#     key notation (Dict["key"]). This class recursively sets Dicts to objects,
#     allowing you to recurse into nested dicts (like: AttrDict.attr.attr)
#     """
#
#     def __init__(self, mapping=None):
#         super(AttrDict, self).__init__()
#         if mapping is not None:
#             for key, value in mapping.items():
#                 self.__setitem__(key, value)
#
#     def __setitem__(self, key, value):
#         if isinstance(value, dict):
#             value = AttrDict(value)
#         super(AttrDict, self).__setitem__(key, value)
#         self.__dict__[key] = value  # for code completion in editors
#
#     def __getattr__(self, item):
#         try:
#             return super().__getitem__(item)
#         except KeyError:
#             raise AttributeError(item)
#
#     __setattr__ = __setitem__


class Config(AttrDict):
    """
    This class loads json files as AttrDicts.

    Provides two ways of accessing configuration parameters:
        - per each added file:
            e.g. you loaded file `general.json` with `load(path)` method.
            then access any related properties coming from this file by simply:
            `config.general.property`
        - per each property alone:
            access any property from any loaded file by simply:
            `config.property`

    Keep in mind that any overlapping properties will be overriden by the latest added one.
    As well as changing property with a global (per property) access will change its value throughout the whole config.
    E.g. assuming that initially:
        `config.general.property == config.property == 1`
        setting `config.property = 2` will set `config.general.property` to 2 as well.

    (!) Note that it returns `None` in case of a missing key.
    Its actually more natural for configuration file as it basically equals to `not stated`.
    """

    custom = AttrDict()
    _bound_parameters = AttrDict()
    _files = []

    def bound(self, parameters: Mapping):
        self._bound_parameters.update(parameters)

    def add_file(self, path):
        path = Path(path)
        if not path.exists() or not path.is_file():
            error = "Can't allocate configuration file with path: `{}`".format(path)
            logging.error(error)
            raise FileNotFoundError(error)
        if path not in self._files:
            self._files.append(path)
        self._order_files_by_date_modified()

    def load(self, path=None):
        if path:
            self.add_file(path)

        for file in self._files:
            self._load_file(file)

        self._reindex_parameters()

    def _reindex_parameters(self):
        # purely for code completion in editors
        for parameter, value in self._bound_parameters.items():
            super(Config, self).__dict__[parameter] = value

    def _order_files_by_date_modified(self):
        self._files.sort(key=os.path.getmtime)

    def _load_file(self, path):
        with open(path.as_posix(), 'r+') as file:
            file_name = path.name.split('.')[0]
            parameters = json.load(file)

        self.bound(parameters)
        self.__setitem__(file_name, parameters)

    def _repair_consistency(self, parameter, value):
        for group, group_parameters in self.items():
            if parameter in group_parameters.keys():
                self[group][parameter] = value

    def __getitem__(self, parameter):
        if parameter in self._bound_parameters.keys():
            return self._bound_parameters[parameter]
        try:
            return super().__getitem__(parameter)
        except KeyError:
            return

    def __getattr__(self, item):
        try:
            super(Config).__getitem__(item)
        except KeyError:
            return

    def __setitem__(self, parameter, value):
        if parameter in self._bound_parameters.keys():
            self._bound_parameters[parameter] = value
            self._repair_consistency(parameter, value)
            self._reindex_parameters()
            return

        super(Config, self).__setitem__(parameter, value)

    # __setattr__ = __setitem__


class DefaultConfig(Config):
    def __init__(self):

        # ==================== General settings ====================
        self.run = None  # required
        self.device = Devices.CPU
        self.seed = None

        self.log = AttrDict()
        self.log.to_tensorboard = True
        self.log.to_csv = True
        self.log.frequency = 1

        self.save = AttrDict()
        self.save.best_state = True
        self.save.state_every_nth_epoch = None

        # ================== Data related settings ==================
        self.data = AttrDict()
        self.data.root = None
        self.data.folder = None  # required
        self.data.type = None  # required
        self.data.train = None  # required in train
        self.data.test = None  # required in test
        self.data.val = None
        self.data.num_workers = os.cpu_count()
        self.data.weighted = False
        self.data.shuffle = True
        self.data.drop_last = True
        self.data.pin_memory = True

        # ================= Images related settings =================
        self.images = AttrDict()
        self.images.size = None  # required if data_type is `images`
        self.images.mean = None
        self.images.std = None

        # todo: Hereafter should be same blocks for images with masks and other types
        # ...

        # ============== Running mode related settings ==============
        # General experiment settings
        self.task = None  # required
        self.label = timestamp()
        self.model = AttrDict()
        self.model.name = None  # required
        self.model.load_from = None  # required
        self.model.parameters = AttrDict()

        self.num_outputs = None  # depends on the task_type
        self.checkpoint = None

        # Training specific settings
        self.train = AttrDict()
        self.train.tune = True
        self.train.validate = True
        self.train.metrics = []  # required (str names for metrics from sklearn or dicts for custom ones)
        self.train.num_epochs = None  # required if run `train`
        self.train.batch_size = 1

        self.train.optimizer = AttrDict()
        self.train.optimizer.name = "r_adam"
        self.train.optimizer.load_from = "custom/optimizers/radam.py"

        self.train.lr = None  # depends on optimizer
        self.train.weight_decay = None
        self.train.scheduler = AttrDict()
        self.train.scheduler.name = None
        self.train.scheduler.load_from = None
        self.train.scheduler.parameters = AttrDict()

        self.test = AttrDict()
        self.test.batch_size = 1
        self.test.metrics = []  # will copy train ones if provided

        super(DefaultConfig).__init__()

    @staticmethod
    def for_regression(metric="MAE", data_type="images"):
        regression_config = DefaultConfig()
        regression_config.task_type = TaskTypes.Regression
        regression_config.num_outputs = 1
        regression_config.metrics.append(metric)

        if is_implemented_type(data_type, DataTypes):
            regression_config.data_type = DataTypes[snake_to_camel(data_type.lower())]
            return regression_config

        regression_config.data_type = DataTypes.Custom
        return regression_config

    @staticmethod
    def for_classification(metric="CrossEntropyLoss", data_type="images", num_classes=2):
        classification_config = DefaultConfig()
        classification_config.task_type = TaskTypes.Classification
        classification_config.num_outputs = num_classes
        classification_config.metrics.append(metric)

        if is_implemented_type(data_type, DataTypes):
            classification_config.data_type = DataTypes[snake_to_camel(data_type.lower())]
            return classification_config

        classification_config.data_type = DataTypes.Custom
        return classification_config



def load_configuration() -> Config:
    root = get_project_root()
    config_paths = (root / CONFIGS_DIR).glob('**/*.json')
    config_paths = list(config_paths)

    if not config_paths:
        error = "Can't locate configuration files. Place `*.json` in `{}` first!".format(CONFIGS_DIR)
        logging.error(error)
        raise FileNotFoundError(error)

    config = Config()

    for config_path in config_paths:
        config.add_file(config_path)

    config.load()
    return config


def timestamp(format=TIMESTAMP_FORMAT):
    return datetime.now().strftime(format)
