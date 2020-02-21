import os

from addict import Dict
from torch import cuda

from core.common.config.mixins import (
    LoadMixin,
    SaveMixin,
    IsSetMixin,
)
from core.common.config.parameter import Parameter
from core.common.consts import CONFIGS_DIR, CONFIG_RESERVED_NAMES
from core.common.types import Devices
from core.utils.helpers import get_project_root, timestamp


class Config(IsSetMixin, SaveMixin, LoadMixin, Dict):
    """Subclass of addict's Dict class (https://github.com/mewwts/addict).
    Loads json files as dict-like objects providing both `Config["key"]` and `Config.key` item access.
    Has some unique features such as on-the-fly type conversion for leaf attributes.
    Check https://github.com/stllfe/pyedpiper to learn more.

    NOTE: It returns an empty dict `{}` in case of a missing key.
    Its actually more natural for a configuration file because of its nested nature.
    Moreover, it basically stands for `not stated` and is easy to check right in client code.

    The usage is something like:
    >>> not config.data.root
    >>> True
    """

    def _process_cache(self):
        for data in self._cache:
            name, parameters = data
            name = name.lower()
            if name in CONFIG_RESERVED_NAMES:
                self.update(parameters)
            else:
                self.update({name: parameters})

    def clear(self):
        super(Config, self).clear()
        self._cache.clear()
        self._files.clear()

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = Config(value)
        else:
            value = Parameter(value)
        super(Config, self).__setitem__(key, value)

        # For code completion in editors
        self.__dict__[key] = value

    @classmethod
    def default(cls):
        config = cls()
        # ==================== General settings ====================
        config.run = Parameter(None, required=True)  # required
        config.device = Devices.GPU if cuda.is_available() else Devices.CPU
        config.seed = None

        config.log = {}
        config.log.to_tensorboard = True
        config.log.to_csv = True
        config.log.frequency = 1

        config.state = {}
        config.state.save_best = True
        config.state.save_every_nth_epoch = None

        # ================== Data related settings ==================
        config.data = {}
        config.data.source = None
        config.data.root_folder = Parameter(None, required=True)  # required
        config.data.type = Parameter(None, required=True)  # required
        config.data.train_folder = None  # required in train
        config.data.test_folder = None  # required in test
        config.data.val_folder = None
        config.data.num_workers = os.cpu_count()
        config.data.weighted = False
        config.data.shuffle = True
        config.data.drop_last = True
        config.data.pin_memory = True

        # ================= Images related settings =================
        config.images = {}
        config.images.size = None  # required if data_type is `images`
        config.images.mean = None
        config.images.std = None

        # todo: Hereafter should be same blocks for images with masks and other types
        # todo: ...

        # ============== Running mode related settings ==============
        # General experiment settings
        config.task = Parameter(None, required=True)  # required
        config.label = timestamp()
        config.model = {}
        config.model.name = Parameter(None, required=True)  # required
        config.model.load_from = Parameter(None, required=True)  # required
        config.model.parameters = {}

        config.num_outputs = None  # depends on the task_type
        config.checkpoint = None

        # Training specific settings
        config.train = {}
        config.train.tune = True
        config.train.validate = True
        config.train.metrics = Parameter([],
                                         required=True)  # required (str names for metrics from sklearn or dicts for custom ones)
        config.train.num_epochs = None  # required if run `train`
        config.train.batch_size = 1

        config.train.optimizer = {}
        config.train.optimizer.name = "r_adam"
        config.train.optimizer.load_from = "custom/optimizers/radam.py"

        config.train.lr = None  # depends on optimizer
        config.train.weight_decay = None
        config.train.scheduler = {}
        config.train.scheduler.name = None
        config.train.scheduler.load_from = None
        config.train.scheduler.parameters = {}

        config.test = {}
        config.test.batch_size = 1
        config.test.metrics = []  # will copy train ones if provided
        return config


def load_configuration() -> Config:
    root = get_project_root()
    configs_dir = (root / CONFIGS_DIR).resolve()
    config = Config()
    config.load(configs_dir)
    return config
