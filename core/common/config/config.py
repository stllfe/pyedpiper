import os

from addict import Dict
from torch import cuda

from core.common.config.mixins import (
    LoadMixin,
    SaveMixin,
    IsSetMixin,
    RequireMixin,
)
from core.common.consts import CONFIGS_DIR, CONFIG_RESERVED_NAMES
from core.common.types import Devices
from core.utils.helpers import get_project_root, timestamp


class Config(LoadMixin, SaveMixin, IsSetMixin, Dict):
    """
    Subclass of addict's Dict class (https://github.com/mewwts/addict).
    Load json files as dict-like objects or save them as you wish.
    Provides both `Config["key"]` and `Config.key` item access.
    Check https://github.com/stllfe/pyedpiper to learn more.

    (!) NOTE: It returns an empty dict `{}` in case of a missing key.
    Its actually more natural for a configuration file because of its nested nature.
    Moreover, it basically stands for `not stated` and is easy to check right in client code.

    The usage is something like:
    >>> not config.data.root
    True
    """

    def __init__(self, *args, **kwargs):
        # workaround to separate class methods and actual attributes from dict keys
        self._set_immutable(True)
        LoadMixin.__init__(self, *args, **kwargs)
        self._set_immutable(False)
        Dict.__init__(self, *args, **kwargs)

    def _set_immutable(self, value: bool):
        object.__setattr__(self, "__immutable__", value)

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

    def _validate_attr(self, name):
        return name not in self.__dict__

    def __setattr__(self, name, value):
        if getattr(self, '__immutable__'):
            object.__setattr__(self, name, value)
            return
        if self._validate_attr(name):
            try:
                Dict.__setattr__(self, name, value)
                return
            except AttributeError:
                pass
        raise AttributeError("'Config' object attribute "
                             "'{}' is read-only".format(name))

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = Config(value)
        Dict.__setitem__(self, name, value)

    @classmethod
    def default(cls):
        config = cls()
        # ==================== General settings ====================
        config.run = None  # required
        config.device = Devices.GPU if cuda.is_available() else Devices.CPU
        config.seed = None

        config.log.to_tensorboard = True
        config.log.to_csv = True
        config.log.frequency = 1

        config.state.save_best = True
        config.state.save_every_nth_epoch = None

        # ================== Data related settings ==================
        config.data.source = None
        config.data.root_folder = None  # required
        config.data.type = None  # required
        config.data.train_folder = None  # required in train
        config.data.test_folder = None  # required in test
        config.data.val_folder = None
        config.data.num_workers = os.cpu_count()
        config.data.weighted = False
        config.data.shuffle = True
        config.data.drop_last = True
        config.data.pin_memory = True

        # ================= Images related settings =================
        config.images.size = None  # required if data_type is `images`
        config.images.mean = None
        config.images.std = None

        # todo: Hereafter should be same blocks for images with masks and other types
        # todo: ...

        # ============== Running mode related settings ==============
        # General experiment settings
        # config.set_required("task")
        config.task = None  # required
        config.label = timestamp()

        # config.set_required("model")
        config.model.name = None  # required
        config.model.load_from = None  # required
        config.model.parameters = {}

        config.num_outputs = None  # depends on the task_type
        config.checkpoint = None

        # Training specific settings
        # config.set_required("train")
        config.train.tune = True
        config.train.validate = True
        config.train.metrics = []  # required (str names for metrics from sklearn or dicts for custom ones)
        config.train.num_epochs = None  # required if run `train`
        config.train.batch_size = 1

        config.train.optimizer.name = "r_adam"
        config.train.optimizer.load_from = "custom/optimizers/radam.py"

        config.train.lr = None  # depends on optimizer
        config.train.weight_decay = None

        config.train.scheduler.name = None
        config.train.scheduler.load_from = None
        config.train.scheduler.parameters = {}

        config.test.batch_size = 1
        config.test.metrics = []  # will copy train ones if provided
        return config


def load_configuration() -> Config:
    root = get_project_root()
    configs_dir = (root / CONFIGS_DIR).resolve()
    config = Config()
    config.load(configs_dir)
    return config


if __name__ == "__main__":
    config = Config.default()
