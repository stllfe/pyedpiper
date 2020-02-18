import json
import os
from pathlib import Path


class AttrDict(dict):
    """ Nested Attribute Dictionary

    A class to convert a nested Dictionary into an object with key-values
    accessible using attribute notation (AttrDict.attribute) in addition to
    key notation (Dict["key"]). This class recursively sets Dicts to objects,
    allowing you to recurse into nested dicts (like: AttrDict.attr.attr)
    """

    def __init__(self, mapping=None):
        super(AttrDict, self).__init__()
        if mapping is not None:
            for key, value in mapping.items():
                self.__setitem__(key, value)

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        super(AttrDict, self).__setitem__(key, value)
        self.__dict__[key] = value  # for code completion in editors

    def __getattr__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    __setattr__ = __setitem__


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

    custom = {}
    _bound_parameters = {}
    _files = []

    def add_file(self, path):
        path = Path(path)
        if not path.exists() or not path.is_file():
            error = "Can't allocate configuration file with path: `%s`" % path
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

        self._bound_parameters.update(parameters)
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
            super().__getitem__(item)
        except KeyError:
            return

    def __setitem__(self, parameter, value):
        if parameter in self._bound_parameters.keys():
            self._bound_parameters[parameter] = value
            self._repair_consistency(parameter, value)
            self._reindex_parameters()
            return

        super(Config, self).__setitem__(parameter, value)

    __setattr__ = __setitem__
