import json
import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import List

from core.common.consts import JSON_SCHEMA_INDENTS
from core.common.decorators import ignore


class LoadMixin(object):
    def __init__(self):
        super(LoadMixin, self).__init__()
        self._cache = []
        self._files = []

    files = property()

    @files.getter
    def files(self):
        return self._files

    def add_file(self, path):
        path = Path(path).resolve()
        if not path.exists() or not path.is_file():
            error = "Can't allocate configuration file with path: `{}`".format(path)
            logging.error(error)
            raise FileNotFoundError(error)
        if path not in self._files:
            self._files.append(path)
        self._order_files_by_date_modified()

    def load(self, path=None):
        if path:
            path = Path(path)
            if path.is_dir():
                config_paths = path.glob('**/*.json')
                config_paths = list(config_paths)
                if not config_paths:
                    error = "Can't locate configuration files. Place `*.json` in there first!"
                    logging.error(error)
                    raise FileNotFoundError(error)
                for config_path in config_paths:
                    self.add_file(config_path)
            else:
                self.add_file(path)
        self._load_files()
        self._process_cache()

    def _order_files_by_date_modified(self):
        self._files.sort(key=os.path.getmtime)

    def _load_file(self, path):
        with open(path.as_posix(), 'r+') as file:
            file_name = path.name.split('.')[0]
            parameters = json.load(file)
            self._cache.append((file_name, parameters))

    def _load_files(self):
        for file in self._files:
            self._load_file(file)

    @abstractmethod
    def _process_cache(self):
        pass


class SaveMixin(object):
    def _dump(self, path):
        try:
            with open(str(path), 'w') as file:
                json.dump(self, file, indent=JSON_SCHEMA_INDENTS)
            info = "Config file saved successfully to: {}".format(path)
            logging.info(info)
        except Exception as e:
            error = "Can't dump file on disk! {}".format(e)
            logging.error(error)
            raise e

    def save(self, path=None, name=None):
        file_name = name if name else self.__class__.__name__
        file_name = '{}.json'.format(file_name.lower())
        if not path:
            path = Path('./').resolve()
            return self._dump(path / file_name)

        path = Path(path).resolve()
        if path.is_dir():
            return self._dump(path / file_name)

        elif path.is_file() and path.name.endswith(".json"):
            return self._dump(path)

        error = "Can't save file with path: `{}`. Make sure path is correct!".format(path)
        logging.error(error)
        raise NotADirectoryError(error)


class IsSetMixin(object):
    @ignore(AttributeError, default=False)
    def is_set(self, item: str) -> bool:
        """
        Get all the attributes recursively and validate them according to python's
        "truthy" and "falsy" evaluations, i. e. `bool(attribute)`.
        :param item: (str) - one or more attributes with dot notation "attr1.attr2.'...'.attrN"
        :return: (bool) - whether or not all the attributes evaluate to True
        """
        item = item.strip().lower()
        results = []
        for item in get_attributes_recursively(self, item):
            results.append(bool(item))
        return all(results)

    def is_not_set(self, item):
        return not self.is_set(item)


class RequireMixin(object):
    def __init__(self, *args, **kwargs):
        super(RequireMixin, self).__init__()
        self._required = {}

    required = property()

    @required.getter
    def required(self):
        return self._required

    def set_required(self, item: str, value: bool = True):
        self._validate_input(item, value)
        item = item.strip().lower()
        items = item.split('.')
        # setting the initial value

        self._required[item] = value
        # todo: handle setting requirements recursively
        # e.g if `model.load_from` is required, then `model` itself is required too
        # but the opposite is not true in general case, so we setdefault instead of straight setting
        for item in reversed(items[:-1]):
            self._required.setdefault(item, value)
        # todo: make recursively update each instance's dict within the container

    @staticmethod
    def _validate_input(item: str, value: bool):
        if not isinstance(item, str) or not isinstance(value, bool):
            error = "`item` should be of type `str` and `value` of type `bool`. Got {}, {} instead."
            error = error.format(type(item), type(value))
            logging.error(error)
            raise TypeError(error)

    @ignore(KeyError, default=False)
    def is_required(self, item: str) -> bool:
        item = item.strip().lower()
        return self._required[item]


def get_attributes_recursively(obj, attributes: str) -> List:
    attributes = attributes.split('.')
    current = obj
    results = []
    for attribute in attributes:
        current = getattr(current, attribute)
        results.append(current)

    return results
