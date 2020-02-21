import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from core.common.consts import JSON_SCHEMA_INDENTS
from core.common.decorators import convert_input


def serializer(obj):
    if isinstance(obj, WrappedObjectMixin):
        return obj.value
    else:
        return obj.__dict__


class LoadMixin(ABC):
    _cache = []
    _files = []

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


class SaveMixin(ABC):
    def _dump(self, path):
        try:
            with open(str(path), 'w') as file:
                json.dump(self, file, default=serializer, indent=JSON_SCHEMA_INDENTS)
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


class IsSetMixin(ABC):
    @property
    def is_set(self):
        return bool(self)

    @property
    def is_not_set(self):
        return not self


class IsRequiredMixin(object):
    def __init__(self, required=False, *args, **kwargs):
        super(IsRequiredMixin, self).__init__(*args, **kwargs)
        self.__required__ = required

    @property
    def is_required(self):
        return self.__required__


class WrappedObjectMixin(object):
    def __init__(self, value=None, *args, **kwargs):
        super(WrappedObjectMixin, self).__init__(*args, **kwargs)
        self.value = value

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return str(self.value)

    @property
    def value(self):
        return self.__value__

    @value.setter
    def value(self, other):
        self.__value__ = other


same_type = WrappedObjectMixin


class WrappedEqualityMixin(WrappedObjectMixin):
    @convert_input(same_type)
    def __eq__(self, other):
        return self.value == other.value

    @convert_input(same_type)
    def __ne__(self, other):
        return self.value != other.value


class WrappedComparisonMixin(WrappedEqualityMixin):
    @convert_input(same_type)
    def __lt__(self, other):
        return self.value < other.value

    @convert_input(same_type)
    def __le__(self, other):
        return self.value <= other.value

    @convert_input(same_type)
    def __gt__(self, other):
        return self.value > other.value

    @convert_input(same_type)
    def __ge__(self, other):
        return self.value >= other.value


class WrappedHashingMixin(WrappedObjectMixin):
    def __hash__(self):
        return hash(self.value)


class WrappedNumericMixin(WrappedEqualityMixin):
    @convert_input(same_type)
    def __add__(self, other):
        return self.value + other.value

    @convert_input(same_type)
    def __sub__(self, other):
        return self.value - other.value

    @convert_input(same_type)
    def __mul__(self, other):
        return self.value * other.value

    @convert_input(same_type)
    def __truediv__(self, other):
        return self.value / other.value

    @convert_input(same_type)
    def __mod__(self, other):
        return self.value % other.value

    @convert_input(same_type)
    def __floordiv__(self, other):
        return self.value // other.value

    @convert_input(same_type)
    def __pow__(self, other):
        return self.value ** other.value

    @convert_input(same_type)
    def __lshift__(self, other):
        return self.value >> other.value

    @convert_input(same_type)
    def __rshift__(self, other):
        return self.value << other.value

    @convert_input(same_type)
    def __and__(self, other):
        return self.value & other.value

    @convert_input(same_type)
    def __xor__(self, other):
        return self.value ^ other.value

    @convert_input(same_type)
    def __or__(self, other):
        return self.value | other.value

    @convert_input(same_type)
    def __iadd__(self, other):
        self.value += other.value
        return self.value

    @convert_input(same_type)
    def __contains__(self, other):
        return other.value in self.value

    @convert_input(same_type)
    def __isub__(self, other):
        self.value -= other.value
        return self.value

    @convert_input(same_type)
    def __imul__(self, other):
        self.value *= other.value
        return self.value

    @convert_input(same_type)
    def __idiv__(self, other):
        self.value /= other.value
        return self.value

    @convert_input(same_type)
    def __ifloordiv__(self, other):
        self.value //= other.value
        return self.value

    @convert_input(same_type)
    def __imod__(self, other):
        self.value %= other.value
        return self.value

    @convert_input(same_type)
    def __ipow__(self, other):
        self.value **= other.value
        return self.value

    @convert_input(same_type)
    def __ixor__(self, other):
        self.value ^= other.value
        return self.value

    @convert_input(same_type)
    def is_(self, other):
        return self.value is other.value


class WrappedPrimitive(WrappedNumericMixin, WrappedHashingMixin, WrappedComparisonMixin, WrappedEqualityMixin):
    pass
