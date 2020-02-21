import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

from core.common.consts import JSON_SCHEMA_INDENTS


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

    def _load_files(self):
        for file in self._files:
            self._load_file(file)

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

    @abstractmethod
    def _process_cache(self):
        pass


class SaveMixin(ABC):
    def _dump(self, path):
        try:
            with open(str(path), 'w') as file:
                json.dump(self, file, indent=JSON_SCHEMA_INDENTS)
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
