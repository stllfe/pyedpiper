import logging

from pathlib import Path
from importlib import import_module
from core.utils.helpers import get_project_root
from importlib.util import (
    module_from_spec,
    spec_from_file_location,
)


class ModuleLoader:
    def __init__(self, module_name, load_from):
        self.module_name = module_name
        self.module_obj = self._load_module()

        self.load_from = load_from
        self.loaded_from = None

    @staticmethod
    def _load_third_party_module(module_name):
        try:
            return import_module(module_name)
        except ModuleNotFoundError:
            return

    @staticmethod
    def _load_local_module(module_path):
        module_path = Path(module_path)
        name = module_path.name.split('.')[0]
        if not module_path.exists():
            return
        spec = spec_from_file_location(
            name=name,
            location=module_path
        )
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _load_module(self):
        load_from = self.load_from
        if load_from:
            logging.info("Loading module `{}`".format(load_from))
            module = self._load_third_party_module(load_from)
        else:
            # if no module provided search locally
            logging.info("Couldn't find  {} in third-party libraries".format(self.module_name))
            logging.info("Searching for `{}` locally ...".format(self.module_name))

            load_from = get_project_root() / self.load_from / '{}.py'.format(self.module_name.lower())
            module = self._load_local_module(load_from)

        if not module:
            error = "Module not found: `{}".format(load_from)
            logging.error(error)
            raise ModuleNotFoundError(error)

        logging.info("Found module with name: {}".format(module.__name__))
        try:
            self.module_obj = getattr(module, self.module_name)
        except AttributeError as e:
            error = "No object `{}` found in module: `{}`".format(self.module_name, load_from)
            logging.error(error)
            raise e

        # save the final destination if successfully loaded
        self.loaded_from = load_from
        logging.info("Module `{}` loaded successfully!".format(self.module_name))
