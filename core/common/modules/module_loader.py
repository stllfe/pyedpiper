import logging

from importlib import import_module
from importlib.util import (
    module_from_spec,
    spec_from_file_location,
)

from core.common.consts import PYTHON_EXTENSIONS
from core.utils.helpers import get_absolute_path, get_relative_path
from core.utils.validations import is_valid_file


class ModuleLoader:
    @staticmethod
    def load_external_module(module_name):
        try:
            return import_module(module_name)
        except ModuleNotFoundError as e:
            error = "Can't load third party module `{}` ".format(module_name)
            message = "Make sure that name is correct and corresponding pip package is properly installed."
            logging.error(error + message)
            raise e

    @staticmethod
    def load_local_module(module_path):
        try:
            module_name = module_path.name.split('.')[0]
            spec = spec_from_file_location(
                name=module_name,
                location=module_path
            )
            module = module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            error = "Can't load local module `{}`. ".format(module_path)
            message = "Original exception was: {}".format(e)
            logging.error(error + message)
            raise e

    @staticmethod
    def load_module(module_path: str):
        if _is_local_module(module_path):
            module_path = _resolve_local_module(module_path)
            logging.info("Loading local module `{}`".format(module_path))
            module = ModuleLoader.load_local_module(module_path)
        elif _is_external_module(module_path):
            module = ModuleLoader.load_external_module(module_path)
        else:
            error = "Can't load module `{}`. Provided module path is incorrect!".format(module_path)
            raise ModuleNotFoundError(error)
        logging.info("Module `{}` loaded successfully!".format(module))
        return module


def _is_external_module(module_path) -> bool:
    return module_path.find('/') == -1 and not module_path.startswith('.')


def _is_relative_import(load_from) -> bool:
    return is_valid_file(get_relative_path(load_from), PYTHON_EXTENSIONS)


def _is_absolute_import(load_from) -> bool:
    return is_valid_file(get_absolute_path(load_from), PYTHON_EXTENSIONS)


def _is_local_module(load_from):
    return _is_absolute_import(load_from) or _is_relative_import(load_from)


def _resolve_local_module(module_path):
    if _is_absolute_import(module_path):
        return get_absolute_path(module_path)
    elif _is_relative_import(module_path):
        return get_relative_path(module_path)
    else:
        error = "Tried to resolve module path `{}` which appears to be nonlocal".format(module_path)
        logging.debug(error)
        raise TypeError(error)
