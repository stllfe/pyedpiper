import logging
import os

from pathlib import Path
from importlib import import_module
from importlib.util import (
    module_from_spec,
    spec_from_file_location,
)


log = logging.getLogger(__name__)


class ModuleLoader:

    @staticmethod
    def load_external_module(module_name):
        try:
            return import_module(module_name)
        except ModuleNotFoundError as e:
            try:
                modules = module_name.split('.')
                submodule, package = modules[-1], '.'.join(modules[:-1])
                return import_module(submodule, package)
            except ModuleNotFoundError:
                error = ("Can't load third party module `{}`. "
                         "Make sure that name is correct or corresponding `pip` package is properly installed.")
                error = error.format(module_name)
                log.error(error)
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
            spec._loader.exec_module(module)
            return module
        except Exception as e:
            error = "Can't load local module `{}`. \n{}".format(module_path, str(e))
            log.error(error)
            raise e

    @classmethod
    def load_module(cls, module_path: str):
        if _is_local_module(module_path):
            log.debug("Loading local module '{}' ...".format(module_path))
            full_path = _resolve_local_module(module_path)
            module = cls.load_local_module(full_path)
        elif _is_external_module(module_path):
            module = cls.load_external_module(module_path)
        else:
            error = "Can't load module `{}`. Provided module path is incorrect!".format(module_path)
            log.error(error)
            raise ModuleNotFoundError(error)
        log.debug("Module `{}` is loaded.".format(module.__name__))
        return module


def _is_external_module(module_path) -> bool:
    return module_path.find('/') == -1 and not module_path.startswith('.')


def _is_relative_import(load_from) -> bool:
    relative_path = _get_relative_path(load_from)
    return relative_path.exists() or relative_path.with_suffix('.py').exists()


def _is_absolute_import(load_from) -> bool:
    absolute_path = _get_absolute_path(load_from)
    return absolute_path.is_absolute() and (absolute_path.exists() or absolute_path.with_suffix('.py').exists())


def _is_local_module(load_from):
    return _is_absolute_import(load_from) or _is_relative_import(load_from)


def _resolve_local_module(module_path):
    path = Path(module_path)

    if _is_absolute_import(module_path):
        path = _get_absolute_path(module_path)

    elif _is_relative_import(module_path):
        path = _get_relative_path(module_path)

    if path.exists():
        return path

    with_suffix = path.with_suffix('.py')

    if with_suffix.exists():
        return with_suffix
    else:
        error = "Tried to resolve local module which doesn't exist!"
        raise ModuleNotFoundError(error)


def _get_project_root() -> Path:
    from hydra.utils import get_original_cwd
    # Check whether hydra runtime is running
    # since it changes current working path
    try:
        hydra_runtime_cwd = get_original_cwd()
    except AttributeError:
        return Path(os.getcwd())
    return Path(hydra_runtime_cwd)


def _get_absolute_path(path: str) -> Path:
    """
    Returns the full path from posix one
    :param path: str or Path-like object
    :return: Path object
    """
    return Path(path).resolve()


def _get_relative_path(path: str) -> Path:
    """
    Returns path relative to the project root
    :param path: str or Path-like object
    :return: Path object
    """
    return (_get_project_root() / Path(path)).resolve()
