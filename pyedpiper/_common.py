import logging
import os
import random
from typing import Any, Tuple, Optional, Iterable

import numpy as np
import torch
from _core.object_builder import ObjectCaller
from omegaconf import DictConfig, OmegaConf

from _core.module_loader import ModuleLoader

log = logging.getLogger(__name__)

_TARGET_NAME = "target"
_PARAMS_NAME = "params"


def resolve_target(target_config: DictConfig) -> type:
    """
    Get concrete class from config entry.
    :param target_config: config entry with 'class', 'params' and (optionally) 'module' specified
    :return: `type` object of corresponding class
    """
    if _TARGET_NAME not in target_config or target_config[_TARGET_NAME] is None:
        error = f"No {_TARGET_NAME} property was provided in config."
        log.error(error)
        raise ValueError(error)

    cls, source = _resolve_target_module(target_config)

    try:
        return getattr(source, cls)
    except AttributeError as e:
        error = "{} '{}' not in module '{}'".format(_TARGET_NAME.title(), cls, target_config.module)
        log.error(error)
        raise e


def _resolve_target_module(class_config: DictConfig) -> Tuple[Any, Any]:
    # Assuming that module and class are written in dot notation
    if 'module' not in class_config or class_config.module is None:
        module, _, cls = class_config[_TARGET_NAME].rpartition('.')
    else:
        module = class_config.module
        cls = class_config[_TARGET_NAME]
    source = ModuleLoader.load_module(module)
    return cls, source


def instantiate(target_config: DictConfig, **kwargs) -> Any:
    """Same as `call()`"""
    return call(target_config, **kwargs)


def call(target_config: DictConfig, **kwargs) -> Any:
    """Resolve the module and call the object with 'params' config keys."""
    assert target_config is not None, "Input config is `None`"

    cls = resolve_target(target_config)
    params = target_config.params if _PARAMS_NAME in target_config else OmegaConf.create()

    # If params are None, make an empty dict as well
    params = params or OmegaConf.create()

    assert isinstance(params, DictConfig), (
        "Input config params are expected to be a mapping, "
        "found {}".format(type(params)))

    params = OmegaConf.to_container(params, resolve=True)
    params.update(kwargs)

    return ObjectCaller.call_from_kwargs(cls, **params)


def set_random_seed(seed: Optional[int] = None) -> int:
    """Fix seed values for pseudo-random number generators.

    Specifically, fixes seed inside:
        PyTorch, Numpy, python.random and sets PYTHONHASHSEED environment variable.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    try:
        if seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
        else:
            seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if (seed > max_seed_value) or (seed < min_seed_value):
        msg = (f"Seed value: {seed} is out of bounds, "
               f"numpy accepts from {min_seed_value} to {max_seed_value}!")

        log.warning(msg)
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return seed


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    seed = random.randint(min_seed_value, max_seed_value)
    log.warning(f"No correct seed found, seed set to {seed}.")
    return seed


def as_numpy(obj) -> np.ndarray:
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()

    if isinstance(obj, Iterable):
        return np.array(obj)

    if isinstance(obj, np.ndarray):
        return obj

    error = (f"Can't convert object of type "
             f"`{type(obj)}` into `np.ndarray`")
    log.error(error)
    raise TypeError(error)


def as_tensor(obj, dtype=None) -> torch.Tensor:
    if isinstance(obj, torch.Tensor):
        if dtype is not None:
            obj = obj.type(dtype)
        return obj

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
        if dtype is not None:
            obj = obj.type(dtype)
        return obj

    if isinstance(obj, (list, tuple)):
        obj = np.ndarray(obj)
        obj = torch.from_numpy(obj)
        if dtype is not None:
            obj = obj.type(dtype)
        return obj

    try:
        return torch.as_tensor(data=obj, dtype=dtype)
    except Exception as e:
        error = (f"Can't convert object of type "
                 f"`{type(obj)}` into `torch.Tensor`: {e}")

        log.error(error)
        raise TypeError(error)
