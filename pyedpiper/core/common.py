import logging
import os
import random

from collections import OrderedDict
from numbers import Number
from typing import (
    Any,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np
import torch
from torch.nn import Module

from .module_loader import ModuleLoader
from .object_caller import ObjectCaller

log = logging.getLogger(__name__)

_TARGET_KEY = "target"
_MODULE_KEY = "module"
_PARAMS_KEY = "params"

__all__ = [
    "as_numpy",
    "as_tensor",
    "call",
    "instantiate",
    "transfer_weights",
    "set_random_seed",
]


def _merge(*dicts, dtype=dict):
    """Merge a collection of mappings."""

    if len(dicts) == 1 and not isinstance(dicts[0], Mapping):
        dicts = dicts[0]

    rv = dtype()
    for d in dicts:
        rv.update(d)

    return rv


def _to_dict(c: Any):
    # If provided object is OmegaConf, try to resolve it.
    try:
        import omegaconf
        if isinstance(c, omegaconf.DictConfig):
            return omegaconf.OmegaConf.to_container(c, resolve=True)
    except ImportError:
        pass

    try:
        if isinstance(c, dict):
            return c
        elif isinstance(c, Mapping):
            return dict(c)
        else:
            TypeError("Provided object should be a mapping.")

    except Exception as e:
        log.error(f"Can't convert to dict! \n{e}")
        raise e


def _get_params(node: Mapping):
    return node.get(_PARAMS_KEY, {}) or {}


def _resolve_target(target_config: Mapping) -> type:
    """Get concrete class from config entry."""

    if _TARGET_KEY not in target_config or target_config[_TARGET_KEY] is None:
        error = f"No {_TARGET_KEY} property found in config."
        log.error(error)
        raise ValueError(error)

    cls, source = _resolve_target_module(target_config)

    try:
        return getattr(source, cls)
    except AttributeError as e:
        error = "{} '{}' not in module '{}'".format(_TARGET_KEY.title(), cls, target_config[_MODULE_KEY])
        log.error(error)
        raise e


def _resolve_target_module(class_config: Mapping) -> Tuple[Any, Any]:
    # Assuming that module and class are written in dot notation
    if _MODULE_KEY not in class_config or class_config[_MODULE_KEY] is None:
        module, _, cls = class_config[_TARGET_KEY].rpartition('.')
    else:
        module = class_config[_MODULE_KEY]
        cls = class_config[_TARGET_KEY]
    source = ModuleLoader.load_module(module)
    return cls, source


def instantiate(target_config: Mapping, **kwargs) -> Any:
    """Same as `call()`, but allows recursive object instantiation."""

    import inspect

    def buildable(o: Any):
        if isinstance(o, Mapping):
            try:
                return _resolve_target(o)
            except ValueError:
                return None

    def postorder_from(node: Mapping):
        obj = buildable(node)

        if obj:
            result = dict()
            params = _get_params(node)
            for param, value in params.items():
                result[param] = postorder_from(value)

            # If there is a method or a function provided as parameter
            if inspect.isfunction(obj):
                return obj

            result = _merge(node, result)
            return ObjectCaller.call_from_kwargs(obj, **result)

        return node

    # Convert to dict if needed
    target_config = _to_dict(target_config)

    # Make sure we can resolve the root first
    _resolve_target(target_config)

    # If params are None, set an empty dict as well
    target_config[_PARAMS_KEY] = _get_params(target_config)

    if kwargs:
        # Inject kwargs into params
        target_config[_PARAMS_KEY].update(kwargs)

    return postorder_from(target_config)


def call(target_config: Mapping, *args, **kwargs) -> Any:
    """Resolves a module and calls an object with provided parameters."""

    assert target_config is not None, "Input config is `None`"

    # Convert to dict if needed
    target_config = _to_dict(target_config)

    cls = _resolve_target(target_config)

    # If params are None, make an empty dict as well
    params = _get_params(target_config)

    assert isinstance(params, Mapping), (
        "Input config params are expected to be a mapping, "
        "found {}".format(type(params)))

    params = _merge(params, kwargs)
    return ObjectCaller.call_from_kwargs(cls, *args, **params)


def set_random_seed(seed: Optional[int] = None) -> int:
    """Fixes seed values in pseudo-random number generators.

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

    log.error(f"Can't convert object of type "
              f"`{type(obj)}` into `np.ndarray`")

    raise TypeError()


def as_tensor(obj, dtype=None) -> torch.Tensor:
    if isinstance(obj, Number):
        return torch.tensor([obj], dtype=dtype)

    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj)
        return obj.to(dtype)

    if isinstance(obj, torch.Tensor):
        return obj.to(dtype)

    if isinstance(obj, (List, Tuple)):
        def are_all(t: type):
            return all(map(lambda el: isinstance(el, t), obj))

        # Check whether it's a sequence of tensors, or arrays
        if are_all(torch.Tensor):
            return torch.cat(obj).to(dtype)

        if are_all(np.ndarray):
            obj = np.array(obj)
            obj = torch.from_numpy(obj)
            return obj.to(dtype)

        # As a last resort, try to convert elements one by one
        return torch.cat([as_tensor(it, dtype=dtype) for it in obj]).to(dtype)

    log.error(f"Can't convert object of type "
              f"`{type(obj)}` into `torch.Tensor`!")

    raise TypeError()


def transfer_weights(model: Module, state_dict: OrderedDict, verbose=False) -> Module:
    """Copies weights from the state dict into the model, skipping layers that are incompatible.

    It's helpful for model surgery and/or partial weights initialization.

    Args:
        model (Module): Model to load weights into
        state_dict (OrderedDict): Model state dict to load weights from
        verbose (bool): whether to print unmatched layers

    Returns:
        Module: The model
    """

    missing_keys = list()
    unexpected_keys = list()
    for name, value in state_dict.items():
        try:
            keys = model.load_state_dict(OrderedDict([(name, value)]), strict=False)
            missing_keys += keys.missing_keys
            unexpected_keys += keys.unexpected_keys
        except Exception as e:
            log.error(f"Error occurred while loading {value} into {name}. \n {e}")
            return model

    if verbose:
        log.info(f"Transfer completed. "
                 f"Unexpected keys: {', '.join(unexpected_keys)}. "
                 f"Missing keys: {', '.join(missing_keys)}.")

    return model
