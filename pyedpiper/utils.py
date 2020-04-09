import logging
import random

import numpy as np
import torch
from omegaconf import DictConfig

from .modules.module_loader import ModuleLoader
from .modules.object_builder import ObjectBuilder

log = logging.getLogger(__name__)


def resolve_class(class_config: DictConfig) -> type:
    """
    Get concrete class from config entry.
    :param class_config: config entry with 'class', 'params' and (optionally) 'module' specified
    :return: `type` object of corresponding class
    """
    if 'class' not in class_config or class_config['class'] is None:
        error = "No 'class' property was provided in config."
        log.error(error)
        raise ValueError(error)

    # Assuming that module and class are written as a dotlist
    if 'module' not in class_config or class_config.module is None:
        module, _, cls = class_config['class'].rpartition('.')
    else:
        module = class_config.module
        cls = class_config['class']

    source = ModuleLoader.load_module(module)
    try:
        return getattr(source, cls)
    except AttributeError as e:
        error = "Class '{}' not in module '{}'".format(cls, class_config.module)
        log.error(error)
        raise e


def instantiate(class_config: DictConfig, **kwargs) -> object:
    """
    Resolve 'class' and build object from 'params' config keys.
    """
    from omegaconf import OmegaConf

    cls = resolve_class(class_config)
    params = OmegaConf.to_container(class_config.params, resolve=True)
    params.update(kwargs)
    return ObjectBuilder.build_from_kwargs(cls, **params)


def average_metrics(outputs: list) -> dict:
    """
    Averages all the outputs, adds 'avg' prefix and inserts
    :param outputs: list of outputs from series of PyTorch Lightning steps such as '$step_epoch_end' input
    :return: dict with every tensor value averaged across all outputs
    """

    # We need only one pass since every dict in list contains the same metrics, hence use only the first item
    def _recursive_average(outputs):
        averages = dict()
        output = outputs[0]
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                average = torch.stack([output[key] for output in outputs]).mean()
                new_key = 'avg' + '_' + key
                averages[new_key] = average
                continue
            elif isinstance(value, dict):
                new_dict = _recursive_average([output[key] for output in outputs])
                averages[key] = new_dict
                continue
            else:
                log.warning('Non-typical value detected in outputs. Are you sure you use this function correctly?')
                averages[key] = value
        return averages

    return _recursive_average(outputs)


def map_outputs_to_label(output: dict, old_label, new_label) -> dict:
    """
    Shorthand to swap outputs labels to a different one. For example swap 'val' metrics prefix to 'test' or vice versa.
    :param output: dict with outputs from PyTorch Lightning step such as 'val_step' or 'test_step'
    :param old_label: str
    :param new_label: str
    :return:
    """

    def _recursive_map(output):
        new_outputs = dict()
        for key, value in output.items():
            new_key = key
            if isinstance(value, dict):
                value = _recursive_map(value)
            elif isinstance(value, torch.Tensor):
                new_key = key.replace(old_label, new_label)
            new_outputs[new_key] = value
        return new_outputs

    return _recursive_map(output)


def extract_unique_metrics(results: dict) -> dict:
    """
    Extract all metrics from dict or nested dicts recursively.
    According to PyTorch Lightning API metrics are those of type 'torch.Tensors'.
    :param results: dict from any step output
    :return: flat dict with all the metrics found
    """

    def _extract_recursive(results):
        metrics = dict()
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value
            elif isinstance(value, dict):
                metrics.update(_extract_recursive(value))
        return metrics

    return _extract_recursive(results)


def average_and_log_metrics(outputs: list):
    """
    Shorthand for '$step_epoch_end' type of steps according to PyTorch Lightning API.
    It averages all the outputs, adds 'avg' prefix and inserts new key 'log' for logger to capture the output.
    :param outputs: outputs from series of PyTorch Lightning steps such as 'validation_epoch_end' input or 'test_epoch_end'
    :return: new dict as specified by description
    """
    results = average_metrics(outputs)
    results.update({'log': extract_unique_metrics(results)})
    return results


def set_random_seed(seed):
    """
    Fix all underlying ML lib's seed values.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
