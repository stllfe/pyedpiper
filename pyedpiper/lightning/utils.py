from typing import Optional
from typing import Sequence

import torch


def reduce(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    """Reduces a given tensor by a given reduction method.

    Args:
        tensor: The tensor to be reduced.
        reduction: A string specifying the reduction method ('mean', 'none', 'sum').

    Returns:
        Reduced Tensor

    Raises:
        ValueError if an invalid reduction parameter was given.
    """

    if reduction == 'mean':

        # Tensors of type `torch.long` can't be averaged.
        if tensor.dtype is torch.long:
            tensor = tensor.float()

        return torch.mean(tensor)

    if reduction == 'none':
        return tensor

    if reduction == 'sum':
        return torch.sum(tensor)

    raise ValueError('Reduction parameter unknown.')


def merge_outputs(outputs: list, multi_dim: int = 0, prefix: str = None) -> dict:
    """Merges outputs from different steps into one dictionary.

    Args:
        outputs: A list of outputs from the series of PyTorch Lightning steps.
        prefix: A prefix string to add for every key in the output dictionary.
        multi_dim: The dimension to use for concatenation if there is more than one (default=0).
    """

    def get_merge_fn(value: torch.Tensor):
        from functools import partial

        if value.ndim == 0:
            return torch.stack

        if value.ndim == 1:
            return torch.cat

        if value.ndim > 1:
            return partial(torch.cat, dim=multi_dim)

    def recursive_concat(outputs_):
        concat = dict()
        if not outputs_:
            return concat

        if len(outputs_) == 1:
            return outputs_[0]

        # We need only one pass since every dict in list contains the same metrics, hence use only the first item
        output = outputs_[0]
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                merger = get_merge_fn(value)
                merged = merger([output[key] for output in outputs_])
                new_key = (prefix + '_' if prefix else '' + key).strip('_')
                concat[new_key] = merged
            elif isinstance(value, dict):
                new_dict = recursive_concat([output[key] for output in outputs_])
                concat[key] = new_dict
            else:
                concat[key] = value
        return concat

    return recursive_concat(outputs)


def reduce_outputs(outputs: list, multi_dim: int = 0, reduction: str = 'mean', prefix: str = None) -> dict:
    """Reduces outputs from a series of steps, forming a new merged dictionary.

    Args:
        outputs: A list of outputs from the series of PyTorch Lightning steps.
        prefix: A prefix string to add for every key in the output dictionary.
        multi_dim: The dimension to use for concatenation if there is more than one (default=0).
        reduction: A string specifying the reduction method ('mean', 'none', 'sum').
                   If 'none' this function is the same as ``merge_outputs``.
    """

    merged = merge_outputs(outputs=outputs, prefix=prefix, multi_dim=multi_dim)
    reduced = {key: reduce(value, reduction) if isinstance(value, torch.Tensor) else value
               for key, value in merged.items()}
    return reduced


def change_prefix(output: dict, old: str, new: str) -> dict:
    """Shorthand for swapping outputs key prefixes.
    
    For example, swap 'val' metrics prefix to 'test' or vice versa for every dict entry.
    
    Args:
        output:  An output dictionary from PyTorch Lightning step.
        old: An old prefix string.
        new: A new prefix string.
    """

    def recursive_map(output_):
        new_outputs = dict()
        for key, value in output_.items():
            new_key = key
            if isinstance(value, dict):
                value = recursive_map(value)
            elif isinstance(value, torch.Tensor):
                new_key = key.replace(old, new)
            new_outputs[new_key] = value
        return new_outputs

    return recursive_map(output)


def extract_unique_metrics(output: dict) -> dict:
    """Extracts all the metrics from a dict or nested dicts recursively.

    Note:
        According to PyTorch Lightning API output metrics are of type ``torch.Tensor``.

    Args:
        output:  An output dictionary from PyTorch Lightning step.

    Returns:
        Flat dict with all the metrics found.
    """

    def extract_recursive(results):
        metrics = dict()
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                metrics[key] = value
            elif isinstance(value, dict):
                metrics.update(extract_recursive(value))
        return metrics

    return extract_recursive(output)


def reduce_and_log(outputs: list,
                   multi_dim: int = 0,
                   reduction: str = 'mean',
                   keys: Optional[Sequence[str]] = None,
                   prefix: str = None):
    """Reduces all the outputs, adds prefix if provided and inserts new key 'log' for logger to capture the output.

    Args:
        outputs: A list of outputs from series of PyTorch Lightning steps.
        prefix: A string prefix to add for every key in averaged dictionary.
        multi_dim: The dimension to use for concatenation if there is more than one (default=0).
        reduction: A string specifying the reduction method ('mean', 'sum') (default='mean').
        keys: Optional; If no provided, adds all the keys found for logging.
    """

    if reduction == 'none':
        from pytorch_lightning.utilities import rank_zero_warn
        reduction = 'mean'
        rank_zero_warn("Reduction can't be 'none', since you won't be able to log a sequence. "
                       "If it was intentional, consider using `merge_outputs`."
                       "Using 'mean' instead.")

    reduced = reduce_outputs(outputs=outputs, multi_dim=multi_dim, reduction=reduction, prefix=prefix)
    metrics = extract_unique_metrics(reduced)

    keys = keys or metrics.keys()
    logs = {key: metrics[key] for key in keys}

    reduced.update({'log': logs})
    return reduced
