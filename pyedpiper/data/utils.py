import logging
import numpy as np

from matplotlib import pyplot as plt
from os.path import basename
from PIL import Image
from typing import Callable

from ..core.common import as_numpy

log = logging.getLogger(__name__)


def get_class_weights(targets, use_max=True):
    # todo: add a docstring
    targets = as_numpy(targets)
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])

    if use_max:
        # (number of occurrences in most common class) / (number of occurrences in rare classes)
        max_class_count = np.max(class_sample_count)
        return max_class_count / class_sample_count

    return 1. / class_sample_count


def file_loader(function: Callable) -> Callable:
    """Decorator function to add `try` block and logging for loaders.

    :return: wrapped function
    """
    from functools import wraps

    @wraps(function)
    def wrapper(path, **kwargs):

        try:
            return function(path, **kwargs)

        except Exception as e:
            filename = basename(path)
            log.error(f"Error reading file `{filename}`: {e}")
            raise e

    return wrapper


@file_loader
def numpy_loader(path):
    return np.load(str(path))


@file_loader
def plt_loader(path):
    return plt.imread(str(path))


@file_loader
def pil_loader(path):
    return Image.open(str(path))
