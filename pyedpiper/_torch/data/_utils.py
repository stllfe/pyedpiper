import numpy as np

from _common import as_numpy


def get_class_weights(targets, use_max=True):
    # todo: add a docstring
    targets = as_numpy(targets)
    class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])

    if use_max:
        # (number of occurrences in most common class) / (number of occurrences in rare classes)
        max_class_count = np.max(class_sample_count)
        return max_class_count / class_sample_count

    return 1. / class_sample_count
