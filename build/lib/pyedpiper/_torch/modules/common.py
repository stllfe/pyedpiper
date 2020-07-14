from typing import Callable

import torch
from torch import nn as nn


class Concat(nn.Module):

    __constants__ = ['dim', 'flatten']

    def __init__(self, *m: nn.Module, dim=1, flatten=True):
        super(Concat, self).__init__()

        # to avoid interfering with Module's own attributes
        self._m = m

        self.dim = dim
        self.flatten = flatten

    def forward(self, input):
        outs = list()

        for m in self._m:
            outs.append(m(input))

        outs = torch.cat(outs, dim=self.dim)
        if self.flatten:
            return outs.flatten(start_dim=1)
        return outs


class Positional(nn.Module):
    """Passes inputs through corresponding _core and returns their outputs in the same order."""

    def __init__(self, *m: nn.Module):
        super(Positional, self).__init__()

        # to avoid interfering with the Module's own attributes
        self._m = m

    def forward(self, *args):
        if len(args) != len(self._m):
            raise RuntimeError(f"Size mismatch! Number of _core: {len(self._m)}, "
                               f"number of args: {len(args)}.")

        outs = list()

        for a, m in zip(args, self._m):
            outs.append(m(a))

        return tuple(outs)


class Identity(nn.Module):
    """Module that doesn't change the input."""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Lambda(nn.Module):
    """Module wrapper for any function."""

    def __init__(self, f: Callable, **kwargs):
        super(Lambda, self).__init__()
        if kwargs:
            from functools import partial
            self._f = partial(f, **kwargs)
        else:
            self._f = f

    def forward(self, *args):
        return self._f(*args)


class Flatten(nn.Module):
    """Flattens a contiguous range of dims into a tensor.

    For use with :class:`~nn.Sequential`.

    Args:
        start_dim: first dim to flatten (default = 1).
        end_dim: last dim to flatten (default = -1).

    Shape:
        - Input: :math:`(N, *dims)`
        - Output: :math:`(N, \prod *dims)` (for the default case).


    Examples::
        >>> m = nn.Sequential(
        >>>     nn.Conv2d(1, 32, 5, 1, 1),
        >>>     Flatten()
        >>> )
    """

    __constants__ = ['start_dim', 'end_dim']

    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)
