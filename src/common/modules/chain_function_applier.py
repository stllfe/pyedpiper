import inspect
import logging
from typing import List


class ChainFunctionApplier:
    """
    Base class for any configurator or validator.
    """

    def __init__(self, functions: List[object], applied_to: type):
        self.applied_to = applied_to
        self.functions = []
        for function in functions:
            self.add(function)

    def __call__(self, obj):
        return self.apply(obj)

    @staticmethod
    def is_applicable(function):
        if callable(function):
            signature = inspect.signature(function)
            return bool(signature.parameters.keys())
        return False

    def add(self, function: object):
        if self.is_applicable(function):
            self.functions.append(function)
        else:
            error = "Object should be callable and accept at least one argument of type: {}"
            error = error.format(function, self.applied_to)
            logging.error(error)
            raise TypeError(error)

    def remove(self, validation):
        self.functions.remove(validation)

    def apply(self, obj):
        if self.functions:
            return all([function(obj) for function in self.functions])
