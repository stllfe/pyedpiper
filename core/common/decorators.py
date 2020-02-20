import logging
from functools import wraps
from pathlib import Path

from core.common.config import Config
from core.common.config import ConfigValidator


def config_validation(function):
    """
    Decorator function to add config validations.
    """

    @wraps(function)
    def wrapper(*args, **kwargs):
        errors = function(*args, **kwargs)
        return {'result': not any(errors), 'errors': errors}

    if ConfigValidator.is_applicable_validation(function):
        ConfigValidator.validations.append(wrapper)
    else:
        error = "{} is not compatible function. Please check documentation at: https://github.com/stllfe/pyedpiper/"
        error = error.format(function, Config)
        raise TypeError(error)


def file_loader(func):
    """
    Decorator function to add `try` block and logging for loaders.
    :return: wrapped function
    """

    @wraps(func)
    def wrapper(path, **kwargs):
        file_name = Path(path).name
        try:
            data = func(path, **kwargs)
            return data
        except Exception:
            error = "File or its path is corrupted! {}".format(file_name)
            logging.error(error)
            raise Exception(error)

    return wrapper
