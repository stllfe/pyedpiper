import logging

from functools import wraps
from pathlib import Path

from core.common.config import Config
from core.common.config import ConfigValidator


def config_validation(function):
    """
    Decorator function to add config validations.
    """

    def _is_compatible_validation_fn(function):
        if ConfigValidator.is_applicable(function):
            try:
                empty_config = Config()
                result = function(empty_config)
                if (
                        result is not None and
                        isinstance(result, dict)
                ):  # sad story bro
                    has_result = isinstance(result['result'], bool)
                    has_errors = isinstance(result['errors'], bool)
                    return has_result and has_errors
            except KeyError:
                pass
        return False

    if _is_compatible_validation_fn(function):
        ConfigValidator.custom_validations.append(function)
    else:
        error = "{} is not compatible function. Please check documentation at: https://github.com/stllfe/pyedpiper/"
        error = error.format(function, Config)
        raise TypeError(error)


def file_loader(func):
    """
    Decorator function to add try-catch block and logging for loaders.
    :return:
        wrapped function
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
