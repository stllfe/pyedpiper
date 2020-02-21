import logging
from functools import wraps
from pathlib import Path


def file_loader(function):
    """
    Decorator function to add `try` block and logging for loaders.
    :return: wrapped function
    """

    @wraps(function)
    def wrapper(path, **kwargs):
        file_name = Path(path).name
        try:
            data = function(path, **kwargs)
            return data
        except Exception:
            error = "File or its path is corrupted! {}".format(file_name)
            logging.error(error)
            raise Exception(error)
    return wrapper


def on_init(action):
    """
    Decorator that runs an action after object initialization or a function call.
    :param action: callable object that accepts arguments
    :return: decorated function
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            result = action(result)
            return result

        return wrapper
    return decorator


def ignore(exception: type):
    """
    Soft wrapper to avoid specific exceptions
    :param exception: type to ignore
    :return: decorated function
    """

    def decorator(function):
        def wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except exception:
                pass

        return wrapper

    return decorator


def convert_input(converter):
    def decorated(function):
        def wrapper(*args, **kwargs):
            new_args = [converter(arg) for arg in args]
            new_kwargs = {k: converter(v) for k, v in kwargs.items()}
            return function(*new_args, **new_kwargs)

        return wrapper

    def self_decorated(function):
        def wrapper(self, *args, **kwargs):
            new_args = [converter(arg) for arg in args]
            new_kwargs = {k: converter(v) for k, v in kwargs.items()}
            return function(self, *new_args, **new_kwargs)

        return wrapper

    if isinstance(converter, type):
        return self_decorated
    return decorated
