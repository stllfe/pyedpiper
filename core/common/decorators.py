import logging
from functools import wraps
from pathlib import Path


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


def on_init(action):
    """
    Decorator that runs an action after object initialization or a function call.
    :param action: callable object that accepts arguments
    :return: decorated function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
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
