from pathlib import Path

from core.common.config import Config
from core.common.consts import TORCH_EXTENSIONS
from core.common.decorators import config_validation
from core.common.types import Modes

from core.utils.helpers import torch_loader
from core.utils.validations import (
    is_valid_directory,
    is_valid_file,
)


@config_validation
def data_validation(config: Config):
    directories = []
    errors = []

    if not config.data_folder:
        errors.append("No `data_folder` specified in config. Please update configuration!")

    full_path = Path(config.data_folder).resolve()

    if config.data_root:
        root = Path(config.data_root).resolve()
        full_path = root / full_path

    directories.append(full_path)

    for mode in Modes:
        mode_path = config[f'data_{mode}']
        if mode_path:
            mode_path = Path(mode_path)
            directories.append(mode_path)

    for directory in directories:
        if not is_valid_directory(directory):
            error = "{} path is provided but is not correct. ".format(directory)
            message = "Make sure path exists and is not a file."
            errors.append(error + message)

    return {'result': not any(errors), 'errors': errors}


@config_validation
def checkpoint_validation(config: Config):
    errors = []
    file_path = config.checkpoint

    if file_path:
        if is_valid_file(file_path, TORCH_EXTENSIONS):
            try:
                torch_loader(file_path, config.device)
            except Exception as error:
                errors.append(str(error))
        error = "File {} is not a valid checkpoint. Path is not correct or file is corrupted.".format(file_path)
        errors.append(error)

    return {'result': not any(errors), 'errors': errors}
