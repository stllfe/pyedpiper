import os
from pathlib import Path

from src.common.config import Config
from src.common.modules import ModelObjectProvider
from src.common.consts import CONFIGS_DIR


def get_project_root() -> Path:
    file_dir = Path(__file__)
    parents = file_dir.parents
    src_dir = next((parent for parent in parents if parent.name == 'src'), None)
    main_py_dir = next((parent for parent in parents if list(parent.glob('main.py'))), None)

    if src_dir:
        return src_dir.parent
    if main_py_dir:
        return main_py_dir

    error = "Can't locate the root folder. Make sure you have a `src` folder or a `main.py` in your project root!"
    raise Exception(error)


def order_by_date_modified(files, latest_first=False):
    """
    Orders files by the last time modified.
    :param files:
        list of PosixPath objects or string paths to sort.
    :param latest_first:
        whether or not to start from the latest file
    :return:
        list with the same objects as in files but with correct order
    """
    files.sort(key=os.path.getmtime, reverse=latest_first)
    return files


def load_configuration() -> Config:
    root = get_project_root()
    config_paths = (root / CONFIGS_DIR).glob(f'**/*.json')
    config_paths = list(config_paths)

    if not config_paths:
        err = f"Can't locate configuration files. Place `file.json` in `{CONFIGS_DIR}` first!"
        raise FileNotFoundError(err)

    config = Config()

    for config_path in config_paths:
        config.add_file(config_path)

    config.load()
    return config


def get_engines(config: Config):
    if config.run == 'train':
        pass

    model = ModelObjectProvider(config).model_obj
