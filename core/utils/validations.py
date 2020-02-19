from pathlib import Path
from typing import Union, List


def is_valid_directory(path: Union[str, Path]) -> bool:
    path = Path(path).resolve()
    return path.exists() and path.is_dir()


def is_valid_file(path: Union[str, Path], allowed_extensions: List) -> bool:
    path = Path(path).resolve()
    return path.exists() and path.is_file() and path.suffixes in allowed_extensions
