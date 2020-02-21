from pathlib import Path
from typing import Union, Iterable

from core.common.types import StrEnum


def is_valid_directory(path: Union[str, Path]) -> bool:
    path = Path(path).resolve()
    return path.exists() and path.is_dir()


def is_valid_file(path: Union[str, Path], allowed_extensions: Iterable[str]) -> bool:
    path = Path(path).resolve()
    return path.exists() and path.is_file() and path.suffix in allowed_extensions


def is_implemented_type(value: str, enumeration: type(StrEnum)) -> bool:
    for implemented_type in enumeration:
        if value.casefold() == implemented_type.casefold():
            return True
    return False
