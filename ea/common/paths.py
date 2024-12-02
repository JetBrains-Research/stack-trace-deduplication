from pathlib import Path
from typing import Union


def pathlib_path(path: Union[str, Path]) -> Path:
    if isinstance(path, str):
        return Path(path).expanduser().resolve()
    return path


def str_path(path: Union[str, Path]) -> str:
    if isinstance(path, str):
        return path
    return str(path.expanduser().resolve())


class EaPaths:
    home = (Path.home() / '.ea').expanduser().resolve()
    models = home / 'models'
    data = home / 'data'
    outs = home / 'outs'
    logs = home / 'logs'
    temp = home / 'temp'
    aws = home / 'aws-data'
