"""Scripts for reading from disk."""

import json
from pathlib import Path


def read_json(dir_name: Path, fname: str):
    """Read data from json.

    Args:
        dir_name (Path): directory path.
        fname (str): filename with extension.

    Returns:
        dict : data.
    """
    # check file exists
    filepath = Path(dir_name).joinpath(fname)

    if not filepath.exists():
        raise FileNotFoundError(f"Cannot find file: {fname} in dir: {dir_name}")

    with open(filepath) as f:
        data = json.load(f)
    return data


def read_jsonl(dir_name: Path, fname: str, lines_to_read:int=-1) -> list:
    """Read jsonl data into a python list.

    Args:
        dir_name (path): directory path.
        fname (str): filename with extension.

    Returns:
        list : list containing dict of data.
    """
    # check file exists
    filepath = Path(dir_name).joinpath(fname)

    if not filepath.exists():
        raise FileNotFoundError(f"Cannot find file: {fname} in dir: {dir_name}")

    data = []
    count = 0
    with open(filepath, mode="r") as fp:
        for line in fp.readlines():
            data.append(json.loads(line))
            count +=1
            if lines_to_read > 0 and count == lines_to_read:
                break

    return data
