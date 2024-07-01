"""Scripts for writing to disk."""
import json
from pathlib import Path


def save_dict_in_jsonl(dict_item: dict, filepath: Path):
    """Append dictionary object to jsonl file.

    Args:
        dict_item (dict): dictionary object to save.
        filepath (Path): path to jsonl file to write in.
    """
    # check file exists
    # if not filepath.exists():
    #     raise FileNotFoundError(f"Cannot find file: {filepath}")

    # check if extension is jsonl
    if not filepath.suffix == ".jsonl":
        raise ValueError("Expecting target file to be jsonl")

    with open(filepath, "a") as outfile:
        json.dump(dict_item, outfile)
        outfile.write("\n")
    return


def save_dict_as_json(dict_item: dict, filepath: Path):
    """Write dictionary as json.

    Args:
    dict_item (dict): dictionary object to save.
    filepath (Path): path to json file to write in.
    """
    # check if extension is jsonl
    if not filepath.suffix == ".json":
        raise ValueError("Expecting target file to be json")

    with open(filepath, "a") as fp:
        json.dump(dict_item, fp)
