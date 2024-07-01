"""Utility functions."""
import hashlib


def create_hash(data):
    """Create hash for data."""
    return hashlib.md5(str(data).encode()).hexdigest()


def extract_data_from_table(table: dict):
    """Extract textual content from a table dcitionary (as output by DeepSearch)."""
    try:
        table_content = [[cell["text"] for cell in row] for row in table["data"]]
    except TypeError:
        table_content = [[]]
    return table_content

