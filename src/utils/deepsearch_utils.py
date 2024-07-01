import re
import os
from typing import Tuple
import pandas as pd
from pathlib import Path


def get_entities() -> dict:
    """Read entities that are relevant.

    Returns:
        dict: dict of sub-collection and entities.
    """
    entity_dict = {}
    datapath = Path("./cps-nlp-llm/scripts/deepsearch_data/taxonomy/esg/")
    files = ["environment", "governance", "social"]

    for file in files:
        df = pd.read_csv(datapath.joinpath(f"{file}.csv"))
        items = df["Field Name"]
        items = [item.lower().strip(" ") for item in items]
        entity_dict[file] = items

    return entity_dict


def find_numbers_in_text(text: str):
    """Finds numbers, even if they are separated by '.' or ',' or '-'."""
    pattern = r"(?P<number>\d+[.,-]?\d*)"
    return re.findall(pattern, text)


def normalize_text(text: str):
    """Normalize text before comparison."""
    text = text.lower()
    return text


def count_numbers_in_table(df_table):
    """Count the number of numbers in a dataframe."""
    sep = "[\.\,]"
    number_pattern = rf"(?P<decimal>\d*{sep}?\d*{sep}?\d*)"
    matches = re.findall(
        pattern=number_pattern, string=df_table.to_csv(columns=None, header=None)
    )
    mod_matches = [item for item in matches if item != ""]
    return len(mod_matches)

def ratio_of_numbers_in_table(df_table):
    """Calculate the ratio of numbers to cells in a dataframe."""
    # nrows, ncols = ds_table["#-rows"], ds_table["#-cols"]
    nrows, ncols = df_table.shape
    if nrows <= 2 or ncols <= 2:
        return 0
    else:
        total_cells = nrows * ncols
        col_row_headers = (
            nrows + ncols
        )  # cols and row headers are numbers so we remove them
        # df_table = pd.DataFrame(extract_data_from_table(ds_table))
        count_numbers = count_numbers_in_table(df_table)
        return (count_numbers - col_row_headers) / total_cells


def sanity_tests_on_text(text: str, min_word_count: int) -> Tuple[bool, bool]:
    """Check if a text is useful via heuristic sanity tests.

        Sanity tests:
        - should have minimum number of words
        - should have numbers

    Args:
        text (str): text to test.
        min_word_count (int): number of minimum words the text should have.

    Returns:
        bool: true, if text passed the sanity tests, false otherwise
    """
    # test 1: number of words
    words = text.split(" ")
    bool_word_count = len(words) >= min_word_count
    # test 2: should have numbers
    numeric_values = find_numbers_in_text(text)
    bool_numeric = len(numeric_values) > 0

    return (bool_word_count, bool_numeric)

def sanity_test_table(df):
    """Sanity tests for structured data."""
    bool_empty_table = df.shape[0] <= 1 or df.shape[1] <= 1

    # test value : column should have numbers
    try:
        count_numbers = count_numbers_in_table(df)
    except (IndexError, ValueError, TypeError, KeyError):
        count_numbers = 0

    bool_numeric_values = count_numbers > 0
    # print(df[2].to_list(),f"count numbers: {count_numbers}")
    # print(f"text_values is {test_values}")
    # print(f"text_ncols is {test_ncols}")
    # print(f"Overall result: {test_ncols and test_values}")

    return (bool_empty_table, bool_numeric_values)




def document_statistics(
    doc, relevant_tables, count_table_fails, relevant_texts, count_text_fails
):
    """Do statistics for single document."""
    doc_stats = {}
    doc_stats["filename"] = doc["_source"]["file-info"]["filename"]
    doc_stats["doc_hash"] = doc["_source"]["file-info"]["document-hash"]
    doc_stats["count_tables"] = len(doc["_source"]["tables"])

    # count paragraphs
    count_paragraphs = 0
    for item in doc["_source"]["main-text"]:
        if item["type"] == "paragraph":
            count_paragraphs += 1

    doc_stats["count_texts"] = count_paragraphs
    doc_stats["relevant_tables"] = len(relevant_tables)
    doc_stats["relevant_texts"] = len(relevant_texts)

    # add failure counts
    for k, v in count_table_fails.items():
        doc_stats[k] = v
    for k, v in count_text_fails.items():
        doc_stats[k] = v

    # now count table with each entities
    doc_stats["table_entities"] = {}
    for table in relevant_tables:
        for ent in table["entities"]:
            if ent not in doc_stats["table_entities"]:
                doc_stats["table_entities"][ent] = 1
            else:
                doc_stats["table_entities"][ent] += 1

    # now count texts with each entities
    doc_stats["text_entities"] = {}
    for text in relevant_texts:
        for ent in text["entities"]:
            if ent not in doc_stats["text_entities"]:
                doc_stats["text_entities"][ent] = 1
            else:
                doc_stats["text_entities"][ent] += 1
    doc_stats["doc_description"] = doc["_source"]["description"]
    return doc_stats