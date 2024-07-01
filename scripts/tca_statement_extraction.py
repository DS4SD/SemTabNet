"""Tca statement extraction for inferences."""

import os, sys
import argparse
from pathlib import Path
import pandas as pd

from tqdm import tqdm
from src.input_data.table import Table
from src.io.read import read_jsonl
from src.io.write import save_dict_in_jsonl
from src.dataset.t5data import T5Data
from src.parsers.t5parser import T5Parser
from copy import deepcopy

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract statements from cell_annotations."
    )

    parser.add_argument(
        "-i",
        "--input-file",
        help="Path to input jsonl file containing data to extract statements from.",
        required=True
    )

    parser.add_argument(
        "-t",
        "--task",
        help=f"Fine-tuning task. Options are: tca1d tca2d",
        required=True
    )
    return parser.parse_args()

def extract_statements_from_labels_table(table:Table, return_md:bool =False)->str:
    """Table including cell annotations
    Args:
        table (Table): instant of Table class containing cell labels
    
    Returns:
        statements (string) : markdown representation of statements
    """
    # check cell annotations exist
    if isinstance(table.cell_annotations, pd.DataFrame):
        try:
            table.convert_to_structured_data()
            if return_md:
                return T5Data.convert_list_of_structured_data_to_string(table.annotations)
            if not  return_md:
                return table.annotations
        except Exception:
            return None
    else:
        raise ValueError("Input table is missing cell labels.")
    
def clean_labels_table(tca2d_prediction:str, parser:T5Parser)->pd.DataFrame:
    """Generate clean labels table from model predictions.

    Raises a ValueError in case it is not possible to convert the markdown to a table.
    Args:
        tca2d_prediction (str): markdown labels table from model predictions
    
    Returns:
        labels_table (dataframe) : clean labels table
    """
    clean_predictions = parser.remove_special_tokens(tca2d_prediction)
    labels_table = parser.convert_markdown_to_dataframe(clean_predictions)
    if not isinstance(labels_table,pd.DataFrame):
        # didn't manage to convert the markdown to a table.
        raise ValueError
    labels_table = parser.parse_cell_annotations(labels_table.values.tolist())
    return pd.DataFrame(labels_table)



def main():

    print("main")
    args = parse_arguments()

    if args.task not in ['tca1d', 'tca2d']:
        raise ValueError("Wrong task")
    
    parser = T5Parser(task=args.task)
    
    rfile_name = str(Path(args.input_file).name).removesuffix('.jsonl')+"_statements.jsonl"
    rpath = Path(args.input_file).parent.joinpath(rfile_name)
    print("Your results are being saved at: ", rpath)


    input_file=Path(args.input_file)
    data=read_jsonl(input_file.parent,input_file.name)

    for item in tqdm(data):
        sitem = deepcopy(item)
        with HiddenPrints():
            if args.task=='tca1d':
                # convert model prediction
                labels_table = pd.DataFrame(parser.parse_cell_annotations(item['model_output']))
                table = Table(table=pd.DataFrame(item['input']),
                            cell_annotations=labels_table)
                sitem['model_output'] = extract_statements_from_labels_table(table)

            elif args.task=='tca2d':            
                # convert model prediction
                try:
                    labels_table = clean_labels_table(item['model_output'],parser)
                    table = Table(table=item['table'],
                                cell_annotations=labels_table)
                    sitem['model_output'] = extract_statements_from_labels_table(table)
                    
                except Exception as e:
                    # when the model output markdown can't be converted to a table.
                    # or when there is a mismatch in shape of cell annotations and table.
                    # raise(e)
                    sitem['model_output'] = 'invalid_markdown'
                
            # save data
            save_dict_in_jsonl(sitem, rpath)


if __name__ == '__main__':
    main()