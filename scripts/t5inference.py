"""Script for making inference (no ground-truth data) with fine-tuned T5 models."""

import argparse
# import re
# from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from src.io.read import read_jsonl
from src.io.write import save_dict_in_jsonl
from src.input_data.table import Table
from src.dataset.t5data import T5Data
from src.parsers.t5parser import T5Parser
from src.models.t5.predictor import T5Predictor
from src.utils.utils import extract_labels_table_from_inference, convert_labels_table_to_statement
import copy
from typing import Union


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Predict a T5 fine-tuned model."
    )

    parser.add_argument(
        "-i",
        "--input-file",
        help="Path to input jsonl file containing data to predict on.",
        required=True
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        help=f"Output directory for files and logs.",
        required=True
    )

    parser.add_argument(
        "-tcf",
        "--training-config-file",
        help="Config file used for training the model",
        required=True
    )

    parser.add_argument(
        "-t",
        "--input-type",
        help="text or table",
        required=True
    )

    parser.add_argument(
        "-c",
        "--checkpoint-path",
        help="Config file defining the model",
        required=True
    )

    return parser.parse_args()

def extract_statements(input_data: Union[pd.DataFrame,str], 
                      predictor:T5Predictor, 
                      input_type:str, 
                      retry:int=5)->Union[list[dict],str]:
    """This function extracts statements from model output.
        If an error occurs, it retries the specified numbert of times.

    Args:
        input_data (dataframe or str): data input to model
        retry (int, optional): Number of retries for model inferencing. Defaults to 3.

    Returns:
        list[dict]: list of statements as dataframe.
        model_output[str]: raw output of the model
    """
    statements, model_output = None, None

    if input_type == 'table':
        parser = T5Parser(task='ud2sd')
        md_table = T5Data.convert_table_to_markdown(input_data)
        model_output = predictor.predict(input=md_table, input_type=input_type)
        labels_table = extract_labels_table_from_inference(model_output, parser=parser)
        statements = convert_labels_table_to_statement(labels_table=labels_table,
                                                        input_data=input_data)        
                
    elif input_type == 'text':
        try:
            parser = T5Parser(task='ud2sd')
            model_output = predictor.predict(input=input_data, input_type=input_type)
            model_statements = parser.parse(model_output)
            df_statements = [parser.convert_markdown_to_dataframe(item) for item in model_statements]
            statements =  [item.to_dict(orient='list') for item in df_statements]
        except:
            pass

    if statements is None or model_output is None:
        if retry == 0:
            return (None, None)
        statements, model_output = extract_statements(input_data=input_data, 
                        input_type=input_type, 
                        predictor = predictor, 
                        retry = retry - 1)        
    return (statements, model_output)


def main():

    print("main")
    args = parse_arguments()
    
    ifile = Path(args.input_file)
    data = read_jsonl(ifile.parent, ifile.name)

    predictor = T5Predictor(
        training_config_file=Path(args.training_config_file), 
        checkpoint_path=Path(args.checkpoint_path),
    )
    
    rfile_name = str(Path(args.checkpoint_path).name).removesuffix('.ckpt')+"_inference.jsonl"
    rpath = Path(args.output_dir).joinpath(rfile_name)

    counts = 0
    for item in tqdm(data, total=len(data)):
        
        if args.input_type == 'table':
            table = Table(item['table'])
            input_data = table.to_dataframe()
            
        elif args.input_type == 'text':
            if 'input' in item.keys():
                input_data = item['input']
            elif 'text' in item.keys():
                input_data = item['text']
            else:
                print(item.keys())
                print("Please tell me which key corresponds to the data!")

        ndict = copy.deepcopy(item)
        statements, model_output = extract_statements(
                                input_data=input_data, 
                                input_type=args.input_type, 
                                predictor = predictor, 
                                retry = 5)
        ndict['model_output'] = model_output
        if statements is not None:
            ndict['statements'] = statements
            #save data:
            save_dict_in_jsonl(ndict, rpath)
            counts +=1

    print(f"=== Total items: {len(data)}")
    print(f"=== Successfully predicted on: {counts} ({100*counts/len(data):.2f})%")


if __name__ == '__main__':
    main()