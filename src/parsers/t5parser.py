from io import StringIO
from typing import List
import pandas as pd
from src.dataset.base import BaseDataset

class T5Parser():

    input_text_start_token = "<text>"
    input_text_stop_token = "</text>"
    input_table_start_token = "<table>"
    input_table_stop_token = "</table>"
    output_start_token = "<response>"
    output_stop_token = "</response>"
    newline_token = "<br>"
    sep_token = "<sep>"
    pad_token = "<pad>"
    eos_token = "</s>"
    allowed_tasks = BaseDataset.allowed_tasks
    allowed_cell_labels = ['subject', 
                           'subject_value', 
                           'property', 
                           'subproperty', 
                           'property_value', 
                           'header_1', 
                           'header_2', 
                           'header_3', 
                           'key', 
                           'key_value', 
                           'unit_property', 
                           'unit_value', 
                           'time_property', 
                           'time_value', 
                           'empty', 
                           'rubbish']

    def __init__(self, task:str) -> None:
        if task in T5Parser.allowed_tasks:
            self.task = task
        else:
            raise ValueError("Task not defined in the model family! Please check.")
        
    def check_eos_tokens(self, model_output:str) -> bool:
        if T5Parser.output_stop_token in model_output:
            return True
        elif T5Parser.eos_token in model_output:
            return True
        else:
            return False


    def remove_special_tokens(self,  model_output:str, ) -> str:
        # remove padding token from left
        x = model_output.replace(T5Parser.pad_token,'')
        # remove eos from right
        x = x.removesuffix(T5Parser.eos_token)
        # remove whitespace from left and right
        x = x.lstrip().rstrip()
        # remove response start and 
        x = x.removeprefix(T5Parser.output_start_token).removesuffix(T5Parser.output_stop_token)
        # remove whitespace from left and right
        x = x.lstrip().rstrip()
        return x


    def parse(self, model_output:str) -> List[str]:
        x = self.remove_special_tokens(model_output=model_output)
        # replace line breaks with new lines and split on separations
        x = x.replace(T5Parser.newline_token,"\n").split(T5Parser.sep_token)
        return x
    
    def convert_markdown_to_dataframe(self, 
                                      md_table:str, 
                                      sep:str="|",
                                      set_headers:bool=True,) -> pd.DataFrame:
        """Attempt to convert markdown to dataframe.

        Args:
            md_table (str): table in markdown format
            sep (str, optional): column separator. Defaults to "|".
            set_headers (bool, optional): Convert first row as headers. Defaults to True.

        Returns:
            pd.DataFrame: dataframe object
        """
        # Read a markdown formmated text, getting the header from the first row and index from the second column
        # Drop the left-most and right-most null columns 
        # Drop the header underline row
        if md_table != '':
            # remove padding whitespace but not newline chars
            md_table = md_table.replace("\n","<br>")
            md_table = '|'.join([item.strip() for item in md_table.split('|')])
            md_table = md_table.replace("<br>","\n")
            
            # convert
            try:
                table = pd.read_csv(StringIO(md_table), 
                                    sep = sep,
                                    header=None, 
                                    index_col=None,
                                    skipinitialspace=True).dropna(axis=1, how='all')
                table.index = range(table.shape[0])
                table.columns = range(table.shape[1])
                table = table.drop(index=1)
                if not set_headers:
                    return table
                else:
                    # convert 1st row to headers
                    table.columns = table.iloc[0]
                    table = table.drop(index=0)
                    table.index = range(table.shape[0])
                    try:
                        table.columns = [str(item).strip() for item in table.columns]
                    except Exception as e:
                        raise ValueError("I failed to convert the first row as header.", e)
                    return table
            except Exception as e:
                raise ValueError("I failed to convert this to a dataframe.", e)
                # print(f"Something is off! Here are the deets: {Exception}:{ee}")
        else:
            return         
    
    def parse_single_cell_annotation(self, cell_label:str):
        clean_label = self.remove_special_tokens(cell_label) 
        if clean_label in T5Parser.allowed_cell_labels:
            return clean_label 
        else: 
            return 'misclassified'
    
    def parse_cell_annotations(self, cell_labels:list[list]):
        labels_table = []
        if isinstance(cell_labels, list):
            if isinstance(cell_labels[0], list):
                for row in cell_labels:
                    clean_row = []
                    for cell_label in row:
                        cell_label = str(cell_label)
                        clean_row.append(self.parse_single_cell_annotation(cell_label=cell_label))
                    labels_table.append(clean_row)
                return labels_table
            else:
                raise TypeError("Expected list of list! Check input")
        else:
            raise TypeError("Expected list of list! Check input")


    @staticmethod
    def normalize_text(text:str)-> str:
        # lower case
        text = text.lower()
        
        # remove newline and sep tokens
        text = text.replace(T5Parser.newline_token,'').replace(T5Parser.sep_token, '')

        # remove markdown elements
        text = text.replace('-','')

        # remove extra spaces
        text = ' '.join([y.rstrip().lstrip() for y in text.split('|')]).lstrip().rstrip()

        return text