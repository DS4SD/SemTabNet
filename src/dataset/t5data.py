"""Dataset class for T5 models for UD2SD task."""
import json
from pathlib import Path

import pandas as pd

from src.dataset.base import BaseDataset


class T5Data(BaseDataset):
    """Dataset for T5 model family."""

    def __init__(self, data_path: Path, task:str):
        """Initialize/read training data for T5 model.

        Args:
            data_path (Path): file path
            task (str) : ud2sd or chemistry
        """
        self.input: list[str] = []
        self.output: list[str] = []
        self.input_type : list[str] = []
        self.augmentation_status : list[str] = []
        
        if task in ['ud2sd_table']:
            self.read_ud2sd_data(data_path)
        elif task =="tca1d":
            self.read_tca1d_data(data_path)
        elif task =="tca2d":
            self.read_tca2d_data(data_path)
        else:
            raise ValueError(f"Incorrect task: {task}! Task must be one of: {T5Data.allowed_tasks}.")
        
        print("Dataset health:")
        print(f"# input data: {len(self.input)}")
        print(f"# output data: {len(self.output)}")        
        return
    
    def __len__(self) -> int:
        """Number of items in dataset."""
        return len(self.input)

    def __getitem__(self, idx) -> dict:
        """Returns dataset object at index."""
        item = {
            "input": self.input[idx],
            "output": self.output[idx],
            "input_type":self.input_type[idx],
            "augmentation_status":self.augmentation_status[idx]
        }
        return item
    
   
    def read_ud2sd_data(self, data_path: Path) -> None:
        """Read unstructured_data and structured_data data.

        Args:
            data_path (Path): path to jsonl file.

        Raises:
            ValueError: file path must be of jsonl type.
        """
        if data_path.suffix != ".jsonl":
            raise ValueError(f"{data_path.suffix} file not supported. Only jsonl.")

        with open(data_path, mode="r") as fp:
            for line in fp.readlines():
                _: dict = json.loads(line)
                
                # fix sd
                if isinstance(_['structured_data'], list):
                    sd = T5Data.convert_list_of_structured_data_to_string(_['structured_data'])
                elif isinstance(_['structured_data'], dict):
                    sd = T5Data.convert_structured_data_to_string(_['structured_data'])

                # fix ud
                if isinstance(_['unstructured_data'], list):
                    ud = T5Data.convert_table_to_markdown(pd.DataFrame(_['unstructured_data']))
                elif isinstance(_["unstructured_data"], str):
                    ud = _['unstructured_data']

                # augmentation_status
                if 'augmentation_status' in _.keys():
                    aug = _['augmentation_status']
                else:
                    aug = ''
                
                self.add_data(
                    input=ud,
                    output=sd,
                    input_type = _['ud_type'],
                    augmentation_status=aug
                    )
        return
    
    def read_tca1d_data(self, data_path: Path) -> None:
        """Read table cell annotation data (1D)

        Args:
            data_path (Path): path to jsonl file.

        Raises:
            ValueError: file path must be of jsonl type.
        """
        if data_path.suffix != ".jsonl":
            raise ValueError(f"{data_path.suffix} file not supported. Only jsonl.")

        with open(data_path, 'r') as file:
            for line in file:
                # read line
                _:dict = json.loads(line)
                # Add to dataset
                self.add_data(
                    input=_["input"],
                    output=_["output"],
                    augmentation_status='original'
                )
        return
    
    def read_tca2d_data(self, data_path: Path) -> None:
        """Read table cell annotation data (1D)

        Args:
            data_path (Path): path to jsonl file.

        Raises:
            ValueError: file path must be of jsonl type.
        """
        if data_path.suffix != ".jsonl":
            raise ValueError(f"{data_path.suffix} file not supported. Only jsonl.")

        with open(data_path, 'r') as file:
            for line in file:
                # read line
                _:dict = json.loads(line)
                # Add to dataset
                self.add_data(
                    input=self.convert_table_to_markdown(pd.DataFrame(_["table"])),
                    output=self.convert_table_to_markdown(pd.DataFrame(_["cell_annotations"])),
                    augmentation_status=_["augmentation_status"]
                )
        return

    def add_data(self, 
                 input: str, 
                 output: str, 
                 input_type: str = 'text',
                 augmentation_status:str = 'None'):
        """Add input and output sample to dataset.

        Args:
            input (str): input to the model 
            input_type (str) : type of input data (text or table), Defaut:'text'
            output (str): expected output from the model
            augmentation_status (str) : data is 'original' or 'augmented'
        """
        self.input.append(input)
        self.output.append(output)
        self.input_type.append(input_type)
        self.augmentation_status.append(augmentation_status)
        return

    @staticmethod
    def convert_table_to_markdown(df: pd.DataFrame):
        """Conver input tables to markdown

        Args:
            df (DataFrame): pandas dataframe table

        Returns:
            str: markdown format
        """
        return df.to_markdown(index=False,tablefmt="github").replace("\n",T5Data.newline_token)

    @staticmethod
    def convert_structured_data_to_string(structured_data:dict) -> str:
        """Convert individual data to markdown via pandas

        Args:
            structured_data (dict): should be convertible to pandas dataframe

        Returns:
            str: markdown table
        """
        df = pd.DataFrame(structured_data)
        cols_to_skip = ["data_hash", "organization","predicate_hash"]
        for col in cols_to_skip:
            if col in df.columns:
                df = df.drop(columns=[col])
        return T5Data.convert_table_to_markdown(df=df)


    @staticmethod
    def convert_list_of_structured_data_to_string(structured_data_list:list[dict],) ->str:
        """String representation (markdown table) of multiple structured data dicts in a list.

        Args:
            structured_data_list (list[dict]): list of structured data with keys like: subject, property.
            col_set (str, optional): token to separate columns, Defaults to "|".

        Returns:
            str: Multiple tables are separated by newline.
        """
        str_sd = ""
        for sd in structured_data_list:
            current_string = T5Data.convert_structured_data_to_string(sd)
            str_sd += current_string + T5Data.sep_token
        return str_sd.removesuffix(T5Data.sep_token)


class T5DataTCA1D(T5Data):
    """Dataset for T5 model family.
    
    Used for the prediction for tca1d task.
    """
    def __init__(self, input: list[str], output: list[str], augmentation_status: list[str]):
        """Initialize training data for T5 model."""
        self.input: list[str] = []
        self.output: list[str] = []
        self.augmentation_status : list[str] = []
        self.input_type: list[str] = []
        
        assert(len(input)==len(output)==len(augmentation_status))

        for _input,_output,_augmentation_status in zip(input,output,augmentation_status):
            self.add_data(_input,_output,_augmentation_status)
        return
