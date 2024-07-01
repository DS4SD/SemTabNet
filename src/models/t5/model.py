""" Class for defining T5 models from HF transformers."""

from pathlib import Path
from typing import Any, Optional, Union

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.models.base import BaseModel
from src.dataset.t5data import T5Data

# Vocabulary: 32000 English word pieces
# T5 comes in different sizes:
# "t5-small"    -> 060 M parameters
# "t5-base"     -> 220 M parameters
# "t5-large"    -> 770 M parameters
# "t5-3b"       -> 2.8 B parameters
# "t5-11b"      -> 011 B parameters

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

class T5Model(BaseModel):
    """T5 model for sequence to sequence modelling."""
    
    def __init__(
        self, 
        append_special_tokens : bool,
        model_config_key: str,
        hparam_config:dict,
        model_config_file:Path,
        optim_params: Optional[dict] = None,
    ) -> None:
        """Intantiate an instance of the model.

        Args:
            model_config_key (str, optional):
            Multiple configs are defined in the config file. Pick one flavour and point
            its key to this argument.
            optim_params (Optional[dict], optional): Parameters for the optimizer.
            hparam_config (dict) : Pass the run config here to save as hyperparameters.
            Defaults to None.
        """
        super().__init__(
                    model_config_key=model_config_key,
                    model_config_file=model_config_file,
                    hparam_config=hparam_config,
                    optim_params=optim_params)

        # set the models
        self.encoder = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=self.model_config["model_name"],
            model_max_length=self.model_config["max_input_length"],
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=self.model_config["model_name"],
        )
        
        self._append_special_tokens = append_special_tokens
        if self._append_special_tokens:
            before_vocab_size = len(self.encoder)
            special_tokens = T5Model.special_tokens
            special_tokens.extend(T5Data.special_tokens)
            self.encoder.add_tokens(special_tokens)
            after_vocab_size = len(self.encoder)
            print(f"Before size: {before_vocab_size}")
            print(f"After size: {after_vocab_size}")
            # if after_vocab_size - before_vocab_size != len(T5Model.special_tokens):
            #     raise ValueError("Vocabulary mismatch after adding new tokens")
            self.model.resize_token_embeddings(len(self.encoder))
        else:
            print("You've set 'append_special_token' to False.")
            print("Special tokens have not been added to the Tokenizer! Performance unpredictable.")

        return


    def _preprocess_input_data(self, input, input_type):
        """Preprocess or normalize input (text).

        Args:
            input (str): raw text data.

        Returns:
            str: normalized text data.
        """
        input = input.strip(" ")
        if self.model_config["input_lowercase"]:
            input = input.lower()
        
        if self._append_special_tokens:
            if input_type == "text":
                input = f"{BaseModel.input_text_start_token}\n{input}\n{BaseModel.input_text_stop_token}"
            elif input_type == "table":
                input = f"{BaseModel.input_table_start_token}\n{input}\n{BaseModel.input_table_stop_token}"
        return input        

    def _preprocess_output_data(
        self,
        output: str,
    ) -> str:
        """Preprocess or normalize outpu (text).

        Args:
            output (str): string representation of data.

        Returns:
            str: normalized text data.
        """
        output = output.strip(" ")
        if self.model_config["input_lowercase"]:
            output = output.lower()

        if self._append_special_tokens:
            output = f"{BaseModel.output_start_token}{output}{BaseModel.output_stop_token}"
        return output

    def _preprocess_batch(self, batch: dict) -> dict:
        """Preprocess individual batch

        Args:
            batch (dict): a sequence of inputs and ouputs in the form:
            {
                "input" : list[str],
                "input_type" : list[str]
                "output" : list[str]
            }

        Returns:
            dict: Processed batch with keys: "input_ids", "attention_mask", "labels".
        """
        # apply preprocessing steps
        batch_input = list(
            map(self._preprocess_input_data, batch["input"], batch["input_type"])
        )
        batch_output = list(
            map(self._preprocess_output_data, batch["output"])
        )

        # input : tokenize and encode
        batch_input = self.encoder(
            text=batch_input,  # text to tokenize + encode
            return_tensors="pt",  # pytorch tensors
            padding="max_length",  # pad to max_length
            is_split_into_words=False,  # tokenize the text first
            max_length=self.model_config["max_input_length"],  # for truncate/pad
            truncation="longest_first",  # truncate tokens
        )

        # output : tokenize and encode
        batch_output = self.encoder(
            text=batch_output,
            return_tensors="pt",  # pytorch tensors
            padding="max_length",  # pad to max_length
            is_split_into_words=False,  # tokenize the text first
            max_length=self.model_config["max_input_length"],  # for truncate/pad
            truncation="longest_first",  # truncate tokens
        )

        # replace padding token id's of the labels by -100 so it's ignored by the loss
        labels = batch_output.input_ids.detach().clone()  # type:ignore
        labels[labels == self.encoder.pad_token_id] = -100

        # create batch
        batch = {
            "input_ids": batch_input.input_ids,  # type:ignore
            "attention_mask": batch_input.attention_mask,  # type:ignore
            "labels": labels,
        }

        return batch

    def predict(
        self, input: str, input_type :str ) -> Union[Any, str]:
        """Method for generating prediction on single input.

        Args:
            input (str): data (text or table as markdown table)
            input_type (str): text or table

        Returns:
            str: predicted output data
        """
        with torch.no_grad():
            self.eval()

            # preprocess the input
            ud = self._preprocess_input_data(input,input_type=input_type)

            # tokenize and encode
            ud = self.encoder(
                text=[ud],
                return_tensors="pt",  # pytorch tensors
                padding="max_length",  # pad to max_length
                is_split_into_words=False,  # tokenize the text first
                max_length=self.model_config["max_input_length"],  # for truncate/pad
                truncation="longest_first",  # truncate tokens
            ).data

            # generate model prediction
            output_token_ids = self.model.generate(
                input_ids=ud["input_ids"].to(device),
                max_new_tokens=4096,
                num_beams=1,
                do_sample=False,
            )

            # decode the prediction
            output = self.encoder.batch_decode(output_token_ids, skip_special_token=True)[0]

            return output_token_ids, output


    def predict_batch(self, batch:dict):
        """Function to make predictions over a single batch.

        Args:
            batch (dict): a sequence of inputs in the form:
            {
                "input" : list[str],
                "input_type" : list[str]
            }        
        """
        with torch.no_grad():
            self.eval()
            # preprocess the input data
            batch_input = list(
            map(self._preprocess_input_data, batch["input"], batch["input_type"]
                ))
            
            # input : tokenize and encode
            batch_input = self.encoder(
                text=batch_input,  # text to tokenize + encode
                return_tensors="pt",  # pytorch tensors
                padding="max_length",  # pad to max_length
                is_split_into_words=False,  # tokenize the text first
                max_length=self.model_config["max_input_length"],  # for truncate/pad
                truncation="longest_first",  # truncate tokens
            )   

            # generate model prediction
            output_token_ids = self.model.generate(
                input_ids=batch_input.data["input_ids"].to(device),
                max_new_tokens=4096,
                num_beams=1,
                do_sample=False,
            )             

            # decode the prediction
            output = self.encoder.batch_decode(output_token_ids, skip_special_token=True)

            return output_token_ids, output