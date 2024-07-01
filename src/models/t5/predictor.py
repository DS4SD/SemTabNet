"""Class for making predictions from a model checkpoint."""
import torch
from pathlib import Path
from src.models.t5.model import T5Model
from src.io.read import read_json
from src.utils.training_utils import get_gpu_details

class T5Predictor:
    """Class for making predictions with a fine-tuned T5Model from a checkpoint"""

    def __init__(self, 
                 training_config_file:Path,
                 checkpoint_path:Path, 
                 model_config_file:Path=None,
                 ):
        """Initialize the T5 

        Args:
            training_config_file (Path): Training config file which was for training the model.
            checkpoint_path (Path): Path to model checkpoint.
        """
        self.training_config_file = training_config_file
        self.training_config = read_json(training_config_file.parent, training_config_file.name)
        # print("==== TRAINING CONFIG")
        # print(self.training_config)
        # print("=== === ===")
        if model_config_file is None:
            self.model_config_file = Path(self.training_config['model_config_file'])
        else:
            self.model_config_file = Path(model_config_file)
        self.checkpoint_path = checkpoint_path
        self.load_model_from_checkpoint()

    def load_model_from_checkpoint(self):
        # Initialize the model
        self.model = T5Model(model_config_key = self.training_config['model_config_key'],
                        append_special_tokens=self.training_config['append_special_tokens'],
                        optim_params={},
                        hparam_config= self.training_config,
                        model_config_file=self.model_config_file
                        )

        # decide on current device:
        if not torch.cuda.is_available():
            ckpt = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        else:
            ckpt = torch.load(self.checkpoint_path)
            # get gpu details
            get_gpu_details(print_details=True)
        
        checkpoint = {}
        for k, v in ckpt['state_dict'].items():
            checkpoint[k.removeprefix('model.')] = v

        # model.model.resize_token_embeddings(32108)
        self.model.model.load_state_dict(checkpoint)  

        if torch.cuda.is_available():
            self.model.to('cuda')
            print(f"Is model on GPU?: {next(self.model.parameters()).is_cuda}")

        return
    
    def predict(self, input:str, input_type:str) -> str:
        """Method for generating predictions for a single data.

        Args:
            input (str): input for the model
            input_type (str): type of input from 'text'/'table'

        Returns:
            str: Model predictions
        """
        output_token_ids, output = self.model.predict(input=input, input_type=input_type)
        return output
    
    def predict_batch(self, batch) -> str:
        """Method for generating predictions for a batch.

        Args:
            batch (dict): a sequence of inputs in the form:
            {
                "input" : list[str],
                "input_type" : list[str]
            }     
        where:
            input (str): input for the model
            input_type (str): type of input from 'text'/'table'
            
        Returns:
            str: Model predictions
        """
        output_token_ids, output = self.model.predict_batch(batch=batch)
        return output_token_ids, output