"""Class for reading a training config and training a model."""

from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset.t5data import T5Data
from src.models.t5.model import T5Model
from src.io.read import read_json
from src.utils.training_utils import validate_config_file, get_gpu_details

class T5Trainer:
    """Class that trains a T5Model"""

    def __init__(
        self,
        run_config_file : Path,
        output_dir : Path
    ):
        """Initializes the trainer class with parameters from the run_config file.
        """
        run_config = read_json(run_config_file.parent, run_config_file.name)
        run_config = validate_config_file(run_config, run_config_file.name)
        self.run_config = run_config

        # Set trainer params
        self.trainer_params = {}
        remove_keys_not_for_trainer = ['run_name',
                                       'task',
                                       'device_stats',
                                       'device',
                                    'model_config_key',
                                    'training_data_path',
                                    'append_special_tokens',
                                    'num_workers',
                                    'training_data_ratio',
                                    'batch_size',
                                    'learning_rate',
                                    'seed_everything',
                                    'model_config_file']
        for k, v in run_config.items():
            if k == 'gpu_devices':
                self.trainer_params["devices"] = run_config["gpu_devices"]
            elif k in remove_keys_not_for_trainer:
                continue
            else:
                self.trainer_params[k] = v

        # print(self.trainer_params)
        self.trainer_params["default_root_dir"] = Path(output_dir)

        # Set optimizer parameters
        self._optim_params = {}
        self._optim_params["lr"] = run_config["learning_rate"]

    def train(self) -> T5Model:
        """Trains a model

        Returns
        -------
        T5Model
            The trained model
        """
        # get gpu details
        if self.run_config['gpu_devices'] == 1:
            get_gpu_details(print_details=True)

        # Load the data
        dataset = T5Data(data_path=Path(self.run_config["training_data_path"]), task = self.run_config['task'])
        train_count = int(self.run_config['training_data_ratio']*len(dataset))
        val_count = len(dataset) - train_count
        dataset = dataset.random_split( dataset=dataset,
                                        lengths=[train_count, val_count],
                                        seed=42)

        # Prepare data loaders
        train_loader = DataLoader(dataset[0], 
                                  batch_size=self.run_config["batch_size"], 
                                  shuffle=True,
                                  num_workers=self.run_config["num_workers"],
                                  pin_memory=True,
                                  persistent_workers=True)
        val_loader = DataLoader(dataset[1], 
                                batch_size=self.run_config["batch_size"], 
                                shuffle=False,
                                num_workers=self.run_config["num_workers"],
                                pin_memory=True,
                                persistent_workers=True)

        # Prepare the model
        model = T5Model(
            model_config_key = self.run_config['model_config_key'],
            append_special_tokens=self.run_config['append_special_tokens'],
            optim_params=self._optim_params,
            hparam_config= self.run_config,
            model_config_file=Path(self.run_config['model_config_file']))

        # Train the model
        model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            trainer_args=self.trainer_params,
        )

        return model
