"""Base class definitions."""


import math
from pathlib import Path
from typing import Any, Optional

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from src.io.read import read_json

class BaseModel(pl.LightningModule):
    """Base class for model implementation."""

    input_text_start_token = "<text>"
    input_text_stop_token = "</text>"
    input_table_start_token = "<table>"
    input_table_stop_token = "</table>"
    output_start_token = "<response>"
    output_stop_token = "</response>"
    newline_token = "<br>"
    sep_token = "<sep>"

    
    special_tokens = [
        input_text_start_token,
        input_text_stop_token,
        input_table_start_token,
        input_table_stop_token,
        output_start_token,
        output_stop_token,
        newline_token,
        sep_token,
        "^" # adding caret symbol to model,        
    ]

    def __init__(
        self, 
        model_config_key: str,
        model_config_file:Path,
        hparam_config:dict,
        optim_params: Optional[dict] = None
    ) -> None:
        
        """Intantiate an instance of the model.

        Args:
            model_config_key (str, optional): model flavour from the config file.
            optim_params (Optional[dict], optional): Parameters for the optimizer.
            hparam_config (dict) : Pass the run config here to save as hyperparameters.
        """
        super().__init__()
        self.hparams_config = hparam_config

        # seed everything?
        if isinstance(self.hparams_config['seed_everything'], int):
            pl.seed_everything(self.hparams_config['seed_everything'], workers=True)

        # read model config and get model flavour
        self._read_model_config(model_config_key, model_config_file)
        self.model_name = self.model_config["model_name"]
        self.optim_params: dict = {} if optim_params is None else optim_params

        # save hyperparameters
        self.save_hyperparameters()
        return
    
    def _read_model_config(self, model_config_key: str, model_config_file:Path) -> None:
        """Read model config from config file.

        Args:
            model_config_key (srt):
            Multiple configs are defined in the config file. Pick one flavour and point
            its key to this argument.
        """
        _ = read_json(dir_name=model_config_file.parent,
                      fname=model_config_file.name)
        self.model_config = _[model_config_key]
        return
    

    def training_step(self, batch: dict, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Training loop"""
        loss = self.model(**batch).loss
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, logger=True)
        return loss

    def validation_step(self, batch: dict, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Validation loop"""
        loss = self.model(**batch).loss
        # Logging to TensorBoard (if installed) by default
        self.log("val_loss", loss, logger=True, sync_dist=True)
        return loss

    def test_step(self, batch: dict, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Testing loop"""
        loss = self.model(**batch).loss
        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss, logger=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.AdamW(self.parameters(), **self.optim_params)

        def lr_scaler(global_step,
                lr_max:float = 1e-5,
                lr_min:float = 1e-6,
                lr_start:float = 1e-10, 
                exp_k : float = 0.0005,
                warm_up_steps:int=-1, 
                flat_steps : int =-1):
            """Returns scalar factor which is multiplied to the learning rate.
            Set warm_up_steps to -1, to remove warmup.
            Set exp_k to -1, to remove lr exponential decay.
            
            - warm-up phase: learning rate linearly increase from lr_start to lr
            - evenly between warm_up_steps.
            - The lr is then constant for flat_steps, after which it starts to decay.

            Args:
                global_step (_type_): Current step in the training loop.
                lr_max (float, optional): max learning rate. Read from training config. 
                lr_min (float, optional): min learning rate. 
                Defaults to self.hparams_config['learning_rate'].
                lr_start (float, optional): Starting learning rate. Defaults to 1e-10.
                exp_k (float, optional): Factor deciding decay rate of lr. Defaults to 0.001
                warm_up_steps (int, optional): Number of warm up steps. Defaults to 500.
                flat_steps (int, optional): Number of flat steps before lr decay. Defaults to 0.

            Returns:
                _type_: _description_
            """
            if global_step < warm_up_steps:
                if  warm_up_steps == -1:
                    lr_scale = 1
                else:    
                    # linear warm up
                    slope = (lr_max-lr_start)/warm_up_steps
                    y_n = lr_start + global_step*slope
                    lr_scale = y_n/lr_max
            
            elif global_step < warm_up_steps + flat_steps:
                lr_scale = 1
            else:
                if exp_k == -1:
                    lr_scale = 1
                else :
                    # exponential decay
                    # decay_factor:float = 0.9990,  
                    # lr_scale = decay_factor ** (global_step-warm_up_steps)
                    lr_scale = (lr_min/lr_max)+ math.exp(-exp_k*global_step)
            return lr_scale


        # lr_warmup = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer,
        #                                               start_factor=0.1,
        #                                               end_factor=1.0,
        #                                               total_iters=1000, # steps
        #                                               last_epoch=-1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer,
                                                         lr_lambda=lr_scaler)
        optim_config = {
            "optimizer" : optimizer,
            "lr_scheduler": {
                            # REQUIRED: The scheduler instance
                            "scheduler": lr_scheduler,
                            # The unit of the scheduler's step size, could also be 'step'.
                            # 'epoch' updates the scheduler on epoch end whereas 'step'
                            # updates it after a optimizer update.
                            "interval": "step",
                            # How many epochs/steps should pass between calls to
                            # `scheduler.step()`. 1 corresponds to updating the learning
                            # rate after every epoch/step.
                            "frequency": 1,
                            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
                            # "monitor": "val_loss",
                            # If set to `True`, will enforce that the value specified 'monitor'
                            # is available when the scheduler is updated, thus stopping
                            # training if not found. If set to `False`, it will only produce a warning
                            # "strict": True,
                            # If using the `LearningRateMonitor` callback to monitor the
                            # learning rate progress, this keyword can be used to specify
                            # a custom logged name
                            # "name": None,
                        }
        }
        return optim_config

    def on_before_batch_transfer(self, batch: dict, dataloader_idx:int) -> dict:
        """Operations to perform before transferring batch to device

        Parameters
        ----------
        batch : dict
        dataloader_idx : int

        Returns
        -------
        dict
            Preprocessed batch
        """
        return self._preprocess_batch(batch)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        trainer_args: Optional[dict] = None,
        ) -> Optional[str]:
        """Method for running the optimizaition of the model.

        Args:
            train_loader (DataLoader): collection of training samples
            val_loader (DataLoader): colleciton of validation samples
            trainer_init_args (Optional[dict], optional): Defaults to None.
            trainer_train_args (Optional[dict], optional): Defaults to None.

        Returns:
            Optional[str]: _description_
        """
        # From pytorch lightning, get a trainer
        trainer = self.get_trainer(trainer_args)

        trainer.fit(  # type:ignore
            self,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
        )

        # Get the best checkpoint path
        callback = trainer.checkpoint_callback
        if isinstance(callback, pl.callbacks.ModelCheckpoint):
            return callback.best_model_path
        return None
    
    
    def get_trainer(self, 
                    trainer_args: dict[Any, Any], 
                    device_stats:bool = False,
                    every_n_epochs:int = 10) -> pl.Trainer:
        """Returns a PyTorchLightning trainer constructed from the given args

        Parameters
        ----------
        trainer_args : Arguments for constructing the trainer, by default None
        device_stats : Record device stats,
        every_n_epochs (int) : checkpoint after every N epoch. None to switch off
     

        Returns
        -------
        pl.Trainer
            The trainer instance
        """
        fname = f"{self.hparams_config['run_name']}-{self.hparams_config['model_config_key']}"
        # Prepare callbacks for checkpointing
        checkpoint_topk_callback = pl.callbacks.ModelCheckpoint(
            filename= fname+"-topk-{epoch}-{step}-{val_loss:.6f}",
            monitor = "val_loss",
            save_top_k = 5,
        )
        callbacks = [checkpoint_topk_callback]
        if every_n_epochs is not None:
            checkpoint_every_epoch_callback = pl.callbacks.ModelCheckpoint(
                filename = fname + "-{epoch}-{step}-{val_loss:.6f}",
                monitor = "val_loss",
                every_n_epochs=50
            )
            callbacks.append(checkpoint_every_epoch_callback)

        # Prepare simple profiler
        simple_profiler = pl.profilers.SimpleProfiler(
            # dirpath = self.hparams_config["default_root_dir"],
            filename=self.hparams_config['run_name']+'_profiler', 
            extended=True)
        
        if device_stats:
            device_stats = pl.callbacks.DeviceStatsMonitor()
            callbacks.append(device_stats)
        
        # Learning Rate Monitor
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval = 'step')
        callbacks.append(lr_monitor)

        # Early Stopping
        # early_stopping = pl.callbacks.EarlyStopping(
        #     monitor = "val_loss",
        #     min_delta = 0.05,
        #     patience = 10, # number of checks on val_loss 
        #     strict = True, # crash training if monitor metric not found
        # )

        # Prepare device parameters
        if "accelerator" not in trainer_args.keys():
            trainer_args["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"
        trainer_args["devices"] = (
            trainer_args.get("devices", torch.cuda.device_count()) if torch.cuda.is_available() else 1
        )

        return pl.Trainer(**trainer_args, 
                          callbacks=callbacks,
                          profiler=simple_profiler)    
  
