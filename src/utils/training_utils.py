"""Collection of utils for training models."""
import torch


def validate_config_file(configs:dict, filename:str) -> dict:
    """This function :
    - verifies that all keys required from a config file are present 
    - adds default where necessary
    """
    # Parameters
    # ----------
    # training_data_path : str
    #     Location of the data directory
    # batch_size : int
    #     Batch size to use for training
    # max_epochs : int
    #     Number of epochs to train for
    # learning_rate : float
    #     Learning rate to use
    # gpu_devices : int
    #     Number of devices to train on
    # num_workers : int
    #     Number of workers available to dataloaders
    # log_every_n_steps : int
    #     Steps after which to log
    # output_dir : str
    #     Location where output directory will be saved
    # model_config_key : str
    #     key specifying model configuration (see model_config.json)
    # fast_dev_run : bool
    #     runs 5 batch of training, validation, test and prediction data 
    #       through your trainer to see if there are any bugs:
    # limit_train_batches, limit_val_batches: shorten an epoch
    keys_to_check = ["run_name",
                     'task',
                     "training_data_path",
                     "append_special_tokens",
                     "batch_size",
                     "max_epochs",
                     "learning_rate",
                     "gpu_devices",
                     "num_workers",
                     "log_every_n_steps",
                     "model_config_key",
                     "training_data_ratio",]
    keys_with_defaults = ["fast_dev_run"]
    for k in keys_to_check:
        if k not in configs.keys():
            if k in keys_with_defaults:
                configs[k] = False
            else:
                raise KeyError(f"Key '{k}' missing from training config file {filename}.")

    return configs

def get_gpu_details(print_details:bool = True):

    device_info = {}
   
    if torch.cuda.is_available():
        ndevices = torch.cuda.device_count()
        for i in range(ndevices):
            dev_memory_info = torch.cuda.mem_get_info(i)
            device_info[i] = {}
            device_info[i]['device_name'] = torch.cuda.get_device_name(i)
            device_info[i]['device_free_memory_bytes'] = dev_memory_info[0]
            device_info[i]['device_total_memory_bytes'] = dev_memory_info[1]
    else:
        print("No CUDA enabled device found!")
    
    if print_details == True:
            for k,v in device_info.items():
                print(f"For GPU device: {k}")
                name = device_info[k]['device_name']
                total = device_info[k]['device_total_memory_bytes']
                free = device_info[k]['device_free_memory_bytes']
                just_space = 50
                print(f"{'Device Name': <{just_space}} : {name}")
                print(F"{'Total Memory (GB)': <{just_space}} : {total/1e9}")
                print(F"{' Free Memory (GB)': <{just_space}} : {free/1e9}")
                print(F"{' Free Memory (%)': <{just_space}} : {100*free/total}")
    return device_info