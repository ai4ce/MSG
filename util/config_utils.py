# config loading helper functions
import yaml
import os
from pathlib import Path
from typing import Dict, Any
import argparse
from datetime import datetime

def load_yaml(file_path: Path) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
    
def load_configs(config_dir: Path) -> Dict[str, Any]:
    config = {}
    for file in config_dir.glob('*.yaml'):
        config.update(load_yaml(file))
    return config

def merge_configs(default_config: Dict[str, Any], custom_config: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in custom_config.items():
        if isinstance(value, dict) and key in default_config:
            merge_configs(default_config[key], value)
        else:
            default_config[key] = value
    return default_config

def override_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    args_dict = vars(args)
    for key in args_dict:
        if args_dict[key] is not None:
            config[key] = args_dict[key]
    return config

def create_experiment_subdir(base_output_dir):
    """Create a unique subdirectory for each experiment run."""
    # Use current date and time as a unique identifier
    run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    subdir = os.path.join(base_output_dir, run_id)
    os.makedirs(subdir, exist_ok=True)
    return subdir, run_id

def get_configs(base_config_dir: str, args: argparse.Namespace, creat_subdir: bool) -> Dict[str, Any]:
    """
    Load configs and update with customized configs (from yml and args) if needed
    If creat_subdir == True: create subdir in the output_dir and update the value
    By default, create_subdir is set to False during evaluation and True during training.
    """
    base_config = load_configs(Path(base_config_dir))
    if args.experiment:
        experiment_config_path = Path(f'configs/experiments/{args.experiment}.yaml')
        experiment_config = load_yaml(experiment_config_path)
        final_config = merge_configs(base_config, experiment_config)
    else:
        # print("simply use base")
        final_config = base_config

    final_config = override_with_args(final_config, args)
    if creat_subdir:
        unique_output_dir, run_id = create_experiment_subdir(final_config["output_dir"])
        final_config["output_dir"] = unique_output_dir
        final_config["run_id"] = run_id
    else:
        final_config["run_id"] = None
    # double check the dir
    os.makedirs(final_config["output_dir"], exist_ok=True)
    return final_config