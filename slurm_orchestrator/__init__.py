import json
import logging
import pathlib
from typing import Dict, Union
from .slurm_utils import srun_launch, sbatch_launch


__all__ = ["launch"]


def launch(config: Union[Dict, str, pathlib.Path], map_config: Dict[str, Dict] = {}):
    if not isinstance(config, Dict):
        with open(config) as file:
            config = json.load(file)
    
    for param_name, map_ in map_config.items():
        value = config[param_name]
        config.update(map_[value])
        config.pop(param_name)

    _make_paths(config)

    n_launches = config.pop("n_launches", 1)
    interactive = config.get("interactive", False)
    if interactive and n_launches != 1:
        logging.warning("Interactive mode: ignoring n_launches argument.")

    if interactive:
        srun_launch(config)
    else:
        sbatch_launch(config, n_launches=n_launches)


def _make_paths(config):
    path = pathlib.Path.cwd()
    config["path"] = path

    out_path = path.joinpath("outs", config.get("group", ""))
    out_path.mkdir(parents=True, exist_ok=True)
    config["out_path"] = out_path
    
    stdout_path = out_path.joinpath("stdout")
    stdout_path.mkdir(exist_ok=True)
    config["stdout_path"] = stdout_path

    model_path = out_path.joinpath("models")
    model_path.mkdir(exist_ok=True)
    config["model_path"] = model_path

    log_path = out_path.joinpath("logs")
    log_path.mkdir(exist_ok=True)
    config["log_path"] = log_path

    profile_path = out_path.joinpath("profile")
    profile_path.mkdir(exist_ok=True)
    config["profile_path"] = profile_path
