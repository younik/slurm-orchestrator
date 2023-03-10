import pathlib
from typing import List
import subprocess
import tempfile
import os


def serialize_main_args(config: dict) -> str:
    args = config.items()

    # delete --key=False
    args = filter(lambda k_v: not isinstance(k_v[1], bool) or k_v[1], args)

    args = filter(lambda k_v: k_v[1] is not None, args)
    return ' '.join(f'--{key}={value}' for key, value in args)


def _get_slurm_args(config: dict) -> List[str]:
    name = config.get("name", "unnamed")
    stdout_path = config["stdout_path"]
    args = [
        f"--job-name={name}",
        f"--output={stdout_path}/{name}_%j.out"
    ]

    if "nodes" in config.keys():
        nodes = config["nodes"]
        args.append(f"--nodes={nodes}")
    if "ntasks_per_node" in config.keys():
        ntasks_per_node = config["ntasks_per_node"]
        args.append(f"--ntasks={ntasks_per_node}")
    if "cpus" in config.keys():
        cpus = config["cpus"]
        args.append(f"--cpus-per-task={cpus}")
    if "mem_per_cpu" in config.keys():
        mem_per_cpu = config["mem_per_cpu"]
        args.append(f"--mem-per-cpu={mem_per_cpu}")
    if "gpus_per_node" in config.keys():
        gpus_per_node = config["gpus_per_node"]
        args.append(f"--gpus-per-node={gpus_per_node}")
    if "timeout" in config.keys():
        timeout = config["timeout"]
        args.append(f"--time={timeout}:00:00")

    return args


def _sbatch_template(config: dict) -> str:
    slurm_args = _get_slurm_args(config)

    batch_script = "#!/bin/bash"
    batch_script += "\n".join(slurm_args)
    batch_script += " \n srun python main.py"
    batch_script += f" {serialize_main_args(config)} --jobid=$SLURM_JOBID"

    return batch_script

def sbatch_launch(config: dict, n_launches: int = 1):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".sbatch", delete=False)
    f.write(_sbatch_template(config))
    f.close()
    for _ in range(n_launches):
        subprocess.call(["sbatch", f.name])
    os.unlink(f.name)


def srun_launch(config: dict):
    slurm_args = _get_slurm_args(config)
    slurm_args = filter(lambda arg: not arg.startswith("--output"), slurm_args)
    slurm_args = list(slurm_args)
    path = pathlib.Path(__file__).parent
    slurm_args.append(f"python {path}/main.py {serialize_main_args(config)}")
    subprocess.call(' '.join(["srun", *slurm_args]), shell=True)
