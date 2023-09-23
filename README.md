# Slurm Orchestrator

Slurm Orchestrator is a Python library designed to simplify the process of running Machine Learning jobs on a Slurm cluster while neatly tracking them using the Weights and Biases (WandB) platform. This README provides an overview of how to use Slurm Orchestrator to streamline your machine learning experiments and effectively manage them on a Slurm cluster.

## Table of Contents

1. [Installation](#Installation)
2. [Getting Started](#Getting-Started)
3. [Usage](#Usage)
4. [Flags](#Flags)
5. [Contributing](#Contributing)
6. [Disclaimer](#Disclaimer)


## Installation

To get started with Slurm Orchestrator, you can install it using pip:

```bash
pip install git+https://github.com/younik/slurm-orchestrator.git
```

## Getting Started

Before using Slurm Orchestrator, you should have access to a Slurm cluster and have the Weights and Biases (WandB) library installed. Make sure you have your Slurm environment properly configured, with `WANDB_API_KEY` properly set.

## Usage

To use Slurm Orchestrator, follow these steps:

Create a Python launcher file (e.g., launcher.py) for your machine learning job. Inside this file, import the slurm_orchestrator module and use the launch function to specify the configuration of your job. Here's an example:


```python
import slurm_orchestrator

if __name__ == "__main__":
    slurm_orchestrator.launch({
        "main": "example_main",
        "name": "example",
        "project": "example",
        "interactive": True,
        "env_name": "Taxi-v3"
    })
```

In your launcher file, specify the main key with the name of the Python file (without the ".py" extension) that contains your machine learning job's main method. This file should accept a configuration parameter.
You can also define other configuration flags using the absl library. Slurm Orchestrator supports various flags to customize your job execution (see Flags section for details).
Execute your launcher script on the Slurm cluster to launch your machine learning job.

## Flags

Here is a list of supported flags that you can add to the launch dictionary:

 - **main**: Name of the Python file (without ".py" extension) containing the main method of your machine learning job (required).
 - **name**: Name of the launch (default: "unnamed").
 - **group**: Group name of the launch.
 - **project**: Project name for tracking the job (required).
 - **disable_wandb**: Disable Weights and Biases (WandB) integration (default: False).
 - **sweep_id**: Wandb sweep ID for the launch.
 - **profile**: Profile code execution (default: False).
 - **interactive**: Launch in interactive mode (default: False).
 - **seed**: Seed for the run.
 - **path**: Path containing the main file (default: current shell directory).


Moreover the standard Slurm flags are available:

 - **nodes**: Number of Slurm nodes.
 - **ntasks_per_node**: Number of tasks per node.
 - **cpus**: Number of CPUs per node.
 - **mem_per_cpu**: Amount of memory per CPU.
 - **gpus_per_node**: Number of GPUs per node.
 - **timeout**: Timeout for the Slurm job.

## Contributing

If you would like to contribute to Slurm Orchestrator or report issues, you are welcome to open an issue or a PR.

## Disclaimer

This project was done for personal purpose while running RL experiments on ETH Zurich cluster. This README file is written by an LLM.

