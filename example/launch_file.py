import slurm_orchestrator


if __name__ == "__main__":
    slurm_orchestrator.launch({
        "main": "example_main",
        "name": "example",
        "project": "example",
        "interactive": True,
        "env_name":"Taxi-v3"
    })