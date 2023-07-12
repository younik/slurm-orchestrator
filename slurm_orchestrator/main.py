import argparse
from absl import app, flags
import importlib
import random
import sys
import traceback
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--main', type=str, help='File name to launch (without .py)')
parser.add_argument('--path', type=str, help='Path used for this run')

config = flags.FLAGS
flags.DEFINE_string('main', None, 'Launch file (without .py)', required=True)
flags.DEFINE_string('name', 'unnamed', 'Name of the launch')
flags.DEFINE_string('group', None, 'Group name of the launch')
flags.DEFINE_string('project', None, 'Project name', required=True)
flags.DEFINE_integer('jobid', None, 'Slurm job id')
flags.DEFINE_string('path', None, 'Path of the launch folder', required=True)
flags.DEFINE_string('out_path', None, 'Path for run outputs')
flags.DEFINE_string('stdout_path', None, 'Path for stdout file')
flags.DEFINE_string('fig_path', None, 'Path for saving figures')
flags.DEFINE_string('model_path', None, 'Path for saving models')
flags.DEFINE_string('log_path', None, 'Path for saving logs')
flags.DEFINE_string('profile_path', None, 'Path for saving profile traces')
flags.DEFINE_bool('disable_wandb', False, 'Disable WandB')
flags.DEFINE_string('sweep_id', None, 'Wandb sweep id of the launch')
flags.DEFINE_bool('profile', False, 'Profile code')
flags.DEFINE_bool('interactive', False, 'Launch in interactive mode')
flags.DEFINE_integer('seed', None, 'Seed of the run')
flags.DEFINE_string('unique_name', None, 'Unique name (you do not need to set this)')


def _import_module(path: str):
    spec = importlib.util.spec_from_file_location("launch_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["launch_module"] = module
    spec.loader.exec_module(module)
    return module

def _launch_module(_):
    def launch():
        if config.seed is None:
            config.seed = random.randint(0, 2 ** 32 - 1)
        if config.unique_name is None:
            config.unique_name = f"{config.name}_{config.jobid}"

        run = None
        if not config.disable_wandb:
            wandb.login()

            run = wandb.init(
                project=config.project,
                name=config.name,
                group=config.group,
                config=config.flag_values_dict(),
                settings=wandb.Settings(start_method='fork'),
                sync_tensorboard=True,
                save_code=True,
            )

            vars(config).update(wandb.config)

        if config.profile:
            import jax
            jax.profiler.start_trace(
                f"{config.profile_path}/{config.unique_name}",
                create_perfetto_trace=True
            )

        module = _import_module(f"{config.path}/{config.main}.py")
        try:
            module.main(config)
        except Exception as e:  # noqa
            print(traceback.print_exc(), file=sys.stderr)
        finally:
            if config.profile:
                jax.profiler.stop_trace()
            if run is not None:
                run.finish()

    if not config.disable_wandb and config.sweep_id is not None:
        wandb.agent(config.sweep_id, launch, config.project)
    else:
        launch()


if __name__ == "__main__":
    parsed, _ = parser.parse_known_args()
    sys.path.append(parsed.path)  # needed for relative import in main
    _import_module(f"{parsed.path}/{parsed.main}.py")
    app.run(_launch_module)