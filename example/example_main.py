from absl import flags
import gym
from stable_baselines3 import PPO


flags.DEFINE_string("env_name", "CartPole-v1", "Name of gym environment")
flags.DEFINE_integer("total_timesteps", 10_000, "Total number of timesteps")
flags.DEFINE_integer("n_steps", 2048, "Number of steps per update")
flags.DEFINE_integer("batch_size", 64, "Minibatch size")
flags.DEFINE_float("gamma", 0.99, "PPO's discount factor")
flags.DEFINE_float("learning_rate", 3e-4, "Learning rate") 
flags.DEFINE_float("gae_lambda", 0.95, "PPO's lambda for GAE")
flags.DEFINE_integer("n_epochs", 10, "Number of epoch for the surrogate loss")

def main(config):
    env = gym.make(config.env_name)
    model = PPO("MlpPolicy", env,
        verbose=1,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        learning_rate=config.learning_rate,
        gae_lambda=config.gae_lambda,
        n_epochs=config.n_epochs
    )
    model.learn(total_timesteps=config.total_timesteps)