import gym
from collect_mineral_shard_env.envs import Collect_Mineral_Shard_Env 
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from absl import flags

FLAGS = flags.FLAGS
FLAGS([''])

# create vectorized environment
# env = gym.make('collect_mineral_shards-v0')
# env = Collect_Mineral_Shard_Env()
env = DummyVecEnv([lambda: Collect_Mineral_Shard_Env()])

# use ppo2 to learn and save the model when finished
model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="log/")
model.learn(total_timesteps=int(1e5), tb_log_name="first_run", reset_num_timesteps=False)
model.save("model/collect_mineral_shard")