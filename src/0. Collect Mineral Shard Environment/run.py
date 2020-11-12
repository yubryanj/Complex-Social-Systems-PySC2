
from collect_mineral_shard_env.envs import Collect_Mineral_Shard_Env 
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from absl import flags


if __name__ == "__main__":

    FLAGS = flags.FLAGS
    FLAGS([''])

    efficiency_incentive = True
    log_name = "efficiency_incentive"

    # create vectorized PySC2 mineral collection environment
    env = DummyVecEnv([lambda: Collect_Mineral_Shard_Env(efficiency_incentive= efficiency_incentive)])

    # use ppo2 to learn and save the model when finished
    model = PPO2(MlpPolicy, env, verbose=False, tensorboard_log="log/")

    # Train them model 
    model.learn(total_timesteps=int(1e5), tb_log_name=log_name, reset_num_timesteps=False)

    # Save the model!
    model.save(f"model/collect_mineral_shard_{log_name}")