# from cloudpickle.cloudpickle import parametrized_type_hint_getinitargs
from collect_mineral_shard_env.envs import Collect_Mineral_Shard_Env 
from agents.random_agent import RandomAgent
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, A2C, DQN
from stable_baselines.deepq.policies import MlpPolicy as DQN_MlpPolicy
from stable_baselines.sac.policies import MlpPolicy as SAC_MlpPolicy
import argparse
from absl import flags
import os
from os import path
import numpy as np

def load_model(environment, parameters):
    """
    Loads the reinforcement learning algorithm
    :param  algorithm   the reinforcement learning algorithm to run
    :param  weights     path to the trained model weights
    :output model       the agent model
    """
    
    # Place holder for the model
    agent = None
    
    # Filename to save training logs
    log_name = parameters['experiment_id'] + '_' + parameters['log_name']

    # Load the agent
    if parameters['algorithm'] == 'PPO':
        agent = PPO2(MlpPolicy, environment, verbose=False, tensorboard_log=f"{parameters['log_dir']}/{parameters['algorithm']}")
    elif parameters['algorithm'] == 'A2C':
        agent = A2C(MlpPolicy, environment, verbose=False, tensorboard_log=f"{parameters['log_dir']}/{parameters['algorithm']}")
    elif parameters['algorithm'] == 'DQN':
        agent = DQN(DQN_MlpPolicy, env=environment, verbose=False, tensorboard_log=f"{parameters['log_dir']}/{parameters['algorithm']}")
    elif parameters['algorithm'] == 'RANDOM':
        agent = RandomAgent(environment)
        return agent
    else:
        assert("Algorithm does not exist!")
    
    # Directory to the model results
    model_dir = f"models/{parameters['algorithm']}"

    # Define path to trained weights, if it exists
    weights_dir = f"models/{parameters['algorithm']}/weights_experiment_{parameters['experiment_id']}.zip"

    # Weights found -- Load the trained model
    if os.path.exists(weights_dir):
        print(f'Loading weights!')
        
        # Load the weights into the agent
        agent.load(weights_dir)

        if parameters['continue_training']:
        
            print(f"Continuing training!")

            # Train the agent further!
            agent.learn(total_timesteps=int(parameters['timesteps']), tb_log_name=log_name, reset_num_timesteps=False)

            print(f"Training completed!")

            # Save the weights!
            agent.save(weights_dir)

            print(f"Updated weights saved!")


    # No weights found -- Train the model!
    else:
        print('Training the model!')
        
        # Train the agent!
        agent.learn(total_timesteps=int(parameters['timesteps']), tb_log_name=log_name, reset_num_timesteps=False)

        # Make sure the weight directory exists, or else saving will fail!
        if not path.exists(model_dir):
            os.makedirs(model_dir)

        # Save the weights!
        agent.save(weights_dir)

        print("Model trained and weights saved!")

    return agent


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS([''])

    base_dir='/Volumes/GoogleDrive/My Drive/Education/Eidgenössische Technische Hochschule Zürich/Complex Social Systems/Project/Complex-Social-Systems/src'

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', type=str, help='id of experiment', default='000')
    parser.add_argument('--total_episodes', type=int, help='number of episodes to run for', default='10')
    parser.add_argument('--apply_incentive', dest='apply_incentive', help='include incentive structure in the rewards', action='store_true')
    parser.add_argument('--continue-training', dest='continue_training', help='to continue training using weights already learned', action='store_true')
    parser.add_argument('--episodic_rewards', dest='episodic_rewards', help='return rewards only at the end of an episode', action='store_true')
    parser.add_argument('--mineral_thresholding', dest='mineral_thresholding', help='stop incurring penalites after a certain mineral collection threshold is reached', action='store_true')
    parser.add_argument('--weights_dir', type=str, help='location of trained model')
    parser.add_argument('--algorithm', type=str, help='(RANDOM,PPO,A2C,DQN)', default='PPO')
    parser.add_argument('--log_dir', type=str, help='name of the log directory', default='logs/')
    parser.add_argument('--log_name', type=str, help='name of the log', default='logs')
    parser.add_argument('--timesteps', type=int, help='number of timesteps to train for', default=1e6) 

    # Convert to a dictionary 
    parameters = vars(parser.parse_args())

    print(f"Beginning the experiment: {parameters['experiment_id']}")
    print(f'Using parameters: {parameters}')

    # create vectorized PySC2 mineral collection environment
    env = DummyVecEnv([lambda: Collect_Mineral_Shard_Env(   efficiency_incentive    = parameters['apply_incentive'],
                                                            mineral_thresholding    = parameters['mineral_thresholding'],
                                                            episodic_rewards        = parameters['episodic_rewards']
                                                            )])

    # Load the agent
    agent = load_model(environment=env, parameters=parameters)

    # Store the rewards of the episode
    episode_rewards = []

    # Step rewards
    step_rewards = []

    # Repeat for the total number of episodes in order to capture the amount of variation
    for _ in range(parameters['total_episodes']):

        # Inititalize the reward for this episode
        episode_reward = 0.0

        # Collect the cumulative episode reward
        cumulative_reward = 0.0

        # Collect the cumulative reward at every step
        step_reward = []
                
        # obtain the initial observation vector
        obs = env.reset()

        # Capture whether the episode is finished or not
        done = False

        # Take actions!
        while not done:
            
            # Decide on an action
            action, _states = agent.predict(obs)
            
            # Take the decided action
            obs, reward, done, info = env.step(action)

            # Update the reward for this episode
            if done:
                episode_reward = info[0]['minerals_collected']

            # Update the cumulative minerals collected
            cumulative_reward += reward[0]

            # Store in the step rewards
            step_reward.append(cumulative_reward)

        # Store the reward of the next episode
        episode_rewards.append(episode_reward)

        # Store the step rewards for the entire episode
        step_rewards.append(step_reward)
        
        # Display rewards of this episode
        print(f'Episode reward: {episode_reward}')

    print("Experiment Completed.")
    print(f'Results: {episode_rewards}')

    print(step_rewards)

    np.savetxt( f'models/{parameters["algorithm"]}/experiment_{parameters["experiment_id"]}_{parameters["algorithm"]}_results.csv', \
                episode_rewards, \
                delimiter="," \
                )

np.savetxt( f'models/{parameters["algorithm"]}/experiment_{parameters["experiment_id"]}_{parameters["algorithm"]}_step_results.csv', \
                step_rewards, \
                delimiter="," \
                )