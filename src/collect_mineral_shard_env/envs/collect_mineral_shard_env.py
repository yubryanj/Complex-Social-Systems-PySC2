# Note: Modeled after https://github.com/fyr91/sc2env

import gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from gym import spaces
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from absl import flags

class Collect_Mineral_Shard_Env(gym.Env):

    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name'  : "CollectMineralShards",
        'players'   : [sc2_env.Agent(sc2_env.Race.terran)],
        'agent_interface_format': features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                    ),
        'realtime'  : False,
        'visualize' : False,
    }


    def __init__(self,  MAXIMUM_NUMBER_OF_MARINES   = 2,
                        MAXIMUM_NUMBER_OF_SHARDS    = 20,
                        efficiency_incentive        = False,
                        episodic_rewards            = False,
                        mineral_thresholding        = False,
                        MINERAL_COLLECTION_CAP      = 1000,
                        STEP_COST                   = 20,
                        **kwargs):
        super().__init__()
        self.kwargs                     = kwargs
        self.env                        = None
        self.marines                    = []
        self.number_of_marines          = 0
        self.number_of_minerals         = 0
        self.steps_taken                = 0
        self.last_mineral_position      = 0
        self.MAXIMUM_NUMBER_OF_MARINES  = MAXIMUM_NUMBER_OF_MARINES
        self.MAXIMUM_NUMBER_OF_SHARDS   = MAXIMUM_NUMBER_OF_SHARDS
        self.MINERAL_COLLECTION_CAP     = MINERAL_COLLECTION_CAP
        self.STEP_COST                  = STEP_COST
        self.observation_shape          = (self.MAXIMUM_NUMBER_OF_MARINES + self.MAXIMUM_NUMBER_OF_SHARDS, 2) # (22,2)
        self.efficiency_incentive       = efficiency_incentive
        self.episodic_rewards           = episodic_rewards
        self.mineral_thresholding       = mineral_thresholding

        # 0 no operation
        # 1 - 8 move
        self.action_space = spaces.Discrete(9)
        
        # 2 marines     [0: x, 1: y]
        # 20 minerals   [0: x, 1: y]
        self.observation_space = spaces.Box(
            low     = 0,
            high    = 64,
            shape   = self.observation_shape, # (22,2)
            dtype   = np.uint8
            )


    def reset(self):
        """
        Reset is called when    (1) first starting the environment and 
                                (2) at the end of every episode
        :param  None
        :output observations
        """
        if self.env is None:
            self.init_env()
        self.marines                = []
        self.number_of_marines      = 0
        self.number_of_minerals     = 0
        self.steps_taken            = 0 
        self.last_mineral_position  = 0


        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)


    def init_env(self):
        """
        Initializes the starcraft environment using the parameters specified in kwargs
        :param  None
        :output None
        """
        args = {**self.default_settings, **self.kwargs}
        self.env =  sc2_env.SC2Env(**args)


    def get_derived_obs(self, raw_obs):
        """
        Retrieves the observation vector to disply to the agent
        :param  raw_obs raw observation vector from the pysc2 environment
        :output obs     observation vector prepared for agent consumption
        """
        obs = np.zeros(self.observation_shape, dtype=np.uint8)
        marines = self.get_units_by_type(raw_obs, units.Terran.Marine, 1)
        minerals = [[unit.x, unit.y] for unit in raw_obs.observation.raw_units if unit.alliance == 3]

        self.marines = []
        self.minerals = []
        self.number_of_minerals = len(minerals)
        self.number_of_marines = len(marines)

        index = 0

        # Populate marine elements in the obs
        for m in marines:
            self.marines.append(m)
            obs[index] = np.array([m.x, m.y])
            index += 1

        # Populate minerals elements in the obs
        for m in minerals:
            self.minerals.append(m)
            obs[index] = np.asarray(m)
            index += 1

        return obs


    def calculate_step_reward(self, minerals_collected, is_last_move = False):
        """
        Compute the reward depending on the incentive scheme
        :param  minerals_collected  amount of minerals collected to this point in time
        :param  is_last_move        whether the episode is over
        :output reward              reward to be distributed to the agents
        """

        # Set the rewards equal to the amount of mineral collected at the end of an episode!
        if self.episodic_rewards:

            # If the episode is over
            if is_last_move:

                # Reward received by the agent contingent on the amount of mineral collected
                reward = minerals_collected

                # NOTE: Calculating incentive by steps taken may not be fruitful for incentivizing useful actions
            else:
                reward = 0
        
        # Incremental rewards for every mineral collected
        else:
            # reward is then the change in mineral position
            reward = minerals_collected - self.last_mineral_position

            # Update the tracking of current mineral position
            self.last_mineral_position = minerals_collected
        
        return reward


    def step(self, action):
        """
        Step is called every time the agent decides on an action
        :param  action          action the agent wishes to take
        :output observation     prepared observation vector resulting from the selected action
        :output reward          reward earned by the agent for taking the action
        :output done            True/False reflecting whether an episode is finished
        :output information     additional information
        """
        # Update the number of steps taken
        self.steps_taken += 1

        # Retrieve the raw observation vector resulting from taking the requested action
        raw_obs = self.take_action(action)

        # Convert the raw Pysc2 observation vector into an agent friendly format
        obs = self.get_derived_obs(raw_obs)

        # Each mineral collected contributes 100 to the mineral collection.
        minerals_collected = raw_obs.observation.player.minerals

        # Compute the reward
        reward = self.calculate_step_reward(minerals_collected, is_last_move = raw_obs.last())

        # Plus a possible efficiency incentive contingent on the amount of time incurred.
        if self.efficiency_incentive:

            # Stop incurring a reward penalty after collecting the prescribed amount of minerals
            if self.mineral_thresholding and minerals_collected > self.MINERAL_COLLECTION_CAP:
                pass
            else:
                reward -= self.STEP_COST

        info = {'minerals_collected':raw_obs.observation.player.minerals}

        return obs, reward, raw_obs.last(), info


    def take_action(self, action):
        """
        Passes the desired action to the pysc2 environment
        :param  action  action be taken
        :output raw_obs raw observation vector returned by the pysc2 environment
        """
        if action == 0:
            action_mapped = actions.RAW_FUNCTIONS.no_op()
        else:
            derived_action = np.floor((action-1)/ self.number_of_marines)
            idx = (action-1)% self.number_of_marines
            if derived_action == 0:
                action_mapped = self.move_up(idx)
            elif derived_action == 1:
                action_mapped = self.move_down(idx)
            elif derived_action == 2:
                action_mapped = self.move_left(idx)
            else:
                action_mapped = self.move_right(idx)

        raw_obs = self.env.step([action_mapped])[0]
        return raw_obs


    def move_up(self, idx):
        """
        Moves the agent up
        :param  idx index of the marine to be moved
        """
        idx = np.floor(idx)
        try:
            selected = self.marines[idx]
            new_pos = [selected.x, selected.y-2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()


    def move_down(self, idx):
        """
        Moves the agent down
        :param  idx index of the marine to be moved
        """
        try:
            selected = self.marines[idx]
            new_pos = [selected.x, selected.y+2]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()


    def move_left(self, idx):
        """
        Moves the agent left
        :param  idx index of the marine to be moved
        """
        try:
            selected = self.marines[idx]
            new_pos = [selected.x-2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()


    def move_right(self, idx):
        """
        Moves the agent right
        :param  idx index of the marine to be moved
        """
        try:
            selected = self.marines[idx]
            new_pos = [selected.x+2, selected.y]
            return actions.RAW_FUNCTIONS.Move_pt("now", selected.tag, new_pos)
        except:
            return actions.RAW_FUNCTIONS.no_op()


    def get_units_by_type(self, obs, unit_type, player_relative=0):
        """
        Retrieves the desired units
        :param  obs raw             observation vector
        :param  unit_type           the unit which to retrieve
        :param  player_relative     value of the unit in relation to the player. Acceptable values are:
                                    (NONE = 0, SELF = 1, ALLY = 2, NEUTRAL = 3, ENEMY = 4)
        """
        return [unit for unit in obs.observation.raw_units
            if unit.unit_type == unit_type
            and unit.alliance == player_relative]


    def close(self):
        """
        Closes the gym environment
        :param  none
        :output none
        """

        if self.env is not None:
            self.env.close()
        super().close()


    def render(self, mode='human'):
        """
        Generates information for the user
        :param  mode    version of rendering to generate
        :ouput  none
        """
        pass


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    FLAGS([''])

    efficiency_incentive = False

    # create vectorized environment
    env = DummyVecEnv([lambda: Collect_Mineral_Shard_Env(efficiency_incentive= efficiency_incentive)])
    log_name = "efficiency_incentive"

    # use ppo2 to learn and save the model when finished
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="log/")
    model.learn(total_timesteps=int(1e5), tb_log_name=log_name, reset_num_timesteps=False)
    model.save(f"model/collect_mineral_shard_{log_name}")