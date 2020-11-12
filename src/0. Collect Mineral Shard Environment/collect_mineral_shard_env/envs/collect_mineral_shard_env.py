# Note: Modeled after https://github.com/fyr91/sc2env

import gym
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from gym import spaces
import numpy as np
import datetime

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from absl import flags


class CollectMineralShardEnv(gym.Env):
    """
    Todo: add docstring
    """

    metadata = {'render.modes': ['human']}
    default_settings = {
        'map_name': "CollectMineralShards",
        'players': [sc2_env.Agent(sc2_env.Race.terran)],
        'agent_interface_format': features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64),
        'realtime': False
    }

    def __init__(self,
                 MAXIMUM_NUMBER_OF_MARINES = 2,
                 MAXIMUM_NUMBER_OF_SHARDS = 20,
                 efficiency_incentive = False,
                 MINERAL_COLLECTION_CAP = 1000,
                 **kwargs):
        super().__init__()
        self.kwargs                     = kwargs
        self.env                        = None
        self.marines                    = []
        self.minerals                   = []
        self.number_of_marines          = 0
        self.number_of_minerals         = 0
        self.MAXIMUM_NUMBER_OF_MARINES  = MAXIMUM_NUMBER_OF_MARINES
        self.MAXIMUM_NUMBER_OF_SHARDS   = MAXIMUM_NUMBER_OF_SHARDS
        self.MINERAL_COLLECTION_CAP     = MINERAL_COLLECTION_CAP
        self.observation_shape          = (self.MAXIMUM_NUMBER_OF_MARINES + self.MAXIMUM_NUMBER_OF_SHARDS, 2) # (22,2)
        self.episode_start_time         = datetime.datetime.now().timestamp()
        self.efficiency_incentive       = efficiency_incentive

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
        self.marines            = []
        self.minerals           = []
        self.number_of_marines  = 0
        self.number_of_minerals = 0
        self.episode_start_time = datetime.datetime.now().timestamp()

        raw_obs = self.env.reset()[0]
        return self.get_derived_obs(raw_obs)

    def init_env(self):
        """
        Initializes the starcraft environment using the parameters specified in kwargs
        :param  None
        :output None
        """
        args = {**self.default_settings, **self.kwargs}
        self.env = sc2_env.SC2Env(**args)

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

    def step(self, action):
        """
        Step is called every time the agent decides on an action
        :param  action          action the agent wishes to take
        :output observation     prepared observation vector resulting from the selected action
        :output reward          reward earned by the agent for taking the action
        :output done            True/False reflecting whether an episode is finished
        :output information     additional information
        """
        raw_obs = self.take_action(action)
        obs = self.get_derived_obs(raw_obs)

        # Set the rewards equal to the amount of mineral collected at the end of an episode!
        # Todo: Does episode mean every action? If not, do we want to include a reward scheme, that just rewards every
        #  mineral collected?
        if raw_obs.last():
            
            # Determine if we are applying an efficiency incentive in the current experiment
            if self.efficiency_incentive:
                
                # Calculates the aount of seconds passed since the episode began
                current_time = datetime.datetime.now().timestamp()
                efficiency_incentive = current_time - self.episode_start_time

            else:
                # If efficiency incentive is not in play, ignore this component of the reward
                efficiency_incentive = 0

            # Each mineral collected contributes 100 to the mineral collection.
            # Scale it down by 100 such that the time component has a larger influence
            minerals_collected = raw_obs.observation.player.minerals * (1.0/100.0)

            # Reward received by the agent contingent on the amount of mineral collected
            # Plus a possible efficiency incentive contingent on the amount of time incurred.
            reward = minerals_collected - efficiency_incentive
            # Todo: Could this result in negative reward values? Do we allow this?

            print(f'reward:  {reward}')

        else:
            reward = 0

        # TODO: i.e. explore investigating incentive structure by setting done to true at 1,000 minerals.

        return obs, reward, raw_obs.last(), {}

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

    # create vectorized environment
    env = DummyVecEnv([lambda: CollectMineralShardEnv()])

    # use ppo2 to learn and save the model when finished
    model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="log/")
    model.learn(total_timesteps=int(1e5), tb_log_name="first_run", reset_num_timesteps=False)
    model.save("model/collect_mineral_shard")