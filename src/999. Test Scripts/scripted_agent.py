# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Scripted agents."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from absl import app

from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS

class CollectMineralShardsFeatureUnits(base_agent.BaseAgent):
  """An agent for solving the CollectMineralShards map with feature units.

  Controls the two marines independently:
  - select marine
  - move to nearest mineral shard that wasn't the previous target
  - swap marine and repeat
  """

  def setup(self, obs_spec, action_spec):
    super(CollectMineralShardsFeatureUnits, self).setup(obs_spec, action_spec)

    if "feature_units" not in obs_spec[0]:
      raise Exception("This agent requires the feature_units observation.")

  def reset(self):
    super(CollectMineralShardsFeatureUnits, self).reset()
    self._marine_selected = False
    self._previous_mineral_xy = [-1, -1]

  def step(self, obs):
    super(CollectMineralShardsFeatureUnits, self).step(obs)

    # Get a list of marines
    marines = [unit for unit in obs.observation.feature_units
               if unit.alliance == _PLAYER_SELF]

    # if no marines are available, return nothing
    if not marines:
      return FUNCTIONS.no_op()

    # Select one marine
    marine_unit = next((m for m in marines
                        if m.is_selected == self._marine_selected), marines[0])

    # Get the position of the marine
    marine_xy = [marine_unit.x, marine_unit.y]

    # Nothing selected or the wrong marine is selected.
    if not marine_unit.is_selected:

      # Enable the marine selected flag 
      self._marine_selected = True

      # select the marine
      return FUNCTIONS.select_point("select", marine_xy)



    if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
      
      # Find and move to the nearest mineral.
      minerals = [[unit.x, unit.y] for unit in obs.observation.feature_units
                  if unit.alliance == _PLAYER_NEUTRAL]

      if self._previous_mineral_xy in minerals:
        # Don't go for the same mineral shard as other marine.
        minerals.remove(self._previous_mineral_xy)

      if minerals:
        # Find the closest mineral.
        distances = numpy.linalg.norm(numpy.array(minerals) - numpy.array(marine_xy), axis=1)
        closest_mineral_xy = minerals[numpy.argmin(distances)]

        # Swap to the other marine.
        self._marine_selected = False
        self._previous_mineral_xy = closest_mineral_xy
        return FUNCTIONS.Move_screen("now", closest_mineral_xy)

    return FUNCTIONS.no_op()



def main(unused_argv):
    agent = CollectMineralShardsFeatureUnits()
    try:
        while True:
            with sc2_env.SC2Env(
                map_name ="CollectMineralShards",
                players=[sc2_env.Agent(sc2_env.Race.terran)],
                agent_interface_format = features.AgentInterfaceFormat(
                    feature_dimensions = features.Dimensions(screen=84, minimap=64),
                    use_feature_units = True
                ),
                step_mul = 16,
                game_steps_per_episode = 0,
                visualize=True
            ) as env:

                agent.setup(env.observation_spec(), env.action_spec())
                timesteps = env.reset()
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)


    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)