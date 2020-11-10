#!/bin/bash
# Scripts for running Collect Minearl Shards minigame with 

# Uses Featured units ( Units on the active screen)
python3 -m pysc2.bin.agent --map CollectMineralShards --agent pysc2.agents.scripted_agent.CollectMineralShardsFeatureUnits

# Uses Featured units ( Units on the active screen)
python3 -m pysc2.bin.agent --map CollectMineralShards --agent pysc2.agents.scripted_agent.CollectMineralShards

# Uses Raw Feature Map
python3 -m pysc2.bin.agent --map CollectMineralShards --agent pysc2.agents.scripted_agent.CollectMineralShardsRaw
