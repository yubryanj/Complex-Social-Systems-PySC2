#!/bin/bash

python -m pysc2.bin.agent \
--map Simple64 \
--agent refined_agent.SparseAgent \
--agent_race terran \
--max_agent_steps 0 \
--norender