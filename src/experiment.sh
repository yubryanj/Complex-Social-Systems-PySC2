#!/bin/bash

# capture the parameters
parameters=$1

# Activate the environment
source /cluster/work/gess/starcraft/pysc2/bin/activate

# Run the code!
python run.py $parameters

