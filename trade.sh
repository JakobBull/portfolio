#!/bin/bash

# Check if the correct number of command-line arguments are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 BUY/SELL [additional arguments]"
    exit 1
fi

# Set the action (BUY or SELL) from the command line
action=$1

# Validate the action
if [ "$action" != "BUY" ] && [ "$action" != "SELL" ]; then
    echo "Action must be BUY or SELL"
    exit 1
fi

# Set the path to the Python interpreter
python_interpreter="/Users/jakobbull/anaconda3/envs/portfolio/bin/python"

# Set the path to the trade.py script
trade_script="trade.py"

# Construct the command to run the trade.py script
command="$python_interpreter $trade_script $@"

# Run the command
eval $command
