#!/bin/bash
. ~/.bash_profile
conda activate portfolio
cd /Users/jakobbull/Documents/Projects/portfolio
python3 missed_execution.py
conda deactivate