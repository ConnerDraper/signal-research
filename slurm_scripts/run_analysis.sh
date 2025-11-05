#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=16G

source /home/connerd4/silverfund/signal-research/.venv/bin/activate
cd /home/connerd4/silverfund/signal-research
python code/analyze.py
