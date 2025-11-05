#!/bin/bash
#SBATCH --job-name=signal_pipeline
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --mail-user=connerd4@byu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

PROJECT_ROOT="/home/connerd4/silverfund/signal-research"

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT

mkdir -p logs

# Read n_cpus from config and pass to SLURM
N_CPUS=$(python3 -c "
import yaml
with open('config_files/research_config.yaml') as f:
    config = yaml.safe_load(f)
print(config['backtest']['n_cpus'])
")
echo "Using $N_CPUS CPUs from config"

# Run with config-driven CPU allocation
python code/pipeline.py
