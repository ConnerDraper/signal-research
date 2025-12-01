#!/bin/bash

# Usage: sbatch scripts/run_single_signal.slurm <path_to_specific_config_file>
# Example: sbatch scripts/run_single_signal.slurm config_files/signals/str_22_low_quality.yaml

#SBATCH --job-name=signal_research
#SBATCH --output=data/logs/signal_research_%j.out
#SBATCH --error=data/logs/signal_research_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --mail-user=connerd4@byu.edu 
#SBATCH --mail-type=BEGIN,END,FAIL 

PROJECT_ROOT="/home/connerd4/silverfund/signal-research"

# 1. Check for config path argument
if [ -z "$1" ]; then
    echo "Error: No config file provided." >&2
    echo "Usage: sbatch $0 <path_to_specific_config_file>" >&2
    exit 1
fi
CONFIG_FILE_PATH=$1

# Ensure the log directory exists
mkdir -p "$PROJECT_ROOT/data/logs"

# 2. Setup Environment
cd "$PROJECT_ROOT" || exit 1

# 3. Extract Signal Name (for logging/positional arg)
SIGNAL_NAME=$(basename "$CONFIG_FILE_PATH" .yaml)

echo "======================================================"
echo "Starting SLURM task for signal: ${SIGNAL_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Using Config: ${CONFIG_FILE_PATH}"
echo "======================================================"

# set up date slots for parallelization
year_starts=(
  "1995-06-27" "1996-06-27" "1997-06-27" "1998-06-27" "1999-06-27"
  "2000-06-27" "2001-06-27" "2002-06-27" "2003-06-27" "2004-06-27"
  "2005-06-27" "2006-06-27" "2007-06-27" "2008-06-27" "2009-06-27"
  "2010-06-27" "2011-06-27" "2012-06-27" "2013-06-27" "2014-06-27"
  "2015-06-27" "2016-06-27" "2017-06-27" "2018-06-27" "2019-06-27"
  "2020-06-27" "2021-06-27" "2022-06-27" "2023-06-27" "2024-06-27"
)

year_ends=(
  "1996-06-26" "1997-06-26" "1998-06-26" "1999-06-26" "2000-06-26"
  "2001-06-26" "2002-06-26" "2003-06-26" "2004-06-26" "2005-06-26"
  "2006-06-26" "2007-06-26" "2008-06-26" "2009-06-26" "2010-06-26"
  "2011-06-26" "2012-06-26" "2013-06-26" "2014-06-26" "2015-06-26"
  "2016-06-26" "2017-06-26" "2018-06-26" "2019-06-26" "2020-06-26"
  "2021-06-26" "2022-06-26" "2023-06-26" "2024-06-26" "2025-09-15"
)

num_signals=${#signals[@]}
num_years=${#year_starts[@]}
total_tasks=$((num_signals * num_years))



# 4. Run the pipeline script
uv run -m src.pipeline --config "${CONFIG_FILE_PATH}"

echo "======================================================"
echo "Finished SLURM task for signal: ${SIGNAL_NAME}"
echo "======================================================"