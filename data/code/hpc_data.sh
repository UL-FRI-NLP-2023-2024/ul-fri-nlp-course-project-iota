ml load Python
source ./../../venv/bin/activate
python data_process.py


# This script is more or less meant for testing if stuff works on HPC. (By being SSHed to a Compute Node)
# Use sbatch_data_process.sh for actual data processing. (From a login node)
# sbatch sbatch_data_process.sh