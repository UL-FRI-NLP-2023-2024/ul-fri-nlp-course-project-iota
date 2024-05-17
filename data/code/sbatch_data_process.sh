#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=2G
#SBATCH --nodelist=gwn05
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --output=logs/nlp-data-process-%J.out
#SBATCH --error=logs/nlp-data-process-%J.err
#SBATCH --job-name="Data processing for NLP"

ml load Python
source ./../../venv/bin/activate
srun python data_process.py