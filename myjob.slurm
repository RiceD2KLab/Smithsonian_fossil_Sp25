#!/bin/bash
#SBATCH --job-name=faster_rcnn_modeling
#SBATCH --mail-user=aoo4@rice.edu
#SBATCH --mail-type=ALL
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --reservation=dsci435
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6000m
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

echo "My job ran on:"
echo $SLURM_NODELIST
pwd

# Load conda environment correctly
source ~/.bashrc  # Or ~/.bash_profile if that's where conda is initialized
conda activate myenv

srun ls -l /storage/hpc/work/dsci435/smithsonian/master_annotations_csv/master_annotation.csv

# Navigate to the script directory
cd /projects/dsci435/smithsonian_sp25/Smithsonian_fossil_Sp25/scripts

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/aoo4/.conda/envs/myenv/lib

# Run the Python script normally
srun python faster_rcnn_modeling.py
