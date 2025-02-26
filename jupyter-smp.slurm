#!/bin/bash
# This value can be adjusted to run in a condo if applicable
# or commons and scavenge if the max walltime is more than 30 minutes
#SBATCH --partition=commons
# To run SMP applications only launch one task, leave this value as is
#SBATCH --reservation=dsci435
#SBATCH --ntasks=1 
# Tell SLURM to only place threads on physical cores, leave this value as is
#SBATCH --threads-per-core=1
# This value can be adjusted if a different thread count is required
#SBATCH --cpus-per-task=8
# This value can be adjusted if a different maximum wall time is required
#SBATCH --time=30:00

# purge any modules that might be automatically loaded
module purge
# XALT causes problems
export XALT_EXECUTABLE_TRACKING=no

# Collect job information
node=$(hostname -s) 
user=$(whoami) 
port=$((49152+RANDOM/2))

# Display instructions 
echo -e " 
Run the following command to set up an ssh tunnel to the compute node:

ssh -NL ${port}:${node}:${port} ${user}@nots.crc.rice.edu

Copy the 127.0.0.1 link below into a browser on your local machine "

# Run Jupyter from Singularity container
#srun singularity exec /opt/apps/examples/Jupyter/scipy-notebook_latest.sif jupyter-notebook --no-browser --port=${port} --ip=${node}
srun singularity exec /opt/apps/examples/Jupyter/tensorflow-notebook_latest.sif jupyter-notebook --no-browser --port=${port} --ip=${node}

