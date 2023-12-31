#!/bin/bash
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --nodes=1               # node count
#SBATCH --ntasks-per-node=2     # total number of tasks per node
#SBATCH --cpus-per-task=32       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:a100:2             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

set -x
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
echo "MASTER_PORT="$MASTER_PORT
#export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_TASKS_PER_NODE))
export WORLD_SIZE=$SLURM_NPROCS
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module purge
module load cesga/system 
eval "$(conda shell.bash hook)"
conda deactivate
source $STORE/mytorchdist/bin/deactivate
source $STORE/mytorchdist/bin/activate
srun ./trainDDPv1.sh