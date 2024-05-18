#!/bin/bash
#SBATCH --array=1-3
#SBATCH --job-name=ARCMG # Job name
#SBATCH --output=ARCMG%j.out     # STDOUT output file
#SBATCH --error=ARCMG%j.err      # STDERR output file (optional)
#SBATCH --partition=p_mischaik_1      # Partition (job queue)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Total number of tasks across all nodes
#SBATCH --cpus-per-task=1             # Number of CPUs (cores) per task (>1 if multithread tasks)
#SBATCH --mem=4000                    # Real memory (RAM) required (MB)
#SBATCH --time=68:00:00               # Total run time limit (hh:mm:ss)
#SBATCH --requeue                     # Return job to the queue if preempted
#SBATCH --export=ALL                  # Export you current env to the job env

# Load necessary modules
#module purge
#module load python/3.8.5

cd /scratch/bg545/phase-space-classification/experiments

#  Run python script with input data

search_dir=tmp_config_pendulum_lqr/$SLURM_ARRAY_TASK_ID/

yourfilenames=`ls $(pwd)/$search_dir*.yaml`

for eachfile in $yourfilenames
do
    N_FILE_EX=$(basename $eachfile)
    N_FILE=${N_FILE_EX%.*}

    echo $N_FILE_EX
    python train.py --config_dir "$search_dir" --config $N_FILE_EX --transfer_learning
done


echo $SLURM_JOBID
sstat --format=MaxRSS,MaxDiskRead,MaxDiskWrite,NodeList -j $SLURM_JOBID

echo $SLURM_ARRAY_JOB_ID
sstat --format=MaxRSS,MaxDiskRead,MaxDiskWrite,NodeList -j $SLURM_ARRAY_JOB_ID

underline="_"
echo $SLURM_ARRAY_JOB_ID$underline$SLURM_ARRAY_TASK_ID
sacct --units=G --format=MaxRSS,MaxDiskRead,MaxDiskWrite,Elapsed,NodeList -j $SLURM_JOBID$underline$SLURM_ARRAY_TASK_ID