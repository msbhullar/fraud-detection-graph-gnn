#!/bin/bash
# E3 v2 training: 6 tasks = 3 settings x (with graph, graph removed), one GPU each, run in parallel.
#SBATCH -A class_cse57382398fall2026
#SBATCH -p public
#SBATCH -q class
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH -t 0-06:00
#SBATCH --array=0-5
#SBATCH -o logs/e3v2_train_%A_%a.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mbhullar@asu.edu
cd /scratch/mbhullar/fraud-graph-project
/scratch/mbhullar/envs/fraud-gpu/bin/python src/ibm_graphsage_v2.py train "$SLURM_ARRAY_TASK_ID"
echo "JOB FINISHED"
