#!/bin/bash
# Score the chosen round-2 variant on test ONCE. Usage: sbatch jobs/ablation_finalize.sh <variant | auto>
#SBATCH -A class_cse57382398fall2026
#SBATCH -p public
#SBATCH -q class
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 0-01:00
#SBATCH -o logs/ablation_finalize_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mbhullar@asu.edu
cd /scratch/mbhullar/fraud-graph-project
/scratch/mbhullar/envs/fraud-gpu/bin/python src/ibm_ablation_finalize.py "$1"
/scratch/mbhullar/envs/fraud-gpu/bin/python src/bootstrap_compare.py ibm_e2p_tuned ibm_e2r_tuned xgboost:xgboost --normal-days --resamples=200
echo "JOB FINISHED"
