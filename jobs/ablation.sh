#!/bin/bash
# Round 2 feature-group ablation, validation only (test files are never loaded). 6 variants x 30 settings on 1 GPU.
#SBATCH -A class_cse57382398fall2026
#SBATCH -p public
#SBATCH -q class
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00
#SBATCH -o logs/ablation_%j.out
cd /scratch/mbhullar/fraud-graph-project
/scratch/mbhullar/envs/fraud-gpu/bin/python src/ibm_ablation.py
echo "JOB FINISHED"
