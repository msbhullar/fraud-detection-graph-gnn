#!/bin/bash
# Round 2 causal features (groups 1-5) on all 5,078,345 IBM transactions. CPU only.
#SBATCH -A class_cse57382398fall2026
#SBATCH -p public
#SBATCH -q class
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 0-01:00
#SBATCH -o logs/window_features_%j.out
cd /scratch/mbhullar/fraud-graph-project
/scratch/mbhullar/envs/fraud-gpu/bin/python src/ibm_window_features.py
