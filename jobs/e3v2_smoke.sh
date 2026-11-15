#!/bin/bash
# E3 v2 code check: 2 short epochs on a small subset (about 10-15 min).
#SBATCH -A class_cse57382398fall2026
#SBATCH -p public
#SBATCH -q class
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 0-01:00
#SBATCH -o logs/e3v2_smoke_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mbhullar@asu.edu
cd /scratch/mbhullar/fraud-graph-project
/scratch/mbhullar/envs/fraud-gpu/bin/python src/ibm_graphsage_v2.py smoke
echo "JOB FINISHED"
