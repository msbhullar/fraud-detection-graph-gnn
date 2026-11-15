#!/bin/bash
# E3 v2 finalize: best setting by validation -> test once (with and without graph), blend with ibm_e2r XGBoost, bootstraps.
#SBATCH -A class_cse57382398fall2026
#SBATCH -p public
#SBATCH -q class
#SBATCH -G 1
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 0-02:00
#SBATCH -o logs/e3v2_finalize_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mbhullar@asu.edu
cd /scratch/mbhullar/fraud-graph-project
PY=/scratch/mbhullar/envs/fraud-gpu/bin/python
$PY src/ibm_graphsage_v2.py finalize
$PY src/bootstrap_compare.py ibm_e3v2_nograph ibm_e3v2 graphsage:graphsage --normal-days --resamples=200
$PY src/bootstrap_compare.py ibm_e2r_tuned ibm_e3v2 xgboost:graphsage --normal-days --resamples=200
$PY src/bootstrap_compare.py ibm_e2r_tuned ibm_e3v2_blend xgboost:graphsage --normal-days --resamples=200
echo "JOB FINISHED"
