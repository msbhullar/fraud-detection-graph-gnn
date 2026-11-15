#!/bin/bash
# E4 arm A2 on Sol with big models (model-size test): same 50 cases, same prompt and JSON schema as the Mac run.
# Submit from /scratch/mbhullar/e4_sol with:  sbatch jobs/e4_generate_sol.sh
#SBATCH -A class_cse57382398fall2026
#SBATCH -p public
#SBATCH -q class
#SBATCH -G a100:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 0-04:00
#SBATCH -o logs/e4_sol_%j.out

cd /scratch/mbhullar/e4_sol
module load ollama/0.5.5
export OLLAMA_MODELS=/scratch/mbhullar/ollama_models      # keep the 60 GB of model files out of the 100 GB home folder
mkdir -p "$OLLAMA_MODELS"
echo "node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv
export OLLAMA_HOST=127.0.0.1:11434      # server and Python run on the same GPU node
ollama serve > "logs/ollama_server_${SLURM_JOB_ID}.log" 2>&1 &      # start the server directly (the ollama-start shortcut does not exist inside batch jobs)
OLLAMA_SERVER_PID=$!
for attempt in $(seq 1 60); do ollama list >/dev/null 2>&1 && break; sleep 2; done      # wait up to 2 minutes for the server
echo "OLLAMA_HOST=$OLLAMA_HOST"; ollama --version
ollama list >/dev/null 2>&1 || { echo "SERVER DID NOT START"; tail -30 "logs/ollama_server_${SLURM_JOB_ID}.log"; kill $OLLAMA_SERVER_PID; exit 1; }
for model in qwen2.5:32b llama3.3:70b; do
  echo "=== pulling $model ($(date)) ==="
  ollama pull "$model" || echo "PULL FAILED: $model"
done
ollama list
echo "=== generating ($(date)) ==="
/scratch/mbhullar/envs/fraud-gpu/bin/python src/e4_generate_v2.py full qwen2.5:32b,llama3.3:70b
echo "=== done ($(date)) ==="
kill $OLLAMA_SERVER_PID
