#!/usr/bin/env bash
#SBATCH --job-name=ds8b_qlora_4gpu
#SBATCH --output=ds8b_qlora_4gpu_%j.out
#SBATCH --error=ds8b_qlora_4gpu_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --gres=gpu:4
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail
set -x
cd "${SLURM_SUBMIT_DIR}"

echo "Start $(date) on $(hostname)"
echo "Node list: ${SLURM_NODELIST}"

# NCCL/runtime
export NCCL_DEBUG=warn
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Writable HF caches. Use $HOME to avoid accidental '@' in path.
CACHE_ROOT="${SLURM_TMPDIR:-$HOME/.cache/hf}"
export HF_HOME="${CACHE_ROOT}"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/transformers"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hub"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

# Paths
SIF=/ceph/home/student.aau.dk/sd48tq/DeepSeek_QLoRA.sif
DSCFG=${SLURM_SUBMIT_DIR}/deepspeed_zero2_qlora.json
LANG_JSON=/ceph/home/student.aau.dk/sd48tq/new_finetune/data/qa_pairs.json
OUTDIR=/ceph/home/student.aau.dk/sd48tq/new_finetune/deepseek-8b-en-10e-qlora
mkdir -p "$OUTDIR"

# Quick visibility checks
srun -u bash -lc 'nvidia-smi || true'
srun -u singularity exec --nv "$SIF" bash -lc 'echo "Container OK"; nvidia-smi || true'
srun -u singularity exec --nv \
	  --env HF_HOME="$HF_HOME" \
	    --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
	      --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
	        --env HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
		  "$SIF" python3.10 - <<'PY'
import os, pathlib
print("HF_HOME:", os.environ.get("HF_HOME"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
pathlib.Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)
print("Writable cache OK")
PY

ACCEL_ARGS=( --num_processes 4 --mixed_precision bf16 )
TRAIN_ARGS=(
	  --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B
	  --qa_json_path "$LANG_JSON"
	  --output_dir "$OUTDIR"
	  --use_qlora true
	  --pack_samples true
	  --max_seq_length 2048
	  --learning_rate 2e-4
	  --num_train_epochs 10
	  --per_device_train_batch_size 1
	  --gradient_accumulation_steps 16
	  --lr_scheduler_type cosine
	  --warmup_ratio 0.03
	  --save_steps 1000
	  --save_total_limit 3
	  --logging_steps 20
	  --bf16 true
	  --gradient_checkpointing true
	  --val_split 0.02
	  --deepspeed "$DSCFG"
	  --flash_attention false
	  )

	  # Launch with caches explicitly injected to override any /scratch defaults set in the image
srun -u singularity exec --nv  --cleanenv \
	  --env HF_HOME="$HF_HOME" \
	  --env TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
	  --env HF_DATASETS_CACHE="$HF_DATASETS_CACHE" \
	  --env HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE" \
	  --env TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM" \
	  "$SIF" \
	  accelerate launch "${ACCEL_ARGS[@]}" train_sft.py "${TRAIN_ARGS[@]}"

echo "Done $(date)"
