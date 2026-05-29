#!/usr/bin/env bash
set -euo pipefail

cd /root/TSRA

RUN_ROOT="/vepfs/tsra_outputs/clutrr_vanilla_classifier_audit/clutrr_vanilla_classifier_audit_$(date +%Y%m%d_%H%M%S)"
QUEUE="$RUN_ROOT/queue.tsv"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
GPUS="${TSRA_VANILLA_AUDIT_GPUS:-0,1}"
GPU_MEM_THRESHOLD_MB="${TSRA_GPU_MEM_THRESHOLD_MB:-1000}"

mkdir -p "$STATUS_DIR" "$LOG_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/clutrr_vanilla_classifier_audit/latest

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

ROBERTA="/vepfs/tsra_models/hf/roberta-base-clean-current"
DEBERTA_V3="/vepfs/tsra_models/hf/deberta-v3-base"

cat > "$QUEUE" <<EOF
name	dataset	model_type	model_path	seed	batch_size	eval_batch_size	epochs
clutrr_089_roberta_vanilla_seed0	data_089907f8	roberta	$ROBERTA	0	16	32	10
clutrr_089_roberta_vanilla_seed1	data_089907f8	roberta	$ROBERTA	1	16	32	10
clutrr_089_roberta_vanilla_seed42	data_089907f8	roberta	$ROBERTA	42	16	32	10
clutrr_089_deberta_v3_vanilla_seed0	data_089907f8	deberta-v3	$DEBERTA_V3	0	16	32	10
clutrr_089_deberta_v3_vanilla_seed1	data_089907f8	deberta-v3	$DEBERTA_V3	1	16	32	10
clutrr_089_deberta_v3_vanilla_seed42	data_089907f8	deberta-v3	$DEBERTA_V3	42	16	32	10
EOF

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUN_ROOT/supervisor.log"
}

free_gpu() {
  local gpu used
  IFS=',' read -ra candidates <<< "$GPUS"
  for gpu in "${candidates[@]}"; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ "${used:-999999}" -lt "$GPU_MEM_THRESHOLD_MB" ]]; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

dispatched_count() {
  find "$STATUS_DIR" -name "*.started" -type f | wc -l
}

run_task() {
  local gpu="$1" name="$2" dataset="$3" model_type="$4" model_path="$5" seed="$6" batch_size="$7" eval_batch_size="$8" epochs="$9"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"
  local log_file="$LOG_DIR/${name}.log"

  {
    echo "task=$name"
    echo "dataset=$dataset"
    echo "model_type=$model_type"
    echo "model_path=$model_path"
    echo "seed=$seed"
    echo "epochs=$epochs"
    echo "gpu=$gpu"
    echo "objective=vanilla_final_label_ce_only"
    echo "uses_tsra_architecture=false"
    echo "uses_entity_spans=false"
    echo "uses_trace_loss=false"
    echo "uses_consistency_loss=false"
    echo "uses_renamed_augmentation=false"
    echo "started_at=$(date '+%F %T')"
  } > "$started"

  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m clutrr.cli.baseline \
      --config configs/clutrr/train_baseline.yaml \
      --dataset "$dataset" \
      --root data \
      --model_type "$model_type" \
      --model_name_or_path "$model_path" \
      --epochs "$epochs" \
      --batch_size "$batch_size" \
      --eval_batch_size "$eval_batch_size" \
      --gpus "$gpu" \
      --seed "$seed" \
      > "$log_file" 2>&1
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 100 "$log_file"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 160 "$log_file"; } > "$failed"
    fi
    exit "$rc"
  ) &
  log "Launched $name on GPU $gpu (pid=$!)"
}

total_tasks=$(( $(wc -l < "$QUEUE") - 1 ))
log "CLUTRR vanilla classifier audit started. total_tasks=$total_tasks run_root=$RUN_ROOT gpus=$GPUS"

while [[ "$(dispatched_count)" -lt "$total_tasks" ]]; do
  gpu="$(free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    log "No free GPU among $GPUS below ${GPU_MEM_THRESHOLD_MB}MB; waiting."
    sleep 120
    continue
  fi

  launched=0
  while IFS=$'\t' read -r name dataset model_type model_path seed batch_size eval_batch_size epochs; do
    [[ "$name" == "name" ]] && continue
    [[ -f "$STATUS_DIR/${name}.started" ]] && continue
    run_task "$gpu" "$name" "$dataset" "$model_type" "$model_path" "$seed" "$batch_size" "$eval_batch_size" "$epochs"
    launched=1
    break
  done < "$QUEUE"

  [[ "$launched" -eq 0 ]] && log "No launchable task found; waiting."
  sleep 120
done

log "All CLUTRR vanilla classifier audit tasks have been dispatched."
