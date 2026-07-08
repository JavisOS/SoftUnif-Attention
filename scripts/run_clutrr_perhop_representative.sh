#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/clutrr_perhop_representative/clutrr_perhop_representative_$(date +%Y%m%d_%H%M%S)"
QUEUE="$RUN_ROOT/queue.tsv"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
GPUS="${TRUA_CLUTRR_PERHOP_GPUS:-0,1,2,3,4,5,6,7}"
GPU_MEM_THRESHOLD_MB="${TRUA_GPU_MEM_THRESHOLD_MB:-1000}"

mkdir -p "$STATUS_DIR" "$LOG_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/clutrr_perhop_representative/latest

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

ROBERTA="/vepfs/tsra_models/hf/roberta-base-clean-current"
DEBERTA="/vepfs/tsra_models/hf/deberta-base"
DEBERTA_V3="/vepfs/tsra_models/hf/deberta-v3-base"

cat > "$QUEUE" <<EOF
name	dataset	model_type	model_path	seed	variant	batch_size	eval_batch_size	epochs
clutrr_089_roberta_vanilla_seed0	data_089907f8	roberta	$ROBERTA	0	vanilla	16	32	10
clutrr_089_roberta_tsra_seed0	data_089907f8	roberta	$ROBERTA	0	tsra	16	32	10
clutrr_089_roberta_vanilla_seed1	data_089907f8	roberta	$ROBERTA	1	vanilla	16	32	10
clutrr_089_roberta_tsra_seed1	data_089907f8	roberta	$ROBERTA	1	tsra	16	32	10
clutrr_089_roberta_vanilla_seed42	data_089907f8	roberta	$ROBERTA	42	vanilla	16	32	10
clutrr_089_roberta_tsra_seed42	data_089907f8	roberta	$ROBERTA	42	tsra	16	32	10
clutrr_089_deberta_vanilla_seed0	data_089907f8	deberta	$DEBERTA	0	vanilla	16	32	10
clutrr_089_deberta_tsra_seed0	data_089907f8	deberta	$DEBERTA	0	tsra	16	32	10
clutrr_089_deberta_vanilla_seed1	data_089907f8	deberta	$DEBERTA	1	vanilla	16	32	10
clutrr_089_deberta_tsra_seed1	data_089907f8	deberta	$DEBERTA	1	tsra	16	32	10
clutrr_089_deberta_vanilla_seed42	data_089907f8	deberta	$DEBERTA	42	vanilla	16	32	10
clutrr_089_deberta_tsra_seed42	data_089907f8	deberta	$DEBERTA	42	tsra	16	32	10
clutrr_089_deberta_v3_vanilla_seed0	data_089907f8	deberta-v3	$DEBERTA_V3	0	vanilla	16	32	10
clutrr_089_deberta_v3_tsra_seed0	data_089907f8	deberta-v3	$DEBERTA_V3	0	tsra	16	32	10
clutrr_089_deberta_v3_vanilla_seed1	data_089907f8	deberta-v3	$DEBERTA_V3	1	vanilla	16	32	10
clutrr_089_deberta_v3_tsra_seed1	data_089907f8	deberta-v3	$DEBERTA_V3	1	tsra	16	32	10
clutrr_089_deberta_v3_vanilla_seed42	data_089907f8	deberta-v3	$DEBERTA_V3	42	vanilla	16	32	10
clutrr_089_deberta_v3_tsra_seed42	data_089907f8	deberta-v3	$DEBERTA_V3	42	tsra	16	32	10
clutrr_db9_deberta_v3_vanilla_seed0	data_db9b8f04	deberta-v3	$DEBERTA_V3	0	vanilla	16	32	10
clutrr_db9_deberta_v3_tsra_seed0	data_db9b8f04	deberta-v3	$DEBERTA_V3	0	tsra	16	32	10
clutrr_db9_deberta_v3_vanilla_seed1	data_db9b8f04	deberta-v3	$DEBERTA_V3	1	vanilla	16	32	10
clutrr_db9_deberta_v3_tsra_seed1	data_db9b8f04	deberta-v3	$DEBERTA_V3	1	tsra	16	32	10
clutrr_db9_deberta_v3_vanilla_seed42	data_db9b8f04	deberta-v3	$DEBERTA_V3	42	vanilla	16	32	10
clutrr_db9_deberta_v3_tsra_seed42	data_db9b8f04	deberta-v3	$DEBERTA_V3	42	tsra	16	32	10
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

started_count() {
  find "$STATUS_DIR" -name "*.started" -type f | wc -l
}

finished_count() {
  local done failed
  done="$(find "$STATUS_DIR" -name "*.done" -type f | wc -l)"
  failed="$(find "$STATUS_DIR" -name "*.failed" -type f | wc -l)"
  echo $((done + failed))
}

run_task() {
  local gpu="$1" name="$2" dataset="$3" model_type="$4" model_path="$5" seed="$6" variant="$7"
  local batch_size="$8" eval_batch_size="$9" epochs="${10}"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"
  local log_file="$LOG_DIR/${name}.log"

  {
    echo "task=$name"
    echo "dataset=$dataset"
    echo "model_type=$model_type"
    echo "model_path=$model_path"
    echo "variant=$variant"
    echo "seed=$seed"
    echo "epochs=$epochs"
    echo "gpu=$gpu"
    echo "started_at=$(date '+%F %T')"
  } > "$started"

  (
    set +e
    if [[ ! -d "$model_path" ]]; then
      echo "Missing model_path: $model_path" > "$log_file"
      rc=66
    elif [[ "$variant" == "vanilla" ]]; then
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
    else
      CUDA_VISIBLE_DEVICES="$gpu" python -u -m clutrr.cli.train \
        --config configs/clutrr/train_trua.yaml \
        --dataset "$dataset" \
        --root data \
        --model_type "$model_type" \
        --model_name_or_path "$model_path" \
        --epochs "$epochs" \
        --batch_size "$batch_size" \
        --eval_batch_size "$eval_batch_size" \
        --gpus "$gpu" \
        --strategy single \
        --seed "$seed" \
        --lambda_nexthop 1.0 \
        --lambda_edge 1.0 \
        --lambda_consistency 5.0 \
        > "$log_file" 2>&1
      rc=$?
    fi

    if [[ "$rc" -eq 0 ]]; then
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 140 "$log_file"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 180 "$log_file"; } > "$failed"
    fi
    exit "$rc"
  ) &
  log "Launched $name on GPU $gpu (pid=$!)"
}

total_tasks=$(( $(wc -l < "$QUEUE") - 1 ))
log "CLUTRR per-hop representative queue started. total_tasks=$total_tasks run_root=$RUN_ROOT gpus=$GPUS"

while [[ "$(started_count)" -lt "$total_tasks" ]]; do
  gpu="$(free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    log "No free GPU among $GPUS below ${GPU_MEM_THRESHOLD_MB}MB; waiting."
    sleep 60
    continue
  fi

  launched=0
  while IFS=$'\t' read -r name dataset model_type model_path seed variant batch_size eval_batch_size epochs; do
    [[ "$name" == "name" ]] && continue
    [[ -f "$STATUS_DIR/${name}.started" ]] && continue
    run_task "$gpu" "$name" "$dataset" "$model_type" "$model_path" "$seed" "$variant" "$batch_size" "$eval_batch_size" "$epochs"
    launched=1
    break
  done < "$QUEUE"

  [[ "$launched" -eq 0 ]] && log "No launchable task found; waiting."
  sleep 45
done

log "All queue entries dispatched. Waiting for completion."
while [[ "$(finished_count)" -lt "$total_tasks" ]]; do
  log "Progress: $(finished_count)/$total_tasks finished."
  sleep 120
done

log "All queue entries finished. Aggregating per-hop results."
python3 scripts/aggregate_clutrr_perhop.py --run-root "$RUN_ROOT" >> "$RUN_ROOT/supervisor.log" 2>&1
log "Aggregation finished."
