#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/crest_clutrr/crest_clutrr_$(date +%Y%m%d_%H%M%S)"
LATEST="/vepfs/tsra_outputs/crest_clutrr/latest"
QUEUE="$RUN_ROOT/queue.tsv"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
STATUS_DIR="$RUN_ROOT/status"
GPUS="${TRUA_CREST_GPUS:-0,1,2}"
GPU_MEM_THRESHOLD_MB="${TRUA_GPU_MEM_THRESHOLD_MB:-1000}"

mkdir -p "$LOG_DIR" "$RESULT_DIR" "$STATUS_DIR"
ln -sfn "$RUN_ROOT" "$LATEST"

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DEBERTA_V3="/vepfs/tsra_models/hf/deberta-v3-base"

cat > "$QUEUE" <<EOF
name	dataset	model_type	model_path	seed	epochs	batch_size	eval_batch_size	cf_weight	consistency_weight
crest_089_deberta_v3_seed0	data_089907f8	deberta-v3	$DEBERTA_V3	0	10	16	64	1.0	0.5
crest_089_deberta_v3_seed1	data_089907f8	deberta-v3	$DEBERTA_V3	1	10	16	64	1.0	0.5
crest_089_deberta_v3_seed42	data_089907f8	deberta-v3	$DEBERTA_V3	42	10	16	64	1.0	0.5
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
  local gpu="$1" name="$2" dataset="$3" model_type="$4" model_path="$5" seed="$6" epochs="$7" batch_size="$8" eval_batch_size="$9" cf_weight="${10}" consistency_weight="${11}"
  local out_file="$RESULT_DIR/${name}.json"
  local log_file="$LOG_DIR/${name}.log"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"

  {
    echo "task=$name"
    echo "method=crest_style_query_reverse_rename"
    echo "official_code=false"
    echo "dataset=$dataset"
    echo "model_type=$model_type"
    echo "model_path=$model_path"
    echo "seed=$seed"
    echo "epochs=$epochs"
    echo "gpu=$gpu"
    echo "cf_weight=$cf_weight"
    echo "consistency_weight=$consistency_weight"
    echo "uses_trua_architecture=false"
    echo "uses_trace_loss=false"
    echo "uses_gold_path=false"
    echo "test_time_input=raw_story_query_only"
    echo "out=$out_file"
    echo "started_at=$(date '+%F %T')"
  } > "$started"

  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python3 -u scripts/crest_clutrr_baseline.py \
      --dataset "$dataset" \
      --root data \
      --model_type "$model_type" \
      --model_name_or_path "$model_path" \
      --epochs "$epochs" \
      --batch_size "$batch_size" \
      --eval_batch_size "$eval_batch_size" \
      --seed "$seed" \
      --gpus "$gpu" \
      --cf_weight "$cf_weight" \
      --consistency_weight "$consistency_weight" \
      --out "$out_file" \
      > "$log_file" 2>&1
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 120 "$log_file"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 180 "$log_file"; } > "$failed"
    fi
    exit "$rc"
  ) &
  log "Launched $name on GPU $gpu (pid=$!)"
}

total_tasks=$(( $(wc -l < "$QUEUE") - 1 ))
log "CREST-style CLUTRR queue started. total_tasks=$total_tasks run_root=$RUN_ROOT gpus=$GPUS"

while [[ "$(started_count)" -lt "$total_tasks" ]]; do
  gpu="$(free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    log "No free GPU among $GPUS below ${GPU_MEM_THRESHOLD_MB}MB; waiting."
    sleep 60
    continue
  fi

  launched=0
  while IFS=$'\t' read -r name dataset model_type model_path seed epochs batch_size eval_batch_size cf_weight consistency_weight; do
    [[ "$name" == "name" ]] && continue
    [[ -f "$STATUS_DIR/${name}.started" ]] && continue
    run_task "$gpu" "$name" "$dataset" "$model_type" "$model_path" "$seed" "$epochs" "$batch_size" "$eval_batch_size" "$cf_weight" "$consistency_weight"
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

log "All queue entries finished. Aggregating."
python3 scripts/aggregate_crest_clutrr.py --run-root "$RUN_ROOT" >> "$RUN_ROOT/supervisor.log" 2>&1
log "Aggregation finished."
