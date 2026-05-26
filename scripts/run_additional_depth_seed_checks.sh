#!/usr/bin/env bash
set -euo pipefail

cd /root/TSRA

RUN_ROOT="/vepfs/tsra_outputs/additional_depth_checks/depth_seed_checks_$(date +%Y%m%d_%H%M%S)"
QUEUE="$RUN_ROOT/queue.tsv"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
GPUS="${TSRA_EXTRA_GPUS:-5,6,7}"
GPU_MEM_THRESHOLD_MB="${TSRA_GPU_MEM_THRESHOLD_MB:-1000}"
MODEL="/vepfs/tsra_models/hf/deberta-base"

mkdir -p "$STATUS_DIR" "$LOG_DIR" "$RESULT_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/additional_depth_checks/latest_depth_seed_checks

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cat > "$QUEUE" <<EOF
kind	name	seed	lambda_trace	lambda_nexthop	lambda_edge	lambda_consistency	dataset	root	train_depths	test_depths	train_qdeps	test_qdeps	max_sents	max_len	batch_size	lr
clutrr	clutrr_db9_deberta_tsra_seed0	0	-	1.0	1.0	5.0	data_db9b8f04	-	-	-	-	-	-	-	16	-
clutrr	clutrr_db9_deberta_label_only_seed0	0	-	0.0	0.0	0.0	data_db9b8f04	-	-	-	-	-	-	-	16	-
clutrr	clutrr_db9_deberta_tsra_seed1	1	-	1.0	1.0	5.0	data_db9b8f04	-	-	-	-	-	-	-	16	-
clutrr	clutrr_db9_deberta_label_only_seed1	1	-	0.0	0.0	0.0	data_db9b8f04	-	-	-	-	-	-	-	16	-
clutrr	clutrr_db9_deberta_tsra_seed42	42	-	1.0	1.0	5.0	data_db9b8f04	-	-	-	-	-	-	-	16	-
clutrr	clutrr_db9_deberta_label_only_seed42	42	-	0.0	0.0	0.0	data_db9b8f04	-	-	-	-	-	-	-	16	-
prop	ruletaker_raw_deberta_tsra_trainq12_seed0	0	1.0	-	-	-	ruletaker_raw	data/rule-reasoning-dataset-V2020.2.5.0/original	1,2	1,2,3,5	1,2	1,2,3,4,5	24	192	16	2e-5
prop	ruletaker_raw_deberta_baseline_trainq12_seed0	0	0.0	-	-	-	ruletaker_raw	data/rule-reasoning-dataset-V2020.2.5.0/original	1,2	1,2,3,5	1,2	1,2,3,4,5	24	192	16	2e-5
prop	ruletaker_raw_deberta_tsra_trainq12_seed1	1	1.0	-	-	-	ruletaker_raw	data/rule-reasoning-dataset-V2020.2.5.0/original	1,2	1,2,3,5	1,2	1,2,3,4,5	24	192	16	2e-5
prop	ruletaker_raw_deberta_baseline_trainq12_seed1	1	0.0	-	-	-	ruletaker_raw	data/rule-reasoning-dataset-V2020.2.5.0/original	1,2	1,2,3,5	1,2	1,2,3,4,5	24	192	16	2e-5
prop	ruletaker_raw_deberta_tsra_trainq12_seed42	42	1.0	-	-	-	ruletaker_raw	data/rule-reasoning-dataset-V2020.2.5.0/original	1,2	1,2,3,5	1,2	1,2,3,4,5	24	192	16	2e-5
prop	ruletaker_raw_deberta_baseline_trainq12_seed42	42	0.0	-	-	-	ruletaker_raw	data/rule-reasoning-dataset-V2020.2.5.0/original	1,2	1,2,3,5	1,2	1,2,3,4,5	24	192	16	2e-5
prop	proofwriter_deberta_baseline_seed42_10ep	42	0.0	-	-	-	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	-	-	16	192	16	2e-5
prop	proofwriter_deberta_tsra_seed42_10ep	42	1.0	-	-	-	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	-	-	16	192	16	2e-5
EOF

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUN_ROOT/supervisor.log"
}

free_gpu() {
  local gpu
  IFS=',' read -ra candidates <<< "$GPUS"
  for gpu in "${candidates[@]}"; do
    local used
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

run_clutrr() {
  local gpu="$1" name="$2" seed="$3" dataset="$4" lambda_nexthop="$5" lambda_edge="$6" lambda_consistency="$7" batch_size="$8"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"
  local log_file="$LOG_DIR/${name}.log"
  {
    echo "kind=clutrr"
    echo "task=$name"
    echo "dataset=$dataset"
    echo "gpu=$gpu"
    echo "seed=$seed"
    echo "epochs=10"
    echo "lambda_nexthop=$lambda_nexthop"
    echo "lambda_edge=$lambda_edge"
    echo "lambda_consistency=$lambda_consistency"
    echo "started_at=$(date '+%F %T')"
  } > "$started"

  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m clutrr.cli.train \
      --config configs/clutrr/train_tsra.yaml \
      --dataset "$dataset" \
      --root data \
      --model_type deberta \
      --model_name_or_path "$MODEL" \
      --epochs 10 \
      --batch_size "$batch_size" \
      --eval_batch_size 32 \
      --gpus "$gpu" \
      --strategy single \
      --seed "$seed" \
      --lambda_nexthop "$lambda_nexthop" \
      --lambda_edge "$lambda_edge" \
      --lambda_consistency "$lambda_consistency" \
      > "$log_file" 2>&1
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 80 "$log_file"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 140 "$log_file"; } > "$failed"
    fi
    exit "$rc"
  ) &
  log "Launched $name on GPU $gpu (pid=$!)"
}

run_prop() {
  local gpu="$1" name="$2" seed="$3" lambda_trace="$4" dataset="$5" root="$6" train_depths="$7" test_depths="$8" train_qdeps="$9" test_qdeps="${10}" max_sents="${11}" max_len="${12}" batch_size="${13}" lr="${14}"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"
  local log_file="$LOG_DIR/${name}.log"
  local out_json="$RESULT_DIR/${name}.json"
  {
    echo "kind=prop"
    echo "task=$name"
    echo "dataset=$dataset"
    echo "gpu=$gpu"
    echo "seed=$seed"
    echo "epochs=10"
    echo "lambda_trace=$lambda_trace"
    echo "train_depths=$train_depths"
    echo "test_depths=$test_depths"
    echo "train_qdeps=$train_qdeps"
    echo "test_qdeps=$test_qdeps"
    echo "out=$out_json"
    echo "started_at=$(date '+%F %T')"
  } > "$started"

  (
    set +e
    args=(
      scripts/transformer_tsra_prop.py
      --dataset "$dataset"
      --root "$root"
      --limit-train 0
      --limit-test 0
      --epochs 10
      --batch-size "$batch_size"
      --lr "$lr"
      --lambda-trace "$lambda_trace"
      --max-sents "$max_sents"
      --max-len "$max_len"
      --model-name "$MODEL"
      --seed "$seed"
      --out "$out_json"
    )
    [[ "$train_depths" != "-" ]] && args+=(--train-depths "$train_depths")
    [[ "$test_depths" != "-" ]] && args+=(--test-depths "$test_depths")
    [[ "$train_qdeps" != "-" ]] && args+=(--train-qdeps "$train_qdeps")
    [[ "$test_qdeps" != "-" ]] && args+=(--test-qdeps "$test_qdeps")
    CUDA_VISIBLE_DEVICES="$gpu" python3 "${args[@]}" > "$log_file" 2>&1
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
log "Additional depth/seed supervisor started. total_tasks=$total_tasks run_root=$RUN_ROOT gpus=$GPUS"

while [[ "$(dispatched_count)" -lt "$total_tasks" ]]; do
  gpu="$(free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    log "No free GPU among $GPUS below ${GPU_MEM_THRESHOLD_MB}MB; waiting."
    sleep 120
    continue
  fi

  launched=0
  while IFS=$'\t' read -r kind name seed lambda_trace lambda_nexthop lambda_edge lambda_consistency dataset root train_depths test_depths train_qdeps test_qdeps max_sents max_len batch_size lr; do
    [[ "$kind" == "kind" ]] && continue
    [[ -f "$STATUS_DIR/${name}.started" ]] && continue
    if [[ "$kind" == "clutrr" ]]; then
      run_clutrr "$gpu" "$name" "$seed" "$dataset" "$lambda_nexthop" "$lambda_edge" "$lambda_consistency" "$batch_size"
    elif [[ "$kind" == "prop" ]]; then
      run_prop "$gpu" "$name" "$seed" "$lambda_trace" "$dataset" "$root" "$train_depths" "$test_depths" "$train_qdeps" "$test_qdeps" "$max_sents" "$max_len" "$batch_size" "$lr"
    else
      log "Unknown task kind: $kind"
      exit 3
    fi
    launched=1
    break
  done < "$QUEUE"

  [[ "$launched" -eq 0 ]] && log "No launchable task found; waiting."
  sleep 120
done

log "All additional depth/seed queue entries have been dispatched."
