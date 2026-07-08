#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/formal_10ep/tsra_formal_10ep_$(date +%Y%m%d_%H%M%S)"
QUEUE="$RUN_ROOT/queue.tsv"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
GPU_MEM_THRESHOLD_MB=1000

mkdir -p "$STATUS_DIR" "$LOG_DIR" "$RESULT_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/formal_10ep/latest_tsra_formal_10ep

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DEBERTA="/vepfs/tsra_models/hf/deberta-base"
ROBERTA="/vepfs/tsra_models/hf/roberta-base-clean-current"

cat > "$QUEUE" <<EOF
kind	name	gpu_hint	model_type	model_path	seed	epochs	lambda_nexthop	lambda_edge	lambda_consistency	dataset	root	train_depths	test_depths	lambda_trace	max_sents	max_len	batch_size	lr
clutrr	clutrr_deberta_full_seed0	-	deberta	$DEBERTA	0	10	1.0	1.0	5.0	-	-	-	-	-	-	-	16	-
clutrr	clutrr_deberta_full_seed1	-	deberta	$DEBERTA	1	10	1.0	1.0	5.0	-	-	-	-	-	-	-	16	-
clutrr	clutrr_deberta_label_only_seed0	-	deberta	$DEBERTA	0	10	0.0	0.0	0.0	-	-	-	-	-	-	-	16	-
clutrr	clutrr_deberta_label_only_seed1	-	deberta	$DEBERTA	1	10	0.0	0.0	0.0	-	-	-	-	-	-	-	16	-
clutrr	clutrr_deberta_no_consistency_seed0	-	deberta	$DEBERTA	0	10	1.0	1.0	0.0	-	-	-	-	-	-	-	16	-
clutrr	clutrr_deberta_no_consistency_seed1	-	deberta	$DEBERTA	1	10	1.0	1.0	0.0	-	-	-	-	-	-	-	16	-
clutrr	clutrr_roberta_full_seed0	-	roberta	$ROBERTA	0	10	1.0	1.0	5.0	-	-	-	-	-	-	-	16	-
clutrr	clutrr_roberta_full_seed1	-	roberta	$ROBERTA	1	10	1.0	1.0	5.0	-	-	-	-	-	-	-	16	-
clutrr	clutrr_roberta_label_only_seed0	-	roberta	$ROBERTA	0	10	0.0	0.0	0.0	-	-	-	-	-	-	-	16	-
clutrr	clutrr_roberta_label_only_seed1	-	roberta	$ROBERTA	1	10	0.0	0.0	0.0	-	-	-	-	-	-	-	16	-
clutrr	clutrr_roberta_no_consistency_seed0	-	roberta	$ROBERTA	0	10	1.0	1.0	0.0	-	-	-	-	-	-	-	16	-
clutrr	clutrr_roberta_no_consistency_seed1	-	roberta	$ROBERTA	1	10	1.0	1.0	0.0	-	-	-	-	-	-	-	16	-
prop	proofwriter_deberta_baseline_10ep	-	-	$DEBERTA	0	10	-	-	-	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	0.0	16	192	16	2e-5
prop	proofwriter_deberta_tsra_10ep	-	-	$DEBERTA	0	10	-	-	-	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	1.0	16	192	16	2e-5
prop	ruletaker_deberta_baseline_10ep	-	-	$DEBERTA	0	10	-	-	-	ruletaker_gfair	external_baselines/GFaiR/data/ruletaker_3ext_sat	-	-	0.0	24	192	16	2e-5
prop	ruletaker_deberta_tsra_10ep	-	-	$DEBERTA	0	10	-	-	-	ruletaker_gfair	external_baselines/GFaiR/data/ruletaker_3ext_sat	-	-	1.0	24	192	16	2e-5
prop	prontoqa_deberta_baseline_10ep	-	-	$DEBERTA	0	10	-	-	-	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	0.0	24	192	16	2e-5
prop	prontoqa_deberta_tsra_10ep	-	-	$DEBERTA	0	10	-	-	-	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	1.0	24	192	16	2e-5
EOF

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUN_ROOT/supervisor.log"
}

free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v th="$GPU_MEM_THRESHOLD_MB" '$2 + 0 < th {gsub(/ /, "", $1); print $1; exit}'
}

dispatched_count() {
  find "$STATUS_DIR" -name "*.started" -type f | wc -l
}

run_clutrr() {
  local gpu="$1" name="$2" model_type="$3" model_path="$4" seed="$5" epochs="$6"
  local lambda_nexthop="$7" lambda_edge="$8" lambda_consistency="$9"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"
  local log_file="$LOG_DIR/${name}.log"

  {
    echo "kind=clutrr"
    echo "task=$name"
    echo "gpu=$gpu"
    echo "model_type=$model_type"
    echo "model_path=$model_path"
    echo "seed=$seed"
    echo "epochs=$epochs"
    echo "lambda_nexthop=$lambda_nexthop"
    echo "lambda_edge=$lambda_edge"
    echo "lambda_consistency=$lambda_consistency"
    echo "started_at=$(date '+%F %T')"
  } > "$started"

  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m clutrr.cli.train \
      --config configs/clutrr/train_trua.yaml \
      --dataset data_089907f8 \
      --root data \
      --model_type "$model_type" \
      --model_name_or_path "$model_path" \
      --epochs "$epochs" \
      --batch_size 16 \
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
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 40 "$log_file"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 100 "$log_file"; } > "$failed"
    fi
    exit "$rc"
  ) &
  log "Launched $name on GPU $gpu (pid=$!)"
}

run_prop() {
  local gpu="$1" name="$2" model_path="$3" epochs="$4" dataset="$5" root="$6"
  local train_depths="$7" test_depths="$8" lambda_trace="$9" max_sents="${10}" max_len="${11}" batch_size="${12}" lr="${13}"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"
  local log_file="$LOG_DIR/${name}.log"
  local out_json="$RESULT_DIR/${name}.json"

  {
    echo "kind=prop"
    echo "task=$name"
    echo "gpu=$gpu"
    echo "model_path=$model_path"
    echo "epochs=$epochs"
    echo "dataset=$dataset"
    echo "root=$root"
    echo "lambda_trace=$lambda_trace"
    echo "started_at=$(date '+%F %T')"
  } > "$started"

  (
    set +e
    args=(
      scripts/transformer_trua_prop.py
      --dataset "$dataset"
      --root "$root"
      --limit-train 0
      --limit-test 0
      --epochs "$epochs"
      --batch-size "$batch_size"
      --lr "$lr"
      --lambda-trace "$lambda_trace"
      --max-sents "$max_sents"
      --max-len "$max_len"
      --model-name "$model_path"
      --out "$out_json"
    )
    if [[ "$train_depths" != "-" ]]; then
      args+=(--train-depths "$train_depths")
    fi
    if [[ "$test_depths" != "-" ]]; then
      args+=(--test-depths "$test_depths")
    fi
    CUDA_VISIBLE_DEVICES="$gpu" python3 "${args[@]}" > "$log_file" 2>&1
    rc=$?
    if [[ "$rc" -eq 0 ]]; then
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; echo "out=$out_json"; tail -n 60 "$log_file"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 120 "$log_file"; } > "$failed"
    fi
    exit "$rc"
  ) &
  log "Launched $name on GPU $gpu (pid=$!)"
}

total_tasks=$(( $(wc -l < "$QUEUE") - 1 ))
log "Formal TRUA 10ep supervisor started. total_tasks=$total_tasks run_root=$RUN_ROOT"

while [[ "$(dispatched_count)" -lt "$total_tasks" ]]; do
  gpu="$(free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    log "No free GPU below ${GPU_MEM_THRESHOLD_MB}MB; waiting."
    sleep 120
    continue
  fi

  launched=0
  while IFS=$'\t' read -r kind name gpu_hint model_type model_path seed epochs lambda_nexthop lambda_edge lambda_consistency dataset root train_depths test_depths lambda_trace max_sents max_len batch_size lr; do
    [[ "$kind" == "kind" ]] && continue
    [[ -f "$STATUS_DIR/${name}.started" ]] && continue
    if [[ "$kind" == "clutrr" ]]; then
      run_clutrr "$gpu" "$name" "$model_type" "$model_path" "$seed" "$epochs" "$lambda_nexthop" "$lambda_edge" "$lambda_consistency"
    elif [[ "$kind" == "prop" ]]; then
      run_prop "$gpu" "$name" "$model_path" "$epochs" "$dataset" "$root" "$train_depths" "$test_depths" "$lambda_trace" "$max_sents" "$max_len" "$batch_size" "$lr"
    else
      log "Unknown task kind for $name: $kind"
      exit 3
    fi
    launched=1
    break
  done < "$QUEUE"

  if [[ "$launched" -eq 0 ]]; then
    log "No launchable task found; waiting."
  fi
  sleep 120
done

log "All formal TRUA 10ep queue entries have been dispatched."
