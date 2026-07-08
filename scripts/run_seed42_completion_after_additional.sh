#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

WAIT_PID_FILE="/vepfs/tsra_outputs/additional_depth_checks/latest_supervisor.pid"
GPUS="${TRUA_SEED42_GPUS:-5,6,7}"
GPU_MEM_THRESHOLD_MB="${TRUA_GPU_MEM_THRESHOLD_MB:-1000}"
RUN_ROOT="/vepfs/tsra_outputs/formal_10ep/seed42_completion_$(date +%Y%m%d_%H%M%S)"
QUEUE="$RUN_ROOT/queue.tsv"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
DEBERTA="/vepfs/tsra_models/hf/deberta-base"
ROBERTA="/vepfs/tsra_models/hf/roberta-base-clean-current"
BERT="/vepfs/tsra_models/hf/bert-base-uncased"

mkdir -p "$STATUS_DIR" "$LOG_DIR" "$RESULT_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/formal_10ep/latest_seed42_completion

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$RUN_ROOT/supervisor.log"
}

if [[ -f "$WAIT_PID_FILE" ]]; then
  wait_pid="$(cat "$WAIT_PID_FILE" || true)"
  if [[ -n "$wait_pid" ]] && kill -0 "$wait_pid" 2>/dev/null; then
    log "Waiting for additional depth/seed queue pid=$wait_pid before starting seed42 completion."
    while kill -0 "$wait_pid" 2>/dev/null; do
      sleep 300
    done
    log "Additional depth/seed queue pid=$wait_pid has exited; starting seed42 completion."
  else
    log "No live additional queue found in $WAIT_PID_FILE; starting immediately."
  fi
else
  log "No wait pid file found; starting immediately."
fi

cat > "$QUEUE" <<EOFQ
kind	name	model_type	model_path	seed	epochs	lambda_nexthop	lambda_edge	lambda_consistency	dataset	root	train_depths	test_depths	lambda_trace	max_sents	max_len	batch_size	lr
clutrr	clutrr_089_deberta_full_seed42	deberta	$DEBERTA	42	10	1.0	1.0	5.0	data_089907f8	-	-	-	-	-	-	16	-
clutrr	clutrr_089_deberta_label_only_seed42	deberta	$DEBERTA	42	10	0.0	0.0	0.0	data_089907f8	-	-	-	-	-	-	16	-
clutrr	clutrr_089_deberta_no_consistency_seed42	deberta	$DEBERTA	42	10	1.0	1.0	0.0	data_089907f8	-	-	-	-	-	-	16	-
clutrr	clutrr_089_roberta_full_seed42	roberta	$ROBERTA	42	10	1.0	1.0	5.0	data_089907f8	-	-	-	-	-	-	16	-
clutrr	clutrr_089_roberta_label_only_seed42	roberta	$ROBERTA	42	10	0.0	0.0	0.0	data_089907f8	-	-	-	-	-	-	16	-
clutrr	clutrr_089_roberta_no_consistency_seed42	roberta	$ROBERTA	42	10	1.0	1.0	0.0	data_089907f8	-	-	-	-	-	-	16	-
prop	proofwriter_bert_baseline_seed42_10ep	-	$BERT	42	10	-	-	-	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	0.0	16	192	16	2e-5
prop	proofwriter_bert_tsra_seed42_10ep	-	$BERT	42	10	-	-	-	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	1.0	16	192	16	2e-5
prop	ruletaker_bert_baseline_seed42_10ep	-	$BERT	42	10	-	-	-	ruletaker_gfair	external_baselines/GFaiR/data/ruletaker_3ext_sat	-	-	0.0	24	192	16	2e-5
prop	ruletaker_bert_tsra_seed42_10ep	-	$BERT	42	10	-	-	-	ruletaker_gfair	external_baselines/GFaiR/data/ruletaker_3ext_sat	-	-	1.0	24	192	16	2e-5
prop	prontoqa_bert_baseline_seed42_10ep	-	$BERT	42	10	-	-	-	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	0.0	24	192	16	2e-5
prop	prontoqa_bert_tsra_seed42_10ep	-	$BERT	42	10	-	-	-	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	1.0	24	192	16	2e-5
prop	proofwriter_roberta_baseline_seed42_10ep	-	$ROBERTA	42	10	-	-	-	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	0.0	16	192	16	2e-5
prop	proofwriter_roberta_tsra_seed42_10ep	-	$ROBERTA	42	10	-	-	-	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	1.0	16	192	16	2e-5
prop	ruletaker_roberta_baseline_seed42_10ep	-	$ROBERTA	42	10	-	-	-	ruletaker_gfair	external_baselines/GFaiR/data/ruletaker_3ext_sat	-	-	0.0	24	192	16	2e-5
prop	ruletaker_roberta_tsra_seed42_10ep	-	$ROBERTA	42	10	-	-	-	ruletaker_gfair	external_baselines/GFaiR/data/ruletaker_3ext_sat	-	-	1.0	24	192	16	2e-5
prop	prontoqa_roberta_baseline_seed42_10ep	-	$ROBERTA	42	10	-	-	-	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	0.0	24	192	16	2e-5
prop	prontoqa_roberta_tsra_seed42_10ep	-	$ROBERTA	42	10	-	-	-	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	1.0	24	192	16	2e-5
prop	ruletaker_deberta_baseline_seed42_10ep	-	$DEBERTA	42	10	-	-	-	ruletaker_gfair	external_baselines/GFaiR/data/ruletaker_3ext_sat	-	-	0.0	24	192	16	2e-5
prop	ruletaker_deberta_tsra_seed42_10ep	-	$DEBERTA	42	10	-	-	-	ruletaker_gfair	external_baselines/GFaiR/data/ruletaker_3ext_sat	-	-	1.0	24	192	16	2e-5
prop	prontoqa_deberta_baseline_seed42_10ep	-	$DEBERTA	42	10	-	-	-	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	0.0	24	192	16	2e-5
prop	prontoqa_deberta_tsra_seed42_10ep	-	$DEBERTA	42	10	-	-	-	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	1.0	24	192	16	2e-5
EOFQ

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

run_clutrr() {
  local gpu="$1" name="$2" model_type="$3" model_path="$4" seed="$5" lambda_nexthop="$6" lambda_edge="$7" lambda_consistency="$8" batch_size="$9" dataset="${10}"
  local started="$STATUS_DIR/${name}.started" done="$STATUS_DIR/${name}.done" failed="$STATUS_DIR/${name}.failed" log_file="$LOG_DIR/${name}.log"
  {
    echo "kind=clutrr"; echo "task=$name"; echo "dataset=$dataset"; echo "gpu=$gpu"; echo "model_type=$model_type"; echo "model_path=$model_path"; echo "seed=$seed"; echo "epochs=10"; echo "lambda_nexthop=$lambda_nexthop"; echo "lambda_edge=$lambda_edge"; echo "lambda_consistency=$lambda_consistency"; echo "started_at=$(date '+%F %T')"
  } > "$started"
  (
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m clutrr.cli.train --config configs/clutrr/train_trua.yaml --dataset "$dataset" --root data --model_type "$model_type" --model_name_or_path "$model_path" --epochs 10 --batch_size "$batch_size" --eval_batch_size 32 --gpus "$gpu" --strategy single --seed "$seed" --lambda_nexthop "$lambda_nexthop" --lambda_edge "$lambda_edge" --lambda_consistency "$lambda_consistency" > "$log_file" 2>&1
    rc=$?
    if [[ "$rc" -eq 0 ]]; then { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 80 "$log_file"; } > "$done"; else { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 160 "$log_file"; } > "$failed"; fi
    exit "$rc"
  ) &
  log "Launched $name on GPU $gpu (pid=$!)"
}

run_prop() {
  local gpu="$1" name="$2" model_path="$3" seed="$4" dataset="$5" root="$6" train_depths="$7" test_depths="$8" lambda_trace="$9" max_sents="${10}" max_len="${11}" batch_size="${12}" lr="${13}"
  local started="$STATUS_DIR/${name}.started" done="$STATUS_DIR/${name}.done" failed="$STATUS_DIR/${name}.failed" log_file="$LOG_DIR/${name}.log" out_json="$RESULT_DIR/${name}.json"
  {
    echo "kind=prop"; echo "task=$name"; echo "dataset=$dataset"; echo "gpu=$gpu"; echo "model_path=$model_path"; echo "seed=$seed"; echo "epochs=10"; echo "lambda_trace=$lambda_trace"; echo "out=$out_json"; echo "started_at=$(date '+%F %T')"
  } > "$started"
  (
    set +e
    args=(scripts/transformer_trua_prop.py --dataset "$dataset" --root "$root" --limit-train 0 --limit-test 0 --epochs 10 --batch-size "$batch_size" --lr "$lr" --lambda-trace "$lambda_trace" --max-sents "$max_sents" --max-len "$max_len" --model-name "$model_path" --seed "$seed" --out "$out_json")
    [[ "$train_depths" != "-" ]] && args+=(--train-depths "$train_depths")
    [[ "$test_depths" != "-" ]] && args+=(--test-depths "$test_depths")
    CUDA_VISIBLE_DEVICES="$gpu" python3 "${args[@]}" > "$log_file" 2>&1
    rc=$?
    if [[ "$rc" -eq 0 ]]; then { cat "$started"; echo "finished_at=$(date '+%F %T')"; tail -n 100 "$log_file"; } > "$done"; else { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 180 "$log_file"; } > "$failed"; fi
    exit "$rc"
  ) &
  log "Launched $name on GPU $gpu (pid=$!)"
}

total_tasks=$(( $(wc -l < "$QUEUE") - 1 ))
log "Seed42 completion supervisor started. total_tasks=$total_tasks run_root=$RUN_ROOT gpus=$GPUS"

while [[ "$(dispatched_count)" -lt "$total_tasks" ]]; do
  gpu="$(free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    log "No free GPU among $GPUS below ${GPU_MEM_THRESHOLD_MB}MB; waiting."
    sleep 120
    continue
  fi
  launched=0
  while IFS=$'\t' read -r kind name model_type model_path seed epochs lambda_nexthop lambda_edge lambda_consistency dataset root train_depths test_depths lambda_trace max_sents max_len batch_size lr; do
    [[ "$kind" == "kind" ]] && continue
    [[ -f "$STATUS_DIR/${name}.started" ]] && continue
    if [[ "$kind" == "clutrr" ]]; then
      run_clutrr "$gpu" "$name" "$model_type" "$model_path" "$seed" "$lambda_nexthop" "$lambda_edge" "$lambda_consistency" "$batch_size" "$dataset"
    else
      run_prop "$gpu" "$name" "$model_path" "$seed" "$dataset" "$root" "$train_depths" "$test_depths" "$lambda_trace" "$max_sents" "$max_len" "$batch_size" "$lr"
    fi
    launched=1
    break
  done < "$QUEUE"
  [[ "$launched" -eq 0 ]] && log "No launchable task found; waiting."
  sleep 120
done

log "All seed42 completion queue entries have been dispatched."
