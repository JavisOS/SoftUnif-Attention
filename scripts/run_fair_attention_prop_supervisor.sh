#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/fair_attention_prop/fair_attention_prop_$(date +%Y%m%d_%H%M%S)"
QUEUE="$RUN_ROOT/queue.tsv"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
RESULT_DIR="$RUN_ROOT/results"
GPU_MEM_THRESHOLD_MB=1200

mkdir -p "$STATUS_DIR" "$LOG_DIR" "$RESULT_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/fair_attention_prop/latest

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

DEBERTA="/vepfs/tsra_models/hf/deberta-base"

cat > "$QUEUE" <<EOF
name	method	dataset	root	train_depths	test_depths	train_qdeps	test_qdeps	max_sents	max_len	batch_size	lr	seed	epochs
pw_dat_s0	dual_attention	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	-	-	16	192	16	2e-5	0	10
pw_dat_s1	dual_attention	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	-	-	16	192	16	2e-5	1	10
pw_dat_s42	dual_attention	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	-	-	16	192	16	2e-5	42	10
rt_dat_s0	dual_attention	ruletaker_raw	data/rule-reasoning-dataset-V2020.2.5.0/original	1,2	1,2,3,5	1,2	1,2,3,4,5	24	192	16	2e-5	0	10
rt_dat_s1	dual_attention	ruletaker_raw	data/rule-reasoning-dataset-V2020.2.5.0/original	1,2	1,2,3,5	1,2	1,2,3,4,5	24	192	16	2e-5	1	10
rt_dat_s42	dual_attention	ruletaker_raw	data/rule-reasoning-dataset-V2020.2.5.0/original	1,2	1,2,3,5	1,2	1,2,3,4,5	24	192	16	2e-5	42	10
pq_dat_s0	dual_attention	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	-	-	24	192	16	2e-5	0	10
pq_dat_s1	dual_attention	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	-	-	24	192	16	2e-5	1	10
pq_dat_s42	dual_attention	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	-	-	24	192	16	2e-5	42	10
pw_abs_s0	abstractor_rca	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	-	-	16	192	16	2e-5	0	10
pw_abs_s1	abstractor_rca	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	-	-	16	192	16	2e-5	1	10
pw_abs_s42	abstractor_rca	proofwriter	data/proofwriter/raw/proofwriter-dataset-V2020.12.3	0,1,2	3,5	-	-	16	192	16	2e-5	42	10
rt_abs_s0	abstractor_rca	ruletaker_raw	data/rule-reasoning-dataset-V2020.2.5.0/original	1,2	1,2,3,5	1,2	1,2,3,4,5	24	192	16	2e-5	0	10
rt_abs_s1	abstractor_rca	ruletaker_raw	data/rule-reasoning-dataset-V2020.2.5.0/original	1,2	1,2,3,5	1,2	1,2,3,4,5	24	192	16	2e-5	1	10
rt_abs_s42	abstractor_rca	ruletaker_raw	data/rule-reasoning-dataset-V2020.2.5.0/original	1,2	1,2,3,5	1,2	1,2,3,4,5	24	192	16	2e-5	42	10
pq_abs_s0	abstractor_rca	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	-	-	24	192	16	2e-5	0	10
pq_abs_s1	abstractor_rca	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	-	-	24	192	16	2e-5	1	10
pq_abs_s42	abstractor_rca	prontoqa	data/prontoqa_ood/processed/generated_ood_data	-	-	-	-	24	192	16	2e-5	42	10
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

run_task() {
  local gpu="$1" name="$2" method="$3" dataset="$4" root="$5" train_depths="$6" test_depths="$7"
  local train_qdeps="$8" test_qdeps="$9" max_sents="${10}" max_len="${11}" batch_size="${12}" lr="${13}" seed="${14}" epochs="${15}"
  local started="$STATUS_DIR/${name}.started"
  local done="$STATUS_DIR/${name}.done"
  local failed="$STATUS_DIR/${name}.failed"
  local log_file="$LOG_DIR/${name}.log"
  local out_json="$RESULT_DIR/${name}.json"

  {
    echo "name=$name"
    echo "method=$method"
    echo "dataset=$dataset"
    echo "gpu=$gpu"
    echo "seed=$seed"
    echo "epochs=$epochs"
    echo "started_at=$(date '+%F %T')"
  } > "$started"

  (
    set +e
    args=(
      scripts/fair_attention_prop.py
      --method "$method"
      --dataset "$dataset"
      --root "$root"
      --model-name "$DEBERTA"
      --limit-train 0
      --limit-test 0
      --epochs "$epochs"
      --batch-size "$batch_size"
      --lr "$lr"
      --max-sents "$max_sents"
      --max-len "$max_len"
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
      { cat "$started"; echo "finished_at=$(date '+%F %T')"; echo "out=$out_json"; tail -n 80 "$log_file"; } > "$done"
    else
      { cat "$started"; echo "failed_at=$(date '+%F %T')"; echo "exit_code=$rc"; tail -n 160 "$log_file"; } > "$failed"
    fi
    exit "$rc"
  ) &
  log "Launched $name on GPU $gpu (pid=$!)"
}

total_tasks=$(( $(wc -l < "$QUEUE") - 1 ))
log "Fair attention baseline supervisor started. total_tasks=$total_tasks run_root=$RUN_ROOT"

while [[ "$(dispatched_count)" -lt "$total_tasks" ]]; do
  gpu="$(free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    log "No free GPU below ${GPU_MEM_THRESHOLD_MB}MB; waiting."
    sleep 120
    continue
  fi

  launched=0
  while IFS=$'\t' read -r name method dataset root train_depths test_depths train_qdeps test_qdeps max_sents max_len batch_size lr seed epochs; do
    [[ "$name" == "name" ]] && continue
    [[ -f "$STATUS_DIR/${name}.started" ]] && continue
    run_task "$gpu" "$name" "$method" "$dataset" "$root" "$train_depths" "$test_depths" "$train_qdeps" "$test_qdeps" "$max_sents" "$max_len" "$batch_size" "$lr" "$seed" "$epochs"
    launched=1
    break
  done < "$QUEUE"

  [[ "$launched" -eq 0 ]] && log "No launchable task found; waiting."
  sleep 120
done

log "All fair attention baseline queue entries have been dispatched."
