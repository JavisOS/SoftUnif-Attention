#!/usr/bin/env bash
set -euo pipefail

cd /root/TSRA

RUN_ROOT="/vepfs/tsra_outputs/clutrr_backbone_sweep/clutrr_backbone_sweep_$(date +%Y%m%d_%H%M%S)"
QUEUE="$RUN_ROOT/queue.tsv"
STATUS_DIR="$RUN_ROOT/status"
LOG_DIR="$RUN_ROOT/logs"
GPU_MEM_THRESHOLD_MB="${TSRA_GPU_MEM_THRESHOLD_MB:-1000}"
GPUS="${TSRA_CLUTRR_SWEEP_GPUS:-4,5,6,7}"

mkdir -p "$STATUS_DIR" "$LOG_DIR"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/clutrr_backbone_sweep/latest

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

BERT="/vepfs/tsra_models/hf/bert-base-uncased"
ROBERTA="/vepfs/tsra_models/hf/roberta-base-clean-current"
DEBERTA_V3="/vepfs/tsra_models/hf/deberta-v3-base"
MODERNBERT="/vepfs/tsra_models/hf/modernbert-base"

cat > "$QUEUE" <<EOF
name	dataset	model_type	model_path	seed	variant	lambda_nexthop	lambda_edge	lambda_consistency	batch_size	eval_batch_size	epochs
clutrr_089_bert_label_only_seed0	data_089907f8	bert	$BERT	0	label_only	0.0	0.0	0.0	16	32	10
clutrr_089_bert_full_seed0	data_089907f8	bert	$BERT	0	full	1.0	1.0	5.0	16	32	10
clutrr_089_bert_label_only_seed1	data_089907f8	bert	$BERT	1	label_only	0.0	0.0	0.0	16	32	10
clutrr_089_bert_full_seed1	data_089907f8	bert	$BERT	1	full	1.0	1.0	5.0	16	32	10
clutrr_089_bert_label_only_seed42	data_089907f8	bert	$BERT	42	label_only	0.0	0.0	0.0	16	32	10
clutrr_089_bert_full_seed42	data_089907f8	bert	$BERT	42	full	1.0	1.0	5.0	16	32	10
clutrr_db9_bert_label_only_seed0	data_db9b8f04	bert	$BERT	0	label_only	0.0	0.0	0.0	16	32	10
clutrr_db9_bert_full_seed0	data_db9b8f04	bert	$BERT	0	full	1.0	1.0	5.0	16	32	10
clutrr_db9_bert_label_only_seed1	data_db9b8f04	bert	$BERT	1	label_only	0.0	0.0	0.0	16	32	10
clutrr_db9_bert_full_seed1	data_db9b8f04	bert	$BERT	1	full	1.0	1.0	5.0	16	32	10
clutrr_db9_bert_label_only_seed42	data_db9b8f04	bert	$BERT	42	label_only	0.0	0.0	0.0	16	32	10
clutrr_db9_bert_full_seed42	data_db9b8f04	bert	$BERT	42	full	1.0	1.0	5.0	16	32	10
clutrr_db9_roberta_label_only_seed0	data_db9b8f04	roberta	$ROBERTA	0	label_only	0.0	0.0	0.0	16	32	10
clutrr_db9_roberta_full_seed0	data_db9b8f04	roberta	$ROBERTA	0	full	1.0	1.0	5.0	16	32	10
clutrr_db9_roberta_label_only_seed1	data_db9b8f04	roberta	$ROBERTA	1	label_only	0.0	0.0	0.0	16	32	10
clutrr_db9_roberta_full_seed1	data_db9b8f04	roberta	$ROBERTA	1	full	1.0	1.0	5.0	16	32	10
clutrr_db9_roberta_label_only_seed42	data_db9b8f04	roberta	$ROBERTA	42	label_only	0.0	0.0	0.0	16	32	10
clutrr_db9_roberta_full_seed42	data_db9b8f04	roberta	$ROBERTA	42	full	1.0	1.0	5.0	16	32	10
clutrr_089_deberta_v3_label_only_seed0	data_089907f8	deberta-v3	$DEBERTA_V3	0	label_only	0.0	0.0	0.0	16	32	10
clutrr_089_deberta_v3_full_seed0	data_089907f8	deberta-v3	$DEBERTA_V3	0	full	1.0	1.0	5.0	16	32	10
clutrr_089_deberta_v3_label_only_seed1	data_089907f8	deberta-v3	$DEBERTA_V3	1	label_only	0.0	0.0	0.0	16	32	10
clutrr_089_deberta_v3_full_seed1	data_089907f8	deberta-v3	$DEBERTA_V3	1	full	1.0	1.0	5.0	16	32	10
clutrr_089_deberta_v3_label_only_seed42	data_089907f8	deberta-v3	$DEBERTA_V3	42	label_only	0.0	0.0	0.0	16	32	10
clutrr_089_deberta_v3_full_seed42	data_089907f8	deberta-v3	$DEBERTA_V3	42	full	1.0	1.0	5.0	16	32	10
clutrr_db9_deberta_v3_label_only_seed0	data_db9b8f04	deberta-v3	$DEBERTA_V3	0	label_only	0.0	0.0	0.0	16	32	10
clutrr_db9_deberta_v3_full_seed0	data_db9b8f04	deberta-v3	$DEBERTA_V3	0	full	1.0	1.0	5.0	16	32	10
clutrr_db9_deberta_v3_label_only_seed1	data_db9b8f04	deberta-v3	$DEBERTA_V3	1	label_only	0.0	0.0	0.0	16	32	10
clutrr_db9_deberta_v3_full_seed1	data_db9b8f04	deberta-v3	$DEBERTA_V3	1	full	1.0	1.0	5.0	16	32	10
clutrr_db9_deberta_v3_label_only_seed42	data_db9b8f04	deberta-v3	$DEBERTA_V3	42	label_only	0.0	0.0	0.0	16	32	10
clutrr_db9_deberta_v3_full_seed42	data_db9b8f04	deberta-v3	$DEBERTA_V3	42	full	1.0	1.0	5.0	16	32	10
clutrr_089_modernbert_label_only_seed0	data_089907f8	modernbert	$MODERNBERT	0	label_only	0.0	0.0	0.0	16	32	10
clutrr_089_modernbert_full_seed0	data_089907f8	modernbert	$MODERNBERT	0	full	1.0	1.0	5.0	16	32	10
clutrr_089_modernbert_label_only_seed1	data_089907f8	modernbert	$MODERNBERT	1	label_only	0.0	0.0	0.0	16	32	10
clutrr_089_modernbert_full_seed1	data_089907f8	modernbert	$MODERNBERT	1	full	1.0	1.0	5.0	16	32	10
clutrr_089_modernbert_label_only_seed42	data_089907f8	modernbert	$MODERNBERT	42	label_only	0.0	0.0	0.0	16	32	10
clutrr_089_modernbert_full_seed42	data_089907f8	modernbert	$MODERNBERT	42	full	1.0	1.0	5.0	16	32	10
clutrr_db9_modernbert_label_only_seed0	data_db9b8f04	modernbert	$MODERNBERT	0	label_only	0.0	0.0	0.0	16	32	10
clutrr_db9_modernbert_full_seed0	data_db9b8f04	modernbert	$MODERNBERT	0	full	1.0	1.0	5.0	16	32	10
clutrr_db9_modernbert_label_only_seed1	data_db9b8f04	modernbert	$MODERNBERT	1	label_only	0.0	0.0	0.0	16	32	10
clutrr_db9_modernbert_full_seed1	data_db9b8f04	modernbert	$MODERNBERT	1	full	1.0	1.0	5.0	16	32	10
clutrr_db9_modernbert_label_only_seed42	data_db9b8f04	modernbert	$MODERNBERT	42	label_only	0.0	0.0	0.0	16	32	10
clutrr_db9_modernbert_full_seed42	data_db9b8f04	modernbert	$MODERNBERT	42	full	1.0	1.0	5.0	16	32	10
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
  local gpu="$1" name="$2" dataset="$3" model_type="$4" model_path="$5" seed="$6" variant="$7"
  local lambda_nexthop="$8" lambda_edge="$9" lambda_consistency="${10}" batch_size="${11}" eval_batch_size="${12}" epochs="${13}"
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
    echo "lambda_nexthop=$lambda_nexthop"
    echo "lambda_edge=$lambda_edge"
    echo "lambda_consistency=$lambda_consistency"
    echo "started_at=$(date '+%F %T')"
  } > "$started"

  (
    set +e
    if [[ ! -d "$model_path" ]]; then
      echo "Missing model_path: $model_path" > "$log_file"
      rc=66
    else
      CUDA_VISIBLE_DEVICES="$gpu" python -u -m clutrr.cli.train \
        --config configs/clutrr/train_tsra.yaml \
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
        --lambda_nexthop "$lambda_nexthop" \
        --lambda_edge "$lambda_edge" \
        --lambda_consistency "$lambda_consistency" \
        > "$log_file" 2>&1
      rc=$?
    fi

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
log "CLUTRR backbone sweep started. total_tasks=$total_tasks run_root=$RUN_ROOT gpus=$GPUS"

while [[ "$(dispatched_count)" -lt "$total_tasks" ]]; do
  gpu="$(free_gpu || true)"
  if [[ -z "$gpu" ]]; then
    log "No free GPU among $GPUS below ${GPU_MEM_THRESHOLD_MB}MB; waiting."
    sleep 120
    continue
  fi

  launched=0
  while IFS=$'\t' read -r name dataset model_type model_path seed variant lambda_nexthop lambda_edge lambda_consistency batch_size eval_batch_size epochs; do
    [[ "$name" == "name" ]] && continue
    [[ -f "$STATUS_DIR/${name}.started" ]] && continue
    run_task "$gpu" "$name" "$dataset" "$model_type" "$model_path" "$seed" "$variant" "$lambda_nexthop" "$lambda_edge" "$lambda_consistency" "$batch_size" "$eval_batch_size" "$epochs"
    launched=1
    break
  done < "$QUEUE"

  [[ "$launched" -eq 0 ]] && log "No launchable task found; waiting."
  sleep 120
done

log "All CLUTRR backbone sweep tasks have been dispatched."
