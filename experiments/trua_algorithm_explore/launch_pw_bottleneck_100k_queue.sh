#!/usr/bin/env bash
set -u

ROOT=/root/TRUA
OUT=/vepfs/tsra_outputs/trua_algorithm_explore
LOG="$OUT/logs"
PW_ROOT="$ROOT/data/proofwriter/raw/proofwriter-dataset-V2020.12.3"
BERT=/vepfs/tsra_models/hf/bert-base-uncased
ROBERTA=/vepfs/tsra_models/hf/roberta-base
mkdir -p "$LOG"

MAX_GPUS=8
GPU_IDS=(0 1 2 3 4 5 6 7)
declare -a PIDS
declare -a NAMES

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

is_running() {
  local pid="$1"
  [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null
}

free_gpu_index() {
  local i pid
  for i in "${!GPU_IDS[@]}"; do
    pid="${PIDS[$i]:-}"
    if ! is_running "$pid"; then
      echo "$i"
      return 0
    fi
  done
  return 1
}

launch_task() {
  local gpu="$1" backbone="$2" model="$3" variant="$4" seed="$5"
  local name="pw_${variant}_${backbone}_100k10ep_s${seed}"
  local out_file="$OUT/${name}.json"
  local log_file="$LOG/${name}.log"

  if [[ -s "$out_file" ]]; then
    echo "$(timestamp) SKIP existing $name"
    return 0
  fi

  if [[ "$variant" == "single" ]]; then
    echo "$(timestamp) START gpu=$gpu $name"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 -u "$ROOT/experiments/trua_algorithm_explore/prop_multihop_trua.py" \
      --dataset proofwriter --root "$PW_ROOT" --model-name "$model" \
      --reasoner single --num-glimpses 4 \
      --limit-train 100000 --limit-test 10000 --epochs 10 --batch-size 8 --seed "$seed" \
      --lambda-trace 1.0 --lambda-trace-coverage 0.25 --lambda-offtrace 0.05 --coverage-penalty 0.2 \
      --train-depths 0,1,2 --test-depths 3,5 \
      --out "$out_file" > "$log_file" 2>&1 &
  else
    echo "$(timestamp) START gpu=$gpu $name"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 -u "$ROOT/experiments/trua_algorithm_explore/prop_bottleneck_trua.py" \
      --dataset proofwriter --root "$PW_ROOT" --model-name "$model" \
      --num-glimpses 4 \
      --limit-train 100000 --limit-test 10000 --epochs 10 --batch-size 8 --seed "$seed" \
      --text-residual-weight 0.2 --bypass-dropout 0.5 --lambda-reason-ce 1.0 --lambda-text-ce 0.0 \
      --lambda-trace 1.0 --lambda-trace-coverage 0.25 --lambda-offtrace 0.05 --coverage-penalty 0.2 \
      --train-depths 0,1,2 --test-depths 3,5 \
      --out "$out_file" > "$log_file" 2>&1 &
  fi

  local pid=$!
  echo "$pid" > "$LOG/${name}.pid"
  echo "$(timestamp) PID $pid $name"
  PIDS[$gpu]="$pid"
  NAMES[$gpu]="$name"
}

wait_for_slot() {
  local idx
  while true; do
    if idx=$(free_gpu_index); then
      echo "$idx"
      return 0
    fi
    sleep 60
  done
}

TASKS=()
for backbone in bert roberta; do
  for variant in single bt_soft; do
    for seed in 0 1 42; do
      TASKS+=("$backbone:$variant:$seed")
    done
  done
done

echo "$(timestamp) QUEUE start total=${#TASKS[@]} max_gpus=$MAX_GPUS"
for task in "${TASKS[@]}"; do
  IFS=: read -r backbone variant seed <<< "$task"
  if [[ "$backbone" == "bert" ]]; then
    model="$BERT"
  else
    model="$ROBERTA"
  fi
  idx=$(wait_for_slot)
  gpu="${GPU_IDS[$idx]}"
  launch_task "$gpu" "$backbone" "$model" "$variant" "$seed"
  sleep 5
done

echo "$(timestamp) All tasks launched; waiting for remaining jobs"
for i in "${!GPU_IDS[@]}"; do
  pid="${PIDS[$i]:-}"
  name="${NAMES[$i]:-}"
  if is_running "$pid"; then
    echo "$(timestamp) WAIT gpu=${GPU_IDS[$i]} pid=$pid $name"
    wait "$pid"
    echo "$(timestamp) DONE gpu=${GPU_IDS[$i]} pid=$pid $name exit=$?"
  fi
done
echo "$(timestamp) QUEUE done"
