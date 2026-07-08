#!/usr/bin/env bash
set -u

ROOT=/root/TRUA
OUT=/vepfs/tsra_outputs/trua_algorithm_explore
LOG="$OUT/logs"
PW_ROOT="$ROOT/data/proofwriter/raw/proofwriter-dataset-V2020.12.3"
ROBERTA=/vepfs/tsra_models/hf/roberta-base-clean-20260520_172733
mkdir -p "$LOG"

GPU_IDS=(6 7)
declare -a PIDS
declare -a NAMES

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }
is_running() { local pid="$1"; [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; }

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

launch_task() {
  local slot="$1" gpu="$2" variant="$3" seed="$4"
  local name="pw_${variant}_roberta_100k10ep_s${seed}"
  local out_file="$OUT/${name}.json"
  local log_file="$LOG/${name}.log"

  if [[ -s "$out_file" ]]; then
    echo "$(timestamp) SKIP existing $name"
    return 0
  fi

  if [[ "$variant" == "single" ]]; then
    echo "$(timestamp) START gpu=$gpu $name"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 -u "$ROOT/experiments/trua_algorithm_explore/prop_multihop_trua.py" \
      --dataset proofwriter --root "$PW_ROOT" --model-name "$ROBERTA" \
      --reasoner single --num-glimpses 4 \
      --limit-train 100000 --limit-test 10000 --epochs 10 --batch-size 8 --seed "$seed" \
      --lambda-trace 1.0 --lambda-trace-coverage 0.25 --lambda-offtrace 0.05 --coverage-penalty 0.2 \
      --train-depths 0,1,2 --test-depths 3,5 \
      --out "$out_file" > "$log_file" 2>&1 &
  else
    echo "$(timestamp) START gpu=$gpu $name"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python3 -u "$ROOT/experiments/trua_algorithm_explore/prop_bottleneck_trua.py" \
      --dataset proofwriter --root "$PW_ROOT" --model-name "$ROBERTA" \
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
  PIDS[$slot]="$pid"
  NAMES[$slot]="$name"
}

TASKS=()
for variant in single bt_soft; do
  for seed in 0 1 42; do
    TASKS+=("$variant:$seed")
  done
done

echo "$(timestamp) ROBERTA QUEUE start total=${#TASKS[@]} gpus=${GPU_IDS[*]}"
for task in "${TASKS[@]}"; do
  IFS=: read -r variant seed <<< "$task"
  idx=$(wait_for_slot)
  gpu="${GPU_IDS[$idx]}"
  launch_task "$idx" "$gpu" "$variant" "$seed"
  sleep 5
done

echo "$(timestamp) All RoBERTa tasks launched; waiting"
for i in "${!GPU_IDS[@]}"; do
  pid="${PIDS[$i]:-}"
  name="${NAMES[$i]:-}"
  if is_running "$pid"; then
    echo "$(timestamp) WAIT gpu=${GPU_IDS[$i]} pid=$pid $name"
    wait "$pid"
    echo "$(timestamp) DONE gpu=${GPU_IDS[$i]} pid=$pid $name exit=$?"
  fi
done
echo "$(timestamp) ROBERTA QUEUE done"
