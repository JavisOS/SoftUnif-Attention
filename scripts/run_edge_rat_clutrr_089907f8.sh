#!/usr/bin/env bash
set -euo pipefail

cd /root/TRUA

RUN_ROOT="/vepfs/tsra_outputs/official_external/edge_rat_clutrr_089907f8_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/status"
ln -sfn "$RUN_ROOT" /vepfs/tsra_outputs/official_external/latest_edge_rat_clutrr_089907f8

cd /root/TRUA/external_baselines/EdgeTransformer/clutrr
mkdir -p logs

(
  echo "started $(date -Is)"
  CUDA_VISIBLE_DEVICES=3 python train.py \
    --model_type edge_transformer \
    --data_path data_089907f8 \
    --epochs 50 \
    --batch_size 400 \
    --seed 42 \
    --log_file "$RUN_ROOT/logs/edge_transformer.csv"
  echo "done $(date -Is)"
) > "$RUN_ROOT/logs/edge_transformer.log" 2>&1 &
echo "$! edge_transformer gpu=3 log=$RUN_ROOT/logs/edge_transformer.log" | tee "$RUN_ROOT/edge_transformer.pid"
touch "$RUN_ROOT/status/edge_transformer.started"

(
  echo "started $(date -Is)"
  CUDA_VISIBLE_DEVICES=5 python train.py \
    --model_type rat \
    --data_path data_089907f8 \
    --epochs 50 \
    --batch_size 400 \
    --seed 42 \
    --log_file "$RUN_ROOT/logs/rat.csv"
  echo "done $(date -Is)"
) > "$RUN_ROOT/logs/rat.log" 2>&1 &
echo "$! rat gpu=5 log=$RUN_ROOT/logs/rat.log" | tee "$RUN_ROOT/rat.pid"
touch "$RUN_ROOT/status/rat.started"

echo "$RUN_ROOT"
