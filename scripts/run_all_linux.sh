#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p ckpt_long

# 1. Start DouZero Server
if ! pgrep -f "tools/run_douzero_serve.py" > /dev/null; then
    echo "Starting DouZero eval server..."
    nohup python3 tools/run_douzero_serve.py > douzero_serve.log 2>&1 &
    sleep 3
fi

# 2. Start Curve Plotter
if ! pgrep -f "tools/plot_eval.py" > /dev/null; then
    echo "Starting eval curve plotter..."
    nohup python3 tools/plot_eval.py --csv ckpt_long/eval_vs_douzero.csv --out ckpt_long/eval_curve.png --watch 60 > plotter.log 2>&1 &
fi

# 3. Check for resume
RESUME_FLAG=""
if [ -f "ckpt_long/meta.txt" ]; then
    echo "Found existing checkpoint in ckpt_long, resuming..."
    RESUME_FLAG="--resume ckpt_long"
fi

# 4. Start training
if ! pgrep -f "build/perfectdou_train" > /dev/null; then
    echo "Starting PerfectDou training..."
    nohup ./build/perfectdou_train \
        --backend cpu \
        --threads 2 \
        --games 128 \
        --mb 2048 \
        --updates 500000 \
        --eval-every 200 \
        --eval-decks 1000 \
        --eval-port 18765 \
        --eval-max-snapshots 20 \
        --snapshot-every 100 \
        --out ckpt_long \
        $RESUME_FLAG > train.log 2>&1 &
    echo "Training started in background (PID: $!). Log: train.log"
else
    echo "Training is already running!"
fi
