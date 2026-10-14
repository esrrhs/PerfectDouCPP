#!/bin/bash
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Stopping PerfectDou training..."
pkill -f "build/perfectdou_train" || true

echo "Stopping eval curve plotter..."
pkill -f "tools/plot_eval.py" || true

echo "Stopping DouZero eval server..."
pkill -f "tools/run_douzero_serve.py" || true

echo "All services stopped."
