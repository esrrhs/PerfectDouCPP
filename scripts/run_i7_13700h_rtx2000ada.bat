@echo off
rem ==============================================================================
rem PerfectDou CPP - Dedicated Launch Script
rem Hardware Configuration: Intel Core i7-13700H (14C/20T) + NVIDIA RTX 2000 Ada Laptop (8GB)
rem Benchmarked Optimal Parameters (hidden 256 / lstm 128, epochs 2, CUDA):
rem   --backend cuda      : GPU PPO update; rollout inference stays on CPU copies
rem   --threads 16        : 16 CPU rollout threads (20 was no faster on this laptop)
rem   --games 2048        : games per update (peak throughput; default 768 is slower)
rem   --mb 2048           : minibatch size (4096 was not faster together with 2048 games)
rem   OMP/MKL threads 1   : each rollout thread must not start its own LibTorch pool
rem Measured on a cold policy: ~165 games/s, ~9900 samples/s. Shorter games later
rem raise games/s without raising samples/s.
rem ==============================================================================

setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

set "TORCH_LIB=D:\deps\libtorch\lib"
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin"
set "PATH=%TORCH_LIB%;%CUDA_BIN%;%PATH%"
set "PY=build-win\douzero-venv\Scripts\python.exe"
set "TRAIN=build-msvc\Release\perfectdou_train.exe"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "PD_CPU_INFER="
set "PD_CHECK_INFER="

echo [PerfectDou] Starting dedicated high-throughput training pipeline...
echo [Hardware] CPU: Intel Core i7-13700H (16 rollout threads)
echo [Hardware] GPU: NVIDIA RTX 2000 Ada Laptop (CUDA LibTorch acceleration)
echo.

start "PerfectDou - DouZero Eval Server" cmd /k "title DouZero Server && %PY% tools\run_douzero_serve.py"
timeout /t 3 /nobreak >nul

start "PerfectDou - Curve Plotter" cmd /k "title Eval Curve Plotter && %PY% tools\plot_eval.py --csv ckpt_long\eval_vs_douzero.csv --out ckpt_long\eval_curve.png --watch 20"

start "PerfectDou - Training" cmd /k "title PerfectDou Training && %TRAIN% --backend cuda --threads 16 --games 2048 --mb 2048 --out ckpt_long %*"

echo [PerfectDou] All components launched in separate titled windows:
echo   1. DouZero Eval Server (TCP port 18765)
echo   2. Eval Curve Plotter (monitoring ckpt_long\eval_vs_douzero.csv)
echo   3. PerfectDou Training (CUDA, threads=16, games=2048, mb=2048)
echo.
endlocal
