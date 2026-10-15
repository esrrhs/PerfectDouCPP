@echo off
rem ==============================================================================
rem PerfectDou CPP - Dedicated Launch Script
rem Hardware Configuration: Intel Core i7-6700K (4C/8T) + NVIDIA GeForce GTX 1070 (8GB)
rem Benchmarked Optimal Parameters:
rem   --backend cuda      : GPU tensor forward/backward & PPO optimization
rem   --threads 6         : 6 CPU rollout threads (leaves 2 threads for GPU/OS/TCP)
rem   --games 768         : Batch size per update chunk
rem   --mb 2048           : Minibatch size for maximal GEMM throughput (~68-100 games/s)
rem ==============================================================================

setlocal
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%.."

set "TORCH_LIB=C:\Users\zx\AppData\Roaming\Python\Python314\site-packages\torch\lib"
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
set "PATH=%TORCH_LIB%;%CUDA_BIN%;%PATH%"

echo [PerfectDou] Starting dedicated high-throughput training pipeline...
echo [Hardware] CPU: Intel Core i7-6700K (6 rollout threads)
echo [Hardware] GPU: NVIDIA GeForce GTX 1070 (CUDA LibTorch acceleration)
echo.

start "PerfectDou - DouZero Eval Server" cmd /k "title DouZero Server && C:\Python314\python.exe tools\run_douzero_serve.py"
timeout /t 3 /nobreak >nul

start "PerfectDou - Curve Plotter" cmd /k "title Eval Curve Plotter && C:\Python314\python.exe tools\plot_eval.py --csv ckpt_long\eval_vs_douzero.csv --out ckpt_long\eval_curve.png --watch 20"

start "PerfectDou - Training" cmd /k "title PerfectDou Training && .\build\perfectdou_train.exe --backend cuda --threads 6 --games 768 --mb 2048 --out ckpt_long %*"

echo [PerfectDou] All components launched in separate titled windows:
echo   1. DouZero Eval Server (TCP port 18765)
echo   2. Eval Curve Plotter (monitoring ckpt_long\eval_vs_douzero.csv)
echo   3. PerfectDou Training (CUDA, threads=6, games=768, mb=2048)
echo.
endlocal
