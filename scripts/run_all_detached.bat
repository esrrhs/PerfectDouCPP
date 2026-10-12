@echo off
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"
cd /d "N:\project\PerfectDouCPP"

start "DouZeroServer" /b python tools\run_douzero_serve.py >> logs\douzero_serve.log 2>&1
timeout /t 3 /nobreak >nul
start "EvalPlotter" /b python tools\plot_eval.py --csv ckpt_long\eval_vs_douzero.csv --out ckpt_long\eval_curve.png --watch 20 >> logs\plot_eval.log 2>&1
start "PerfectDouTrain" /b .\build\perfectdou_train.exe >> logs\train_cuda.log 2>> logs\train_cuda.err
