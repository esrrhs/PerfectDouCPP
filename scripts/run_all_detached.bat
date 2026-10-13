@echo off
set "PATH=C:\Users\zx\AppData\Roaming\Python\Python314\site-packages\torch\lib;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"
cd /d "N:\project\PerfectDouCPP"

start "PerfectDou - DouZero Eval Server" cmd /k "title DouZero Server && C:\Python314\python.exe tools\run_douzero_serve.py"
timeout /t 3 /nobreak >nul
start "PerfectDou - Curve Plotter" cmd /k "title Eval Curve Plotter && C:\Python314\python.exe tools\plot_eval.py --csv ckpt_long\eval_vs_douzero.csv --out ckpt_long\eval_curve.png --watch 20"
start "PerfectDou - Training" cmd /k "title PerfectDou Training && .\build\perfectdou_train.exe --backend cuda --threads 8 --buffer-chunks 2 --games 512 --mb 2048 --out ckpt_long"
