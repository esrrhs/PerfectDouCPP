@echo off
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"
cd /d "N:\project\PerfectDouCPP"

echo ========================================================
echo Running Benchmark Suite to Maximize Training Throughput
echo ========================================================

echo.
echo [Test 1/4] Baseline: threads=8, games=512, mb=2048, buffer=2
echo --------------------------------------------------------
.\build\perfectdou_train.exe --updates 5 --games 512 --threads 8 --mb 2048 --buffer 2 --epochs 2 --out ckpt_bench_1 --no-eval > logs\bench_test1.log 2>&1

echo.
echo [Test 2/4] More Threads: threads=16, games=512, mb=2048, buffer=2
echo --------------------------------------------------------
.\build\perfectdou_train.exe --updates 5 --games 512 --threads 16 --mb 2048 --buffer 2 --epochs 2 --out ckpt_bench_2 --no-eval > logs\bench_test2.log 2>&1

echo.
echo [Test 3/4] Deep Buffer: threads=16, games=512, mb=2048, buffer=4
echo --------------------------------------------------------
.\build\perfectdou_train.exe --updates 5 --games 512 --threads 16 --mb 2048 --buffer 4 --epochs 2 --out ckpt_bench_3 --no-eval > logs\bench_test3.log 2>&1

echo.
echo [Test 4/4] Larger Batch: threads=16, games=768, mb=2048, buffer=4
echo --------------------------------------------------------
.\build\perfectdou_train.exe --updates 5 --games 768 --threads 16 --mb 2048 --buffer 4 --epochs 2 --out ckpt_bench_4 --no-eval > logs\bench_test4.log 2>&1

echo.
echo ========================================================
echo Benchmark Completed! Cleaning up benchmark checkpoints...
echo ========================================================
rd /s /q ckpt_bench_1 ckpt_bench_2 ckpt_bench_3 ckpt_bench_4 >nul 2>&1
echo Done!
