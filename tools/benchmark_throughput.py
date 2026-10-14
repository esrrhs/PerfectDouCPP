import subprocess
import time
import re
import os
import shutil
from pathlib import Path

# Candidates to benchmark on CUDA GPU with default model size (hidden=256, lstm=128)
# Environment: Intel i7-6700K (4 cores, 8 threads), NVIDIA GTX 1070
CONFIGS = [
    {"name": "M1: games=256, th=4, mb=2048",  "games": 256,  "threads": 4,  "mb": 2048},
    {"name": "M2: games=256, th=8, mb=2048",  "games": 256,  "threads": 8,  "mb": 2048},
    {"name": "M3: games=512, th=6, mb=2048",  "games": 512,  "threads": 6,  "mb": 2048},
    {"name": "M4: games=512, th=8, mb=2048",  "games": 512,  "threads": 8,  "mb": 2048},
    {"name": "M5: games=512, th=8, mb=4096",  "games": 512,  "threads": 8,  "mb": 4096},
    {"name": "M6: games=768, th=6, mb=2048",  "games": 768,  "threads": 6,  "mb": 2048},
    {"name": "M7: games=768, th=8, mb=2048",  "games": 768,  "threads": 8,  "mb": 2048},
    {"name": "M8: games=1024, th=8, mb=2048", "games": 1024, "threads": 8,  "mb": 2048},
]

TORCH_LIB = r"C:\Users\zx\AppData\Roaming\Python\Python314\site-packages\torch\lib"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
env = os.environ.copy()
env["PATH"] = f"{TORCH_LIB};{CUDA_BIN};" + env.get("PATH", "")

def run_bench(cfg, updates=2):
    out_dir = Path(f"ckpt_bench_tmp_{cfg['games']}_{cfg['threads']}_{cfg['mb']}")
    cmd = [
        r".\build\perfectdou_train.exe",
        "--backend", "cuda",
        "--updates", str(updates),
        "--games", str(cfg["games"]),
        "--threads", str(cfg["threads"]),
        "--mb", str(cfg["mb"]),
        "--epochs", "2",
        "--out", str(out_dir),
        "--no-eval"
    ]
    t0 = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=r"N:\project\PerfectDouCPP")
    elapsed = time.time() - t0
    
    # Clean up checkpoint
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
        
    # Parse wall time and learn time from log
    lines = res.stdout.strip().split("\n")
    wall_times = []
    learn_times = []
    for line in lines:
        m = re.search(r"wall\s+([\d\.]+)s\s+learn\s+([\d\.]+)s", line)
        if m:
            wall_times.append(float(m.group(1)))
            learn_times.append(float(m.group(2)))
            
    if not wall_times:
        print(f"Error in {cfg['name']}: {res.stderr[:200]}")
        return None
        
    avg_wall = sum(wall_times) / len(wall_times)
    avg_learn = sum(learn_times) / len(learn_times)
    games_per_sec = cfg["games"] / avg_wall
    return {
        "cfg": cfg,
        "avg_wall": avg_wall,
        "avg_learn": avg_learn,
        "games_per_sec": games_per_sec,
        "total_elapsed": elapsed
    }

def main():
    print("=" * 80)
    print("Clean-Room CUDA Benchmark Suite (Default Model: hidden=256, lstm=128)")
    print("=" * 80)
    results = []
    for i, cfg in enumerate(CONFIGS, 1):
        print(f"\n[{i}/{len(CONFIGS)}] Testing: {cfg['name']} ...", flush=True)
        r = run_bench(cfg, updates=2)
        if r:
            print(f"   -> Wall/upd: {r['avg_wall']:.2f}s | Learn: {r['avg_learn']:.2f}s | Throughput: {r['games_per_sec']:.2f} games/s")
            results.append(r)
            
    print("\n" + "=" * 80)
    print("BENCHMARK FINAL SUMMARY & COMPARISON")
    print("=" * 80)
    print(f"{'Config Name':<38} | {'Wall/Upd':<10} | {'Learn':<10} | {'Throughput (games/s)'}")
    print("-" * 80)
    best = None
    for r in results:
        print(f"{r['cfg']['name']:<38} | {r['avg_wall']:>8.2f}s | {r['avg_learn']:>8.2f}s | {r['games_per_sec']:>12.2f} games/s")
        if best is None or r['games_per_sec'] > best['games_per_sec']:
            best = r
            
    print("=" * 80)
    print(f"Optimal Configuration: {best['cfg']['name']}")
    print(f"Maximum Throughput:    {best['games_per_sec']:.2f} games/second")
    print(f"Optimal Wall Time/Upd: {best['avg_wall']:.2f} seconds")
    print("=" * 80)

if __name__ == "__main__":
    main()
