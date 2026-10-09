#!/usr/bin/env python3
"""Periodic PerfectDou vs DouZero-ADP evaluation (CPU).

Every INTERVAL minutes: copy actors from the training output dir, play N random
decks against DouZero-ADP (each deck twice with role swap), append WP/ADP to
CSV, and refresh a trend plot with a polynomial fit.

Setup on a new machine
----------------------
1. Build this repo (Windows needs perfectdou_eval.exe).
2. Clone DouZero next to the repo or pass --douzero-root:
     git clone --depth 1 https://github.com/kwai/DouZero.git third_party/DouZero
3. Place ADP weights under baselines/douzero_ADP/:
     landlord.ckpt  landlord_up.ckpt  landlord_down.ckpt
   Mirror: https://huggingface.co/palemoky/douzero-baselines
   (checkpoints/douzero_ADP/)
4. Python deps (CPU torch is enough; keep CUDA_VISIBLE_DEVICES empty):
     pip install torch numpy matplotlib
     # torch CPU wheel, e.g. --index-url https://download.pytorch.org/whl/cpu

Example (while training writes to ckpt_long/)
--------------------------------------------
  # terminal A: train
  ./build/perfectdou_train --backend cuda --updates 1000000 \\
      --games 256 --threads 8 --snapshot-every 5 --out ckpt_long

  # terminal B: eval loop
  set PYTHONPATH=third_party/DouZero
  python tools/eval_vs_douzero_loop.py \\
      --train-dir ckpt_long --out-dir eval_runs \\
      --interval-sec 600 --decks 100

Outputs: eval_runs/vs_douzero.csv  eval_runs/vs_douzero.png
         eval_runs/snapshots/<label>/actor{0,1,2}.bin
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def wait_models(train_dir: Path, timeout: float = 3600) -> None:
    need = [train_dir / f"actor{i}.bin" for i in range(3)]
    t0 = time.time()
    while True:
        if all(p.is_file() and p.stat().st_size > 0 for p in need):
            return
        if time.time() - t0 > timeout:
            raise TimeoutError(f"models not ready in {train_dir}")
        time.sleep(2)


def copy_models(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in [f"actor{i}.bin" for i in range(3)] + [f"critic{i}.bin" for i in range(3)]:
        s = src / name
        if s.is_file():
            shutil.copy2(s, dst / name)


def plot_curve(csv_path: Path, out_png: Path) -> None:
    if not csv_path.is_file():
        return
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    (
                        float(row["minutes"]),
                        float(row["wp"]),
                        float(row["adp"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    if len(rows) < 1:
        return
    rows.sort(key=lambda r: r[0])
    x = np.array([r[0] for r in rows], dtype=float)
    wp = np.array([r[1] for r in rows], dtype=float)
    adp = np.array([r[2] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, y, title, ylabel in (
        (axes[0], wp, "Win rate vs DouZero-ADP", "WP"),
        (axes[1], adp, "ADP vs DouZero-ADP", "ADP"),
    ):
        ax.plot(x, y, "o-", color="#1f77b4", label="eval", markersize=5)
        if len(x) >= 2:
            deg = 1 if len(x) < 4 else min(2, len(x) - 1)
            coef = np.polyfit(x, y, deg)
            xx = np.linspace(x.min(), x.max(), 200)
            ax.plot(xx, np.polyval(coef, xx), "--", color="#d62728", label=f"fit deg={deg}")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
    axes[1].set_xlabel("minutes since train start")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def run_one_eval(
    *,
    models: Path,
    decks: int,
    port: int,
    label: str,
    eval_csv: Path,
    douzero_ckpt: Path,
    eval_bin: Path,
    serve_py: Path,
    douzero_root: Path,
) -> tuple[float, float]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(douzero_root) + os.pathsep + env.get("PYTHONPATH", "")
    # Serve listens then blocks on accept; start it first.
    serve = subprocess.Popen(
        [
            sys.executable,
            str(serve_py),
            "--port",
            str(port),
            "--ckpt-dir",
            str(douzero_ckpt),
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        # Wait until server prints ready
        ready = False
        assert serve.stdout is not None
        t0 = time.time()
        while time.time() - t0 < 120:
            line = serve.stdout.readline()
            if not line and serve.poll() is not None:
                break
            if "ready" in line:
                ready = True
                break
        if not ready:
            out = serve.stdout.read() if serve.stdout else ""
            raise RuntimeError(f"DouZero server failed to start:\n{out}")

        eval_csv.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(eval_bin),
            "--models",
            str(models),
            "--decks",
            str(decks),
            "--port",
            str(port),
            "--label",
            label,
            "--csv",
            str(eval_csv),
            "--seed",
            str(int(time.time()) & 0x7FFFFFFF),
        ]
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"perfectdou_eval failed ({proc.returncode})")
    finally:
        if serve.poll() is None:
            serve.terminate()
            try:
                serve.wait(timeout=10)
            except subprocess.TimeoutExpired:
                serve.kill()

    # Read last CSV row for wp/adp
    with eval_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError("eval csv empty")
    last = rows[-1]
    return float(last["wp"]), float(last["adp"])


def append_summary(
    summary_csv: Path,
    *,
    minutes: float,
    label: str,
    decks: int,
    wp: float,
    adp: float,
) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not summary_csv.is_file() or summary_csv.stat().st_size == 0
    with summary_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["minutes", "timestamp", "label", "decks", "wp", "adp"])
        w.writerow(
            [
                f"{minutes:.2f}",
                datetime.now().isoformat(timespec="seconds"),
                label,
                decks,
                f"{wp:.4f}",
                f"{adp:.4f}",
            ]
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", default=str(ROOT / "ckpt_long"))
    p.add_argument("--out-dir", default=str(ROOT / "eval_runs"))
    p.add_argument("--interval-sec", type=int, default=600)
    p.add_argument("--decks", type=int, default=100)
    p.add_argument("--port", type=int, default=18765)
    p.add_argument(
        "--douzero-ckpt",
        default=str(ROOT / "baselines" / "douzero_ADP"),
    )
    p.add_argument("--eval-bin", default=str(ROOT / "build" / "perfectdou_eval.exe"))
    p.add_argument("--serve-py", default=str(ROOT / "tools" / "douzero_serve.py"))
    p.add_argument(
        "--douzero-root", default=str(ROOT / "third_party" / "DouZero")
    )
    args = p.parse_args()

    train_dir = Path(args.train_dir)
    out_dir = Path(args.out_dir)
    summary_csv = out_dir / "vs_douzero.csv"
    plot_png = out_dir / "vs_douzero.png"
    detail_csv = out_dir / "eval_detail.csv"

    print(f"waiting for models in {train_dir} ...", flush=True)
    wait_models(train_dir)
    t_start = time.time()
    round_i = 0
    print("models ready; starting eval loop", flush=True)

    while True:
        round_i += 1
        minutes = (time.time() - t_start) / 60.0
        label = f"t{int(minutes):05d}_r{round_i:04d}"
        snap = out_dir / "snapshots" / label
        print(f"\n=== eval {label} (+{minutes:.1f} min) ===", flush=True)
        copy_models(train_dir, snap)
        try:
            wp, adp = run_one_eval(
                models=snap,
                decks=args.decks,
                port=args.port,
                label=label,
                eval_csv=detail_csv,
                douzero_ckpt=Path(args.douzero_ckpt),
                eval_bin=Path(args.eval_bin),
                serve_py=Path(args.serve_py),
                douzero_root=Path(args.douzero_root),
            )
            append_summary(
                summary_csv,
                minutes=minutes,
                label=label,
                decks=args.decks,
                wp=wp,
                adp=adp,
            )
            plot_curve(summary_csv, plot_png)
            print(
                f"recorded WP={wp:.4f} ADP={adp:.4f} -> {summary_csv} / {plot_png}",
                flush=True,
            )
        except Exception as e:
            print(f"eval failed: {e}", flush=True)

        # Align to wall-clock interval from loop start (eval time counts).
        next_at = t_start + round_i * args.interval_sec
        sleep_s = max(5.0, next_at - time.time())
        print(f"sleep {sleep_s:.0f}s until next eval", flush=True)
        time.sleep(sleep_s)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("stopped", flush=True)
        raise SystemExit(0)
