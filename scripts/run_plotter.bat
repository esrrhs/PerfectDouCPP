@echo off
cd /d "N:\project\PerfectDouCPP"
title Eval Curve Plotter
C:\Python314\python.exe tools\plot_eval.py --csv ckpt_long\eval_vs_douzero.csv --out ckpt_long\eval_curve.png --watch 20
