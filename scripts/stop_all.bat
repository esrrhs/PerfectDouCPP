@echo off
echo Stopping all PerfectDou background processes...

taskkill /F /IM perfectdou_train.exe >nul 2>&1
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*run_douzero_serve.py*' -or $_.CommandLine -like '*plot_eval.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }" >nul 2>&1

echo All background processes stopped.
