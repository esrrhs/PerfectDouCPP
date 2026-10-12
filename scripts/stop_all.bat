@echo off
echo Stopping all PerfectDou background processes...

taskkill /F /IM perfectdou_train.exe >nul 2>&1
wmic process where "name='python.exe' and (commandline like '%%run_douzero_serve.py%%' or commandline like '%%plot_eval.py%%')" delete >nul 2>&1

echo All background processes stopped.
