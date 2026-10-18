@echo off
rem ==============================================================================
rem PerfectDou CPP - One-Click Resumption Script
rem Automatically resumes training from the latest checkpoint in ckpt_long/
rem ==============================================================================
echo [PerfectDou] Resuming training from ckpt_long...
call "%~dp0run_i7_6700k_gtx1070.bat" --resume ckpt_long %*
