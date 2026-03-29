@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

call "%SCRIPT_DIR%test_run_manual_control_emg_0_9_16.bat" normal --scenario lane_keep_5min --eval-log-dir eval_metrics\out\lane_keep_eval %*
set "CLIENT_EXIT=%ERRORLEVEL%"

endlocal & exit /b %CLIENT_EXIT%
