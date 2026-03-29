@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

call "%SCRIPT_DIR%test_run_manual_control_emg_0_9_16.bat" normal --scenario highway_overtake --eval-log-dir eval_metrics\out\highway_overtake_eval %*
set "CLIENT_EXIT=%ERRORLEVEL%"

endlocal & exit /b %CLIENT_EXIT%
