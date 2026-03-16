@echo off
setlocal

set "PYTHON_EXE=C:\Users\igdal\anaconda3\envs\capstone-emg\python.exe"
set "GRAPHICS_MODE=low"
set "SRC_ROOT=%~dp0.."

if /I "%~1"=="medium" (
    set "GRAPHICS_MODE=normal"
    shift
) else if /I "%~1"=="low" (
    set "GRAPHICS_MODE=low"
    shift
)

if not exist "%PYTHON_EXE%" (
    echo capstone-emg python not found at "%PYTHON_EXE%"
    pause
    exit /b 1
)

pushd "%SRC_ROOT%"
"%PYTHON_EXE%" carla_integration\manual_control_emg.py --host 127.0.0.1 --port 2000 --graphics %GRAPHICS_MODE% --client-fps 30 --res 960x540 --camera-res 640x360 --camera-fps 10 %*
set "CLIENT_EXIT=%ERRORLEVEL%"
popd

echo.
echo manual_control_emg exit code: %CLIENT_EXIT%
echo Usage: run_manual_control_emg_0_9_16.bat [low^|medium]
pause
exit /b %CLIENT_EXIT%

endlocal
