@echo off
setlocal EnableDelayedExpansion

set "PYTHON_EXE=C:\Users\igdal\anaconda3\envs\capstone-emg\python.exe"
set "GRAPHICS_MODE=normal"
set "SRC_ROOT=%~dp0.."
set "EXTRA_ARGS="

if /I "%~1"=="low" (
    set "GRAPHICS_MODE=low"
    shift
) else if /I "%~1"=="normal" (
    set "GRAPHICS_MODE=normal"
    shift
)

:collect_args
if "%~1"=="" goto run_client
set "EXTRA_ARGS=!EXTRA_ARGS! "%~1""
shift
goto collect_args

:run_client

if not exist "%PYTHON_EXE%" (
    echo capstone-emg python not found at "%PYTHON_EXE%"
    pause
    exit /b 1
)

set SDL_VIDEO_FULLSCREEN_DISPLAY=2
pushd "%SRC_ROOT%"
"%PYTHON_EXE%" carla_integration\manual_control_emg.py --host 127.0.0.1 --port 2000 --graphics %GRAPHICS_MODE% --client-fps 30 --res 1920x1080 --show-hud --map Town02_Opt !EXTRA_ARGS!
set "CLIENT_EXIT=%ERRORLEVEL%"
popd

echo.
echo manual_control_emg exit code: %CLIENT_EXIT%
echo Usage: test_run_manual_control_emg_0_9_16.bat [normal^|low]
pause
exit /b %CLIENT_EXIT%

endlocal
