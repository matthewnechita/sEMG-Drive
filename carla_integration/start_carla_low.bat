@echo off
setlocal

set "CARLA_ROOT=C:\Users\matth\Desktop\CARLA_0.9.15"

if not exist "%CARLA_ROOT%\CarlaUE4.exe" (
    echo CARLA executable not found at "%CARLA_ROOT%\CarlaUE4.exe"
    exit /b 1
)

echo Starting CARLA from "%CARLA_ROOT%" in low-quality mode...
start "" "%CARLA_ROOT%\CarlaUE4.exe" -quality-level=Low %*

endlocal
