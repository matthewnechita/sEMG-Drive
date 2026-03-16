@echo off
setlocal

set "CARLA_ROOT=%~dp0..\..\CARLA_0.9.16"
set "CARLA_EXE=%CARLA_ROOT%\CarlaUE4.exe"
set "MAP_NAME=Town03"
set "QUALITY_LABEL=low"
set "QUALITY_ARG=-quality-level=Low"

if /I "%~1"=="medium" (
    set "QUALITY_LABEL=medium"
    set "QUALITY_ARG="
)

if /I "%~1"=="low" (
    set "QUALITY_LABEL=low"
    set "QUALITY_ARG=-quality-level=Low"
)

if not "%~2"=="" (
    set "MAP_NAME=%~2"
)

if not exist "%CARLA_EXE%" (
    echo CARLA executable not found at "%CARLA_EXE%"
    pause
    exit /b 1
)

echo Starting clean CARLA 0.9.16 on %MAP_NAME% with %QUALITY_LABEL% graphics...
start "CARLA 0.9.16" "%CARLA_EXE%" /Game/Carla/Maps/%MAP_NAME% %QUALITY_ARG% -ResX=960 -ResY=540 -windowed

echo Server launch command sent.
echo If SmartScreen appears, approve the unsigned executable manually.
echo Usage: start_clean_carla_server_0_9_16.bat [low^|medium] [Town01_Opt]
pause

endlocal
