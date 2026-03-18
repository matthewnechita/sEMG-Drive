@echo off
setlocal

set "CARLA_ROOT=%~dp0..\..\CARLA_0.9.16"
set "CARLA_EXE=%CARLA_ROOT%\CarlaUE4.exe"
set "MAP_NAME=Town02_Opt"
set "QUALITY_LABEL=normal"
set "QUALITY_ARG="

if /I "%~1"=="low" (
    set "QUALITY_LABEL=low"
    set "QUALITY_ARG=-quality-level=Low"
    shift
) else if /I "%~1"=="normal" (
    set "QUALITY_LABEL=normal"
    set "QUALITY_ARG="
    shift
)

if not "%~1"=="" (
    set "MAP_NAME=%~1"
)

if not exist "%CARLA_EXE%" (
    echo CARLA executable not found at "%CARLA_EXE%"
    pause
    exit /b 1
)

echo Starting CARLA 0.9.16 on %MAP_NAME% with %QUALITY_LABEL% graphics...
start "CARLA 0.9.16" "%CARLA_EXE%" /Game/Carla/Maps/%MAP_NAME% %QUALITY_ARG% -ResX=1280 -ResY=720 -windowed

echo Server launch command sent.
echo Usage: test_start_carla_server_0_9_16.bat [normal^|low] [Town03]
pause

endlocal
