@echo off
rem start.bat - open localcode, the interactive session, from a double-click or a prompt.
rem
rem   start.bat                         the model from LOCALCODE_MODEL, or %USERPROFILE%\model\GLM-5.3-Flash
rem   start.bat D:\models\GLM-5.3-Flash  that model directory
rem   start.bat --reasoning low         any localcode option (localcode --help lists them)
rem
rem Set LOCALCODE_INT4 to an int4 container (--write-int4) to read the experts from it:
rem faster, and an approximation of the checkpoint (docs/USAGE.md).
setlocal
set "ROOT=%~dp0"
set "EXE=%ROOT%build\Release\localcode.exe"

if not exist "%EXE%" (
    echo localcode is not built yet. From %ROOT%:
    echo   cmake -B build -DCMAKE_BUILD_TYPE=Release -DGLM53F_CUDA=ON
    echo   cmake --build build --config Release -j
    goto :fail
)

rem A model directory given as the first argument wins; otherwise LOCALCODE_MODEL.
if not "%~1"=="" if exist "%~1\config.json" set "LOCALCODE_MODEL=%~1"
if not defined LOCALCODE_MODEL if exist "%USERPROFILE%\model\GLM-5.3-Flash\config.json" (
    set "LOCALCODE_MODEL=%USERPROFILE%\model\GLM-5.3-Flash"
)
if not defined LOCALCODE_MODEL (
    set /p "LOCALCODE_MODEL=Model directory (the one with config.json): "
)
if not exist "%LOCALCODE_MODEL%\config.json" (
    echo No config.json in "%LOCALCODE_MODEL%".
    echo Give the model directory as an argument, or set LOCALCODE_MODEL.
    goto :fail
)

set "EXTRA="
if defined LOCALCODE_INT4 set "EXTRA=--int4-dir "%LOCALCODE_INT4%""

rem The model directory travels in LOCALCODE_MODEL, so a directory given as the first
rem argument is dropped from what localcode receives.
set "ARGS=%*"
if not "%~1"=="" if exist "%~1\config.json" call :drop_first %*

rem Files named in /file are relative to the working directory: start where the script is.
pushd "%ROOT%"
"%EXE%" %EXTRA% %ARGS%
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" (
    echo.
    echo localcode ended with code %RC%.
    goto :fail
)
exit /b 0

:drop_first
set "ARGS="
shift
:drop_loop
if "%~1"=="" exit /b 0
set "ARGS=%ARGS% %1"
shift
goto :drop_loop

:fail
rem Keep the window open when started with a double-click, so the message can be read.
echo %CMDCMDLINE% | find /i "/c" >nul && pause
exit /b 1
