@echo off
setlocal
cd /d "%~dp0"
set "PRECISION=%~1"
if "%PRECISION%"=="" set "PRECISION=f16"
if /i "%PRECISION%"=="f16" goto f16
if /i "%PRECISION%"=="f32" goto f32
echo Usage: convert.cmd [f16^|f32]
exit /b 2
:f16
set "PRECISION=f16"
goto valid
:f32
set "PRECISION=f32"
:valid
set "PYTHONUTF8=1"
if exist ".venv\Scripts\python.exe" goto install
py -3 -c "import sys; assert sys.version_info >= (3, 11), 'Python 3.11 or newer is required'"
if errorlevel 1 goto failed
py -3 -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install --only-binary=:all: numpy==2.3.5 safetensors==0.7.0
if errorlevel 1 goto failed
".venv\Scripts\python.exe" convert_anime_whisper.py --download --model-dir anime-whisper-source --precision %PRECISION% --output converted\ggml-anime-whisper-local-%PRECISION%.bin
if errorlevel 1 goto failed
echo.
echo Conversion finished. Copy the generated .toml into the ASR2RPP model definitions folder.
echo Keep the generated .bin at its current path. No application defaults have been changed.
exit /b 0
:failed
echo.
echo Conversion failed. Read the error above. Existing output files are not overwritten.
exit /b 1
