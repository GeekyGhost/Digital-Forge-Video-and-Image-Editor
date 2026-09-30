@echo off
REM Runs the foley V2A adapter exactly as ai.rs::generate_bridge does, but in the console so
REM stderr is visible. The editor discards stderr on a SUCCESSFUL run, which is precisely when
REM the "[ltx-foley] scaled-FP8 ..." line matters -- the scaled-FP8 LoRA fusion fails quietly,
REM producing degraded audio rather than an error.
REM
REM Usage:  test-foley.bat <input.mp4> ["prompt"]
REM Pass --dry-run as the only argument to validate paths and layout without loading the model.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\vendor\LTX-2\.venv\Scripts\python.exe"
set "ADAPTER=%ROOT%\ai-runtime\aiv_runtime\ltx_foley_v2a.py"
set "CKPT=%ROOT%\models\LTX-2.3\ltx-2.3-22b-distilled-fp8.safetensors"
set "GEMMA=%ROOT%\models\gemma-3-12b"
set "LORA=%ROOT%\models\loras\ltx\ltx-2.3-22b-lora-foley-v2a-1.0.safetensors"

if "%~1"=="" (
  echo Usage: test-foley.bat ^<input.mp4^> ["prompt"]
  exit /b 1
)

if "%~1"=="--dry-run" (
  echo Dry run needs a video too: test-foley.bat --dry-run is not supported.
  exit /b 1
)

set "IN=%~f1"
set "PROMPT=%~2"
if "%PROMPT%"=="" set "PROMPT=clear synchronised sound effects matching the on-screen action"
set "OUT=%ROOT%\cache\foley-test.mp4"
REM 0 = ignore the source clip's own audio (correct for foley). Raise toward 1 only to
REM deliberately keep some of the original bed. Third argument overrides it.
set "AUDIOREF=%~3"
if "%AUDIOREF%"=="" set "AUDIOREF=0"

echo Input      : %IN%
echo Checkpoint : %CKPT%
echo Output     : %OUT%
echo.

"%PY%" "%ADAPTER%" ^
  --checkpoint-path "%CKPT%" ^
  --gemma-root "%GEMMA%" ^
  --lora-path "%LORA%" ^
  --video-path "%IN%" ^
  --output-path "%OUT%" ^
  --prompt "%PROMPT%" ^
  --seed 42 ^
  --cfg-scale 1.0 ^
  --audio-cfg-scale 1.5 ^
  --lora-strength 1.0 ^
  --reference-strength 1.0 ^
  --audio-reference-strength %AUDIOREF% ^
  --offload disk

echo.
echo Exit code: %ERRORLEVEL%
endlocal
