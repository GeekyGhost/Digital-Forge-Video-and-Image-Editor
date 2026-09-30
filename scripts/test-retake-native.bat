@echo off
REM Probe whether LTX Retake runs at NATIVE resolution on 16 GB, by spending the VRAM budget on
REM pixels instead of duration.
REM
REM The adapter's 768x768 pixel guard is not a model limit -- cost scales with pixels TIMES latent
REM frames. The validated profile (768x512, 121 frames) peaked at 9,336 MiB. A 1280x704 window of
REM 49 frames works out to roughly the same token count, so it SHOULD fit. This measures it.
REM
REM Usage:  test-retake-native.bat <clip.mp4> [startSeconds] [frames] ["prompt"] [offload]
REM   cfg: argument 6, video CFG scale (default 2.5). 1.0 = unguided, the old behaviour.
REM   offload: disk (default) or cpu. Every torch_cpu.dll crash so far has been on the disk
REM   streaming path, whose pinned-buffer recycling is the prime suspect -- "cpu" is the control.
REM   frames must be 8n+1: 25 (~1s), 41, 49 (~2s), 73, 97 (~4s), 121, 193
setlocal enabledelayedexpansion
set "ROOT=%~dp0.."
set "PY=%ROOT%\vendor\LTX-2\.venv\Scripts\python.exe"
set "ADAPTER=%ROOT%\ai-runtime\aiv_runtime\ltx_retake_16gb.py"
set "CKPT=%ROOT%\models\LTX-2.3\ltx-2.3-22b-distilled-fp8.safetensors"
set "GEMMA=%ROOT%\models\gemma-3-12b"
set "WORK=%ROOT%\cache\retake-probe"

if "%~1"=="" ( echo Usage: test-retake-native.bat ^<clip.mp4^> [startSeconds] [frames] ["prompt"] & exit /b 1 )
if not exist "%WORK%" mkdir "%WORK%"
set "IN=%~f1"
set "START=%~2"
if "%START%"=="" set "START=0"
set "FRAMES=%~3"
if "%FRAMES%"=="" set "FRAMES=49"
set "PROMPT=%~4"
set "OFFLOAD=%~5"
set "CFG=%~6"
if "%CFG%"=="" set "CFG=2.5"
if "%OFFLOAD%"=="" set "OFFLOAD=disk"
if "%PROMPT%"=="" set "PROMPT=the same scene, cleanly rendered"

echo === source ===
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of default=nw=1 "%IN%"

echo.
echo === conditioning: snap to /32 at NATIVE scale, cut a %FRAMES%-frame window from %START%s ===
REM Only the dimensions are touched, and only to the nearest multiple of 32 (1280x720 -^> 1280x704).
REM Resolution is NOT reduced -- that is the whole point of the probe.
ffmpeg -hide_banner -loglevel error -y -ss %START% -i "%IN%" -frames:v %FRAMES% ^
  -vf "scale=trunc(iw/32)*32:trunc(ih/32)*32:flags=lanczos,setsar=1" ^
  -c:v libx264 -crf 14 -pix_fmt yuv420p -c:a aac -b:a 192k "%WORK%\window.mp4"
if not exist "%WORK%\window.mp4" ( echo Conditioning failed. & pause & exit /b 1 )
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of default=nw=1 "%WORK%\window.mp4"

echo.
echo === running retake with the pixel guard raised, offload=%OFFLOAD%, cfg=%CFG% ===
REM 4 million is simply "no practical ceiling"; the point is to let the run proceed and see what it
REM actually costs. If this OOMs, the fixed budget plus a spatial-upscale round trip is the answer.
set "AIV_RETAKE_MAX_PIXELS=4000000"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "START_TS=%TIME%"
"%PY%" "%ADAPTER%" ^
  --checkpoint-path "%CKPT%" ^
  --gemma-root "%GEMMA%" ^
  --video-path "%WORK%\window.mp4" ^
  --output-path "%WORK%\retaken.mp4" ^
  --prompt "%PROMPT%" ^
  --start-time 0.25 ^
  --end-time 1.25 ^
  --seed 42 ^
  --mode video ^
  --cfg-scale %CFG% ^
  --offload %OFFLOAD%
echo.
echo Started %START_TS%  finished %TIME%
echo Exit code: %ERRORLEVEL%
echo Output: %WORK%\retaken.mp4
echo.
echo Look for the "[ltx-retake] peak VRAM" line above. Under ~15,000 MiB means native-resolution
echo retake is viable for short windows and the guard should become a token budget, not a pixel one.
pause
endlocal
