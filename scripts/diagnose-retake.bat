@echo off
REM Find out WHY a partial-range LTX Retake comes back unchanged, by measurement rather than guess.
REM
REM Runs the same clip twice and prints three instrumented traces from the adapter each time:
REM
REM   [ltx-retake][mask]   how many tokens the time window actually made denoisable.
REM                        0 = the window resolved to nothing and the output IS the input.
REM   [ltx-retake][cfg]    the mean |cond - uncond| the guider receives. CFG contributes exactly
REM                        (scale - 1) * (cond - uncond). This checkpoint is guidance-DISTILLED,
REM                        so if that gap is under ~1% of |cond| then cfg-scale 2 and cfg-scale 5
REM                        are both multiplying zero -- which is the reported symptom exactly.
REM   [ltx-retake][step]   how far the latent moved INSIDE the mask, per step. Live mask plus a
REM                        near-zero movement means the model is reconstructing the pinned
REM                        surroundings, not ignoring the prompt.
REM
REM Both passes now name their --guidance preset explicitly. They did not, and between the day
REM the reference values became argparse defaults and the day the presets landed, Pass A silently
REM inherited stg 1.0 / modality 3.0 too -- so "CFG only" ran the same guidance as Pass B and the
REM whole A/B contrast measured nothing.
REM
REM Pass A reproduces the failing setup: a 1s window, CFG only.
REM Pass B is the corrected setup: a 2.5s window (over the 2s minimum Lightricks' own hosted
REM retake enforces and the OSS pipeline does not) with the three guidance axes LTX-2.3's own
REM reference config uses and which do NOT depend on a cond/uncond gap surviving distillation.
REM
REM Usage:  diagnose-retake.bat <clip.mp4> [startSeconds] [frames] ["prompt"]
REM   frames must be 8n+1 and long enough to hold a 2.5s window: 97 (~4s) or 121 (~5s).
setlocal enabledelayedexpansion
set "ROOT=%~dp0.."
set "PY=%ROOT%\vendor\LTX-2\.venv\Scripts\python.exe"
set "ADAPTER=%ROOT%\ai-runtime\aiv_runtime\ltx_retake_16gb.py"
set "CKPT=%ROOT%\models\LTX-2.3\ltx-2.3-22b-distilled-fp8.safetensors"
set "GEMMA=%ROOT%\models\gemma-3-12b"
set "WORK=%ROOT%\cache\retake-diagnose"

if "%~1"=="" ( echo Usage: diagnose-retake.bat ^<clip.mp4^> [startSeconds] [frames] ["prompt"] & exit /b 1 )
if not exist "%WORK%" mkdir "%WORK%"
set "IN=%~f1"
set "START=%~2"
if "%START%"=="" set "START=0"
set "FRAMES=%~3"
if "%FRAMES%"=="" set "FRAMES=97"
set "PROMPT=%~4"
if "%PROMPT%"=="" set "PROMPT=the rider raises one arm high above his head"

echo === source ===
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of default=nw=1 "%IN%"

echo.
echo === conditioning: snap to /32, cut a %FRAMES%-frame window from %START%s ===
ffmpeg -hide_banner -loglevel error -y -ss %START% -i "%IN%" -frames:v %FRAMES% ^
  -vf "scale=trunc(iw/32)*32:trunc(ih/32)*32:flags=lanczos,setsar=1" ^
  -c:v libx264 -crf 14 -pix_fmt yuv420p -c:a aac -b:a 192k "%WORK%\window.mp4"
if not exist "%WORK%\window.mp4" ( echo Conditioning failed. & pause & exit /b 1 )
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,nb_frames -of default=nw=1 "%WORK%\window.mp4"

set "AIV_RETAKE_MAX_PIXELS=4000000"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

echo.
echo ============================================================
echo == PASS A: 1.0s window, CFG only -- the reported failure   ==
echo ============================================================
"%PY%" "%ADAPTER%" ^
  --checkpoint-path "%CKPT%" --gemma-root "%GEMMA%" ^
  --video-path "%WORK%\window.mp4" --output-path "%WORK%\pass-a.mp4" ^
  --prompt "%PROMPT%" --start-time 0.5 --end-time 1.5 --seed 42 --mode video ^
  --guidance fast --cfg-scale 2.5 --offload cpu
echo Pass A exit code: %ERRORLEVEL%

echo.
echo ============================================================
echo == PASS B: 2.5s window, CFG + STG + modality + rescale     ==
echo ============================================================
"%PY%" "%ADAPTER%" ^
  --checkpoint-path "%CKPT%" --gemma-root "%GEMMA%" ^
  --video-path "%WORK%\window.mp4" --output-path "%WORK%\pass-b.mp4" ^
  --prompt "%PROMPT%" --start-time 0.5 --end-time 3.0 --seed 42 --mode video ^
  --guidance quality --cfg-scale 2.5 --stg-scale 1.0 --stg-blocks 28 --modality-scale 3.0 ^
  --rescale-scale 0.7 --offload cpu
echo Pass B exit code: %ERRORLEVEL%

echo.
echo Outputs: %WORK%\pass-a.mp4  and  %WORK%\pass-b.mp4
echo.
echo Read the traces in this order:
echo   1. If [mask] says 0 tokens, stop -- the window never became denoisable.
echo   2. If [cfg] is under 1%% of ^|cond^|, CFG is dead on this distilled checkpoint and
echo      raising --cfg-scale will never do anything. STG and modality are the live levers.
echo   3. Compare [step] movement between Pass A and Pass B. If B moves and A does not, the
echo      window length was the problem, not the prompt.
pause
endlocal
