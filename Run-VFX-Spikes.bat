@echo off
setlocal
cd /d "%~dp0"
title Digital Forge - VFX Phase 0 spikes

rem Starts the newest built Digital Forge with AIV_SPIKES=1, which opens a second window with the
rem Phase 0 spike page. Build first with Build-AIV-Studio.bat so the exe contains the spike code.
rem Results are saved to scratch\vfx-spikes\results-*.json. Delete this file with spikes\.

set "EXE="
for /f "delims=" %%F in ('dir /b /o:-d "Releases\aiv-studio-*.exe" 2^>nul') do (
  set "EXE=%%F"
  goto :found
)
echo No Releases\aiv-studio-*.exe found. Run Build-AIV-Studio.bat first.
pause
exit /b 1

:found
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg was not found on PATH. The decode, readback and alpha spikes need it.
  pause
  exit /b 1
)
echo Starting Releases\%EXE% with AIV_SPIKES=1
echo In the "VFX Phase 0 spikes" window press "Run all". It takes a few minutes; the first run
echo also encodes the test media into scratch\vfx-spikes\media.
set "AIV_SPIKES=1"
start "" /D "%~dp0" "%~dp0Releases\%EXE%"
exit /b 0
