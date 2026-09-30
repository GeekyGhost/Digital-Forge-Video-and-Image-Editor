@echo off
setlocal
title Digital Forge Video Studio Setup and Launcher
echo Digital Forge Video Studio diagnostic launcher
echo For normal console-free startup, double-click Launch-AIV-Studio.vbs
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-and-launch.ps1"
if errorlevel 1 (
  echo.
  echo Digital Forge setup or launch failed. Review the message above.
  pause
  exit /b 1
)
