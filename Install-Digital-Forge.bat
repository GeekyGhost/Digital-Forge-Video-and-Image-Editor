@echo off
setlocal
cd /d "%~dp0"
title Digital Forge Video and Image Editor - Installer

rem Guided first-time setup. Safe to run again: finished steps are skipped, and running it again
rem is also how you add optional AI features later. Pass options through, for example:
rem   Install-Digital-Forge.bat -Components image-studio,photo -Prebuilt

where powershell.exe >nul 2>nul
if errorlevel 1 (
  echo Windows PowerShell was not found on PATH.
  pause
  exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" %*
set "AIV_INSTALL_EXIT=%errorlevel%"

echo.
if "%AIV_INSTALL_EXIT%"=="0" (
  echo Installation finished.
) else (
  echo Installation did not finish cleanly ^(exit code %AIV_INSTALL_EXIT%^).
  echo The details are above and in the newest .aiv\install-*.log file.
)
pause
exit /b %AIV_INSTALL_EXIT%
