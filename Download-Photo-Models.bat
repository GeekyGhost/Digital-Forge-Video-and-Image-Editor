@echo off
setlocal
cd /d "%~dp0"

title Download Digital Forge Photo Edit Models

if /I "%~1"=="help" goto :usage
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="/?" goto :usage

set "DEPTH_VARIANT=Small"
if /I "%~1"=="base" (
  set "DEPTH_VARIANT=Base"
) else if not "%~1"=="" (
  echo Unknown option: %~1
  echo.
  goto :usage_error
)

where powershell.exe >nul 2>nul
if errorlevel 1 (
  echo Windows PowerShell was not found on PATH.
  pause
  exit /b 1
)

if not exist "%~dp0scripts\download-photo-models.ps1" (
  echo Missing scripts\download-photo-models.ps1.
  echo Run this script from the Digital Forge project checkout.
  pause
  exit /b 1
)

echo Fetching the optional Photo Edit models (depth variant: %DEPTH_VARIANT%)...
echo Photo Edit works without them; they only enable the Relight and Remove cards.
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\download-photo-models.ps1" -DepthVariant %DEPTH_VARIANT%
if errorlevel 1 (
  echo.
  echo MODEL DOWNLOAD FAILED. Review the errors above.
  pause
  exit /b 1
)

echo.
echo MODELS READY. Relaunch Digital Forge to pick them up.
pause
exit /b 0

:usage_error
echo Usage: Download-Photo-Models.bat [base]
pause
exit /b 2

:usage
echo Usage: Download-Photo-Models.bat [base]
echo.
echo   no argument   Depth Anything V2 Small, about 100 MB.
echo   base          Depth Anything V2 Base instead, better quality, about 380 MB.
echo.
echo   LaMa is already installed and verified, so it is skipped either way.
exit /b 0
