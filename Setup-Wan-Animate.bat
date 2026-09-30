@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup-wan-animate.ps1" %*
set "AIV_SETUP_EXIT=%errorlevel%"
if not "%AIV_SETUP_EXIT%"=="0" (
  echo.
  echo Wan Animate 2 setup FAILED ^(exit code %AIV_SETUP_EXIT%^). The error is printed above.
  pause
)
exit /b %AIV_SETUP_EXIT%
