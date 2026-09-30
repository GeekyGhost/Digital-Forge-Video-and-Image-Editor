@echo off
setlocal
cd /d "%~dp0"

echo Building and qualifying AIV Studio...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\build-release.ps1"
if errorlevel 1 (
  echo.
  echo BUILD FAILED. Review the errors above.
  pause
  exit /b 1
)

echo.
echo BUILD COMPLETE. The versioned executable is in Releases.
pause
exit /b 0
