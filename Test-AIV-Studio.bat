@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\test-all.ps1"
if errorlevel 1 (
  echo.
  echo TESTS FAILED. Review the errors above.
  pause
  exit /b 1
)

echo.
echo ALL TESTS PASSED.
pause
exit /b 0
