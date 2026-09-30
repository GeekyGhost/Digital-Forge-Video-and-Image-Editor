@echo off
setlocal
cd /d "%~dp0"

echo Installing NVIDIA Video Effects super-resolution into the Image Studio runtime.
echo.
echo This is NVIDIA's own software under NVIDIA's own proprietary licence. It is not
echo covered by this project's licence, and Digital Forge redistributes no part of it --
echo it downloads from NVIDIA's package index, onto this machine, after you accept
echo their terms. See the Third-Party Components section of LICENSE.md.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\setup-rtx-upscale.ps1" %*

REM Exit code 2 is the licence gate declining to proceed, not a failure. Anything else
REM non-zero means the install or the verification probe genuinely did not work.
if errorlevel 3 goto :failed
if errorlevel 2 goto :licence
if errorlevel 1 goto :failed

echo.
echo NVIDIA VFX super-resolution is ready. It appears in the upscale model list
echo alongside Real-ESRGAN.
pause
exit /b 0

:licence
echo.
echo Not installed -- NVIDIA's licence has not been accepted. Review the links above,
echo then run this again as:
echo.
echo     Install-RTX-Upscale.bat -AcceptLicense
echo.
pause
exit /b 2

:failed
echo.
echo INSTALL FAILED. Review the errors above. Real-ESRGAN upscaling is unaffected and
echo remains available.
pause
exit /b 1
