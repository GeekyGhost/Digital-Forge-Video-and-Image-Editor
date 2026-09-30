@echo off
setlocal
cd /d "%~dp0"

title Compile Digital Forge Rust Shell

if /I "%~1"=="help" goto :usage
if /I "%~1"=="--help" goto :usage
if /I "%~1"=="/?" goto :usage

set "CARGO_PROFILE=debug"
set "CARGO_FLAG="
if /I "%~1"=="release" (
  set "CARGO_PROFILE=release"
  set "CARGO_FLAG=--release"
) else if not "%~1"=="" (
  echo Unknown option: %~1
  echo.
  goto :usage_error
)

where cargo.exe >nul 2>nul
if errorlevel 1 (
  echo Rust Cargo was not found on PATH.
  echo Install Rust with rustup, restart Windows Terminal, and rerun this script.
  pause
  exit /b 1
)

if not exist "%~dp0src-tauri\Cargo.toml" (
  echo Missing src-tauri\Cargo.toml.
  echo Run this script from the Digital Forge project checkout.
  pause
  exit /b 1
)

echo Compiling the Digital Forge Rust shell using the %CARGO_PROFILE% profile...
echo.
cargo build --manifest-path "%~dp0src-tauri\Cargo.toml" %CARGO_FLAG%
if errorlevel 1 (
  echo.
  echo RUST BUILD FAILED. Review the Cargo errors above.
  pause
  exit /b 1
)

echo.
echo RUST BUILD COMPLETE.
echo Executable: %~dp0src-tauri\target\%CARGO_PROFILE%\aiv-studio.exe
pause
exit /b 0

:usage_error
echo Usage: Compile-Rust.bat [release]
pause
exit /b 2

:usage
echo Usage: Compile-Rust.bat [release]
echo.
echo   no argument   Compile a faster debug build.
echo   release       Compile an optimized release build.
exit /b 0
