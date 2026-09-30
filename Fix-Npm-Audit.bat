@echo off
setlocal
cd /d "%~dp0"
title Digital Forge - npm audit fix

echo ============================================================
echo  Digital Forge - npm audit fix
echo  Project: %CD%
echo ============================================================
echo.

call npm.cmd audit fix
set "AUDIT_FIX_EXIT=%ERRORLEVEL%"

echo.
echo ============================================================
echo  Final npm audit report
echo ============================================================
echo.
call npm.cmd audit
set "AUDIT_REPORT_EXIT=%ERRORLEVEL%"

echo.
echo ============================================================
echo  npm audit fix exit code: %AUDIT_FIX_EXIT%
echo  final audit exit code:   %AUDIT_REPORT_EXIT%
echo ============================================================
echo.

if not "%AUDIT_FIX_EXIT%"=="0" (
  echo npm audit fix reported an error. Review the output above.
) else if not "%AUDIT_REPORT_EXIT%"=="0" (
  echo npm audit still reports one or more vulnerabilities.
) else (
  echo npm audit completed without reported vulnerabilities.
)

echo.
pause
endlocal
