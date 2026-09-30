@echo off
REM Triage for the LTX worker dying with 0xC0000005 (access violation).
REM
REM An access violation is a NATIVE crash, so the useful evidence is not in Python's traceback --
REM there isn't one. Windows records the faulting MODULE, which names the library that actually
REM died (torch_cuda.dll, cudnn, a flash-attention kernel, nvcuda...). That single line decides
REM whether this is a driver/library fault or memory pressure.
REM
REM Also reports commit headroom, because this machine has form: on 2026-08-10 LTX died for a day
REM on the Windows commit limit when C: filled and the pagefile could not expand.
setlocal
echo ============ 1. FAULTING MODULE (the decisive one) ============
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Get-WinEvent -FilterHashtable @{LogName='Application';ProviderName='Application Error'} -MaxEvents 6 -ErrorAction SilentlyContinue | Where-Object { $_.Message -match 'python|aiv|ffmpeg' } | ForEach-Object { '--- {0} ---' -f $_.TimeCreated; ($_.Message -split \"`n\" | Select-Object -First 12) -join \"`n\"; '' }"

echo.
echo ============ 2. .NET / WER reports ============
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Get-WinEvent -FilterHashtable @{LogName='Application';Id=1001} -MaxEvents 4 -ErrorAction SilentlyContinue | Where-Object { $_.Message -match 'python' } | ForEach-Object { '--- {0} ---' -f $_.TimeCreated; ($_.Message -split \"`n\" | Select-String -Pattern 'Fault' | Select-Object -First 10) -join \"`n\"; '' }"

echo.
echo ============ 3. MEMORY / COMMIT ============
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$os = Get-CimInstance Win32_OperatingSystem;" ^
  "$commitLimitGB = [math]::Round($os.TotalVirtualMemorySize/1MB,2);" ^
  "$commitFreeGB  = [math]::Round($os.FreeVirtualMemory/1MB,2);" ^
  "$ramGB = [math]::Round($os.TotalVisibleMemorySize/1MB,2);" ^
  "'RAM total        : {0} GB' -f $ramGB;" ^
  "'Commit limit     : {0} GB' -f $commitLimitGB;" ^
  "'Commit available : {0} GB' -f $commitFreeGB;" ^
  "if ($commitFreeGB -lt 30) { 'WARNING: under 30 GB free commit. A single safe_open of the 27.5 GiB checkpoint charges ~27.6 GB.' } else { 'Commit headroom looks sufficient for the 27.6 GB checkpoint open.' };" ^
  "Get-CimInstance Win32_PageFileUsage | ForEach-Object { 'Pagefile         : {0}  current {1} MB  peak {2} MB' -f $_.Name, $_.CurrentUsage, $_.PeakUsage }"

echo.
echo ============ 4. DISK (pagefile can only grow if C: has room) ============
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='C:'\" | ForEach-Object { 'C: free {0} GB of {1} GB' -f [math]::Round($_.FreeSpace/1GB,1), [math]::Round($_.Size/1GB,1) }"

echo.
echo ============ 5. GPU ============
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv 2>nul || echo nvidia-smi not on PATH

echo.
pause
endlocal
