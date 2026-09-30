<#
.SYNOPSIS
    Narrows down which part of safetensors' checkpoint open is hitting os error 1455.

.DESCRIPTION
    The LTX loader fails at sft_loader.py:30:

        safetensors.safe_open(shard_path, framework="pt", device="cuda")

    That call does two separable things: it memory-maps a 27.5 GiB file, and it prepares tensors
    for a CUDA device. This script tests them apart, so we stop guessing which one is failing.

      1. header-only key scan (mmap, no tensor data)
      2. safe_open with device=cpu
      3. safe_open with device=cuda        <-- the exact failing call
      4. actually materialise one small tensor on cuda

    Each stage prints the system commit charge before and after, so we can see how much commit the
    call really reserves. If stage 2 passes and stage 3 fails, the fault is in the CUDA path, not
    the pagefile -- which points at the torch/CUDA version rather than system memory.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\probe-safetensors.ps1
#>
[CmdletBinding()]
param()

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LtxPython   = Join-Path $ProjectRoot 'vendor\LTX-2\.venv\Scripts\python.exe'
$Checkpoint  = Join-Path $ProjectRoot 'models\LTX-2.3\ltx-2.3-22b-distilled-fp8.safetensors'

if (-not (Test-Path -LiteralPath $LtxPython))  { throw "LTX interpreter not found: $LtxPython" }
if (-not (Test-Path -LiteralPath $Checkpoint)) { throw "FP8 checkpoint not found: $Checkpoint" }

# Committed Bytes / Commit Limit are the counters that actually govern error 1455. WMI's
# FreeVirtualMemory is derived and has been misleading here, so read the counters directly.
function Get-CommitGB {
    $committed = (Get-Counter '\Memory\Committed Bytes' -ErrorAction SilentlyContinue).CounterSamples[0].CookedValue
    $limit     = (Get-Counter '\Memory\Commit Limit'    -ErrorAction SilentlyContinue).CounterSamples[0].CookedValue
    [pscustomobject]@{
        Committed = [math]::Round($committed / 1GB, 2)
        Limit     = [math]::Round($limit / 1GB, 2)
        Available = [math]::Round(($limit - $committed) / 1GB, 2)
    }
}

$before = Get-CommitGB
Write-Host ''
Write-Host 'Commit state before probe' -ForegroundColor Cyan
Write-Host ("  committed {0} GB / limit {1} GB  ->  {2} GB available" -f $before.Committed, $before.Limit, $before.Available)
Write-Host ("  checkpoint {0:N1} GB" -f ((Get-Item -LiteralPath $Checkpoint).Length / 1GB)) -ForegroundColor DarkGray

# The probe runs entirely inside one interpreter so each stage inherits the previous stage's
# address space, which is how the real loader behaves.
$Probe = @'
import ctypes, sys, time
import safetensors

path = sys.argv[1]

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

def commit():
    m = MEMORYSTATUSEX(); m.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    # ullAvailPageFile is the commit headroom the allocator actually checks.
    return m.ullAvailPageFile / 1024**3

def stage(label, fn):
    start = commit()
    t0 = time.time()
    try:
        result = fn()
        print("  PASS  %-34s %5.1fs   commit headroom %.1f -> %.1f GB   %s"
              % (label, time.time() - t0, start, commit(), result))
        return True
    except Exception as error:
        print("  FAIL  %-34s %5.1fs   commit headroom %.1f -> %.1f GB"
              % (label, time.time() - t0, start, commit()))
        print("        %s: %s" % (type(error).__name__, error))
        return False

print()
print("Probing", path)

def header_only():
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        return "%d keys" % len(f.keys())

def open_cpu():
    f = safetensors.safe_open(path, framework="pt", device="cpu")
    globals()["_cpu"] = f
    return "%d keys" % len(f.keys())

def open_cuda():
    f = safetensors.safe_open(path, framework="pt", device="cuda")
    globals()["_cuda"] = f
    return "%d keys" % len(f.keys())

def read_one():
    f = globals().get("_cuda") or globals().get("_cpu")
    name = sorted(f.keys())[0]
    t = f.get_tensor(name)
    return "%s %s %s" % (name[:38], tuple(t.shape), t.device)

ok = stage("1. header scan (cpu, scoped)", header_only)
ok = stage("2. safe_open device=cpu", open_cpu) and ok

import torch
print("  --    torch %s / cuda %s / available %s" % (torch.__version__, torch.version.cuda, torch.cuda.is_available()))

ok = stage("3. safe_open device=cuda", open_cuda) and ok
ok = stage("4. materialise one tensor", read_one) and ok
print()
print("RESULT:", "all stages passed" if ok else "see the first FAIL above")
'@

$ProbeFile = Join-Path $env:TEMP ('aiv-safetensors-probe-{0}.py' -f $PID)
Set-Content -LiteralPath $ProbeFile -Value $Probe -Encoding UTF8

$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8       = '1'
try {
    & $LtxPython $ProbeFile $Checkpoint
} finally {
    Remove-Item -LiteralPath $ProbeFile -ErrorAction SilentlyContinue
}

$after = Get-CommitGB
Write-Host ''
Write-Host 'Commit state after probe' -ForegroundColor Cyan
Write-Host ("  committed {0} GB / limit {1} GB  ->  {2} GB available" -f $after.Committed, $after.Limit, $after.Available)
Write-Host ''
