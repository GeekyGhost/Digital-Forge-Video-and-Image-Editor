<#
.SYNOPSIS
    Installs SageAttention 2 into the LTX venv, and nothing else's.

.DESCRIPTION
    The LTX transformer runs on PyTorch SDPA and has no way not to. Its vendored selector reaches
    FlashAttention only on sm90 (Hopper) and sm100 (datacenter Blackwell); this machine is an RTX
    4090 Laptop, which is Ada / sm89. Installing FlashAttention would therefore change nothing --
    the branch that would use it is never taken. Sage is the kernel built for sm89.

    The pinned wheel matches the LTX venv exactly: CUDA 12.8, torch 2.7.1, CPython 3.10, Windows
    x64. Nothing about torch changes, which matters, because the whole 16 GB FP8 streaming
    profile was qualified on 2.7.1.

    Everything lands in vendor\LTX-2\.venv. The application venv, the MiniMax-H3 venv and any
    Python outside this project are untouched -- the H3 venv already has its own sageattention
    built against ITS torch, and the two must not be mixed.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-ltx-attention.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-ltx-attention.ps1 -Verify
#>
[CmdletBinding()]
param(
    # Check what is installed and print the verdict, without installing anything.
    [switch]$Verify
)

# 'Continue', not 'Stop'. Under 'Stop' any line an external program writes to stderr becomes a
# terminating error, and pip and torch both use stderr for ordinary progress and warnings --
# common.ps1 carries the same note for the same reason. Every external call below is checked by
# $LASTEXITCODE instead, which is what actually says whether it worked.
$ErrorActionPreference = 'Continue'

$Root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')
$Spec = (Get-AivRuntimeMatrix).Ltx
$VenvPython = Join-Path $Root 'vendor\LTX-2\.venv\Scripts\python.exe'

# The LTX venv is built by `uv` and has no pip in it -- `python -m pip` there reports "No module
# named pip". Installing into it therefore goes through the application venv's uv with
# `--python`, which is exactly how setup-ltx.ps1 puts torch there in the first place.
$Uv = Resolve-AivUv -Persist

# Pinned, not floating. A different wheel means a different torch ABI, and pip will happily
# install one that imports and then faults inside a CUDA kernel.
$SageWheel = $Spec.SageWheel
$TritonPin = "triton-windows==$($Spec.TritonWindows)"

if (-not (Test-Path $VenvPython)) {
    Write-Host "The LTX venv is missing: $VenvPython" -ForegroundColor Red
    Write-Host "Run scripts\setup-ltx.ps1 first." -ForegroundColor Yellow
    exit 1
}
# The wheel is built for one exact interpreter and one exact torch. Check both before installing
# rather than after, because pip's own check passes on a mismatched torch and the failure then
# surfaces as a crash mid-denoise.
$probeSource = @'
import sys, json
info = {"python": "%d.%d" % sys.version_info[:2]}
try:
    import torch
    info["torch"] = torch.__version__
except Exception as error:
    info["torch"] = "missing: %s" % error
try:
    import sageattention
    info["sage"] = getattr(sageattention, "__version__", "installed")
except Exception:
    info["sage"] = None
try:
    import triton
    info["triton"] = triton.__version__
except Exception:
    info["triton"] = None
print(json.dumps(info))
'@

# Written to a file rather than passed with `-c`, matching probe-safetensors.ps1: a multi-line
# argument through PowerShell's quoting is a well-known way to lose newlines on Windows.
$probeFile = Join-Path $env:TEMP ('aiv-ltx-attention-probe-{0}.py' -f $PID)
Set-Content -LiteralPath $probeFile -Value $probeSource -Encoding UTF8
try {
    $probe = & $VenvPython $probeFile
} finally {
    Remove-Item -LiteralPath $probeFile -ErrorAction SilentlyContinue
}
$state = $probe | ConvertFrom-Json

$sageState = if ($state.sage) { $state.sage } else { 'not installed' }
$tritonState = if ($state.triton) { $state.triton } else { 'not installed' }
Write-Host "LTX venv:  Python $($state.python), torch $($state.torch)"
Write-Host "installed: sageattention $sageState, triton $tritonState"

if ($state.python -ne '3.10') {
    Write-Host "The pinned wheel is cp310 and this venv is $($state.python)." -ForegroundColor Red
    Write-Host "Pick the matching wheel from the v2.2.0-windows release instead of forcing this one." -ForegroundColor Yellow
    exit 1
}
if ($state.torch -notlike '2.7.*') {
    Write-Host "The pinned wheel is built for torch 2.7.x and this venv has $($state.torch)." -ForegroundColor Red
    Write-Host "Do not bump torch to suit the wheel: the 16 GB FP8 profile was qualified on 2.7.1." -ForegroundColor Yellow
    exit 1
}

if ($Verify) {
    if ($state.sage -and $state.triton) {
        Write-Host "Sage is installed. Set Attention to Sage in the AI panel." -ForegroundColor Green
        Write-Host "The worker log line names the number of attention sites swapped; zero there" -ForegroundColor DarkGray
        Write-Host "means it fell back to native whatever this says." -ForegroundColor DarkGray
    } elseif ($state.sage) {
        Write-Host "sageattention is present but triton is not, and Sage compiles its kernels" -ForegroundColor Yellow
        Write-Host "through triton. Run this script without -Verify to install the pinned pair." -ForegroundColor Yellow
    } else {
        Write-Host "Sage is not installed. Run this script without -Verify." -ForegroundColor Yellow
    }
    exit 0
}

Write-Host ''
Write-Host "Installing $TritonPin" -ForegroundColor Cyan
    & $Uv pip install --python $VenvPython $TritonPin
if ($LASTEXITCODE -ne 0) { Write-Host 'triton install failed.' -ForegroundColor Red; exit 1 }

Write-Host ''
Write-Host 'Installing SageAttention 2.2.0 (cu128, torch 2.7.1, cp310)' -ForegroundColor Cyan
# --no-deps on purpose. The wheel declares a torch requirement, and a resolver that decides to
# satisfy it would fetch a DEFAULT-index torch and quietly replace the cu128 build this venv was
# set up with -- turning an attention install into a silent torch downgrade. triton is installed
# above by name, which is the only other thing Sage needs at run time.
    & $Uv pip install --python $VenvPython --no-deps $SageWheel
if ($LASTEXITCODE -ne 0) { Write-Host 'sageattention install failed.' -ForegroundColor Red; exit 1 }

# Importing is not the same as running. Sage compiles a triton kernel on first call, and that is
# where a mismatched toolchain actually fails -- so exercise one real attention call here rather
# than discovering it seven minutes into a retake.
Write-Host ''
Write-Host 'Running one real attention call' -ForegroundColor Cyan
$checkSource = @'
import torch
from sageattention import sageattn

# The install must not have moved torch. A resolver that swapped the cu128 build for a default
# one would leave everything importable and fail later, inside the FP8 streaming path.
if not torch.__version__.startswith("2.7.1"):
    raise SystemExit(f"torch is now {torch.__version__}; the install moved it. Reinstall with "
                     "scripts\\setup-ltx.ps1 before using this venv.")
if "cu128" not in torch.__version__ and torch.version.cuda != "12.8":
    raise SystemExit(f"torch is no longer a cu128 build (cuda {torch.version.cuda}).")

if not torch.cuda.is_available():
    raise SystemExit("no CUDA device visible from the LTX venv")

name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
print(f"device: {name} (sm{major}{minor})")

# Both shapes LTX actually calls with. The first check here used one square self-attention
# case, which is exactly why it passed while the adapter crashed on the first block: text
# cross-attention has 6144 video queries against 1024 text keys, and nothing aligned tested
# that. head dim 128, bf16, NHD, unpadded -- the kernel takes any sequence length.
def check(label, q_len, kv_len, heads=32, dim_head=128):
    shape = lambda length: (1, length, heads, dim_head)
    q = torch.randn(*shape(q_len), device="cuda", dtype=torch.bfloat16)
    k = torch.randn(*shape(kv_len), device="cuda", dtype=torch.bfloat16)
    v = torch.randn(*shape(kv_len), device="cuda", dtype=torch.bfloat16)
    out = sageattn(q, k, v, tensor_layout="NHD", is_causal=False)
    torch.cuda.synchronize()
    if tuple(out.shape) != shape(q_len):
        raise SystemExit(f"{label}: got {tuple(out.shape)}, expected {shape(q_len)}")
    reference = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
    ).transpose(1, 2)
    error = (out.float() - reference).abs().mean().item()
    scale = reference.abs().mean().item()
    print(f"{label}: q {q_len} x kv {kv_len} -> mean |sage - sdpa| "
          f"{error:.6f} ({error / scale:.3%} of |sdpa|)")
    if error / scale > 0.05:
        raise SystemExit(f"{label}: Sage disagrees with SDPA by more than 5%")

check("self-attention ", 6144, 6144)
check("cross-attention", 6144, 1024)
check("odd lengths    ", 4097, 1003)
print("OK")
'@

$checkFile = Join-Path $env:TEMP ('aiv-ltx-attention-check-{0}.py' -f $PID)
Set-Content -LiteralPath $checkFile -Value $checkSource -Encoding UTF8
try {
    & $VenvPython $checkFile
} finally {
    Remove-Item -LiteralPath $checkFile -ErrorAction SilentlyContinue
}
if ($LASTEXITCODE -ne 0) {
    Write-Host ''
    Write-Host 'Sage installed but did not run correctly. Leave the backend on native.' -ForegroundColor Red
    exit 1
}

Write-Host ''
Write-Host 'Done. Set Attention to Sage in the AI panel, or pass --attention-backend sage.' -ForegroundColor Green
Write-Host 'The first call of each run compiles a triton kernel, so time the SECOND run.' -ForegroundColor Yellow
