<#
.SYNOPSIS
  Build the MiniMax-H3 virtualenv.

.DESCRIPTION
  H3 gets its OWN environment, at vendor\MiniMax-H3\.venv, for the same reason LTX has one: it
  needs Diffusers' MiniMax-H3 pipeline plus quantization kernels, and installing that beside either existing
  venv risks the two things on this machine that already work. Demucs has clobbered the CUDA torch
  in .venv once already.

  Three environments after this runs:
    vendor\LTX-2\.venv       LTX adapters       torch 2.7.1+cu128
    vendor\MiniMax-H3\.venv  H3 adapter         torch 2.11.0+cu128, diffusers 0.40
    .venv                    application worker Whisper, Kiwi-Edit, model installer

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-h3.ps1
#>
[CmdletBinding()]
param(
  [switch]$Recreate,
  [switch]$SkipSage
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')
$spec = (Get-AivRuntimeMatrix).H3
$home_ = Join-Path $root "vendor\MiniMax-H3"
$venv = Join-Path $home_ ".venv"
$python = Join-Path $venv "Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $home_ | Out-Null
$uv = Resolve-AivUv -Persist

if ((Test-Path -LiteralPath $venv) -and -not (Test-PythonInterpreter $python)) {
  Write-Warning "The H3 environment cannot start and will be recreated: $venv"
  $Recreate = $true
}
if ($Recreate -and (Test-Path -LiteralPath $venv)) {
  Write-Host "Removing the existing environment at $venv"
  Remove-AivVenv -Path $venv -ProjectRoot $root -Confirm:$false
}

if (-not (Test-Path $python)) {
  & $uv python install $spec.Python
  Write-Host "Creating $venv with uv (Python $($spec.Python))"
  & $uv venv --python $spec.Python $venv
}
if (-not (Test-Path $python)) { throw "The environment was not created at $venv" }

# Installs go through uv when it is available and through pip only as a fallback.
#
# Two things went wrong before this shape. First, `uv venv` does not install pip, so every
# `python -m pip` failed. Second -- and worse -- probing for pip with `& $python -m pip --version
# 2>&1` is itself fatal: with $ErrorActionPreference = "Stop", redirecting a native command's
# stderr turns each stderr line into a terminating ErrorRecord, so the probe killed the script
# before the repair it was guarding could run.
#
# `uv pip install --python <interpreter>` needs no pip in the target environment at all, which
# removes the whole problem rather than working around it. It is also considerably faster.
# Arguments are passed as an explicit array, NOT as loose tokens: PowerShell tries to bind a bare
# `--index-url` as a parameter name and fails with "A parameter cannot be found that matches
# parameter name 'index-url'".
function Invoke-Install {
  param([string[]]$Packages)
  & $uv pip install --python $python @Packages
  if ($LASTEXITCODE -ne 0) { throw "Installing $($Packages -join ' ') failed with exit code $LASTEXITCODE" }
}

Write-Host "`nInstalling pinned PyTorch from $($spec.TorchIndex)"
Invoke-Install @("torch==$($spec.Torch)", "torchvision==$($spec.TorchVision)", "--index-url", $spec.TorchIndex)

Write-Host "`nInstalling the released MiniMax-H3 Diffusers runtime"
Invoke-Install @("diffusers==$($spec.Diffusers)")

Write-Host "`nInstalling the rest"
# Transformers 5's checkpoint loader keeps enough Windows commit charged during on-load torchao
# conversion that this 32B conditioner dies partway through the bake. 4.57.6 is the final v4
# release, includes Qwen3-VL, and is intentionally pinned for the isolated H3 environment.
Invoke-Install @("transformers==$($spec.Transformers)", "huggingface-hub==0.36.2", "accelerate>=1.6", "torchao>=0.14", "comfy-kitchen==0.2.31", "safetensors",
  "peft>=0.17", "sentencepiece", "protobuf", "pillow", "imageio[ffmpeg]", "soundfile", "av", "numpy")

if (-not $SkipSage) {
  Write-Host "`nInstalling optional Sage Attention 2 for the H3-only environment"
  # Torch 2.10 and 2.11 both pair with Triton 3.6. The cp310-abi3 Sage wheel works on this
  # environment's Python 3.12 and is built for cu128 + Torch 2.10 or newer. Sage is deliberately
  # optional: the worker probes it before generation and falls back to native SDPA if installation
  # or kernel startup is unavailable on a different GPU.
  try {
    Invoke-Install @("triton-windows>=3.6,<3.7")
    Invoke-Install @("https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post6/sageattention-2.2.0%2Bcu128torch2.10.0andhigher.post6-cp310-abi3-win_amd64.whl")
  } catch {
    Write-Warning "Sage Attention could not be installed; H3 will keep using native PyTorch SDPA. $($_.Exception.Message)"
  }
}

Write-Host "`nVerifying"
& $python -c @"
import torch
import importlib.util
from diffusers import MiniMaxH3Transformer3DModel, ModularPipeline
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0),
          round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1), 'GiB')
print('diffusers has MiniMaxH3Transformer3DModel')
print('peft', bool(importlib.util.find_spec('peft')), 'pyav', bool(importlib.util.find_spec('av')), 'comfy-kitchen', bool(importlib.util.find_spec('comfy_kitchen')), 'sageattention', bool(importlib.util.find_spec('sageattention')))
"@
if ($LASTEXITCODE -ne 0) { throw "Verification failed" }

Write-Host "`nDone. Next:"
Write-Host "  .venv\Scripts\python.exe scripts\models.py install minimax-h3 --accept-licenses"
