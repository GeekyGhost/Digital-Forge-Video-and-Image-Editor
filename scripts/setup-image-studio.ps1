<# Build the isolated ZiT / Qwen Image Edit / BiRefNet environment. #>
[CmdletBinding()]
param(
  [switch]$Recreate,
  [switch]$SkipSam2
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')
$spec = (Get-AivRuntimeMatrix).ImageStudio
$runtimeRoot = Join-Path $root "vendor\Image-Studio"
$venv = Join-Path $runtimeRoot ".venv"
$python = Join-Path $venv "Scripts\python.exe"
$uv = Resolve-AivUv -Persist

if ((Test-Path -LiteralPath $venv) -and -not (Test-PythonInterpreter $python)) {
  Write-Warning "The Image Studio environment cannot start and will be recreated: $venv"
  $Recreate = $true
}

if ($Recreate -and (Test-Path -LiteralPath $venv)) {
  Remove-AivVenv -Path $venv -ProjectRoot $root -Confirm:$false
}
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
if (-not (Test-Path -LiteralPath $python)) {
  & $uv python install $spec.Python
  & $uv venv --python $spec.Python $venv
  if ($LASTEXITCODE -ne 0) { throw "Could not create the Image Studio environment" }
}

function Install-Packages([string[]]$Packages) {
  & $uv pip install --python $python @Packages
  if ($LASTEXITCODE -ne 0) { throw "Installing $($Packages -join ' ') failed" }
}

Write-Host "Installing the proven Torch 2.10 CUDA 12.8 build"
Install-Packages @("torch==$($spec.Torch)", "torchvision==$($spec.TorchVision)", "--index-url", $spec.TorchIndex)

Write-Host "Installing the pinned Diffusers revision used by the working Digital Forge pipeline"
Install-Packages @("git+https://github.com/huggingface/diffusers.git@$($spec.DiffusersCommit)")

Write-Host "Installing image pipeline dependencies"
Install-Packages @(
  "transformers==$($spec.Transformers)", "accelerate==1.12.0", "peft==0.18.1", "safetensors==0.7.0",
  "pillow==12.1.1", "numpy<2", "sentencepiece", "protobuf", "huggingface-hub", "tqdm",
  "timm", "kornia", "einops", "scipy", "scikit-image", "opencv-python-headless", "prettytable"
)

# Real-ESRGAN runs through spandrel, not the upstream realesrgan/basicsr packages. basicsr imports
# torchvision.transforms.functional_tensor, which torchvision removed in 0.17, and upstream has
# left that broken. Spandrel is inference-only, reads the same published .pth state dicts, and
# detects architecture and scale from the weights -- so no per-model network parameters to get
# wrong. All of its dependencies are already installed above; this adds only spandrel itself.
Write-Host "Installing the Real-ESRGAN upscale backend"
Install-Packages @("spandrel")

# SAM 2 shares this CUDA/Pillow/OpenCV stack cleanly and is intentionally kept out of the LTX and
# MOSS environments. The same install powers still-image selection and tracked video masks.
if (-not $SkipSam2) {
  Write-Host "Installing the official SAM 2 package"
  Install-Packages @("git+https://github.com/facebookresearch/sam2.git")
}

$env:PYTHONPATH = Join-Path $root "ai-runtime"
& $python -c "import torch, timm, kornia; from diffusers import ZImagePipeline, QwenImageEditPlusPipeline; from transformers import AutoModelForImageSegmentation; from spandrel import ModelLoader; import aiv_runtime.image_studio.worker, aiv_runtime.upscale; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'image studio + upscale imports OK')"
if ($LASTEXITCODE -ne 0) { throw "Image Studio verification failed" }
if (-not $SkipSam2) {
  & $python -c "import sam2; print('SAM 2 import OK')"
  if ($LASTEXITCODE -ne 0) { throw "SAM 2 verification failed" }
}
Write-Host "Image Studio runtime ready at $venv"
