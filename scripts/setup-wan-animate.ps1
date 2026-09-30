<# Build the isolated Wan Animate 2 runtime (VFX Phase 5: character insertion).

Everything stays inside the project:
  vendor\Wan-Animate\.venv         its own Python environment (Diffusers 0.40.0 + triton-windows)
  models\Wan-Animate-2\distilled   Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers at the pinned revision
  models\Wan-Animate-2\baked       the 14B DiT stored as FP8 (about 16 GB), made once from the BF16 shards
  cache\huggingface                Hugging Face metadata and download cache for this script

Nothing is shared with the Spritesheet Maker or any other project; only its 16 GB pattern is reused.
Downloads are about 46 GB; the bake reads the 31 GB of BF16 transformer shards once.
The matte step also uses SAM 2 and BiRefNet, which the app already has (Image Studio / utility venv).
#>
[CmdletBinding()]
param(
  [switch]$Recreate,
  [switch]$SkipModels,
  [switch]$SkipBake
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')
$spec = (Get-AivRuntimeMatrix).WanAnimate
$runtimeRoot = Join-Path $root 'vendor\Wan-Animate'
$venv = Join-Path $runtimeRoot '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$readyMarker = Join-Path $runtimeRoot '.aiv-ready.json'
$models = Join-Path $root 'models\Wan-Animate-2'
$uv = Resolve-AivUv -Persist

# Windows PowerShell turns redirected native stderr into a terminating error under 'Stop', so every
# native call whose stderr is captured runs under 'Continue' and is judged by its exit code alone.
function Invoke-NativeCaptured([scriptblock]$Command) {
  $previous = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    $output = & $Command 2>&1 | ForEach-Object { "$_" }
    return [pscustomobject]@{ ExitCode = $LASTEXITCODE; Text = ($output -join [Environment]::NewLine) }
  } finally {
    $ErrorActionPreference = $previous
  }
}

if (Test-Path -LiteralPath $readyMarker) { Remove-Item -LiteralPath $readyMarker -Force }
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null

# ---- environment
if ((Test-Path -LiteralPath $venv) -and -not (Test-PythonInterpreter $python)) { $Recreate = $true }
if ($Recreate -and (Test-Path -LiteralPath $venv)) {
  Remove-AivVenv -Path $venv -ProjectRoot $root -Confirm:$false
}
if (-not (Test-Path -LiteralPath $python)) {
  & $uv python install $spec.Python
  & $uv venv --python $spec.Python $venv
  if ($LASTEXITCODE -ne 0) { throw 'Could not create the Wan Animate environment' }
}

function Install-Packages([string[]]$Packages) {
  & $uv pip install --python $python @Packages
  if ($LASTEXITCODE -ne 0) { throw "Installing $($Packages -join ' ') failed" }
}

Install-Packages @("torch==$($spec.Torch)", "torchvision==$($spec.TorchVision)", '--index-url', $spec.TorchIndex)
Install-Packages @(
  "diffusers==$($spec.Diffusers)", "transformers==$($spec.Transformers)", 'accelerate==1.12.0',
  'safetensors==0.8.0', 'huggingface-hub', 'sentencepiece', 'ftfy', 'regex', 'protobuf',
  'numpy<2', 'pillow', 'tqdm', "triton-windows==$($spec.TritonWindows)"
)

$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HOME = Join-Path $root 'cache\huggingface'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$runtimeDir = Join-Path $root 'ai-runtime'

$importCheck = Invoke-NativeCaptured {
  Push-Location $runtimeDir
  try {
    & $python -c "import torch, diffusers, triton; from torch.utils._triton import has_triton; from diffusers import WanAnimate2DistilledBlocks, WanAnimate2Transformer3DModel; from diffusers.hooks import apply_group_offloading; from aiv_runtime import wan_animate, wan_animate_cache; assert has_triton(), 'Triton cannot compile here'; print('wan-animate imports OK', torch.__version__, torch.cuda.is_available(), diffusers.__version__, triton.__version__)"
  } finally { Pop-Location }
}
Write-Host $importCheck.Text
if ($importCheck.ExitCode -ne 0) { throw 'Wan Animate environment verification failed' }

# ---- weights
if (-not $SkipModels) {
  $drive = (Get-Item -LiteralPath $root).PSDrive
  $freeGb = [math]::Round($drive.Free / 1GB, 1)
  $bakedAlready = Test-Path -LiteralPath (Join-Path $models 'baked\wan-animate-bake.json')
  $neededGb = if ($bakedAlready) { 20 } else { 70 }
  Write-Host "Free space on $($drive.Name): $freeGb GB (about $neededGb GB needed)"
  if ($freeGb -lt $neededGb) { throw "Not enough free disk space for Wan Animate 2 ($freeGb GB free, about $neededGb GB needed)" }
  if ($bakedAlready) {
    Write-Host 'Bake present: checking the text encoder, CLIP, VAE and configs only (the BF16 transformer shards are skipped)...'
  } else {
    Write-Host 'Downloading Wan2.2-Animate-2-14B-Distilled (about 46 GB; resumes if interrupted)...'
  }
  Push-Location $runtimeDir
  try {
    & $python -m aiv_runtime.wan_animate download --revision $spec.Revision
    if ($LASTEXITCODE -ne 0) { throw 'Wan Animate 2 model download failed; rerun Setup-Wan-Animate.bat to resume' }
  } finally { Pop-Location }
}

# ---- bake: the DiT stored as FP8, once
if (-not $SkipBake) {
  Write-Host 'Baking the Animate 2 transformer into FP8 shards (reads 31 GB once; several minutes)...'
  Push-Location $runtimeDir
  try {
    & $python -m aiv_runtime.wan_animate bake
    if ($LASTEXITCODE -ne 0) { throw 'The Wan Animate 2 bake failed; the error is printed above' }
  } finally { Pop-Location }
}

@{
  schemaVersion = 1
  python = $spec.Python
  torch = $spec.Torch
  diffusers = $spec.Diffusers
  revision = $spec.Revision
  baked = (Test-Path -LiteralPath (Join-Path $models 'baked\wan-animate-bake.json'))
} | ConvertTo-Json | Set-Content -LiteralPath $readyMarker -Encoding Ascii

Write-Host "Wan Animate 2 runtime ready at $venv"
Write-Host 'The BF16 transformer shards in models\Wan-Animate-2\distilled\transformer\diffusion_pytorch_model-*.safetensors'
Write-Host '(31 GB) are only read by the bake and may be deleted once it exists. A rerun of this script will not fetch'
Write-Host 'them again; to re-bake, run the worker with "download --with-transformer --revision ..." then "bake --force".'
