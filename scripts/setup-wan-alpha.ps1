<# Build the isolated Wan-Alpha v2.0 runtime (VFX Phase 0, task 0.6).

Everything stays inside the project:
  vendor\Wan-Alpha\repo     WeChatCV/Wan-Alpha at the pinned commit (source only)
  vendor\Wan-Alpha\.venv    its own Python environment
  models\Wan-Alpha\...      Wan2.1-T2V-14B, Wan-Alpha v2.0, the LightX2V 4-step LoRA
  models\Wan-Alpha\baked    the DoRA + LightX2V merge, stored as FP8 (about 14 GB)
  cache\huggingface         Hugging Face metadata and download cache for this script

Nothing is shared with the spritesheet maker or any other project.
Downloads are about 71 GB; the bake reads the 57 GB base once and writes about 14 GB.
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
$spec = (Get-AivRuntimeMatrix).WanAlpha
$runtimeRoot = Join-Path $root 'vendor\Wan-Alpha'
$repo = Join-Path $runtimeRoot 'repo'
$venv = Join-Path $runtimeRoot '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$readyMarker = Join-Path $runtimeRoot '.aiv-ready.json'
$models = Join-Path $root 'models\Wan-Alpha'
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

# ---- source: pinned clone, never modified
if (-not (Test-Path -LiteralPath (Join-Path $repo '.git'))) {
  New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
  git clone --filter=blob:none --no-checkout https://github.com/WeChatCV/Wan-Alpha.git $repo
  if ($LASTEXITCODE -ne 0) { throw 'Could not clone WeChatCV/Wan-Alpha' }
}
git -C $repo fetch --depth 1 origin $spec.Commit
if ($LASTEXITCODE -ne 0) { throw 'Could not fetch the pinned Wan-Alpha commit' }
git -C $repo checkout --detach --force $spec.Commit
if ($LASTEXITCODE -ne 0) { throw 'Could not check out the pinned Wan-Alpha commit' }

# ---- environment
if ((Test-Path -LiteralPath $venv) -and -not (Test-PythonInterpreter $python)) { $Recreate = $true }
if ($Recreate -and (Test-Path -LiteralPath $venv)) {
  Remove-AivVenv -Path $venv -ProjectRoot $root -Confirm:$false
}
if (-not (Test-Path -LiteralPath $python)) {
  & $uv python install $spec.Python
  & $uv venv --python $spec.Python $venv
  if ($LASTEXITCODE -ne 0) { throw 'Could not create the Wan-Alpha environment' }
}

function Install-Packages([string[]]$Packages) {
  & $uv pip install --python $python @Packages
  if ($LASTEXITCODE -ne 0) { throw "Installing $($Packages -join ' ') failed" }
}

Install-Packages @("torch==$($spec.Torch)", "torchvision==$($spec.TorchVision)", '--index-url', $spec.TorchIndex)
Install-Packages @(
  "diffusers==$($spec.Diffusers)", "transformers==$($spec.Transformers)", 'accelerate==1.12.0',
  'safetensors==0.8.0', 'huggingface-hub', 'einops', 'easydict', 'ftfy', 'regex', 'sentencepiece',
  'protobuf', 'numpy<2', 'pillow', 'tqdm', 'scipy'
)

$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HOME = Join-Path $root 'cache\huggingface'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$runtimeDir = Join-Path $root 'ai-runtime'

$importCheck = Invoke-NativeCaptured {
  Push-Location $runtimeDir
  try {
    & $python -c "import torch, diffusers, accelerate, einops, easydict, ftfy; from diffusers.hooks import apply_group_offloading; from aiv_runtime import wan_alpha; wan_alpha.wan_module(wan_alpha.DEFAULT_REPO, 'modules.model_dora_lightx2v'); print('wan-alpha imports OK', torch.__version__, torch.cuda.is_available(), diffusers.__version__)"
  } finally { Pop-Location }
}
Write-Host $importCheck.Text
if ($importCheck.ExitCode -ne 0) { throw 'Wan-Alpha environment verification failed' }

# ---- weights
if (-not $SkipModels) {
  $drive = (Get-Item -LiteralPath $root).PSDrive
  $freeGb = [math]::Round($drive.Free / 1GB, 1)
  # With a valid bake the fp32 base shards are not needed (and may have been deleted on purpose).
  $bakedAlready = Test-Path -LiteralPath (Join-Path $models 'baked\wan-alpha-bake.json')
  $neededGb = if ($bakedAlready) { 20 } else { 90 }
  Write-Host "Free space on $($drive.Name): $freeGb GB (about $neededGb GB needed)"
  if ($freeGb -lt $neededGb) { throw "Not enough free disk space for Wan-Alpha ($freeGb GB free, about $neededGb GB needed)" }

  if ($bakedAlready) {
    Write-Host 'Bake present: checking the text encoder, VAE, Wan-Alpha files and LoRA only (the fp32 DiT shards are skipped)...'
  } else {
    Write-Host 'Downloading Wan2.1-T2V-14B, Wan-Alpha v2.0 and the LightX2V LoRA (about 71 GB; resumes if interrupted)...'
  }
  Push-Location $runtimeDir
  try {
    & $python -m aiv_runtime.wan_alpha download --base-revision $spec.BaseRevision --alpha-revision $spec.AlphaRevision --lightx2v-revision $spec.LightX2VRevision
    if ($LASTEXITCODE -ne 0) { throw 'Wan-Alpha model download failed; rerun Setup-Wan-Alpha.bat to resume' }
  } finally { Pop-Location }
}

# ---- bake: DoRA + LightX2V merged once, stored FP8
if (-not $SkipBake) {
  Write-Host 'Baking Wan-Alpha into FP8 shards (reads the 57 GB base once; several minutes)...'
  Push-Location $runtimeDir
  try {
    & $python -m aiv_runtime.wan_alpha bake
    if ($LASTEXITCODE -ne 0) { throw 'The Wan-Alpha bake failed; the error is printed above' }
  } finally { Pop-Location }
}

@{
  schemaVersion = 1
  commit = $spec.Commit
  python = $spec.Python
  torch = $spec.Torch
  diffusers = $spec.Diffusers
  baked = (Test-Path -LiteralPath (Join-Path $models 'baked\wan-alpha-bake.json'))
} | ConvertTo-Json | Set-Content -LiteralPath $readyMarker -Encoding Ascii

Write-Host "Wan-Alpha runtime ready at $venv"
Write-Host 'The fp32 base shards in models\Wan-Alpha\Wan2.1-T2V-14B\diffusion_pytorch_model-*.safetensors (57 GB)'
Write-Host 'are only read by the bake and may be deleted once it exists. A rerun of this script will not fetch them'
Write-Host 'again; to re-bake, run the worker with "download --with-base ..." and then "bake --force".'
