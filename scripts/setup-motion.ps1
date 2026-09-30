<# Build the isolated motion runtime (VFX Phase 4): CoTracker3 point tracking and Video Depth
Anything Small depth sequences.

Everything stays inside the project:
  vendor\co-tracker               facebookresearch/co-tracker at the pinned commit (source only)
  vendor\Video-Depth-Anything     DepthAnything/Video-Depth-Anything at the pinned commit (source only)
  runtimes\motion\.venv           its own Python environment
  models\motion\                  scaled_offline.pth (CoTracker3) and video_depth_anything_vits.pth
  cache\huggingface               Hugging Face metadata and download cache for this script

Licences: CoTracker3 is CC-BY-NC 4.0 (non-commercial use only). Video Depth Anything Small is
Apache-2.0 (the Base and Large checkpoints are not, and are not downloaded). Neither upstream
requirements file is used: Video Depth Anything pins torch 2.1.1 and xformers 0.0.23, and xformers
is optional there. Downloads are about 0.25 GB of weights on top of the shared torch wheels.
#>
[CmdletBinding()]
param(
  [switch]$Recreate,
  [switch]$SkipModels
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')
$spec = (Get-AivRuntimeMatrix).Motion
$runtimeRoot = Join-Path $root 'runtimes\motion'
$venv = Join-Path $root $spec.Venv
$python = Join-Path $venv 'Scripts\python.exe'
$readyMarker = Join-Path $runtimeRoot '.aiv-ready.json'
$models = Join-Path $root 'models\motion'
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

function Sync-PinnedClone([string]$Url, [string]$Path, [string]$Commit) {
  if (-not (Test-Path -LiteralPath (Join-Path $Path '.git'))) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    git clone --filter=blob:none --no-checkout $Url $Path
    if ($LASTEXITCODE -ne 0) { throw "Could not clone $Url" }
  }
  git -C $Path fetch --depth 1 origin $Commit
  if ($LASTEXITCODE -ne 0) { throw "Could not fetch the pinned commit of $Url" }
  git -C $Path checkout --detach --force $Commit
  if ($LASTEXITCODE -ne 0) { throw "Could not check out the pinned commit of $Url" }
}

New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
if (Test-Path -LiteralPath $readyMarker) { Remove-Item -LiteralPath $readyMarker -Force }

# ---- source: pinned clones, never modified
Sync-PinnedClone 'https://github.com/facebookresearch/co-tracker.git' (Join-Path $root 'vendor\co-tracker') $spec.CoTrackerCommit
Sync-PinnedClone 'https://github.com/DepthAnything/Video-Depth-Anything.git' (Join-Path $root 'vendor\Video-Depth-Anything') $spec.VideoDepthCommit

# ---- environment
if ((Test-Path -LiteralPath $venv) -and -not (Test-PythonInterpreter $python)) { $Recreate = $true }
if ($Recreate -and (Test-Path -LiteralPath $venv)) {
  Remove-AivVenv -Path $venv -ProjectRoot $root -Confirm:$false
}
if (-not (Test-Path -LiteralPath $python)) {
  & $uv python install $spec.Python
  & $uv venv --python $spec.Python $venv
  if ($LASTEXITCODE -ne 0) { throw 'Could not create the motion environment' }
}

function Install-Packages([string[]]$Packages) {
  & $uv pip install --python $python @Packages
  if ($LASTEXITCODE -ne 0) { throw "Installing $($Packages -join ' ') failed" }
}

Install-Packages @("torch==$($spec.Torch)", "torchvision==$($spec.TorchVision)", '--index-url', $spec.TorchIndex)
Install-Packages @('numpy<2', 'opencv-python-headless', 'einops', 'easydict', 'tqdm', 'pillow', 'imageio', 'huggingface-hub')

$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HOME = Join-Path $root 'cache\huggingface'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$runtimeDir = Join-Path $root 'ai-runtime'

$importCheck = Invoke-NativeCaptured {
  Push-Location $runtimeDir
  try {
    & $python -c "import sys, torch; sys.path[:0] = [r'$root\vendor\co-tracker', r'$root\vendor\Video-Depth-Anything']; from cotracker.predictor import CoTrackerPredictor; from video_depth_anything.video_depth import VideoDepthAnything; from aiv_runtime import motion_track, video_depth; print('motion imports OK', torch.__version__, torch.cuda.is_available())"
  } finally { Pop-Location }
}
Write-Host $importCheck.Text
if ($importCheck.ExitCode -ne 0) { throw 'Motion environment verification failed' }

# ---- weights
$checkpoints = @()
if (-not $SkipModels) {
  New-Item -ItemType Directory -Force -Path $models | Out-Null
  foreach ($entry in @($spec.CoTrackerCheckpoint, $spec.DepthCheckpoint)) {
    $repoId, $file = $entry -split ':', 2
    Write-Host "Downloading $file from $repoId ..."
    $download = Invoke-NativeCaptured {
      & $python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='$repoId', filename='$file', local_dir=r'$models'))"
    }
    Write-Host $download.Text
    if ($download.ExitCode -ne 0) { throw "Downloading $file failed; rerun Setup-Motion.bat to resume" }
    $path = Join-Path $models $file
    $checkpoints += @{ file = $file; repo = $repoId; bytes = (Get-Item -LiteralPath $path).Length; sha256 = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower() }
  }
}

@{
  schemaVersion = 1
  python = $spec.Python
  torch = $spec.Torch
  coTrackerCommit = $spec.CoTrackerCommit
  videoDepthCommit = $spec.VideoDepthCommit
  checkpoints = $checkpoints
  licences = @('CoTracker3: CC-BY-NC 4.0 (non-commercial use only)', 'Video Depth Anything Small: Apache-2.0')
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $readyMarker -Encoding Ascii

Write-Host "Motion runtime ready at $venv"
Write-Host 'CoTracker3 is CC-BY-NC 4.0: non-commercial use only. The built-in Lucas-Kanade tracker needs no model.'
