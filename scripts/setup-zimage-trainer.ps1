<# Build the isolated Z-Image LoRA captioning and training environment. #>
[CmdletBinding()]
param([switch]$Recreate)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')
$spec = (Get-AivRuntimeMatrix).ZImageTrainer
$runtimeRoot = Join-Path $root 'vendor\Z-Image-Trainer'
$source = Join-Path $runtimeRoot 'diffusers'
$venv = Join-Path $runtimeRoot '.venv'
$python = Join-Path $venv 'Scripts\python.exe'
$readyMarker = Join-Path $runtimeRoot '.aiv-ready.json'
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

if ((Test-Path -LiteralPath $venv) -and -not (Test-PythonInterpreter $python)) { $Recreate = $true }
if ($Recreate -and (Test-Path -LiteralPath $venv)) {
  Remove-AivVenv -Path $venv -ProjectRoot $root -Confirm:$false
}
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
if (-not (Test-Path -LiteralPath $python)) {
  & $uv python install $spec.Python
  & $uv venv --python $spec.Python $venv
  if ($LASTEXITCODE -ne 0) { throw 'Could not create the Z-Image trainer environment' }
}

if (-not (Test-Path -LiteralPath (Join-Path $source '.git'))) {
  git clone --filter=blob:none --no-checkout https://github.com/huggingface/diffusers.git $source
  if ($LASTEXITCODE -ne 0) { throw 'Could not clone Diffusers for the Z-Image trainer' }
}
$trainerPatch = Join-Path $root 'scripts\patches\zimage-aiv-training.patch'
$trainerRelative = 'examples/dreambooth/train_dreambooth_lora_z_image.py'
# Restore the pristine trainer before checkout and patching. Reverse-applying the current patch only
# works when the installed copy carries this exact patch; after the patch is revised, an older copy
# would block the refresh. The trainer file is the only file the AIV patch touches.
if (Test-Path -LiteralPath (Join-Path $source $trainerRelative)) {
  git -C $source checkout -- $trainerRelative
  if ($LASTEXITCODE -ne 0) { throw 'Could not restore the pristine Z-Image trainer before patching' }
}
git -C $source fetch --depth 1 origin $spec.DiffusersCommit
if ($LASTEXITCODE -ne 0) { throw 'Could not fetch the pinned Diffusers trainer revision' }
git -C $source checkout --detach $spec.DiffusersCommit
if ($LASTEXITCODE -ne 0) { throw 'Could not check out the pinned Diffusers trainer revision' }
git -C $source checkout -- $trainerRelative
if ($LASTEXITCODE -ne 0) { throw 'Could not restore the pristine Z-Image trainer before patching' }
$patchCheck = Invoke-NativeCaptured { git -C $source apply --check $trainerPatch }
if ($patchCheck.ExitCode -ne 0) {
  Write-Host $patchCheck.Text
  throw 'The pinned Z-Image trainer no longer matches the AIV integration patch'
}
git -C $source apply $trainerPatch
if ($LASTEXITCODE -ne 0) { throw 'Could not apply the AIV trainer patch (pause/resume, continuation, timestep shift, Turbo adapter)' }

function Install-Packages([string[]]$Packages) {
  & $uv pip install --python $python @Packages
  if ($LASTEXITCODE -ne 0) { throw "Installing $($Packages -join ' ') failed" }
}

Install-Packages @("torch==$($spec.Torch)", "torchvision==$($spec.TorchVision)", '--index-url', $spec.TorchIndex)
Install-Packages @('-e', $source)
Install-Packages @(
  "transformers==$($spec.Transformers)", 'accelerate==1.12.0', 'peft==0.18.1', 'safetensors==0.8.0',
  'pillow==12.1.1', 'numpy<2', 'sentencepiece', 'protobuf', 'huggingface-hub', 'tqdm',
  'datasets', 'bitsandbytes', 'torchao==0.17.0', 'ftfy', 'tensorboard', 'Jinja2'
)

$env:PYTHONPATH = Join-Path $root 'ai-runtime'
$env:PYTHONIOENCODING = 'utf-8'
# torch, torchao and diffusers print harmless warnings on stderr (triton missing, Windows redirects,
# torchao tensor subclasses); only the exit code decides success.
$importCheck = Invoke-NativeCaptured { & $python -c "import torch, bitsandbytes, torchao; from diffusers import ZImagePipeline; import aiv_runtime.zimage_training, aiv_runtime.zimage_caption_worker; print('trainer imports OK', torch.__version__, torch.cuda.is_available())" }
Write-Host $importCheck.Text
if ($importCheck.ExitCode -ne 0) { throw 'Z-Image trainer verification failed' }
# ostris' Z-Image-Turbo training adapter (~340 MB). Turbo LoRA training keeps it active and frozen so
# the few-step distillation survives; it is never loaded for generation. Stored inside the project.
$adapterDir = Join-Path $root 'models\Z-Image-Turbo-Training-Adapter'
$adapterFile = Join-Path $adapterDir 'zimage_turbo_training_adapter_v2.safetensors'
if (-not (Test-Path -LiteralPath $adapterFile)) {
  New-Item -ItemType Directory -Force -Path $adapterDir | Out-Null
  $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
  Write-Host 'Downloading the Z-Image-Turbo training adapter (about 340 MB)...'
  $download = Invoke-NativeCaptured { & $python -c "import sys; from huggingface_hub import hf_hub_download; hf_hub_download('ostris/zimage_turbo_training_adapter', 'zimage_turbo_training_adapter_v2.safetensors', local_dir=sys.argv[1])" $adapterDir }
  if ($download.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $adapterFile)) {
    Write-Host $download.Text
    throw 'Could not download the Z-Image-Turbo training adapter'
  }
}

$trainer = Join-Path $source 'examples\dreambooth\train_dreambooth_lora_z_image.py'
if (-not (Test-Path -LiteralPath $trainer)) { throw "Pinned Z-Image trainer script missing: $trainer" }
$trainerHelp = Invoke-NativeCaptured { & $python $trainer --help }
if ($trainerHelp.ExitCode -ne 0 -or $trainerHelp.Text -notmatch '--aiv_pause_file' -or $trainerHelp.Text -notmatch '--aiv_timestep_shift' -or $trainerHelp.Text -notmatch '--aiv_assistant_lora_path') {
  Write-Host $trainerHelp.Text
  throw 'The Z-Image trainer AIV integration (pause/resume, timestep shift, Turbo adapter) is missing'
}
@{
  schemaVersion = 1
  diffusersCommit = $spec.DiffusersCommit
  python = $spec.Python
  torch = $spec.Torch
  transformers = $spec.Transformers
} | ConvertTo-Json | Set-Content -LiteralPath $readyMarker -Encoding Ascii
Write-Host "Z-Image trainer runtime ready at $venv"
