[CmdletBinding()]
param(
    [switch]$NoFetch,
    [switch]$Recreate,
    [switch]$RepairSource
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')
$Spec = (Get-AivRuntimeMatrix).Ltx
$LtxRoot = Join-Path $ProjectRoot 'vendor\LTX-2'
$LtxVenv = Join-Path $ProjectRoot $Spec.Venv
$LtxPython = Join-Path $LtxVenv 'Scripts\python.exe'
$LockPath = Join-Path $LtxRoot 'uv.lock'
$LtxRepo = 'https://github.com/Lightricks/LTX-2.git'
$Uv = Resolve-AivUv -Persist

if (-not (Test-Path -LiteralPath $LockPath -PathType Leaf)) {
    if ($NoFetch) { throw 'LTX source is missing and -NoFetch was specified.' }
    if (-not (Get-Command 'git' -ErrorAction SilentlyContinue)) { throw 'Git is required to fetch LTX.' }
    New-Item -ItemType Directory -Force -Path $LtxRoot | Out-Null
    if (-not (Test-Path -LiteralPath (Join-Path $LtxRoot '.git'))) {
        & git init --quiet $LtxRoot
        & git -C $LtxRoot remote add origin $LtxRepo
    }
    & git -C $LtxRoot fetch --depth 1 origin $Spec.Commit
    if ($LASTEXITCODE -ne 0) { throw 'Fetching the pinned LTX source failed.' }
    & git -C $LtxRoot checkout --quiet --detach FETCH_HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Checking out the pinned LTX source failed.' }
}

$Head = (& git -C $LtxRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $Head -ne $Spec.Commit) {
    throw "LTX source is $Head; expected $($Spec.Commit)."
}

if ($RepairSource) {
    & git -C $LtxRoot restore --source HEAD -- uv.lock
    if ($LASTEXITCODE -ne 0) { throw 'Restoring the pristine LTX lock failed.' }
}
$LockHash = (Get-FileHash -LiteralPath $LockPath -Algorithm SHA256).Hash
if ($LockHash -ne $Spec.LockSha256) {
    throw "LTX uv.lock is modified ($LockHash). Expected $($Spec.LockSha256). Use -RepairSource to restore it."
}

$Broken = (Test-Path -LiteralPath $LtxVenv) -and -not (Test-PythonInterpreter $LtxPython)
if ($Broken) {
    Write-Warning "The LTX environment cannot start and will be recreated: $LtxVenv"
    $Recreate = $true
}
if ($Recreate -and (Test-Path -LiteralPath $LtxVenv)) {
    Remove-AivVenv -Path $LtxVenv -ProjectRoot $ProjectRoot -Confirm:$false
}

Push-Location -LiteralPath $LtxRoot
try {
    & $Uv python install $Spec.Python
    if ($LASTEXITCODE -ne 0) { throw "Could not install Python $($Spec.Python)." }
    & $Uv sync --frozen --no-dev --python $Spec.Python `
        --no-install-package torch --no-install-package torchvision --no-install-package torchaudio
    if ($LASTEXITCODE -ne 0) { throw 'Installing the pristine LTX lock failed.' }
} finally {
    Pop-Location
}

& $Uv pip install --python $LtxPython --reinstall `
    "torch==$($Spec.Torch)" "torchvision==$($Spec.TorchVision)" "torchaudio==$($Spec.TorchAudio)" `
    --index-url $Spec.TorchIndex
if ($LASTEXITCODE -ne 0) { throw 'Installing the validated LTX CUDA runtime failed.' }

& $Uv pip install --python $LtxPython "triton-windows==$($Spec.TritonWindows)"
if ($LASTEXITCODE -ne 0) { throw 'Installing the pinned LTX Triton runtime failed.' }
& $Uv pip install --python $LtxPython --no-deps $Spec.SageWheel
if ($LASTEXITCODE -ne 0) { throw 'Installing the pinned LTX SageAttention wheel failed.' }

& $LtxPython -c "import sys, torch, torchaudio, torchvision, triton, sageattention; assert sys.version.startswith('$($Spec.Python)'); assert torch.__version__ == '$($Spec.Torch)'; assert torch.cuda.is_available(); print(sys.version.split()[0], torch.__version__, torchaudio.__version__, torchvision.__version__, triton.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'LTX environment verification failed.' }

Write-Host "LTX runtime ready at $LtxVenv" -ForegroundColor Green
