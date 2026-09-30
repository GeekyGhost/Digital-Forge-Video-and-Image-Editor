[CmdletBinding()]
param(
    [switch]$Recreate,
    [switch]$SkipTorch
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')
$Spec = (Get-AivRuntimeMatrix).Moss
$RuntimeRoot = Join-Path $ProjectRoot $Spec.Venv
$RuntimePython = Join-Path $RuntimeRoot 'Scripts\python.exe'
$PackageRoot = Join-Path $ProjectRoot 'ai-runtime\moss_soundeffect_v2'
$ReadyMarker = Join-Path $RuntimeRoot '.moss-sfx-ready'
$Uv = Resolve-AivUv -Persist

if (-not (Test-Path -LiteralPath (Join-Path $PackageRoot 'pyproject.toml') -PathType Leaf)) {
    throw "The MOSS runtime package is missing from $PackageRoot"
}

$Broken = (Test-Path -LiteralPath $RuntimeRoot) -and -not (Test-PythonInterpreter $RuntimePython)
if ($Broken) {
    Write-Warning "The MOSS environment cannot start and will be recreated: $RuntimeRoot"
    $Recreate = $true
}
if ($Recreate -and (Test-Path -LiteralPath $RuntimeRoot)) {
    $Expected = [System.IO.Path]::GetFullPath((Join-Path $ProjectRoot $Spec.Venv)).TrimEnd('\')
    $Target = [System.IO.Path]::GetFullPath($RuntimeRoot).TrimEnd('\')
    if (-not [string]::Equals($Expected, $Target, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove unexpected MOSS runtime path: $Target"
    }
    Remove-Item -LiteralPath $Target -Recurse -Force
}

if (-not (Test-PythonInterpreter $RuntimePython)) {
    & $Uv python install $Spec.Python
    if ($LASTEXITCODE -ne 0) { throw "Could not install Python $($Spec.Python)." }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $RuntimeRoot) | Out-Null
    & $Uv venv --python $Spec.Python $RuntimeRoot
    if ($LASTEXITCODE -ne 0) { throw "Could not create $RuntimeRoot." }
}

if (-not $SkipTorch) {
    & $Uv pip install --python $RuntimePython --reinstall `
        "torch==$($Spec.Torch)" "torchaudio==$($Spec.TorchAudio)" --index-url $Spec.TorchIndex
    if ($LASTEXITCODE -ne 0) { throw 'Installing the MOSS CUDA runtime failed.' }
}
& $Uv pip install --python $RuntimePython $PackageRoot
if ($LASTEXITCODE -ne 0) { throw 'Installing MOSS-SoundEffect failed.' }

& $RuntimePython -c "import sys, torch, torchaudio; from moss_soundeffect_v2 import MossSoundEffectPipeline; assert sys.version.startswith('$($Spec.Python)'); assert torch.__version__ == '$($Spec.Torch)'; assert torch.cuda.is_available(); print(sys.version.split()[0], torch.__version__, torchaudio.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'MOSS runtime verification failed.' }
Set-Content -LiteralPath $ReadyMarker -Value (Get-Date).ToUniversalTime().ToString('o') -Encoding utf8
Write-Host "MOSS-SoundEffect runtime ready at $RuntimeRoot" -ForegroundColor Green
