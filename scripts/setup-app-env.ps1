[CmdletBinding()]
param([switch]$Recreate)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')
$Matrix = Get-AivRuntimeMatrix
$Spec = $Matrix.Application
$Venv = Join-Path $ProjectRoot $Spec.Venv
$Python = Join-Path $Venv 'Scripts\python.exe'
$Uv = Resolve-AivUv -Persist

$Broken = (Test-Path -LiteralPath $Venv) -and -not (Test-PythonInterpreter $Python)
if ($Broken) {
    Write-Warning "The application environment cannot start and will be recreated: $Venv"
    $Recreate = $true
}
if ($Recreate -and (Test-Path -LiteralPath $Venv)) {
    Remove-AivVenv -Path $Venv -ProjectRoot $ProjectRoot -Confirm:$false
}

if (-not (Test-PythonInterpreter $Python)) {
    & $Uv python install $Spec.Python
    if ($LASTEXITCODE -ne 0) { throw "Could not install Python $($Spec.Python)." }
    & $Uv venv --python $Spec.Python $Venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create $Venv." }
}

& $Uv pip install --python $Python --reinstall `
    "torch==$($Spec.Torch)" "torchvision==$($Spec.TorchVision)" `
    --index-url $Spec.TorchIndex
if ($LASTEXITCODE -ne 0) { throw 'Installing the application CUDA runtime failed.' }

& $Uv pip install --python $Python -e "$ProjectRoot\ai-runtime[dev,inference]"
if ($LASTEXITCODE -ne 0) { throw 'Installing the application worker failed.' }

& $Python -c "import sys, torch, torchvision, aiv_runtime; assert sys.version.startswith('$($Spec.Python)'); assert torch.__version__ == '$($Spec.Torch)'; assert torch.cuda.is_available(); print(sys.version.split()[0], torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'Application environment verification failed.' }

Write-Host "Application runtime ready at $Venv" -ForegroundColor Green
