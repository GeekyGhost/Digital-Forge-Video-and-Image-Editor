[CmdletBinding()]
param(
    [switch]$ForceSetup,
    [switch]$NoLaunch
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')
$RuntimeMatrix = Get-AivRuntimeMatrix
$StateRoot = Join-Path $ProjectRoot '.aiv'
$StatePath = Join-Path $StateRoot 'setup-state.json'
$AppVenv = Join-Path $ProjectRoot '.venv'
$AppPython = Join-Path $AppVenv 'Scripts\python.exe'
$LtxRoot = Join-Path $ProjectRoot 'vendor\LTX-2'
$LtxPython = Join-Path $LtxRoot '.venv\Scripts\python.exe'
$ReleasesDir = Join-Path $ProjectRoot 'Releases'
$BuildDir = Join-Path $ProjectRoot 'src-tauri\target\release'

function Require-Command([string]$Name, [string]$Help) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found. $Help"
    }
}

function Get-HashOrEmpty([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return '' }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

function Run-Checked([string]$Description, [scriptblock]$Action) {
    Write-Host "[$Description]" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) { throw "$Description failed with exit code $LASTEXITCODE" }
}

# Runs a probe in an interpreter and reports pass/fail as a plain boolean.
#
# Two PowerShell traps this avoids. First, $ErrorActionPreference = 'Stop' turns *any* stderr
# output from a native command into a terminating error, so a probe that is meant to answer
# "is this installed?" would crash the launcher the moment the answer was no. Second, the script
# has to be a single line: PowerShell mangles embedded newlines when passing an argument to a
# native executable, and python reports the result as a syntax error in "<string>".
function Test-PythonProbe([string]$Interpreter, [string]$Script) {
    if (-not (Test-Path -LiteralPath $Interpreter -PathType Leaf)) { return $false }
    if ($Script -match "`n") { throw 'Test-PythonProbe requires a single-line script.' }
    $Previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Interpreter -c $Script *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $Previous
    }
}

New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
$SavedState = $null
if (Test-Path -LiteralPath $StatePath) {
    try { $SavedState = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json } catch { $SavedState = $null }
}
$HasSavedState = $null -ne $SavedState

$PackageHash = Get-HashOrEmpty (Join-Path $ProjectRoot 'package-lock.json')
$RuntimeHash = Get-HashOrEmpty (Join-Path $ProjectRoot 'ai-runtime\pyproject.toml')
$LtxLockHash = Get-HashOrEmpty (Join-Path $LtxRoot 'uv.lock')
$RuntimeMatrixHash = Get-HashOrEmpty (Join-Path $PSScriptRoot 'runtime-matrix.psd1')

Write-Host 'Digital Forge environment check' -ForegroundColor Green
Write-Host "Root: $ProjectRoot"

Require-Command 'ffmpeg' 'Install FFmpeg and add its bin directory to PATH.'
Require-Command 'ffprobe' 'Install FFprobe and add its bin directory to PATH.'
Require-Command 'nvidia-smi' 'Install or update the NVIDIA Studio driver.'

$Gpu = (& nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | Select-Object -First 1).Trim()
Write-Host "GPU: $Gpu"

$NodeSetupNeeded = $ForceSetup -or -not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'node_modules')) -or ($HasSavedState -and $SavedState.packageLockHash -ne $PackageHash)
if ($NodeSetupNeeded) {
    Require-Command 'node' 'Install Node.js before building Digital Forge.'
    Require-Command 'npm.cmd' 'Install npm with Node.js.'
    Push-Location -LiteralPath $ProjectRoot
    try { Run-Checked 'Installing JavaScript dependencies' { npm.cmd ci } } finally { Pop-Location }
} else {
    Write-Host '[JavaScript dependencies ready]' -ForegroundColor DarkGreen
}

$AppBroken = (Test-Path -LiteralPath $AppVenv) -and -not (Test-PythonProbe $AppPython 'import sys; sys.exit(0)')
$PythonSetupNeeded = $ForceSetup -or $AppBroken -or -not (Test-Path -LiteralPath $AppPython) -or
    ($HasSavedState -and ($SavedState.aiRuntimeHash -ne $RuntimeHash -or $SavedState.runtimeMatrixHash -ne $RuntimeMatrixHash))
if (-not $PythonSetupNeeded) {
    $ExpectedTorch = $RuntimeMatrix.Application.Torch
    $ProbeScript = "import sys, aiv_runtime, torch, diffusers, transformers, einops, torchvision; sys.exit(0 if torch.cuda.is_available() and torch.__version__ == '$ExpectedTorch' and int(transformers.__version__.split('.')[0]) < 5 else 1)"
    $PythonSetupNeeded = -not (Test-PythonProbe $AppPython $ProbeScript)
}
if ($PythonSetupNeeded) {
    $SetupArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'setup-app-env.ps1'))
    if ($ForceSetup -or $AppBroken) { $SetupArgs += '-Recreate' }
    Run-Checked 'Creating and validating application Python environment' { & powershell.exe @SetupArgs }
} else {
    Write-Host '[Application Python environment ready]' -ForegroundColor DarkGreen
}

# The optional AI packs each live in their own environment and are each installed by their own
# script. Report every one of them, including the ones this launcher does not install: a fresh
# clone has no vendor\ at all, and staying silent about that produced an editor whose AI panels
# simply read "not ready" with nothing explaining why.
$LtxSourcePresent = Test-Path -LiteralPath (Join-Path $LtxRoot 'uv.lock')
$LtxCudaReady = Test-PythonProbe $LtxPython 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'
$LtxSetupNeeded = $LtxSourcePresent -and ($ForceSetup -or -not $LtxCudaReady -or ($HasSavedState -and $SavedState.ltxLockHash -ne $LtxLockHash))
if ($LtxSetupNeeded) {
    $LtxArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'setup-ltx.ps1'))
    if ($ForceSetup) { $LtxArgs += '-Recreate' }
    Run-Checked 'Installing and validating the LTX CUDA environment' { powershell.exe @LtxArgs }
} elseif ($LtxCudaReady) {
    Write-Host '[LTX CUDA environment ready]' -ForegroundColor DarkGreen
} elseif (-not $LtxSourcePresent) {
    Write-Warning 'LTX is not installed: vendor\LTX-2 is absent.'
    Write-Warning '  Run scripts\setup-ltx.ps1 to fetch the pinned source and build its environment.'
    Write-Warning '  LTX generation, retake and foley stay disabled until then; the editor is otherwise fully usable.'
}

if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'vendor\MiniMax-H3\.venv'))) {
    Write-Warning 'MiniMax-H3 is not installed. Run scripts\setup-h3.ps1 to enable it.'
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'vendor\Image-Studio\.venv'))) {
    Write-Warning 'Image Studio is not installed. Run scripts\setup-image-studio.ps1 to enable it.'
}

$env:VIRTUAL_ENV = $AppVenv
$env:PATH = (Join-Path $AppVenv 'Scripts') + ';' + $env:PATH
$env:AIV_ROOT = $ProjectRoot
$env:AIV_MODELS = Join-Path $ProjectRoot 'models'
$env:AIV_CACHE = Join-Path $ProjectRoot 'cache'
$env:AIV_LTX_ROOT = $LtxRoot

$LtxCheckpoint = Join-Path $ProjectRoot 'models\LTX-2.3\ltx-2.3-22b-distilled-fp8.safetensors'
$GemmaIndex = Join-Path $ProjectRoot 'models\gemma-3-12b\model.safetensors.index.json'
$LtxPackReady = (Test-Path -LiteralPath $LtxCheckpoint -PathType Leaf) -and (Test-Path -LiteralPath $GemmaIndex -PathType Leaf)
if (-not $LtxPackReady) {
    Write-Warning 'LTX generation is disabled until the official distilled FP8 checkpoint and gated Gemma encoder are installed.'
}

$CurrentState = [ordered]@{
    schemaVersion = 1
    updatedAt = (Get-Date).ToUniversalTime().ToString('o')
    packageLockHash = $PackageHash
    aiRuntimeHash = $RuntimeHash
    ltxLockHash = $LtxLockHash
    runtimeMatrixHash = $RuntimeMatrixHash
    gpu = $Gpu
}
$CurrentState | ConvertTo-Json | Set-Content -LiteralPath $StatePath -Encoding UTF8

if ($NoLaunch) {
    Write-Host 'Environment validation completed; launch skipped.' -ForegroundColor Green
    exit 0
}

# Launch the most recently produced binary rather than a hard-coded version, so a fresh
# `npm run release` always wins over an older file sitting in Releases\.
$Candidates = @()
foreach ($Directory in @($ReleasesDir, $BuildDir)) {
    if (Test-Path -LiteralPath $Directory) {
        $Candidates += Get-ChildItem -LiteralPath $Directory -Filter '*.exe' -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -notmatch 'setup|install|uninstall|webview' }
    }
}
$Executable = $Candidates | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1

# A binary older than the source it was built from is the classic "my changes did not appear"
# trap, so say so instead of silently launching a stale build.
if ($Executable) {
    $SourcePaths = @('src', 'src-tauri\src', 'index.html', 'package.json') |
        ForEach-Object { Join-Path $ProjectRoot $_ } | Where-Object { Test-Path -LiteralPath $_ }
    $NewestSource = $SourcePaths |
        ForEach-Object { Get-ChildItem -LiteralPath $_ -Recurse -File -ErrorAction SilentlyContinue } |
        Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    if ($NewestSource -and $NewestSource.LastWriteTimeUtc -gt $Executable.LastWriteTimeUtc) {
        Write-Warning "$($Executable.Name) was built before the current source ($($NewestSource.Name) is newer)."
        Write-Warning 'Run "npm.cmd run release" to rebuild, or your latest changes will not be visible.'
    }
    Write-Host "Launching $($Executable.FullName)" -ForegroundColor Green
    Start-Process -FilePath $Executable.FullName -WorkingDirectory $ProjectRoot
    exit 0
}

Require-Command 'cargo' 'Install Rust with rustup, restart the terminal, and rerun this launcher.'
Require-Command 'npm.cmd' 'Install Node.js and npm.'
Write-Host 'No compiled executable found; starting Tauri development mode.' -ForegroundColor Yellow
Push-Location -LiteralPath $ProjectRoot
try { npm.cmd run tauri dev } finally { Pop-Location }
