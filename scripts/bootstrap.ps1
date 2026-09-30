[CmdletBinding()]
param(
    [switch]$SkipNpm,
    [switch]$SkipPython,
    [switch]$RecreatePython
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')
Set-Location -LiteralPath $ProjectRoot

function Require-Command([string]$Name, [string]$Help) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found. $Help"
    }
}

Require-Command 'node' 'Install a current Node.js LTS release.'
Require-Command 'npm.cmd' 'Install npm with Node.js.'
Require-Command 'ffmpeg' 'Install FFmpeg and add its bin directory to PATH.'
Require-Command 'ffprobe' 'Install FFprobe with FFmpeg and add it to PATH.'
Require-Command 'nvidia-smi' 'Install an NVIDIA Studio driver.'

$GpuInfo = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
Write-Host "GPU: $GpuInfo"
$MemoryText = (& nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | Select-Object -First 1).Trim()
$VramMiB = [int]$MemoryText
if ($VramMiB -lt 15360) {
    throw "Digital Forge's selected model profile requires a 16 GB-class GPU. Detected $VramMiB MiB."
}

if (-not (Get-Command 'cargo' -ErrorAction SilentlyContinue)) {
    Write-Warning 'Rust is not installed. Install rustup from https://rustup.rs, restart the terminal, then rerun this script.'
} else {
    Write-Host "Rust: $(& rustc --version)"
}

New-Item -ItemType Directory -Force -Path '.\models', '.\cache', '.\projects', '.\exports' | Out-Null

if (-not $SkipNpm) {
    npm.cmd install
}

if (-not $SkipPython) {
    $SetupArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'setup-app-env.ps1'))
    if ($RecreatePython) { $SetupArgs += '-Recreate' }
    & powershell.exe @SetupArgs
    if ($LASTEXITCODE -ne 0) { throw 'Application Python environment setup failed.' }
}

Write-Host ''
Write-Host 'Digital Forge foundation is ready.' -ForegroundColor Green
Write-Host 'Run: npm.cmd run tauri dev'
Write-Host 'List models: .\.venv\Scripts\python.exe .\scripts\models.py list'
