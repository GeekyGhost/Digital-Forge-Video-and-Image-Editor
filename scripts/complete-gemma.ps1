[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

Write-Host 'Before continuing:' -ForegroundColor Yellow
Write-Host '1. Open https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized'
Write-Host '2. Sign in and accept Google Gemma access terms.'
Write-Host '3. Authenticate this machine with: .\.venv\Scripts\hf.exe auth login'
Write-Host ''

& $Python (Join-Path $ProjectRoot 'scripts\models.py') install ltx-core --accept-licenses

