[CmdletBinding()]
param(
    [switch]$SkipInstall,
    [switch]$SkipTests
)

# Compiles the current source into Releases\aiv-studio-<version>.exe.
# If that file is in use, publish a timestamped side-by-side executable instead. The launcher
# runs whichever binary is newest, so a running prior build does not block a successful release.

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ReleasesDir = Join-Path $ProjectRoot 'Releases'
$BuildDir = Join-Path $ProjectRoot 'src-tauri\target\release'

function Require-Command([string]$Name, [string]$Help) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name was not found. $Help"
    }
}

Require-Command 'npm.cmd' 'Install Node.js and npm.'
Require-Command 'cargo' 'Install Rust with rustup, restart the terminal, and rerun this script.'

# The Tauri CLI picks its application directory by finding a tauri.conf.json. If one appears at the
# project root it wins over src-tauri\, and cargo then tries to parse a root manifest that has no
# lib.rs behind it. Catch that here with a readable message instead of a cargo metadata failure.
foreach ($Stray in @('Cargo.toml', 'tauri.conf.json')) {
    $StrayPath = Join-Path $ProjectRoot $Stray
    if (Test-Path -LiteralPath $StrayPath -PathType Leaf) {
        throw "$Stray exists at the project root and shadows src-tauri\$Stray. The root is the React interface, not a Rust crate; move the stray file into .stale-root-crate\ and rerun."
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $ProjectRoot 'src-tauri\tauri.conf.json') -PathType Leaf)) {
    throw 'src-tauri\tauri.conf.json is missing; the native shell cannot be built.'
}

$Version = (Get-Content -LiteralPath (Join-Path $ProjectRoot 'package.json') -Raw | ConvertFrom-Json).version
Write-Host "Building Digital Forge Video Studio $Version" -ForegroundColor Cyan

Push-Location -LiteralPath $ProjectRoot
try {
    if (-not $SkipInstall) {
        Write-Host 'Restoring locked JavaScript dependencies' -ForegroundColor Green
        npm.cmd ci
        if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
    }

    if (-not $SkipTests) {
        Write-Host 'Running release qualification tests' -ForegroundColor Green
        & (Join-Path $PSScriptRoot 'test-all.ps1') -SkipInstall -SkipFrontendBuild
        if ($LASTEXITCODE -ne 0) { throw 'Release qualification failed.' }
    }

    Write-Host 'Type-checking and bundling the interface' -ForegroundColor Green
    npm.cmd run build
    if ($LASTEXITCODE -ne 0) { throw 'The frontend build failed; fix the reported errors and rerun.' }

    Write-Host 'Compiling the native shell (this takes a few minutes on a cold cache)' -ForegroundColor Green
    npm.cmd run tauri build
    if ($LASTEXITCODE -ne 0) { throw 'The Tauri build failed; fix the reported errors and rerun.' }
} finally {
    Pop-Location
}

# Tauri names the main binary from the Cargo package or the configured product name depending on
# version, so take the newest non-installer executable it produced rather than guessing.
$Built = Get-ChildItem -LiteralPath $BuildDir -Filter '*.exe' -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notmatch 'setup|install|uninstall|webview' } |
    Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
if (-not $Built) { throw "No executable was produced in $BuildDir." }

if (-not (Test-Path -LiteralPath $ReleasesDir)) { New-Item -ItemType Directory -Path $ReleasesDir | Out-Null }
$Destination = Join-Path $ReleasesDir "aiv-studio-$Version.exe"
$PublishedDestination = $Destination
try {
    Copy-Item -LiteralPath $Built.FullName -Destination $Destination -Force
} catch {
    if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
        throw
    }

    $Timestamp = Get-Date -Format 'yyyyMMdd-HHmmssfff'
    $PublishedDestination = Join-Path $ReleasesDir "aiv-studio-$Version-$Timestamp.exe"
    Write-Warning "$Destination could not be replaced, most likely because it is currently running."
    Write-Warning "Publishing this build side by side as $PublishedDestination."
    Copy-Item -LiteralPath $Built.FullName -Destination $PublishedDestination -Force
}

Write-Host ''
Write-Host "Published $PublishedDestination" -ForegroundColor Green
Write-Host 'Double-click Launch-AIV-Studio.vbs to run it.' -ForegroundColor Green
