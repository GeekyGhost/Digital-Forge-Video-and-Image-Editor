# Shared helpers for the Digital Forge setup scripts.
# Dot-source this from any script under scripts\:  . (Join-Path $PSScriptRoot 'common.ps1')

# Keep every uv-managed Python interpreter and the uv package cache inside the project folder, so
# the whole install is self-contained and deleting the folder removes all of it. uv would otherwise
# put both under %LOCALAPPDATA%/%APPDATA%. An explicit value set by the user still wins.
$AivCommonRoot = Split-Path -Parent $PSScriptRoot
if (-not $env:UV_PYTHON_INSTALL_DIR) { $env:UV_PYTHON_INSTALL_DIR = Join-Path $AivCommonRoot '.aiv\python' }
if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $AivCommonRoot '.aiv\uv-cache' }

function Get-AivRuntimeMatrix {
    $MatrixPath = Join-Path $PSScriptRoot 'runtime-matrix.psd1'
    if (-not (Test-Path -LiteralPath $MatrixPath -PathType Leaf)) {
        throw "Runtime matrix is missing: $MatrixPath"
    }
    return Import-PowerShellDataFile -LiteralPath $MatrixPath
}

function Test-PythonInterpreter {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Path)

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $Previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Path -c 'import sys; sys.exit(0)' *> $null
        return $LASTEXITCODE -eq 0
    } catch {
        return $false
    } finally {
        $ErrorActionPreference = $Previous
    }
}

function Install-AivUv {
    [CmdletBinding()]
    param([Parameter(Mandatory)][string]$Destination)

    $Url = 'https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip'
    $ToolDir = Split-Path -Parent $Destination
    $Temp = Join-Path $ToolDir 'uv-download'
    New-Item -ItemType Directory -Force -Path $Temp | Out-Null
    $Zip = Join-Path $Temp 'uv.zip'
    Write-Host "uv was not found; downloading it into $ToolDir" -ForegroundColor Cyan
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    $PreviousProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $Url -OutFile $Zip -UseBasicParsing
        Expand-Archive -LiteralPath $Zip -DestinationPath $Temp -Force
        $Exe = Get-ChildItem -LiteralPath $Temp -Filter 'uv.exe' -Recurse -File | Select-Object -First 1
        if (-not $Exe) { throw "the download did not contain uv.exe: $Url" }
        Copy-Item -LiteralPath $Exe.FullName -Destination $Destination -Force
    } finally {
        $ProgressPreference = $PreviousProgress
        Remove-Item -LiteralPath $Temp -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Resolve-AivUv {
    [CmdletBinding()]
    param([switch]$Persist)

    $ProjectRoot = Split-Path -Parent $PSScriptRoot
    $PersistentPath = Join-Path $ProjectRoot '.aiv\tools\uv.exe'
    $Candidates = @()
    $Command = Get-Command 'uv' -ErrorAction SilentlyContinue
    if ($Command) { $Candidates += $Command.Source }
    $Candidates += $PersistentPath
    $Candidates += (Join-Path $ProjectRoot '.venv\Scripts\uv.exe')

    $Uv = $Candidates | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
        Select-Object -First 1
    if (-not $Uv) {
        # A fresh machine has no uv. Fetch the official standalone build into .aiv\tools rather
        # than installing it system-wide, so it stays inside the project like everything else.
        try {
            Install-AivUv -Destination $PersistentPath
        } catch {
            throw "uv was not found and could not be downloaded ($($_.Exception.Message)). Install uv from https://docs.astral.sh/uv/ and rerun setup."
        }
        $Uv = $PersistentPath
    }

    if ($Persist -and -not [string]::Equals($Uv, $PersistentPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $PersistentPath) | Out-Null
        Copy-Item -LiteralPath $Uv -Destination $PersistentPath -Force
        $Uv = $PersistentPath
    }
    return $Uv
}

function Resolve-ProjectPython {
    [CmdletBinding()]
    param([string]$Version = (Get-AivRuntimeMatrix).Application.Python)

    if (Get-Command 'py' -ErrorAction SilentlyContinue) {
        $Previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $Found = & py "-$Version" -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $Found) {
                return ($Found | Select-Object -First 1).ToString().Trim()
            }
        } finally {
            $ErrorActionPreference = $Previous
        }
    }

    $Uv = Resolve-AivUv -Persist
    & $Uv python install $Version
    if ($LASTEXITCODE -ne 0) { throw "uv could not install Python $Version." }
    $Found = & $Uv python find $Version
    if ($LASTEXITCODE -ne 0 -or -not $Found) { throw "uv could not resolve Python $Version." }
    return ($Found | Select-Object -First 1).ToString().Trim()
}

function Remove-AivVenv {
    [CmdletBinding(SupportsShouldProcess)]
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$ProjectRoot
    )

    $Root = [System.IO.Path]::GetFullPath($ProjectRoot).TrimEnd('\')
    $Target = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    if (-not $Target.StartsWith($Root + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove an environment outside the project: $Target"
    }
    if ([System.IO.Path]::GetFileName($Target) -ne '.venv') {
        throw "Refusing to remove a path that is not a .venv directory: $Target"
    }
    if ((Test-Path -LiteralPath $Target) -and $PSCmdlet.ShouldProcess($Target, 'remove virtual environment')) {
        Remove-Item -LiteralPath $Target -Recurse -Force
    }
}
