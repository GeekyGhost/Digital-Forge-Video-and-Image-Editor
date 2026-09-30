<#
.SYNOPSIS
  Guided installer for Digital Forge Video and Image Editor.

.DESCRIPTION
  Normally started by double-clicking Install-Digital-Forge.bat in the project folder.

  Everything this installer creates lives INSIDE the project folder: the uv tool, the Python
  interpreters, every virtual environment, the uv package cache, model weights, caches, logs and
  the compiled app. Deleting the folder removes all of it.

  System prerequisites (NVIDIA driver, FFmpeg, Node.js, Git, WebView2, and - only when building
  from source - Rust and the Visual C++ build tools) are checked first. Missing ones can be
  installed with winget after you confirm.

  Re-running is safe. Every step checks what is already present and only does what is missing,
  so this is also how you add optional features later.

.PARAMETER Components
  Optional features to install without the menu, comma separated. Any of:
    core-models, kiwi-edit, ltx, h3, image-studio, zimage-trainer, moss, motion,
    wan-alpha, wan-animate, photo, rtx-upscale
  or 'all'. Use 'none' to install only the editor.

.PARAMETER DownloadModels
  Also download the model packs that belong to the chosen components. Requires -AcceptLicenses
  when running with -NonInteractive.

.PARAMETER AcceptLicenses
  Confirms that you have reviewed the licence of every model pack (and, for rtx-upscale,
  NVIDIA's licence) that this run downloads.

.PARAMETER InstallPrerequisites
  Install missing system prerequisites with winget without asking.

.PARAMETER Build
  Compile the desktop app from source (needs Rust and the Visual C++ build tools).

.PARAMETER Prebuilt
  Download the prebuilt desktop app from the project's latest GitHub release instead of building.

.PARAMETER NoBuild
  Skip getting the desktop app entirely.

.PARAMETER NonInteractive
  Never prompt. Anything not requested by a parameter is skipped.

.EXAMPLE
  Install-Digital-Forge.bat

.EXAMPLE
  Install-Digital-Forge.bat -Components image-studio,photo -DownloadModels -AcceptLicenses -Prebuilt
#>
[CmdletBinding()]
param(
    [string[]]$Components = @(),
    [switch]$DownloadModels,
    [switch]$AcceptLicenses,
    [switch]$InstallPrerequisites,
    [switch]$Build,
    [switch]$Prebuilt,
    [switch]$NoBuild,
    [switch]$NonInteractive
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'common.ps1')

$GitHubRepo = 'GeekyGhost/Digital-Forge-Video-and-Image-Editor'
$Interactive = -not $NonInteractive
$AppPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$ReleasesDir = Join-Path $ProjectRoot 'Releases'
$Failures = New-Object System.Collections.Generic.List[string]
$Completed = New-Object System.Collections.Generic.List[string]

$StateRoot = Join-Path $ProjectRoot '.aiv'
New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
$LogPath = Join-Path $StateRoot ('install-{0}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
$Transcribing = $false
try { Start-Transcript -LiteralPath $LogPath -Append | Out-Null; $Transcribing = $true } catch { }

# ------------------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------------------

function Write-Banner([string]$Text) {
    Write-Host ''
    Write-Host ('=' * 78) -ForegroundColor DarkCyan
    Write-Host " $Text" -ForegroundColor Cyan
    Write-Host ('=' * 78) -ForegroundColor DarkCyan
}

function Write-Step([string]$Text) { Write-Host ''; Write-Host "==> $Text" -ForegroundColor Cyan }
function Write-Ok([string]$Text) { Write-Host "  [ok]      $Text" -ForegroundColor Green }
function Write-Missing([string]$Text) { Write-Host "  [missing] $Text" -ForegroundColor Yellow }
function Write-Note([string]$Text) { Write-Host "            $Text" -ForegroundColor DarkGray }

function Ask-YesNo([string]$Question, [bool]$Default = $true) {
    if (-not $Interactive) { return $Default }
    $Hint = if ($Default) { '[Y/n]' } else { '[y/N]' }
    while ($true) {
        $Answer = (Read-Host "$Question $Hint").Trim().ToLowerInvariant()
        if ($Answer -eq '') { return $Default }
        if ($Answer -eq 'y' -or $Answer -eq 'yes') { return $true }
        if ($Answer -eq 'n' -or $Answer -eq 'no') { return $false }
    }
}

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

# winget and rustup change PATH in the registry, not in this process. Re-read it so a tool that was
# just installed is usable without restarting the installer.
function Update-SessionPath {
    $Machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $User = [Environment]::GetEnvironmentVariable('Path', 'User')
    $Parts = @($Machine, $User, $env:Path) | Where-Object { $_ }
    $Seen = @{}
    $Merged = foreach ($Entry in ($Parts -join ';').Split(';')) {
        $Key = $Entry.Trim().TrimEnd('\').ToLowerInvariant()
        if ($Key -and -not $Seen.ContainsKey($Key)) { $Seen[$Key] = $true; $Entry.Trim() }
    }
    $env:Path = $Merged -join ';'
    $Cargo = Join-Path $env:USERPROFILE '.cargo\bin'
    if ((Test-Path -LiteralPath $Cargo) -and ($env:Path -notlike "*$Cargo*")) { $env:Path = "$Cargo;$env:Path" }
}

# Runs a native command without letting stderr output become a terminating error, and returns
# its exit code.
function Invoke-Native([scriptblock]$Command) {
    $Previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $Command
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $Previous
    }
}

# Starts a program attached directly to this console and waits for it. Unlike calling it inline,
# this keeps its colours, progress bars and interactive prompts (a piped child cannot show a prompt
# that has no trailing newline). Arguments are quoted for the Windows command line, because this
# folder's path may contain spaces.
function ConvertTo-ArgumentString([string[]]$Arguments) {
    $Quoted = foreach ($A in $Arguments) {
        if ($A -eq '') { '""' }
        elseif ($A -match '[\s"]') { '"' + ($A -replace '(\\*)"', '$1$1\"' -replace '(\\+)$', '$1$1') + '"' }
        else { $A }
    }
    return ($Quoted -join ' ')
}

function Invoke-Console([string]$FilePath, [string[]]$Arguments = @()) {
    $Start = @{ FilePath = $FilePath; NoNewWindow = $true; PassThru = $true; WorkingDirectory = $ProjectRoot }
    if ($Arguments.Count -gt 0) { $Start.ArgumentList = ConvertTo-ArgumentString $Arguments }
    $Process = Start-Process @Start
    $null = $Process.Handle  # cache the handle, or ExitCode can come back empty
    $Process.WaitForExit()
    return $Process.ExitCode
}

# Runs one of the scripts\ setup scripts in its own PowerShell process, exactly as its .bat
# launcher would, and records the outcome instead of aborting the whole install.
function Invoke-SetupScript([string]$Label, [string]$ScriptName, [string[]]$Arguments = @(), [switch]$Critical) {
    Write-Step $Label
    $Path = Join-Path $PSScriptRoot $ScriptName
    $AllArgs = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $Path) + $Arguments
    $Code = Invoke-Console 'powershell.exe' $AllArgs
    if ($Code -eq 0) {
        $Completed.Add($Label)
        return 0
    }
    if ($Critical) { throw "$Label failed with exit code $Code." }
    $Failures.Add("$Label (exit code $Code)")
    Write-Warning "$Label failed with exit code $Code. Continuing with the rest of the install."
    return $Code
}

function Get-FreeGiB([string]$Path) {
    try {
        $Drive = (Get-Item -LiteralPath $Path).PSDrive
        return [math]::Round($Drive.Free / 1GB, 1)
    } catch {
        return -1
    }
}

# ------------------------------------------------------------------------------------------------
# Prerequisite detection
# ------------------------------------------------------------------------------------------------

function Get-NodeMajor {
    if (-not (Test-Command 'node')) { return 0 }
    $script:NodeVersionText = ''
    [void](Invoke-Native { $script:NodeVersionText = (& node --version 2>$null) })
    $Version = "$script:NodeVersionText"
    if ($Version -match '^v(\d+)\.') { return [int]$Matches[1] }
    return 0
}

function Test-MsvcBuildTools {
    $VsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $VsWhere -PathType Leaf)) { return $false }
    $script:VsFound = $null
    [void](Invoke-Native {
        $script:VsFound = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
    })
    return [bool]$script:VsFound
}

function Test-WebView2 {
    $Guid = '{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}'
    $Keys = @(
        "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\$Guid",
        "HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\$Guid",
        "HKCU:\SOFTWARE\Microsoft\EdgeUpdate\Clients\$Guid"
    )
    foreach ($Key in $Keys) {
        $Pv = (Get-ItemProperty -LiteralPath $Key -Name pv -ErrorAction SilentlyContinue).pv
        if ($Pv -and $Pv -ne '0.0.0.0') { return $true }
    }
    return $false
}

function Get-GpuInfo {
    if (-not (Test-Command 'nvidia-smi')) { return $null }
    $script:GpuLine = $null
    $Code = Invoke-Native { $script:GpuLine = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null }
    if ($Code -ne 0 -or -not $script:GpuLine) { return $null }
    $First = ($script:GpuLine | Select-Object -First 1).ToString()
    $Parts = $First.Split(',')
    $Mib = 0
    [void][int]::TryParse($Parts[-1].Trim(), [ref]$Mib)
    return [pscustomobject]@{ Name = $Parts[0].Trim(); MemoryMiB = $Mib }
}

function Get-Prerequisites {
    return @(
        [pscustomobject]@{
            Key = 'nvidia'; Name = 'NVIDIA driver (nvidia-smi)'; Need = 'required'
            Test = { Test-Command 'nvidia-smi' }
            Winget = $null; WingetArgs = @()
            Help = 'Install the current NVIDIA Studio or Game Ready driver from https://www.nvidia.com/Download/index.aspx'
        },
        [pscustomobject]@{
            Key = 'ffmpeg'; Name = 'FFmpeg and FFprobe'; Need = 'required'
            Test = { (Test-Command 'ffmpeg') -and (Test-Command 'ffprobe') }
            Winget = 'Gyan.FFmpeg'; WingetArgs = @()
            Help = 'Install a full FFmpeg build and put its bin folder on PATH: https://www.gyan.dev/ffmpeg/builds/'
        },
        [pscustomobject]@{
            Key = 'node'; Name = 'Node.js 20 or newer (with npm)'; Need = 'required'
            Test = { ((Get-NodeMajor) -ge 20) -and (Test-Command 'npm.cmd') }
            Winget = 'OpenJS.NodeJS.LTS'; WingetArgs = @()
            Help = 'Install the current LTS from https://nodejs.org'
        },
        [pscustomobject]@{
            Key = 'git'; Name = 'Git'; Need = 'features'
            Test = { Test-Command 'git' }
            Winget = 'Git.Git'; WingetArgs = @()
            Help = 'Needed by LTX and several AI runtimes that install pinned sources. https://git-scm.com'
        },
        [pscustomobject]@{
            Key = 'webview2'; Name = 'Microsoft Edge WebView2 Runtime'; Need = 'required'
            Test = { Test-WebView2 }
            Winget = 'Microsoft.EdgeWebView2Runtime'; WingetArgs = @()
            Help = 'Preinstalled on Windows 11. https://developer.microsoft.com/microsoft-edge/webview2/'
        },
        [pscustomobject]@{
            Key = 'rust'; Name = 'Rust toolchain (cargo)'; Need = 'build'
            Test = { Test-Command 'cargo' }
            Winget = 'Rustlang.Rustup'; WingetArgs = @()
            Help = 'Only needed to build the app from source. https://rustup.rs'
        },
        [pscustomobject]@{
            Key = 'msvc'; Name = 'Visual C++ build tools'; Need = 'build'
            Test = { Test-MsvcBuildTools }
            Winget = 'Microsoft.VisualStudio.2022.BuildTools'
            WingetArgs = @('--override', '--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended')
            Help = 'Only needed to build from source. Install "Desktop development with C++" from https://visualstudio.microsoft.com/visual-cpp-build-tools/'
        }
    )
}

function Install-WithWinget([pscustomobject]$Item) {
    if (-not $Item.Winget) { return $false }
    if (-not (Test-Command 'winget')) {
        Write-Warning 'winget is not available on this machine (install "App Installer" from the Microsoft Store).'
        return $false
    }
    Write-Step "Installing $($Item.Name) with winget ($($Item.Winget))"
    $WingetArgs = @('install', '--id', $Item.Winget, '--exact', '--accept-source-agreements', '--accept-package-agreements') + $Item.WingetArgs
    $Code = Invoke-Console 'winget' $WingetArgs
    Update-SessionPath
    if ($Item.Key -eq 'rust' -and (Test-Command 'rustup')) {
        [void](Invoke-Console 'rustup' @('default', 'stable'))
    }
    # winget returns non-zero for "already installed" too, so judge by the probe instead.
    $Present = & $Item.Test
    if (-not $Present -and $Code -ne 0) { Write-Warning "winget exited with code $Code." }
    return [bool]$Present
}

# ------------------------------------------------------------------------------------------------
# Optional components and their model packs
# ------------------------------------------------------------------------------------------------

$Catalog = [ordered]@{
    'core-models'    = @{ Label = 'Editor utility models: Whisper captions, SAM 2 tracking, Real-ESRGAN upscale'; Script = $null; Packs = @('utilities', 'real-esrgan'); Note = '' }
    'kiwi-edit'      = @{ Label = 'AI clip editor (Kiwi-Edit 5B, runs in the editor environment)'; Script = $null; Packs = @('kiwi-edit'); Note = '' }
    'ltx'            = @{ Label = 'LTX 2.3 video generation, retake and foley'; Script = 'setup-ltx.ps1'; Packs = @('ltx-core', 'ltx-loras'); Note = 'Uses gated Hugging Face models (Gemma encoder, foley LoRA).' }
    'h3'             = @{ Label = 'MiniMax-H3 joint video + audio generation'; Script = 'setup-h3.ps1'; Packs = @('minimax-h3', 'minimax-h3-ref2va'); Note = '' }
    'image-studio'   = @{ Label = 'Image Studio: Z-Image Turbo, Qwen Image Edit, BiRefNet, SAM 2'; Script = 'setup-image-studio.ps1'; Packs = @('image-studio'); Note = '' }
    'zimage-trainer' = @{ Label = 'Z-Image Base creative generation and LoRA training'; Script = 'setup-zimage-trainer.ps1'; Packs = @('z-image-base'); Note = '' }
    'moss'           = @{ Label = 'MOSS sound-effects runtime'; Script = 'install-moss-sfx.ps1'; Packs = @(); Note = 'Weights are placed by hand at models\moss-sfx-v2 (see docs\INSTALL.md).' }
    'motion'         = @{ Label = 'Motion: CoTracker3 point tracking + video depth'; Script = 'setup-motion.ps1'; Packs = @(); Note = 'Downloads its own weights. CoTracker3 is CC-BY-NC (non-commercial).' }
    'wan-alpha'      = @{ Label = 'Wan-Alpha transparent VFX elements'; Script = 'setup-wan-alpha.ps1'; Packs = @(); Note = 'Downloads its own weights (large).' }
    'wan-animate'    = @{ Label = 'Wan Animate 2 character insertion'; Script = 'setup-wan-animate.ps1'; Packs = @(); Note = 'Downloads its own weights (large).' }
    'photo'          = @{ Label = 'Photo Edit relight and remove models (depth + LaMa)'; Script = 'download-photo-models.ps1'; Packs = @(); Note = 'About 100 MB.' }
    'rtx-upscale'    = @{ Label = 'NVIDIA RTX Video Super Resolution upscaling'; Script = 'setup-rtx-upscale.ps1'; Packs = @(); Note = 'NVIDIA proprietary licence. Needs Image Studio.' }
}
$Requires = @{ 'rtx-upscale' = @('image-studio'); 'zimage-trainer' = @('image-studio') }

function Get-Manifest([string]$PackId) {
    $Path = Join-Path $ProjectRoot "models\manifests\$PackId.json"
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

function Get-PackSizeText([string[]]$Packs) {
    $Total = 0.0
    foreach ($Pack in $Packs) {
        $Manifest = Get-Manifest $Pack
        if ($Manifest) { $Total += [double]$Manifest.estimatedSizeGiB }
    }
    if ($Total -le 0) { return '' }
    if ($Total -lt 1) { return '< 1 GB models' }
    return ('~{0} GB models' -f [math]::Round($Total))
}

function Resolve-ComponentSelection([string[]]$Raw) {
    $Keys = @($Catalog.Keys)
    $Tokens = @()
    foreach ($Item in $Raw) { $Tokens += ($Item -split '[,; ]+' | Where-Object { $_ }) }
    $Selected = New-Object System.Collections.Generic.List[string]
    foreach ($Token in $Tokens) {
        $T = $Token.Trim().ToLowerInvariant()
        if ($T -eq 'none') { return @() }
        if ($T -eq 'all') { return $Keys }
        $Index = 0
        if ([int]::TryParse($T, [ref]$Index)) {
            if ($Index -ge 1 -and $Index -le $Keys.Count) { $T = $Keys[$Index - 1] } else { Write-Warning "No option $T."; continue }
        }
        if ($Keys -notcontains $T) { Write-Warning "Unknown component '$Token' ignored."; continue }
        if (-not $Selected.Contains($T)) { $Selected.Add($T) }
    }
    # Pull in dependencies, then keep catalogue order so dependencies run first.
    foreach ($Key in @($Selected)) {
        if ($Requires.ContainsKey($Key)) {
            foreach ($Dependency in $Requires[$Key]) {
                if (-not $Selected.Contains($Dependency)) {
                    Write-Host "  Adding $Dependency, which $Key needs." -ForegroundColor DarkYellow
                    $Selected.Add($Dependency)
                }
            }
        }
    }
    return @($Keys | Where-Object { $Selected.Contains($_) })
}

function Show-ComponentMenu {
    Write-Banner 'Optional AI features'
    Write-Host ' The editor (timeline, trim, transitions, titles, captions UI, export) works with none'
    Write-Host ' of these. Each one gets its own isolated environment inside this folder.'
    Write-Host ''
    $Number = 1
    foreach ($Key in $Catalog.Keys) {
        $Entry = $Catalog[$Key]
        $Size = Get-PackSizeText $Entry.Packs
        $SizeText = if ($Size) { "  ($Size)" } else { '' }
        Write-Host ('  {0,2}) {1,-15} {2}{3}' -f $Number, $Key, $Entry.Label, $SizeText)
        if ($Entry.Note) { Write-Host ('      {0,-15} {1}' -f '', $Entry.Note) -ForegroundColor DarkGray }
        $Number++
    }
    Write-Host ''
    Write-Host ' Enter numbers or names separated by commas (for example: 1,5,11), "all",'
    Write-Host ' or just press Enter to skip. You can run this installer again later to add more.'
    $Answer = Read-Host ' Features to install'
    if (-not $Answer.Trim()) { return @() }
    return Resolve-ComponentSelection @($Answer)
}

function Invoke-HuggingFaceLogin {
    $Candidates = @(
        (Join-Path $ProjectRoot '.venv\Scripts\hf.exe'),
        (Join-Path $ProjectRoot '.venv\Scripts\huggingface-cli.exe')
    )
    $Cli = $Candidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    if (-not $Cli) {
        Write-Warning 'The Hugging Face CLI was not found in .venv. Log in later with: .\.venv\Scripts\hf.exe auth login'
        return
    }
    $LoginArgs = if ($Cli -like '*hf.exe') { @('auth', 'login') } else { @('login') }
    [void](Invoke-Console $Cli $LoginArgs)
}

function Install-ModelPack([string]$PackId) {
    $Label = "Model pack: $PackId"
    Write-Step "Downloading $Label"
    $Code = Invoke-Console $AppPython @((Join-Path $ProjectRoot 'scripts\models.py'), 'install', $PackId, '--accept-licenses')
    if ($Code -eq 0) { $Completed.Add($Label); return $true }
    $Failures.Add("$Label (exit code $Code)")
    Write-Warning "$Label failed with exit code $Code. Re-run the installer to resume it."
    return $false
}

# ------------------------------------------------------------------------------------------------
# Getting the desktop app
# ------------------------------------------------------------------------------------------------

function Get-ExistingExecutable {
    $Candidates = @()
    foreach ($Directory in @($ReleasesDir, (Join-Path $ProjectRoot 'src-tauri\target\release'))) {
        if (Test-Path -LiteralPath $Directory) {
            $Candidates += Get-ChildItem -LiteralPath $Directory -Filter '*.exe' -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -notmatch 'setup|install|uninstall|webview' }
        }
    }
    return $Candidates | Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
}

function Get-PrebuiltExecutable {
    Write-Step "Looking for a prebuilt app in the latest release of $GitHubRepo"
    [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    try {
        $Release = Invoke-RestMethod -Uri "https://api.github.com/repos/$GitHubRepo/releases/latest" -Headers @{ 'User-Agent' = 'digital-forge-installer' } -UseBasicParsing
    } catch {
        Write-Warning "No published release was found ($($_.Exception.Message))."
        return $false
    }
    $Asset = $Release.assets | Where-Object { $_.name -match '^aiv-studio.*\.exe$' -and $_.name -notmatch 'setup|install' } | Select-Object -First 1
    if (-not $Asset) {
        Write-Warning "Release $($Release.tag_name) has no aiv-studio-*.exe asset."
        return $false
    }
    $Version = (Get-Content -LiteralPath (Join-Path $ProjectRoot 'package.json') -Raw | ConvertFrom-Json).version
    if ($Asset.name -notmatch [regex]::Escape($Version)) {
        Write-Warning "The release asset $($Asset.name) may not match this source (version $Version)."
    }
    New-Item -ItemType Directory -Force -Path $ReleasesDir | Out-Null
    $Target = Join-Path $ReleasesDir $Asset.name
    Write-Host "  Downloading $($Asset.name) ($([math]::Round($Asset.size / 1MB)) MB)"
    $PreviousProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'
    try {
        Invoke-WebRequest -Uri $Asset.browser_download_url -OutFile $Target -UseBasicParsing
    } catch {
        Write-Warning "Download failed: $($_.Exception.Message)"
        return $false
    } finally {
        $ProgressPreference = $PreviousProgress
    }
    try { Unblock-File -LiteralPath $Target } catch { }
    $Completed.Add("Prebuilt app $($Asset.name)")
    return $true
}

# ================================================================================================
# 1. Welcome and system checks
# ================================================================================================

try {
    Write-Banner 'Digital Forge Video and Image Editor - installer'
    Write-Host " Install folder: $ProjectRoot"
    Write-Host " Log file:       $LogPath"
    Write-Host ''
    Write-Host ' Everything is installed inside this folder. Nothing is uploaded anywhere.'

    if ($env:OS -ne 'Windows_NT') { throw 'Digital Forge runs on Windows 10 or 11 only.' }
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'A 64-bit Windows installation is required.' }

    if ($ProjectRoot.Length -gt 80) {
        Write-Warning "The install path is $($ProjectRoot.Length) characters long. Python packages nest deeply; a shorter path such as C:\DigitalForge avoids Windows path-length failures."
    }
    $LongPaths = (Get-ItemProperty -LiteralPath 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -ErrorAction SilentlyContinue).LongPathsEnabled
    if ($LongPaths -ne 1) {
        Write-Note 'Windows long-path support is off. If a setup step reports a path that is too long,'
        Write-Note 'enable it (see docs\INSTALL.md) or move this folder to a shorter path.'
    }

    Update-SessionPath

    Write-Step 'Checking system prerequisites'
    $Prerequisites = Get-Prerequisites
    $MissingRequired = @()
    $MissingOptional = @()
    foreach ($Item in $Prerequisites) {
        if (& $Item.Test) {
            Write-Ok $Item.Name
        } else {
            Write-Missing ("{0}  ({1})" -f $Item.Name, $Item.Need)
            Write-Note $Item.Help
            if ($Item.Need -eq 'required') { $MissingRequired += $Item } else { $MissingOptional += $Item }
        }
    }

    $Installable = @($MissingRequired + $MissingOptional | Where-Object { $_.Winget })
    if ($Installable.Count -gt 0) {
        $Names = ($Installable | ForEach-Object { $_.Name }) -join ', '
        $DoInstall = $InstallPrerequisites -or (Ask-YesNo "Install the missing prerequisites with winget now? ($Names)" $true)
        if ($DoInstall) {
            foreach ($Item in $Installable) {
                if (Install-WithWinget $Item) { Write-Ok "$($Item.Name) installed" } else { Write-Missing "$($Item.Name) is still missing" }
            }
        }
    }

    $StillMissing = @($Prerequisites | Where-Object { $_.Need -eq 'required' -and -not (& $_.Test) })
    if ($StillMissing.Count -gt 0) {
        Write-Host ''
        foreach ($Item in $StillMissing) { Write-Host "  Still missing: $($Item.Name) - $($Item.Help)" -ForegroundColor Red }
        throw 'Required prerequisites are missing. Install them, then run Install-Digital-Forge.bat again. (If you just installed them, close this window first so PATH refreshes.)'
    }

    $Gpu = Get-GpuInfo
    if ($Gpu) {
        Write-Host ''
        Write-Host ("  GPU: {0}, {1:N0} MiB" -f $Gpu.Name, $Gpu.MemoryMiB)
        if ($Gpu.MemoryMiB -gt 0 -and $Gpu.MemoryMiB -lt 15360) {
            Write-Warning 'The AI model profiles are tuned for a 16 GB NVIDIA GPU. The editor itself will work; most AI features will not fit on this card.'
            if (-not (Ask-YesNo 'Continue anyway?' $true)) { throw 'Stopped at the GPU check.' }
        }
    }

    $Free = Get-FreeGiB $ProjectRoot
    if ($Free -ge 0) {
        Write-Host "  Free disk space on this drive: $Free GB"
        if ($Free -lt 40) { Write-Warning 'The editor and its base Python environment need about 20-40 GB. AI model packs need far more.' }
    }

    # ============================================================================================
    # 2. The editor itself: uv, JavaScript dependencies, application Python environment
    # ============================================================================================

    Write-Step 'Preparing the uv package manager (kept in .aiv\tools)'
    $Uv = Resolve-AivUv -Persist
    Write-Ok "uv: $Uv"
    Write-Note "Python interpreters: $env:UV_PYTHON_INSTALL_DIR"
    Write-Note "Package cache:       $env:UV_CACHE_DIR"

    New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot 'models'), (Join-Path $ProjectRoot 'cache'), (Join-Path $ProjectRoot 'projects'), (Join-Path $ProjectRoot 'exports') | Out-Null

    [void](Invoke-SetupScript 'Editor: JavaScript dependencies and application Python environment (CUDA PyTorch)' 'setup-and-launch.ps1' @('-NoLaunch') -Critical)

    # ============================================================================================
    # 3. The desktop app executable
    # ============================================================================================

    Write-Banner 'Desktop app'
    $Existing = Get-ExistingExecutable
    $CanBuild = (Test-Command 'cargo') -and (Test-MsvcBuildTools)
    $Mode = 'skip'
    if ($NoBuild) {
        $Mode = 'skip'
    } elseif ($Build) {
        $Mode = 'build'
    } elseif ($Prebuilt) {
        $Mode = 'prebuilt'
    } elseif ($Existing -and -not $Interactive) {
        $Mode = 'skip'
    } elseif ($Interactive) {
        if ($Existing) { Write-Host " Found $($Existing.Name). You can keep it or replace it." }
        Write-Host '  1) Download the prebuilt app from GitHub Releases (fastest, no compilers needed)'
        if ($CanBuild) {
            Write-Host '  2) Build it from this source (5-15 minutes the first time)'
        } else {
            Write-Host '  2) Build it from this source (needs Rust and the Visual C++ build tools - not detected)'
        }
        Write-Host '  3) Skip'
        $Default = if ($Existing) { '3' } elseif ($CanBuild) { '2' } else { '1' }
        $Choice = (Read-Host " Choose 1, 2 or 3 [$Default]").Trim()
        if (-not $Choice) { $Choice = $Default }
        switch ($Choice) { '1' { $Mode = 'prebuilt' } '2' { $Mode = 'build' } default { $Mode = 'skip' } }
    }

    if ($Mode -eq 'prebuilt') {
        if (-not (Get-PrebuiltExecutable)) {
            if ($CanBuild -and (Ask-YesNo 'Build from source instead?' $true)) { $Mode = 'build' }
            else { $Failures.Add('Desktop app (no prebuilt release was available)') }
        }
    }
    if ($Mode -eq 'build') {
        if (-not $CanBuild) {
            $Failures.Add('Desktop app build (Rust and the Visual C++ build tools are required)')
            Write-Warning 'Cannot build: install Rust (https://rustup.rs) and the Visual C++ build tools, then run Build-AIV-Studio.bat.'
        } else {
            [void](Invoke-SetupScript 'Building the desktop app' 'build-release.ps1' @('-SkipInstall', '-SkipTests'))
        }
    }

    # ============================================================================================
    # 4. Optional AI features
    # ============================================================================================

    if ($Components.Count -gt 0) {
        $Selection = @(Resolve-ComponentSelection $Components)
    } elseif ($Interactive) {
        $Selection = @(Show-ComponentMenu)
    } else {
        $Selection = @()
    }

    if ($Selection.Count -gt 0) {
        # Decide on model downloads up front, so the install can then run unattended.
        $Packs = @()
        foreach ($Key in $Selection) { $Packs += $Catalog[$Key].Packs }
        $Packs = @($Packs | Select-Object -Unique)
        $GetPacks = $false
        if ($Packs.Count -gt 0) {
            Write-Banner 'Model packs for the selected features'
            $Total = 0.0
            foreach ($Pack in $Packs) {
                $Manifest = Get-Manifest $Pack
                if (-not $Manifest) { continue }
                $Total += [double]$Manifest.estimatedSizeGiB
                Write-Host ("  {0,-19} ~{1,5} GB  {2}" -f $Pack, $Manifest.estimatedSizeGiB, $Manifest.name)
                Write-Host ("  {0,-19} licence: {1}" -f '', $Manifest.license.name) -ForegroundColor DarkGray
                Write-Host ("  {0,-19} {1}" -f '', $Manifest.license.url) -ForegroundColor DarkGray
            }
            Write-Host ''
            Write-Host ("  Total: about {0} GB. Free on this drive: {1} GB." -f [math]::Round($Total), (Get-FreeGiB $ProjectRoot))
            Write-Host '  Each model has its own licence; some restrict commercial use. Review them before'
            Write-Host '  downloading. Answering yes records that you accept them for these packs.'
            if ($Interactive -and -not $DownloadModels) {
                $GetPacks = Ask-YesNo 'Download these model packs now?' $false
            } elseif ($DownloadModels) {
                if ($AcceptLicenses -or (Ask-YesNo 'Do you accept the licences listed above?' $false)) { $GetPacks = $true }
                else { Write-Warning 'Model packs skipped: licences were not accepted.' }
            }
        }

        if ($GetPacks -and ($Selection -contains 'ltx')) {
            Write-Banner 'Gated models for LTX'
            Write-Host '  LTX uses two gated Hugging Face repositories. Before downloading:'
            Write-Host '   1. Sign in at https://huggingface.co'
            Write-Host '   2. Accept the terms at https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized'
            Write-Host '   3. Accept the terms of the LTX foley LoRA (linked in models\manifests\ltx-loras.json)'
            Write-Host '   4. Create a read token at https://huggingface.co/settings/tokens'
            if ($Interactive -and (Ask-YesNo 'Log this machine in to Hugging Face now?' $true)) { Invoke-HuggingFaceLogin }
            elseif (-not $Interactive) { Write-Host '  Non-interactive run: make sure this machine is already logged in (.\.venv\Scripts\hf.exe auth login).' }
        }

        foreach ($Key in $Selection) {
            $Entry = $Catalog[$Key]
            Write-Banner "$Key - $($Entry.Label)"
            $ScriptOk = $true

            if ($Entry.Script) {
                if ($Key -eq 'rtx-upscale') {
                    $RtxArgs = @()
                    if ($AcceptLicenses) { $RtxArgs += '-AcceptLicense' }
                    $Code = Invoke-SetupScript 'RTX Video Super Resolution' $Entry.Script $RtxArgs
                    if ($Code -eq 2 -and -not $AcceptLicenses) {
                        [void]$Failures.Remove('RTX Video Super Resolution (exit code 2)')
                        if (Ask-YesNo "Do you accept NVIDIA's licence shown above?" $false) {
                            $Code = Invoke-SetupScript 'RTX Video Super Resolution' $Entry.Script @('-AcceptLicense')
                        } else {
                            Write-Host '  Skipped. Run Install-RTX-Upscale.bat -AcceptLicense later if you change your mind.'
                        }
                    }
                    $ScriptOk = ($Code -eq 0)
                } else {
                    $ScriptOk = ((Invoke-SetupScript "Runtime: $Key" $Entry.Script) -eq 0)
                }
            }

            if ($GetPacks -and $Entry.Packs.Count -gt 0) {
                if (-not (Test-Path -LiteralPath $AppPython -PathType Leaf)) {
                    $Failures.Add("Model packs for $Key (the application environment is missing)")
                } else {
                    $H3Ready = $true
                    foreach ($Pack in $Entry.Packs) {
                        $Ok = Install-ModelPack $Pack
                        if ($Pack -eq 'minimax-h3' -and -not $Ok) { $H3Ready = $false }
                    }
                    # H3 needs its text encoder baked once, with its own interpreter.
                    if ($Key -eq 'h3' -and $H3Ready -and $ScriptOk) {
                        $H3Python = Join-Path $ProjectRoot 'vendor\MiniMax-H3\.venv\Scripts\python.exe'
                        Write-Step 'Baking the MiniMax-H3 text encoder (one time)'
                        $Code = Invoke-Console $H3Python @((Join-Path $ProjectRoot 'scripts\bake-h3-encoder.py'), '--prune')
                        if ($Code -eq 0) { $Completed.Add('MiniMax-H3 encoder bake') } else { $Failures.Add("MiniMax-H3 encoder bake (exit code $Code)") }
                    }
                }
            }
        }
    }

    # ============================================================================================
    # 5. Summary
    # ============================================================================================

    Write-Banner 'Summary'
    foreach ($Item in $Completed) { Write-Ok $Item }
    foreach ($Item in $Failures) { Write-Host "  [failed]  $Item" -ForegroundColor Red }
    Write-Host ''
    if ($Failures.Count -gt 0) {
        Write-Host ' Some steps did not finish. Fix what is reported above and run Install-Digital-Forge.bat' -ForegroundColor Yellow
        Write-Host ' again - finished steps are skipped, so it picks up where it stopped.' -ForegroundColor Yellow
    }
    Write-Host ' Start the editor by double-clicking Launch-AIV-Studio.vbs'
    Write-Host ' (or Launch-AIV-Studio.bat to watch its startup checks).'
    Write-Host ' Add more AI features any time by running this installer again.'
    Write-Host " Full log: $LogPath"

    $AppReady = [bool](Get-ExistingExecutable)
    if ($AppReady -and $Interactive -and (Ask-YesNo 'Launch Digital Forge now?' $true)) {
        Start-Process -FilePath 'wscript.exe' -ArgumentList ('"{0}"' -f (Join-Path $ProjectRoot 'Launch-AIV-Studio.vbs')) -WorkingDirectory $ProjectRoot
    } elseif (-not $AppReady) {
        Write-Host ''
        Write-Host ' No app executable is installed yet. Either put the aiv-studio-*.exe from the GitHub' -ForegroundColor Yellow
        Write-Host ' release into the Releases folder, or install Rust + Visual C++ build tools and run' -ForegroundColor Yellow
        Write-Host ' Build-AIV-Studio.bat. Launch-AIV-Studio.bat falls back to development mode.' -ForegroundColor Yellow
    }

    if ($Failures.Count -gt 0) { $ExitCode = 2 } else { $ExitCode = 0 }
} catch {
    Write-Host ''
    Write-Host "INSTALL STOPPED: $($_.Exception.Message)" -ForegroundColor Red
    $ExitCode = 1
} finally {
    if ($Transcribing) { try { Stop-Transcript | Out-Null } catch { } }
}
exit $ExitCode
