<#
.SYNOPSIS
    Repairs the Windows pagefile so LTX can memory-map its 27.5 GiB checkpoint.

.DESCRIPTION
    LTX loads the FP8 checkpoint through safetensors' memory-mapped reader. On Windows that
    reservation is charged against the system commit limit (physical RAM + pagefile). When the
    pagefile is small, the map fails with:

        OSError: The paging file is too small for this operation to complete. (os error 1455)

    Windows shrinks a system-managed pagefile when its host volume runs low on space, and it does
    NOT grow it back automatically once space is returned. That is the state this machine is in:
    the One-to-All install filled C: to ~2 GB free on 2026-08-09, the pagefile collapsed, the
    models were later removed, but the pagefile stayed collapsed.

    This script reports the current commit limit and pagefile configuration, then pins an explicit
    pagefile large enough for the checkpoint. A reboot is required for the new size to take effect.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\fix-pagefile.ps1 -ReportOnly
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\fix-pagefile.ps1
#>
[CmdletBinding()]
param(
    # 24 GiB floor / 32 GiB ceiling. With 64 GB of RAM this is ample headroom for a 27.5 GiB map
    # without reserving a wasteful amount of disk.
    [int]$InitialMB = 24576,
    [int]$MaximumMB = 32768,
    [string]$Drive = 'C:',
    # Print the diagnosis and stop, without changing anything.
    [switch]$ReportOnly
)

$ErrorActionPreference = 'Stop'

function Write-Field([string]$Label, [string]$Value, [string]$Colour = 'Gray') {
    Write-Host ("  {0,-26} {1}" -f $Label, $Value) -ForegroundColor $Colour
}

Write-Host ''
Write-Host 'Pagefile diagnosis' -ForegroundColor Cyan

$os   = Get-CimInstance Win32_OperatingSystem
$cs   = Get-CimInstance Win32_ComputerSystem
$ramGB = [math]::Round($cs.TotalPhysicalMemory / 1GB, 1)

# Commit limit is what actually governs error 1455. TotalVirtualMemorySize is in KiB.
$commitLimitGB = [math]::Round($os.TotalVirtualMemorySize / 1MB, 1)
$commitFreeGB  = [math]::Round($os.FreeVirtualMemory / 1MB, 1)
$pagefileGB    = [math]::Round($commitLimitGB - $ramGB, 1)

Write-Field 'Physical RAM'      ("{0} GB" -f $ramGB)
Write-Field 'Commit limit'      ("{0} GB" -f $commitLimitGB)
Write-Field 'Commit available'  ("{0} GB" -f $commitFreeGB)
Write-Field 'Implied pagefile'  ("{0} GB" -f $pagefileGB) $(if ($pagefileGB -lt 16) { 'Red' } else { 'DarkGreen' })

$volume = Get-CimInstance Win32_LogicalDisk -Filter ("DeviceID='{0}'" -f $Drive)
$freeGB = [math]::Round($volume.FreeSpace / 1GB, 1)
Write-Field ("Free space on {0}" -f $Drive) ("{0} GB" -f $freeGB)

Write-Host ''
Write-Host 'Current pagefile configuration' -ForegroundColor Cyan
Write-Field 'System-managed' ([string]$cs.AutomaticManagedPagefile)

$settings = @(Get-CimInstance Win32_PageFileSetting -ErrorAction SilentlyContinue)
if ($settings.Count -eq 0) {
    Write-Field 'Explicit settings' 'none (fully system-managed)' 'Yellow'
} else {
    foreach ($s in $settings) {
        Write-Field $s.Name ("initial {0} MB / max {1} MB" -f $s.InitialSize, $s.MaximumSize)
    }
}

# The live pagefile can differ from the configured one; this is the size actually in use now.
foreach ($u in @(Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue)) {
    Write-Field ('In use: ' + $u.Name) ("{0} MB allocated, {1} MB peak" -f $u.AllocatedBaseSize, $u.PeakUsage) 'DarkGray'
}

# The checkpoint the map has to cover.
$Checkpoint = Join-Path (Split-Path -Parent $PSScriptRoot) 'models\LTX-2.3\ltx-2.3-22b-distilled-fp8.safetensors'
if (Test-Path -LiteralPath $Checkpoint) {
    $ckptGB = [math]::Round((Get-Item -LiteralPath $Checkpoint).Length / 1GB, 1)
    Write-Host ''
    Write-Field 'FP8 checkpoint' ("{0} GB" -f $ckptGB) 'DarkGray'
}

if ($ReportOnly) {
    Write-Host ''
    Write-Host 'Report only; nothing was changed.' -ForegroundColor Green
    exit 0
}

# --- capacity guard -----------------------------------------------------------------------------
# Reserving more than the volume can hold would fail at reboot, leaving no pagefile at all -- which
# is strictly worse than the current state. Refuse rather than risk that.
$requiredGB = [math]::Round($MaximumMB / 1024, 1)
if ($freeGB -lt ($requiredGB + 5)) {
    Write-Host ''
    Write-Host ("ABORTED: {0} needs about {1} GB free to host a {2} GB pagefile safely, but only {3} GB is available." -f $Drive, ($requiredGB + 5), $requiredGB, $freeGB) -ForegroundColor Red
    Write-Host 'Free more space first, or rerun with a smaller -MaximumMB, or host the pagefile on another drive.' -ForegroundColor Yellow
    exit 1
}

# --- elevation ----------------------------------------------------------------------------------
$identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = New-Object Security.Principal.WindowsPrincipal($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Host ''
    Write-Host 'ABORTED: changing the pagefile requires an elevated prompt.' -ForegroundColor Red
    Write-Host 'Reopen PowerShell with "Run as administrator" and rerun this script.' -ForegroundColor Yellow
    exit 1
}

Write-Host ''
Write-Host 'Applying' -ForegroundColor Cyan

# An explicit setting cannot coexist with the system-managed flag; clear it first.
if ($cs.AutomaticManagedPagefile) {
    Set-CimInstance -InputObject $cs -Property @{ AutomaticManagedPagefile = $false }
    Write-Field 'System-managed' 'disabled' 'DarkGreen'
}

$target = $settings | Where-Object { $_.Name -like ($Drive + '*') } | Select-Object -First 1
if ($target) {
    Set-CimInstance -InputObject $target -Property @{ InitialSize = $InitialMB; MaximumSize = $MaximumMB }
    Write-Field 'Updated' ("{0}  initial {1} MB / max {2} MB" -f $target.Name, $InitialMB, $MaximumMB) 'DarkGreen'
} else {
    $path = Join-Path $Drive 'pagefile.sys'
    New-CimInstance -ClassName Win32_PageFileSetting -Property @{
        Name         = $path
        InitialSize  = $InitialMB
        MaximumSize  = $MaximumMB
    } | Out-Null
    Write-Field 'Created' ("{0}  initial {1} MB / max {2} MB" -f $path, $InitialMB, $MaximumMB) 'DarkGreen'
}

Write-Host ''
Write-Host 'Pagefile reconfigured. REBOOT is required before the new size exists.' -ForegroundColor Green
Write-Host 'After rebooting, verify and retest:' -ForegroundColor Green
Write-Host '  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\fix-pagefile.ps1 -ReportOnly' -ForegroundColor Gray
Write-Host '  powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\diagnose-ltx.ps1 -Mode text' -ForegroundColor Gray
