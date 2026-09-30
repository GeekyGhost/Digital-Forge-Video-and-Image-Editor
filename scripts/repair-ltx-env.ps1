<# Rebuild LTX from the same canonical matrix used by normal setup. #>
[CmdletBinding()]
param([switch]$WhatIfOnly)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$Spec = (Get-AivRuntimeMatrix).Ltx

Write-Host 'Canonical LTX repair target' -ForegroundColor Green
Write-Host "  Python       $($Spec.Python)"
Write-Host "  Torch        $($Spec.Torch)"
Write-Host "  TorchVision  $($Spec.TorchVision)"
Write-Host "  TorchAudio   $($Spec.TorchAudio)"
Write-Host "  Source       $($Spec.Commit)"
Write-Host "  Lock SHA256  $($Spec.LockSha256)"

if ($WhatIfOnly) {
    Write-Host 'WhatIf only; nothing was changed.' -ForegroundColor Green
    exit 0
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'setup-ltx.ps1') -Recreate -RepairSource
if ($LASTEXITCODE -ne 0) { throw 'LTX repair failed.' }
