[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RegistryPath = Join-Path $ProjectRoot 'src\domain\transitions.ts'

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    throw 'FFmpeg is required to validate transitions.'
}

$Registry = Get-Content -LiteralPath $RegistryPath -Raw
$Matches = [regex]::Matches($Registry, 'transition\("([^"]+)"')
$Transitions = @($Matches | ForEach-Object { $_.Groups[1].Value } | Where-Object { $_ -ne 'none' } | Sort-Object -Unique)
if ($Transitions.Count -ne 57) {
    throw "Expected 57 qualified native transitions in the registry; found $($Transitions.Count)."
}

$Failures = [System.Collections.Generic.List[string]]::new()
$Index = 0
foreach ($Transition in $Transitions) {
    $Index += 1
    Write-Progress -Activity 'Validating Digital Forge transitions' -Status "$Index / $($Transitions.Count): $Transition" -PercentComplete (($Index / $Transitions.Count) * 100)
    $Filter = "[0:v][1:v]xfade=transition=$($Transition):duration=0.25:offset=0.3"
    $Arguments = @(
        '-hide_banner', '-loglevel', 'error',
        '-f', 'lavfi', '-i', 'color=c=red:s=320x180:r=24:d=0.8',
        '-f', 'lavfi', '-i', 'color=c=blue:s=320x180:r=24:d=0.8',
        '-filter_complex', $Filter,
        '-frames:v', '14', '-f', 'null', '-'
    )
    & ffmpeg @Arguments 2>$null
    if ($LASTEXITCODE -ne 0) {
        $Failures.Add($Transition)
    }
}
Write-Progress -Activity 'Validating Digital Forge transitions' -Completed

if ($Failures.Count -gt 0) {
    throw "Transition render validation failed: $($Failures -join ', ')"
}

Write-Host "Validated all $($Transitions.Count) native transitions." -ForegroundColor Green
