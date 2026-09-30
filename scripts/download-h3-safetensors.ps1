<#
.SYNOPSIS
  Resumably download and verify the published MiniMax-H3 safetensors used by production.

.DESCRIPTION
  Hugging Face Xet and HTTP/2 ranged transfers have both stalled on this Windows host. This helper
  uses bounded independent HTTP/1.1 ranges, verifies every range length, assembles in order, and
  publishes the file only after its official SHA-256 matches. It never accepts .pt or .bin models.
#>
param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("fl2va", "ref2va", "conditioner")]
  [string]$Artifact,
  [string]$ModelRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "models\MiniMax-H3"),
  [ValidateRange(1, 32)][int]$Connections = 4,
  [ValidateRange(4, 1024)][int]$PartSizeMiB = 256,
  [string]$VerifiedPrefix = "",
  [switch]$UseModelScopeMirror,
  [switch]$Anonymous
)

$ErrorActionPreference = "Stop"
$specs = @{
  fl2va = @{
    RelativePath = "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
    Bytes = [int64]20970379616
    Sha256 = "e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a"
  }
  ref2va = @{
    RelativePath = "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
    Bytes = [int64]20970379616
    Sha256 = "9255f52b6677845ad238f20dfaafa94727053694127ab7f255c048f0f9365779"
  }
  conditioner = @{
    RelativePath = "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    Bytes = [int64]27141342152
    Sha256 = "bc2ced0fbea64757fa9acddccfc0b3f4819d1dcf1da6c124d690d368be283923"
  }
}
$spec = $specs[$Artifact]
$root = [IO.Path]::GetFullPath($ModelRoot)
$destination = [IO.Path]::GetFullPath((Join-Path $root ($spec.RelativePath -replace "/", "\")))
if (-not $destination.StartsWith($root + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
  throw "Unsafe destination outside model root: $destination"
}
if ([IO.Path]::GetExtension($destination) -ne ".safetensors") {
  throw "Refusing a non-safetensors model destination: $destination"
}
New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null

if (Test-Path -LiteralPath $destination) {
  $existing = Get-Item -LiteralPath $destination
  if ($existing.Length -eq $spec.Bytes -and
      (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -eq $spec.Sha256) {
    Write-Host "$Artifact is already complete and hash-verified: $destination"
    exit 0
  }
  throw "Destination exists but does not match the published size/hash: $destination"
}

$leaf = Split-Path -Leaf $destination
$partsDirectory = Join-Path (Split-Path -Parent $destination) ("." + $leaf + ".parts")
New-Item -ItemType Directory -Path $partsDirectory -Force | Out-Null
$segmentsPath = Join-Path $partsDirectory "segments.json"
$segments = @()

if (Test-Path -LiteralPath $segmentsPath) {
  $segments = @(Get-Content -LiteralPath $segmentsPath -Raw | ConvertFrom-Json)
} else {
  [int64]$start = 0
  [int]$index = 0
  if ($VerifiedPrefix) {
    $prefix = Get-Item -LiteralPath ([IO.Path]::GetFullPath($VerifiedPrefix))
    if ($prefix.Length -le 0 -or $prefix.Length -ge $spec.Bytes) {
      throw "VerifiedPrefix has an invalid length: $($prefix.Length)"
    }
    $partName = "part-{0:D4}" -f $index
    $partPath = Join-Path $partsDirectory $partName
    if (-not (Test-Path -LiteralPath $partPath)) {
      New-Item -ItemType HardLink -Path $partPath -Target $prefix.FullName | Out-Null
    }
    if ((Get-Item -LiteralPath $partPath).Length -ne $prefix.Length) {
      throw "Verified prefix part length changed"
    }
    $segments += [pscustomobject]@{ Index = $index; Start = [int64]0; End = [int64]($prefix.Length - 1); Name = $partName }
    $start = $prefix.Length
    $index++
  }
  [int64]$partBytes = $PartSizeMiB * 1MB
  while ($start -lt $spec.Bytes) {
    [int64]$end = [Math]::Min($spec.Bytes - 1, $start + $partBytes - 1)
    $segments += [pscustomobject]@{ Index = $index; Start = $start; End = $end; Name = ("part-{0:D4}" -f $index) }
    $start = $end + 1
    $index++
  }
  $segments | ConvertTo-Json | Set-Content -LiteralPath $segmentsPath -Encoding utf8
}

if (-not $segments.Count -or [int64]$segments[0].Start -ne 0 -or
    [int64]$segments[-1].End -ne ($spec.Bytes - 1)) {
  throw "Invalid or incomplete segment manifest: $segmentsPath"
}
for ($i = 0; $i -lt $segments.Count; $i++) {
  if ([int]$segments[$i].Index -ne $i) { throw "Non-contiguous segment index at $i" }
  if ($i -gt 0 -and [int64]$segments[$i].Start -ne ([int64]$segments[$i - 1].End + 1)) {
    throw "Non-contiguous byte ranges at segment $i"
  }
}

$url = "https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/$($spec.RelativePath)?download=true"
$mirrorUrl = "https://modelscope.cn/models/Comfy-Org/MiniMax-H3/resolve/master/$($spec.RelativePath)"
$huggingFaceToken = ""
if (-not $Anonymous) {
  if ($env:HF_TOKEN) {
    $huggingFaceToken = $env:HF_TOKEN.Trim()
  } else {
    $tokenPath = if ($env:HF_TOKEN_PATH) {
      [IO.Path]::GetFullPath($env:HF_TOKEN_PATH)
    } elseif ($env:HF_HOME) {
      [IO.Path]::GetFullPath((Join-Path $env:HF_HOME "token"))
    } else {
      [IO.Path]::GetFullPath((Join-Path $env:USERPROFILE ".cache\huggingface\token"))
    }
    if (Test-Path -LiteralPath $tokenPath) {
      $huggingFaceToken = ([IO.File]::ReadAllText($tokenPath)).Trim()
    }
  }
}
if ($huggingFaceToken -match '[\r\n]') {
  throw "The Hugging Face token contains an unexpected line break"
}
if ($huggingFaceToken) {
  Write-Host "Authentication: Hugging Face token (value is never printed or placed in command arguments)"
} else {
  Write-Host "Authentication: anonymous"
}
$started = [Diagnostics.Stopwatch]::StartNew()
$initialComplete = 0
foreach ($segment in $segments) {
  $part = Join-Path $partsDirectory $segment.Name
  if (Test-Path -LiteralPath $part) { $initialComplete += (Get-Item -LiteralPath $part).Length }
}
$pending = @($segments | Where-Object {
  $part = Join-Path $partsDirectory $_.Name
  $expected = [int64]$_.End - [int64]$_.Start + 1
  -not (Test-Path -LiteralPath $part) -or (Get-Item -LiteralPath $part).Length -ne $expected
})

function Start-RangeDownload($segment) {
  $part = Join-Path $partsDirectory $segment.Name
  $temporaryName = $segment.Name + ".download"
  $temporary = Join-Path $partsDirectory $temporaryName
  $stderr = $part + ".stderr"
  if (Test-Path -LiteralPath $temporary) { Remove-Item -LiteralPath $temporary -Force }
  if (Test-Path -LiteralPath $stderr) { Remove-Item -LiteralPath $stderr -Force }
  $sourceUrl = if ($UseModelScopeMirror -and ([int]$segment.Index % 2 -eq 1)) { $mirrorUrl } else { $url }
  $arguments = @(
    "--http1.1", "--location", "--fail", "--retry", "8", "--retry-all-errors",
    "--retry-delay", "5", "--max-time", "180", "--speed-time", "60", "--speed-limit", "1024",
    "--remove-on-error", "--silent", "--show-error",
    "--range", "$($segment.Start)-$($segment.End)", "--output", $temporaryName, $sourceUrl
  )
  $environment = @{}
  if ($huggingFaceToken -and $sourceUrl -eq $url) {
    # curl imports the token from its child-only environment and expands it internally. This keeps
    # the credential out of the process command line, logs, source tree, and temporary files.
    $environment.HF_TOKEN = $huggingFaceToken
    # Start-Process joins ArgumentList entries into a Windows command line, so retain quotes around
    # the expanded header value to prevent its spaces from becoming separate curl arguments.
    $arguments = @("--variable", "%HF_TOKEN", "--expand-header", '"Authorization: Bearer {{HF_TOKEN:trim}}"') + $arguments
  }
  $startArguments = @{
    FilePath = "curl.exe"
    ArgumentList = $arguments
    WorkingDirectory = $partsDirectory
    WindowStyle = "Hidden"
    RedirectStandardError = $stderr
    PassThru = $true
  }
  if ($environment.Count) { $startArguments.Environment = $environment }
  $process = Start-Process @startArguments
  return [pscustomobject]@{ Process = $process; Segment = $segment; Temporary = $temporary; Part = $part; Stderr = $stderr }
}

function Complete-RangeDownload($item) {
  $item.Process.Refresh()
  if ($item.Process.ExitCode -ne 0) {
    $message = if (Test-Path -LiteralPath $item.Stderr) { Get-Content -LiteralPath $item.Stderr -Raw } else { "" }
    throw "Range $($item.Segment.Index) failed with curl exit $($item.Process.ExitCode): $message"
  }
  [int64]$expected = [int64]$item.Segment.End - [int64]$item.Segment.Start + 1
  [int64]$actual = (Get-Item -LiteralPath $item.Temporary).Length
  if ($actual -ne $expected) { throw "Range $($item.Segment.Index) is $actual bytes, expected $expected" }
  Move-Item -LiteralPath $item.Temporary -Destination $item.Part -Force
  if (Test-Path -LiteralPath $item.Stderr) {
    # On Windows the exited curl process can retain its redirected stderr handle for a few
    # milliseconds after HasExited becomes true. The range has already passed its byte-length
    # check and been promoted, so a delayed empty-log cleanup must never abort the download.
    for ($attempt = 0; $attempt -lt 5; $attempt++) {
      try {
        Remove-Item -LiteralPath $item.Stderr -Force -ErrorAction Stop
        break
      } catch {
        if ($attempt -eq 4) { break }
        Start-Sleep -Milliseconds 100
      }
    }
  }
}

function Write-RangeProgress {
  [int64]$complete = 0
  foreach ($segment in $segments) {
    $part = Join-Path $partsDirectory $segment.Name
    if (Test-Path -LiteralPath $part) { $complete += (Get-Item -LiteralPath $part).Length }
  }
  $percent = [Math]::Round(100.0 * $complete / $spec.Bytes, 2)
  $rate = if ($started.Elapsed.TotalSeconds -gt 0) { [Math]::Round((($complete - $initialComplete) / 1MB) / $started.Elapsed.TotalSeconds, 2) } else { 0 }
  Write-Host "${Artifact}: $percent% ($([Math]::Round($complete / 1GB, 2)) GiB) - $rate MiB/s this run"
}

$queue = [Collections.Generic.Queue[object]]::new()
foreach ($segment in $pending) { $queue.Enqueue($segment) }
$active = [Collections.ArrayList]::new()
$completedSinceReport = 0
try {
  while ($queue.Count -gt 0 -or $active.Count -gt 0) {
    while ($queue.Count -gt 0 -and $active.Count -lt $Connections) {
      [void]$active.Add((Start-RangeDownload $queue.Dequeue()))
    }
    $finishedItems = @()
    foreach ($candidate in @($active)) {
      $candidate.Process.Refresh()
      if ($candidate.Process.HasExited) { $finishedItems += $candidate }
    }
    if (-not $finishedItems.Count) {
      # A worker can exit between collecting its id and entering Wait-Process. Suppress that benign
      # race, refresh the Process objects, and loop again if the exit is not visible yet.
      Wait-Process -Id @($active | ForEach-Object { $_.Process.Id }) -Any -ErrorAction SilentlyContinue
      foreach ($candidate in @($active)) {
        $candidate.Process.Refresh()
        if ($candidate.Process.HasExited) { $finishedItems += $candidate }
      }
      if (-not $finishedItems.Count) { Start-Sleep -Milliseconds 100; continue }
    }
    foreach ($item in $finishedItems) {
      Complete-RangeDownload $item
      [void]$active.Remove($item)
      $completedSinceReport++
    }
    if ($completedSinceReport -ge $Connections -or ($queue.Count -eq 0 -and $active.Count -eq 0)) {
      Write-RangeProgress
      $completedSinceReport = 0
    }
  }
} catch {
  foreach ($item in $active) {
    if (-not $item.Process.HasExited) { Stop-Process -Id $item.Process.Id -Force -ErrorAction SilentlyContinue }
  }
  throw
}

foreach ($segment in $segments) {
  $part = Join-Path $partsDirectory $segment.Name
  [int64]$expected = [int64]$segment.End - [int64]$segment.Start + 1
  if (-not (Test-Path -LiteralPath $part) -or (Get-Item -LiteralPath $part).Length -ne $expected) {
    throw "Cannot assemble: range $($segment.Index) is incomplete"
  }
}

$assembled = $destination + ".assembling"
if (Test-Path -LiteralPath $assembled) { Remove-Item -LiteralPath $assembled -Force }
$output = [IO.File]::Open($assembled, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {
  foreach ($segment in $segments) {
    $input = [IO.File]::OpenRead((Join-Path $partsDirectory $segment.Name))
    try { $input.CopyTo($output, 16MB) } finally { $input.Dispose() }
  }
} finally {
  $output.Dispose()
}
if ((Get-Item -LiteralPath $assembled).Length -ne $spec.Bytes) {
  throw "Assembled file has the wrong length"
}
$hash = (Get-FileHash -LiteralPath $assembled -Algorithm SHA256).Hash.ToLowerInvariant()
if ($hash -ne $spec.Sha256) {
  throw "SHA-256 mismatch for ${Artifact}: got $hash, expected $($spec.Sha256)"
}
Move-Item -LiteralPath $assembled -Destination $destination

$resolvedParts = (Resolve-Path -LiteralPath $partsDirectory).Path
if ((Split-Path -Parent $resolvedParts) -ne (Split-Path -Parent $destination)) {
  throw "Refusing unsafe parts cleanup: $resolvedParts"
}
Remove-Item -LiteralPath $resolvedParts -Recurse -Force
Write-Host "$Artifact complete: $destination"
Write-Host "SHA-256: $hash"
