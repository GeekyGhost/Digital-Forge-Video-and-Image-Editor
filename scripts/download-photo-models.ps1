<#
.SYNOPSIS
  Fetch the two optional models Photo Edit uses: monocular depth and LaMa inpainting.

.DESCRIPTION
  Both are optional. Photo Edit grades, masks, saves and exports with neither of them installed --
  they only light up the Relight and Remove cards, which report themselves missing until the weights
  are here. SAM 2 click-select needs nothing: it shares the pack the video tracker already uses.

  Run this from a normal Windows shell rather than any sandboxed one. The depth model lives on
  Hugging Face, and some sandboxes proxy that host to a 403 while leaving GitHub reachable.

.EXAMPLE
  .\scripts\download-photo-models.ps1
  .\scripts\download-photo-models.ps1 -DepthVariant Base   # better depth, ~380 MB instead of ~100
#>
param(
  [ValidateSet("Small", "Base")][string]$DepthVariant = "Small",
  [string]$ModelRoot = (Join-Path (Split-Path -Parent $PSScriptRoot) "models"),
  [switch]$SkipDepth,
  [switch]$SkipInpaint
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot

# The TorchScript export IOPaint ships. The worker in ai-runtime/aiv_runtime/photo_inpaint.py is
# written against its contract -- model(image, mask), float32 NCHW in 0..1, returning 0..255 -- so a
# different export of the same weights may need the call adjusted.
$lamaUrl = "https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt"
$lamaSha = "344c77bbcb158f17dd143070d1e789f38a66c04202311ae3a258ef66667a9ea9"
$lamaBytes = 205669692

$depthRepo = if ($DepthVariant -eq "Base") {
  "depth-anything/Depth-Anything-V2-Base-hf"
} else {
  "depth-anything/Depth-Anything-V2-Small-hf"
}

function Write-Step([string]$text) { Write-Host "==> $text" -ForegroundColor Cyan }
function Write-Skip([string]$text) { Write-Host "    $text" -ForegroundColor DarkGray }

# ---------------------------------------------------------------- inpainting
if (-not $SkipInpaint) {
  $lamaDir = Join-Path $ModelRoot "lama"
  $lamaPath = Join-Path $lamaDir "big-lama.pt"
  if ((Test-Path $lamaPath) -and ((Get-Item $lamaPath).Length -eq $lamaBytes)) {
    Write-Skip "LaMa already installed at $lamaPath"
  } else {
    Write-Step "Downloading LaMa (~196 MB)"
    New-Item -ItemType Directory -Force -Path $lamaDir | Out-Null
    $part = "$lamaPath.part"
    # Published only once it verifies, so an interrupted run never leaves a half file that loads.
    Invoke-WebRequest -Uri $lamaUrl -OutFile $part -UseBasicParsing
    $actual = (Get-FileHash -Path $part -Algorithm SHA256).Hash.ToLower()
    if ($actual -ne $lamaSha) {
      Remove-Item $part -Force
      throw "LaMa checksum mismatch: expected $lamaSha, got $actual"
    }
    Move-Item -Force $part $lamaPath
    Write-Host "    ok $lamaPath" -ForegroundColor Green
  }
}

# --------------------------------------------------------------------- depth
if (-not $SkipDepth) {
  $depthDir = Join-Path $ModelRoot "depth-anything-v2"
  if (Test-Path (Join-Path $depthDir "config.json")) {
    Write-Skip "Depth model already installed at $depthDir"
  } else {
    Write-Step "Downloading $depthRepo"
    # Uses the utility venv's huggingface_hub, which is already there for the other model packs, so
    # this adds no dependency. A transformers directory is only usable with its config, which is why
    # the readiness check looks for config.json rather than the weights.
    $python = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path $python)) { throw "utility venv not found at $python" }
    & $python -c @"
from huggingface_hub import snapshot_download
path = snapshot_download(
    '$depthRepo',
    local_dir=r'$depthDir',
    allow_patterns=['*.json', '*.safetensors', '*.txt'],
)
print('downloaded to', path)
"@
    if ($LASTEXITCODE -ne 0) { throw "depth model download failed" }
    Write-Host "    ok $depthDir" -ForegroundColor Green
  }
}

Write-Step "Done. Relaunch Digital Forge; the Relight and Remove cards read their state at startup."
