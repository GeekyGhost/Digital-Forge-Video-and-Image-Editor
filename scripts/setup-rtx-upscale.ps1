<#
.SYNOPSIS
  Install NVIDIA's Video Effects super-resolution into the Image Studio runtime.

.DESCRIPTION
  This installs `nvidia-vfx`, NVIDIA's own Python bindings for the Maxine VFX SDK. It is a very
  different proposition from what integrating this used to require:

    - no developer-portal login, and no gated SDK zip
    - no per-GPU-architecture redistributable installer
    - no DirectX or Vulkan surface interop, and no native bridge to write
    - the VFX libraries are bundled inside the wheel

  The effect takes a CUDA tensor and returns one, so it drops into the same worker shape the
  Real-ESRGAN stage already uses.

  NOTHING NVIDIA-OWNED IS REDISTRIBUTED BY THIS PROJECT. This script downloads from NVIDIA's own
  package index, on your machine, after you have accepted NVIDIA's terms. See the Third-Party
  Components section of LICENSE.md.

  Keep this file pure ASCII. Windows PowerShell decodes a .ps1 as the machine's ANSI codepage
  unless it carries a UTF-8 BOM, and a stray em dash then becomes three bytes, one of which is a
  double quote -- which closes whatever string it sits in and reports a parse error elsewhere.
  scripts\test-all.ps1 enforces this.

.PARAMETER AcceptLicense
  Required to proceed. Passing it is your acceptance of NVIDIA's licence terms, printed below.

.PARAMETER Force
  Reinstall even when a working install is already detected.

.EXAMPLE
  .\scripts\setup-rtx-upscale.ps1 -AcceptLicense
#>
[CmdletBinding()]
param(
  [switch]$AcceptLicense,
  [switch]$Force,
  [string]$IndexUrl = "https://pypi.nvidia.com"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root "vendor\Image-Studio\.venv\Scripts\python.exe"
$uv = Get-Command uv -ErrorAction SilentlyContinue

<#
  Run a native command without letting its stderr become a terminating error.

  Windows PowerShell 5.1 converts ANY stderr output from a native command into a NativeCommandError
  when $ErrorActionPreference is 'Stop' -- including when the caller wrote 2>$null to discard it.
  That is not academic here: the "is nvidia-vfx already installed?" probe writes a ModuleNotFoundError
  traceback to stderr *by design*, so the first version of this script died on precisely the outcome
  the probe existed to detect, before ever reaching the install. PowerShell 7 does not behave this
  way, which is exactly why it has to be handled rather than tested away on a dev machine.

  Scoping the preference to 'Continue' for the duration of the call and reading $LASTEXITCODE is the
  portable fix. Callers get the exit code and the combined output, and decide for themselves.
#>
function Invoke-Native {
  param([Parameter(Mandatory)][string]$Exe, [string[]]$Arguments = @())
  $previous = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    $lines = & $Exe @Arguments 2>&1 | ForEach-Object { $_.ToString() }
    return [pscustomobject]@{
      ExitCode = $LASTEXITCODE
      Output   = (($lines) -join [Environment]::NewLine)
    }
  } finally {
    $ErrorActionPreference = $previous
  }
}

# pip and uv both write progress and warnings to stderr, so installs go through the same path.
function Install-Packages([string[]]$Packages) {
  $result = if ($uv) {
    Invoke-Native "uv" (@("pip", "install", "--python", $python) + $Packages)
  } else {
    Invoke-Native $python (@("-m", "pip", "install") + $Packages)
  }
  if ($result.Output) { Write-Host $result.Output }
  if ($result.ExitCode -ne 0) { throw "Installing $($Packages -join ' ') failed with exit code $($result.ExitCode)." }
}

if (-not (Test-Path $python)) {
  throw "The Image Studio runtime is not set up yet. Run scripts\setup-image-studio.ps1 first; nvidia-vfx installs into that same environment so it can hand tensors straight to the upscale worker."
}

# ---------------------------------------------------------------- licence gate
if (-not $AcceptLicense) {
  Write-Host ""
  Write-Host "NVIDIA Video Effects - licence acceptance required" -ForegroundColor Yellow
  Write-Host ""
  Write-Host "  This installs software published by NVIDIA under NVIDIA's own proprietary terms."
  Write-Host "  It is NOT covered by this project's licence, and Digital Forge does not"
  Write-Host "  redistribute any part of it. Review these before continuing:"
  Write-Host ""
  Write-Host "    License Agreement for NVIDIA Software Development Kits"
  Write-Host "      https://developer.download.nvidia.com/gameworks/dlss/NVIDIA_RTX_SDK_EULA_(1Feb2021).pdf"
  Write-Host "    License For Customer Use of NVIDIA Software"
  Write-Host "      https://www.nvidia.com/en-us/drivers/nvidia-license/"
  Write-Host "    Package: https://pypi.org/project/nvidia-vfx/  (LicenseRef-NvidiaProprietary)"
  Write-Host ""
  Write-Host "  Points worth knowing before you build on this:"
  Write-Host "    - commercial release requires notifying NVIDIA beforehand"
  Write-Host "    - cloud/hosted deployment is outside NVIDIA's standard terms"
  Write-Host "    - if you fork this project you redistribute nothing of NVIDIA's by doing so;"
  Write-Host "      your users run this script themselves, exactly as you are about to"
  Write-Host ""
  Write-Host "  Re-run with -AcceptLicense once you have reviewed them." -ForegroundColor Yellow
  Write-Host ""
  exit 2
}

# ---------------------------------------------------------------- preflight
# Each of these fails opaquely at runtime if it is only discovered then: a missing wheel reads as a
# broken install, and a stale driver surfaces as an effect-load error rather than a version complaint.
Write-Host "Checking prerequisites" -ForegroundColor Cyan

$versionProbe = Invoke-Native $python @("-c", "import sys; print('%d.%d' % sys.version_info[:2])")
if ($versionProbe.ExitCode -ne 0) { throw "Could not run the Image Studio Python: $($versionProbe.Output)" }
$pyVersion = $versionProbe.Output.Trim()
Write-Host "  Python           $pyVersion  (nvidia-vfx requires 3.10 or later)"
if ([version]$pyVersion -lt [version]"3.10") {
  throw "nvidia-vfx requires Python 3.10 or later; the Image Studio runtime is on $pyVersion."
}

# Driver floor is 570.65 on Windows. nvidia-smi ships with the driver, so its absence is a finding.
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  $gpuProbe = Invoke-Native "nvidia-smi" @("--query-gpu=name,driver_version", "--format=csv,noheader")
  if ($gpuProbe.ExitCode -eq 0 -and $gpuProbe.Output) {
    $fields = ($gpuProbe.Output -split "`n")[0] -split ",\s*"
    Write-Host "  GPU              $($fields[0])"
    Write-Host "  Driver           $($fields[1])  (requires 570.65 or later)"
    try {
      if ([version]$fields[1] -lt [version]"570.65") {
        Write-Warning "Driver $($fields[1]) is below NVIDIA's documented 570.65 floor for the VFX super-resolution effect. The install will succeed and the effect will fail to load. Update the driver first."
      }
    } catch {
      Write-Warning "Could not compare driver version '$($fields[1])' against the 570.65 floor; continuing."
    }
  }
} else {
  Write-Warning "nvidia-smi was not found, so the driver version could not be checked. NVIDIA VFX needs an RTX GPU with Tensor Cores (Turing or newer) and driver 570.65+."
}

if (-not $Force) {
  # A non-zero exit here is the ordinary "not installed yet" answer, not an error.
  $installed = Invoke-Native $python @("-c", "import nvvfx")
  if ($installed.ExitCode -eq 0) {
    Write-Host ""
    Write-Host "nvidia-vfx is already installed. Re-run with -Force to reinstall." -ForegroundColor Green
    exit 0
  }
}

# ---------------------------------------------------------------- install
# The PyPI entry for nvidia-vfx is a `wheel-stub`: a 2.6 KB source package whose only job is to
# redirect the real download to NVIDIA's index. Without --extra-index-url the install fails with a
# build error rather than a useful message, so the index is passed explicitly.
Write-Host ""
Write-Host "Installing nvidia-vfx from $IndexUrl" -ForegroundColor Cyan
Install-Packages @("nvidia-vfx", "--extra-index-url", $IndexUrl)

# ---------------------------------------------------------------- verify
# Importing is not enough: the wheel can install cleanly and still fail to load its effect when the
# driver or GPU is unsuitable. Build the effect and run one real frame.
Write-Host ""
Write-Host "Verifying the effect actually loads and runs" -ForegroundColor Cyan
$probeSource = @'
import sys
try:
    import torch
    from nvvfx import VideoSuperRes
except Exception as error:
    print("IMPORT-FAILED %s: %s" % (type(error).__name__, error))
    sys.exit(1)

if not torch.cuda.is_available():
    print("NO-CUDA torch cannot see a CUDA device")
    sys.exit(1)

try:
    levels = [name for name in dir(VideoSuperRes.QualityLevel) if not name.startswith("_")]
    frame = torch.rand(3, 180, 320, device="cuda", dtype=torch.float32)
    effect = VideoSuperRes(quality=VideoSuperRes.QualityLevel.HIGH)
    effect.output_width, effect.output_height = 640, 360
    effect.load()
    # The returned buffer is reused by the next run(), so clone before touching it.
    out = torch.from_dlpack(effect.run(frame).image).clone()
    print("OK %s quality-levels=%d" % (tuple(out.shape), len(levels)))
    print("QUALITY %s" % ",".join(sorted(levels)))
except Exception as error:
    print("RUN-FAILED %s: %s" % (type(error).__name__, error))
    sys.exit(1)
'@
$probeFile = Join-Path ([System.IO.Path]::GetTempPath()) "aiv-nvvfx-probe.py"
Set-Content -Path $probeFile -Value $probeSource -Encoding ASCII
$verify = Invoke-Native $python @($probeFile)
Remove-Item $probeFile -ErrorAction SilentlyContinue

if ($verify.Output) { Write-Host $verify.Output }
if ($verify.ExitCode -ne 0) {
  Write-Host ""
  Write-Warning "nvidia-vfx installed but the effect did not run. The AI upscale stage will stay disabled and Real-ESRGAN remains available. Most common causes: a driver below 570.65, a GPU without Tensor Cores, or no wheel published for Python $pyVersion on Windows."
  exit 1
}

Write-Host ""
Write-Host "NVIDIA VFX super-resolution is ready." -ForegroundColor Green
Write-Host "It appears in the upscale model list alongside Real-ESRGAN, and follows the same"
Write-Host "chain rule: AI stage first, resample second."
