<#
.SYNOPSIS
    Runs the LTX single-stage adapter directly, exactly as the editor invokes it.

.DESCRIPTION
    The editor spawns this adapter as a child process and captures its output. When the worker
    dies without writing a readable message, that output never reaches the UI in a useful form.
    This script removes the editor from the picture entirely: same interpreter, same arguments,
    same working directory and environment, but with stdout and stderr going straight to your
    console and to a log file.

    It reproduces the environment the launcher hands the editor, including the VIRTUAL_ENV and
    PATH entries that point at the *application* venv rather than the LTX one, because that
    difference is itself a candidate cause.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\diagnose-ltx.ps1
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\diagnose-ltx.ps1 -Offload cpu
    powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\diagnose-ltx.ps1 -DryRunOnly
#>
[CmdletBinding()]
param(
    [ValidateSet('first-last', 'image', 'text')]
    [string]$Mode = 'first-last',
    [ValidateSet('disk', 'cpu')]
    [string]$Offload = 'disk',
    [int]$NumFrames = 121,
    [int]$Width = 768,
    [int]$Height = 512,
    [string]$Prompt = 'a slow cinematic push in, natural light, gentle motion',
    # 1.0 is unguided and selects SimpleDenoiser. Anything above 1.0 selects GuidedDenoiser,
    # which runs a second unconditional pass per step and roughly doubles peak VRAM.
    [double]$CfgScale = 1.0,
    [double]$AudioCfgScale = 1.5,
    # 'fp8' uses the scaled-FP8 streaming path (fp8-scaled-mm). 'bf16' uses the 46 GB checkpoint,
    # which routes through fp8-cast and DiffusionStage.from_checkpoint instead - a completely
    # different loader. This is the A/B test for a fault inside the scaled-FP8 path.
    [ValidateSet('fp8', 'bf16')]
    [string]$Checkpoint = 'fp8',
    [switch]$DryRunOnly,
    # Skip the launcher's environment and run with a clean one, to test whether the inherited
    # VIRTUAL_ENV / PATH from the launcher is what breaks the worker.
    [switch]$CleanEnvironment
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LtxPython   = Join-Path $ProjectRoot 'vendor\LTX-2\.venv\Scripts\python.exe'
$Adapter     = Join-Path $ProjectRoot 'ai-runtime\aiv_runtime\ltx_single_stage.py'
$CheckpointFile = if ($Checkpoint -eq 'bf16') {
    Join-Path $ProjectRoot 'models\LTX-2.3\ltx-2.3-22b-distilled-1.1.safetensors'
} else {
    Join-Path $ProjectRoot 'models\LTX-2.3\ltx-2.3-22b-distilled-fp8.safetensors'
}
$Gemma       = Join-Path $ProjectRoot 'models\gemma-3-12b'
$First       = Join-Path $ProjectRoot 'cache\aiv-default-first.png'
$Last        = Join-Path $ProjectRoot 'cache\aiv-default-last.png'
$OutDir      = Join-Path $ProjectRoot 'cache'
$Output      = Join-Path $OutDir ('ltx-diagnose-{0}.mp4' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
$LogPath     = Join-Path $ProjectRoot '.aiv\ltx-diagnose.log'

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null

function Assert-Path([string]$Path, [string]$What) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "$What not found: $Path" }
    Write-Host ("  OK  {0,-22} {1}" -f $What, $Path) -ForegroundColor DarkGreen
}

Write-Host ''
Write-Host 'Preflight' -ForegroundColor Cyan
Assert-Path $LtxPython  'LTX interpreter'
Assert-Path $Adapter    'adapter'
Assert-Path $CheckpointFile "$Checkpoint checkpoint"
Assert-Path $Gemma      'Gemma encoder'
if ($Mode -ne 'text') { Assert-Path $First 'first frame' }
if ($Mode -eq 'first-last') { Assert-Path $Last 'last frame' }

# Free space matters: the FP8 path streams weights and the muxer needs scratch.
$Drive = (Get-Item $ProjectRoot).PSDrive
Write-Host ("  --  free space           {0:N1} GB on {1}:" -f ($Drive.Free / 1GB), $Drive.Name) -ForegroundColor DarkGray

# From here on we invoke native commands that legitimately write to stderr (torch and
# transformers emit warnings there). Under 'Stop' that would abort the script on a warning, so
# switch to explicit $LASTEXITCODE checks instead - which every call below already does.
$ErrorActionPreference = 'Continue'

Write-Host ''
Write-Host 'Interpreter and CUDA' -ForegroundColor Cyan
& $LtxPython -c "import sys, torch; print('  python      ', sys.version.split()[0]); print('  torch       ', torch.__version__); print('  cuda build  ', torch.version.cuda); print('  cuda avail  ', torch.cuda.is_available()); print('  device      ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('  free VRAM   ', f'{torch.cuda.mem_get_info()[0]/1024**3:.2f} GiB' if torch.cuda.is_available() else 'n/a')"
if ($LASTEXITCODE -ne 0) { throw "The LTX interpreter could not start or torch failed to import (exit $LASTEXITCODE). That alone is the bug." }

Write-Host ''
Write-Host 'LTX package imports' -ForegroundColor Cyan
& $LtxPython -c "import ltx_core, ltx_pipelines; from ltx_core.model.transformer import LTXModelConfigurator, X0Model, LTXV_MODEL_COMFY_RENAMING_MAP; from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number; from ltx_core.block_streaming import DISK_CPU_SLOTS, StreamingModelBuilder; from ltx_pipelines.utils.blocks import AudioDecoder, DiffusionStage, ImageConditioner, PromptEncoder, VideoDecoder; print('  all adapter imports resolved')"
if ($LASTEXITCODE -ne 0) { throw "The LTX packages failed to import (exit $LASTEXITCODE). That is the bug - the traceback above names the symbol." }

$Arguments = @(
    # An access violation kills the interpreter before Python can raise, which is why this crash
    # has produced no traceback so far. faulthandler installs a native fault handler that dumps
    # the Python stack on SIGSEGV / EXCEPTION_ACCESS_VIOLATION, so the crashing frame is named
    # even though no exception is ever raised. This costs nothing when nothing crashes.
    '-X', 'faulthandler'
    $Adapter
    '--checkpoint-path', $CheckpointFile
    '--gemma-root',      $Gemma
    '--mode',            $Mode
    '--output-path',     $Output
    '--prompt',          $Prompt
    '--width',           $Width
    '--height',          $Height
    '--num-frames',      $NumFrames
    '--frame-rate',      '24'
    '--seed',            '42'
    '--guide-strength',  '0.8'
    '--cfg-scale',       $CfgScale
    '--audio-cfg-scale', $AudioCfgScale
    '--offload',         $Offload
)
if ($Mode -ne 'text')        { $Arguments += @('--first-frame', $First) }
if ($Mode -eq 'first-last')  { $Arguments += @('--last-frame',  $Last) }

# The editor sets exactly these three; the launcher additionally leaks VIRTUAL_ENV and a PATH
# entry for the *application* venv into the child, so mirror that unless asked not to.
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8       = '1'
if ($CleanEnvironment) {
    Remove-Item Env:\VIRTUAL_ENV -ErrorAction SilentlyContinue
    Write-Host ''
    Write-Host 'Running with a CLEAN environment (no inherited VIRTUAL_ENV)' -ForegroundColor Yellow
} else {
    $env:VIRTUAL_ENV = Join-Path $ProjectRoot '.venv'
    $env:PATH = (Join-Path $ProjectRoot '.venv\Scripts') + ';' + $env:PATH
}

Write-Host ''
Write-Host 'Validating arguments (--dry-run)' -ForegroundColor Cyan
& $LtxPython @Arguments --dry-run
if ($LASTEXITCODE -ne 0) { throw "Argument validation failed (exit $LASTEXITCODE). The traceback above is the bug." }
if ($DryRunOnly) { Write-Host ''; Write-Host 'Dry run only; stopping here.' -ForegroundColor Green; exit 0 }

Write-Host ''
Write-Host "Running the real generation - output goes to $Output" -ForegroundColor Cyan
Write-Host "Full transcript: $LogPath" -ForegroundColor DarkGray
Write-Host ('-' * 70) -ForegroundColor DarkGray

# The worker writes progress to stderr. Two PowerShell traps make that hostile:
#
#   1. $ErrorActionPreference = 'Stop' turns *any* stderr line from a native command into a
#      terminating error, so the very first progress message aborts the run.
#   2. Piping `2>&1` into Tee-Object wraps stderr lines as NativeCommandError records rather
#      than plain text, which mangles a traceback into PowerShell error noise.
#
# Start-Process sidesteps both: it takes the arguments as an array (no shell quoting to get
# wrong), writes each stream straight to a file, and returns the real exit code. Windows
# PowerShell 5.1 mangles embedded quotes when handing a command string to cmd.exe, so this is
# the only approach that is reliable for paths containing spaces - which every path here has.
$StdOutFile = Join-Path $env:TEMP ('aiv-ltx-out-{0}.txt' -f $PID)
$StdErrFile = Join-Path $env:TEMP ('aiv-ltx-err-{0}.txt' -f $PID)
$Previous = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
try {
    # -ArgumentList joins an array with spaces and does NOT quote the elements, so every path
    # containing a space is split into separate arguments. Quote each one explicitly. argparse
    # is happy with quoted flags and values alike, so quoting everything is safe.
    $Quoted = $Arguments | ForEach-Object { '"{0}"' -f $_ }
    $Process = Start-Process -FilePath $LtxPython -ArgumentList $Quoted `
        -WorkingDirectory $ProjectRoot -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $StdOutFile -RedirectStandardError $StdErrFile
    $WorkerExit = $Process.ExitCode
} finally {
    $ErrorActionPreference = $Previous
}

# Merge both streams into the transcript. stderr goes last because that is where the traceback
# lands, and it is what matters most when reading the tail of a failed run.
$Transcript = @()
if (Test-Path -LiteralPath $StdOutFile) { $Transcript += Get-Content -LiteralPath $StdOutFile }
if (Test-Path -LiteralPath $StdErrFile) {
    $ErrLines = Get-Content -LiteralPath $StdErrFile
    if ($ErrLines) { $Transcript += '--- worker stderr ---'; $Transcript += $ErrLines }
}
$Transcript | Set-Content -LiteralPath $LogPath -Encoding UTF8
$Transcript | ForEach-Object { Write-Host $_ }
Remove-Item -LiteralPath $StdOutFile, $StdErrFile -ErrorAction SilentlyContinue

Write-Host ('-' * 70) -ForegroundColor DarkGray
Write-Host ''
if ($WorkerExit -eq 0 -and (Test-Path -LiteralPath $Output)) {
    $Size = (Get-Item -LiteralPath $Output).Length
    Write-Host ("SUCCESS - produced {0} ({1:N1} MB)" -f $Output, ($Size / 1MB)) -ForegroundColor Green
    Write-Host 'LTX itself is healthy; the fault is in how the editor invokes or reports it.' -ForegroundColor Green
} else {
    # Windows returns NTSTATUS values as a *signed* int, so an access violation arrives as
    # -1073741819. Casting that straight to [uint32] throws; mask to 32 bits first.
    $Unsigned = [uint32]($WorkerExit -band 0xFFFFFFFFL)
    $Hex = '0x{0:X8}' -f $Unsigned
    Write-Host ("FAILED - worker exit code $WorkerExit ($Hex)") -ForegroundColor Red
    switch ($Unsigned) {
        0xC0000005 { Write-Host '  Access violation: a native crash inside torch/CUDA.' -ForegroundColor Yellow }
        0xC0000017 { Write-Host '  Out of memory: the system commit limit was exhausted.' -ForegroundColor Yellow }
        0xC0000135 { Write-Host '  A dependency DLL was not found.' -ForegroundColor Yellow }
        0xC0000142 { Write-Host '  A dependency DLL failed to initialise - same fault as the launcher popup.' -ForegroundColor Yellow }
        1          { Write-Host '  Python raised an exception; the traceback is above.' -ForegroundColor Yellow }
        default    { Write-Host '  See the transcript above for the failing stage.' -ForegroundColor Yellow }
    }
    Write-Host ''
    Write-Host "Send me $LogPath and I can work from the actual failure." -ForegroundColor Cyan
}
exit $WorkerExit
