<#
.SYNOPSIS
  Run the MiniMax-H3 Ref2VA reference-mix matrix and record wall time.

.DESCRIPTION
  H3 rejects a truly audio-only Ref2VA request, so the audio-focused case uses the supplied image
  as the required visual anchor and assigns the audio as the only timed reference. The matrix is:
  image-only, video-only, audio-focused (image + audio), and mixed (image + video + audio).
  Base at 20 NFE with native attention is the correctness baseline. Turbo and Sage are explicit
  comparisons only after that baseline is visually approved.
#>
param(
  [Parameter(Mandatory = $true)][string]$Image,
  [Parameter(Mandatory = $true)][string]$Video,
  [Parameter(Mandatory = $true)][string]$Audio,
  [string]$OutputDirectory = ".aiv\benchmarks\h3-ref2va",
  [ValidateSet("native", "sage")][string]$Attention = "native",
  [ValidateSet("base", "turbo-4")][string]$SamplingMode = "base",
  [int]$Steps = 20,
  [ValidateSet("performance", "balanced", "minimal")][string]$Offload = "balanced",
  [int]$Width = 960,
  [int]$Height = 544,
  [int]$Frames = 124,
  [int]$Seed = 42
)

$ErrorActionPreference = "Stop"
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root "vendor\MiniMax-H3\.venv\Scripts\python.exe"
$worker = Join-Path $root "ai-runtime\aiv_runtime\minimax_h3_ref.py"
$pack = Join-Path $root "models\MiniMax-H3"
$transformer = Join-Path $pack "diffusion_models\minimax_h3_ref2va_pruned_int8_convrot.safetensors"
$encoder = Join-Path $pack "text_encoders\qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
$lora = Join-Path $pack "loras\minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
$destination = [IO.Path]::GetFullPath((Join-Path $root $OutputDirectory))

if ($SamplingMode -eq "turbo-4" -and $Steps -ne 4) {
  throw "turbo-4 requires exactly 4 transformer evaluations"
}
if ($SamplingMode -eq "base" -and ($Steps -lt 10 -or $Steps -gt 60)) {
  throw "base sampling requires 10-60 transformer evaluations"
}

$requiredPaths = @($python, $worker, $transformer, $encoder, $Image, $Video, $Audio)
if ($SamplingMode -eq "turbo-4") { $requiredPaths += $lora }
foreach ($required in $requiredPaths) {
  if (-not (Test-Path -LiteralPath $required)) { throw "Required path not found: $required" }
}
New-Item -ItemType Directory -Force -Path $destination | Out-Null

$cases = @(
  @{ Name = "image-only"; References = @(@{ kind = "image"; path = $Image; includeEmbeddedAudio = $true }) },
  @{ Name = "video-only"; References = @(@{ kind = "video"; path = $Video; includeEmbeddedAudio = $true }) },
  @{ Name = "audio-focused"; References = @(
      @{ kind = "image"; path = $Image; includeEmbeddedAudio = $true },
      @{ kind = "audio"; path = $Audio; includeEmbeddedAudio = $true }) },
  @{ Name = "mixed"; References = @(
      @{ kind = "image"; path = $Image; includeEmbeddedAudio = $true },
      @{ kind = "video"; path = $Video; includeEmbeddedAudio = $false },
      @{ kind = "audio"; path = $Audio; includeEmbeddedAudio = $true }) }
)

$results = foreach ($case in $cases) {
  $videoOut = Join-Path $destination "$($case.Name).mp4"
  $audioOut = Join-Path $destination "$($case.Name).wav"
  $references = ConvertTo-Json -InputObject @($case.References) -Compress
  $watch = [Diagnostics.Stopwatch]::StartNew()
  & $python $worker `
    --repo-root $pack --transformer-path $transformer --text-encoder-path $encoder `
    --turbo-lora-path $lora --references-json $references `
    --sampling-mode $SamplingMode --steps $Steps `
    --prompt "Benchmark shot. Follow the supplied references and generate synchronized natural audio." `
    --output-path $videoOut --audio-path $audioOut --width $Width --height $Height `
    --num-frames $Frames --seed $Seed --attention-backend $Attention --offload $Offload
  $code = $LASTEXITCODE
  $watch.Stop()
  [pscustomobject]@{
    case = $case.Name
    success = $code -eq 0
    exit_code = $code
    elapsed_seconds = [math]::Round($watch.Elapsed.TotalSeconds, 2)
    sampling_mode = $SamplingMode
    steps = $Steps
    attention = $Attention
    video = $videoOut
    audio = $audioOut
  }
}

$report = Join-Path $destination "results.csv"
$results | Export-Csv -NoTypeInformation -Path $report
$results | Format-Table -AutoSize
Write-Host "Report: $report"
if ($results.success -contains $false) { exit 1 }
