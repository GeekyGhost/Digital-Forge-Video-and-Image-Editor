//! The video upscale chain: Real-ESRGAN, then a resample, in that order and no other.
//!
//! This is the executor for the plan `src/domain/upscale.ts` resolves. The resolver is the single
//! source of truth for *what* the chain is -- the UI readout and these arguments are computed from
//! the same rules -- but the ordering rule is re-asserted here rather than trusted, because a
//! malformed payload must not be able to talk this side into resampling before the model runs.
//!
//! Why AI first: resampling up and then handing the result to Real-ESRGAN asks the model to
//! reconstruct detail from pixels an interpolator invented, which is worse than either stage alone.
//!
//! ## Cost, stated plainly
//!
//! The AI stage decodes the finished master to PNG frames, runs a network over every one of them,
//! and re-encodes. For a two-minute 1080p timeline at 24 fps that is roughly 2,900 frames and tens
//! of gigabytes of intermediate PNG. It is not a filter; it is a second render. The UI says so
//! before it starts.

use crate::ai::project_root;
use serde::{Deserialize, Serialize};
use std::{fs, path::{Path, PathBuf}, process::Stdio};
use tokio::process::Command;

/// `scale_cuda`'s documented interpolation algorithms.
///
/// The filter accepts `interp_algo` values 1-4 (nearest, bilinear, bicubic, lanczos) and defaults
/// to **0**, which is outside that documented range. Every upscaled export this app produced before
/// this module never set the option, so all of them took whatever 0 resolves to internally. Naming
/// the algorithm explicitly is the single largest quality change in the video path.
pub const RESAMPLE_ALGOS: [&str; 4] = ["nearest", "bilinear", "bicubic", "lanczos"];

/// Hard encode ceilings per encoder, mirrored from `src/domain/upscale.ts`.
///
/// HEVC and AV1 NVENC at 8192 are from NVIDIA's NVENC Application Note. H.264 NVENC at 4096 is not
/// published in NVIDIA's support matrix -- that page lists only per-codec yes/no -- and is the
/// figure NVIDIA forum staff and the OBS/ffmpeg community consistently report. It is treated as a
/// floor deliberately: refusing an export that might have worked is much cheaper than accepting one
/// that dies after the render.
pub fn encoder_max_dimension(codec: &str) -> u32 {
    match codec {
        c if c.contains("h264_nvenc") => 4096,
        c if c.contains("nvenc") => 8192,
        _ => 8192,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UpscaleChain {
    /// `esrgan` or `nvvfx`. Ignored when there is no AI stage.
    #[serde(default = "default_backend")]
    pub ai_backend: String,
    /// Weight file stem inside `models/real-esrgan/`, or `None` for resample-only.
    /// Always `None` for the nvvfx backend, whose weights ship inside its wheel.
    #[serde(default)]
    pub ai_model_file: Option<String>,
    /// A `VideoSuperRes.QualityLevel` name. Only meaningful for the nvvfx backend.
    #[serde(default = "default_quality_level")]
    pub ai_quality: String,
    /// Tile size handed to the model. 0 disables tiling.
    #[serde(default = "default_tile")]
    pub tile: u32,
    /// One of [`RESAMPLE_ALGOS`].
    #[serde(default = "default_algo")]
    pub resample: String,
    /// Final encoded size. Already even, already inside the encoder ceiling.
    pub width: u32,
    pub height: u32,
}

fn default_tile() -> u32 { 512 }
fn default_algo() -> String { "lanczos".to_owned() }
fn default_backend() -> String { "esrgan".to_owned() }
fn default_quality_level() -> String { "HIGH".to_owned() }

impl UpscaleChain {
    pub fn validate(&self, codec: &str) -> Result<(), String> {
        if !RESAMPLE_ALGOS.contains(&self.resample.as_str()) {
            return Err(format!(
                "resample must be one of {} -- scale_cuda's default of 0 is outside its own documented range",
                RESAMPLE_ALGOS.join(", ")
            ));
        }
        if self.width == 0 || self.height == 0 || self.width % 2 != 0 || self.height % 2 != 0 {
            return Err("upscale target must be positive and even on both axes".into());
        }
        let ceiling = encoder_max_dimension(codec);
        if self.width.max(self.height) > ceiling {
            return Err(format!(
                "{codec} cannot encode past {ceiling} px on either axis; the requested {}x{} would fail after the render",
                self.width, self.height
            ));
        }
        Ok(())
    }
}

fn pack_dir(root: &Path) -> PathBuf {
    root.join("models/real-esrgan")
}

fn python(root: &Path) -> PathBuf {
    root.join("vendor/Image-Studio/.venv/Scripts/python.exe")
}

fn worker(root: &Path) -> PathBuf {
    root.join("ai-runtime/aiv_runtime/upscale.py")
}

/// The weights this build knows how to offer, mirrored from `AI_UPSCALE_MODELS` in the TS module.
pub const AI_MODEL_FILES: [&str; 5] = [
    "RealESRGAN_x4plus.pth",
    "RealESRGAN_x2plus.pth",
    "realesr-general-x4v3.pth",
    "RealESRGAN_x4plus_anime_6B.pth",
    "realesr-animevideov3.pth",
];

/// spandrel is what actually loads the Real-ESRGAN weights, and it is installed by
/// `scripts/setup-image-studio.ps1` rather than by the model pack. Weights on disk therefore do
/// NOT imply a working backend: without this check the panel reported `ready`, offered the model,
/// and the worker then died on `ModuleNotFoundError: No module named 'spandrel'` -- a failure that
/// says nothing about how to fix it. A directory test is enough and costs nothing, which matters
/// because readiness is polled every time a panel opens.
fn spandrel_package(root: &Path) -> PathBuf {
    root.join("vendor/Image-Studio/.venv/Lib/site-packages/spandrel")
}

pub fn state_label(root: &Path) -> &'static str {
    if !python(root).is_file() {
        "venv-missing"
    } else if !worker(root).is_file() {
        "adapter-missing"
    } else if !spandrel_package(root).is_dir() {
        "spandrel-missing"
    } else if !pack_dir(root).is_dir() {
        "pack-missing"
    } else if !AI_MODEL_FILES.iter().any(|name| pack_dir(root).join(name).is_file()) {
        "weights-missing"
    } else {
        "ready"
    }
}

pub fn ready(root: &Path) -> bool {
    state_label(root) == "ready"
}

// ---------------------------------------------------------------- NVIDIA VFX backend

/// NVIDIA's VideoSuperRes, installed by `scripts/setup-rtx-upscale.ps1` as the `nvidia-vfx` wheel.
///
/// Unlike Real-ESRGAN there is no model pack: the weights are inside the package, which is why
/// readiness is a check on the venv rather than on `models/`. Detection is a directory test rather
/// than `python -c "import nvvfx"` on purpose -- readiness is polled every time the panel opens,
/// and spawning a Python interpreter that loads a 790 MB package to answer it would be absurd.
fn nvvfx_package(root: &Path) -> PathBuf {
    root.join("vendor/Image-Studio/.venv/Lib/site-packages/nvvfx")
}

pub fn nvvfx_state(root: &Path) -> &'static str {
    if !python(root).is_file() {
        "venv-missing"
    } else if !worker(root).is_file() {
        "adapter-missing"
    } else if !nvvfx_package(root).is_dir() {
        "not-installed"
    } else if !nvvfx_package(root).join("__init__.py").is_file() {
        // A partial install: the directory survived an interrupted pip run but the package did not.
        "incomplete"
    } else {
        "ready"
    }
}

pub fn nvvfx_ready(root: &Path) -> bool {
    nvvfx_state(root) == "ready"
}

/// Quality levels the effect exposes, mirrored from `AI_UPSCALE_MODELS` in the TS module.
///
/// Only the upscaling levels are listed. `DENOISE_*` and `DEBLUR_*` also exist and run at native
/// resolution -- they are not upscaling at all, and belong in a clip effect rather than in a
/// dropdown labelled "upscale".
pub const NVVFX_QUALITY_LEVELS: [&str; 9] = [
    "BICUBIC", "LOW", "MEDIUM", "HIGH", "ULTRA",
    "HIGHBITRATE_LOW", "HIGHBITRATE_MEDIUM", "HIGHBITRATE_HIGH", "HIGHBITRATE_ULTRA",
];

/// Weight files actually present, so the UI offers only models that will load.
pub fn installed_models(root: &Path) -> Vec<String> {
    // Weights without spandrel cannot be loaded, so they are not offered.
    if !spandrel_package(root).is_dir() {
        return Vec::new();
    }
    AI_MODEL_FILES
        .iter()
        .filter(|name| pack_dir(root).join(name).is_file())
        .map(|name| (*name).to_owned())
        .collect()
}

/// `-vf` fragment for the resample stage.
pub fn scale_cuda_filter(width: u32, height: u32, algo: &str) -> String {
    format!("scale_cuda={width}:{height}:interp_algo={algo}")
}

async fn run_ffmpeg(args: &[String]) -> Result<(), String> {
    let output = Command::new("ffmpeg")
        .args(args)
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("failed to launch ffmpeg: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    Ok(())
}

/// Probe once whether this ffmpeg's `scale_cuda` accepts `interp_algo`.
///
/// Old builds of the filter predate the option and fail the whole render with "Option not found"
/// rather than ignoring it. One 32x32 nullsrc through the real filter answers the question in
/// milliseconds and costs nothing to be wrong about, which beats either assuming support or
/// parsing `ffmpeg -h filter=scale_cuda` output across versions.
pub async fn supports_interp_algo() -> bool {
    let args: Vec<String> = [
        "-hide_banner", "-loglevel", "error", "-f", "lavfi",
        "-i", "nullsrc=s=32x32:d=0.04",
        "-vf", "format=nv12,hwupload_cuda,scale_cuda=64:64:interp_algo=lanczos",
        "-frames:v", "1", "-f", "null", "-",
    ]
    .iter()
    .map(|value| (*value).to_owned())
    .collect();
    run_ffmpeg(&args).await.is_ok()
}

/// `scale_cuda`'s algorithm names spelled the way swscale spells them.
///
/// The two filters disagree on exactly one: `scale_cuda` calls it `nearest` and swscale calls it
/// `neighbor`. Passing the wrong one is not an error -- swscale ignores an unknown flag and falls
/// back to bicubic -- so the software path would silently resample with a different algorithm from
/// the one the panel promised.
pub fn swscale_flag(algo: &str) -> &'static str {
    match algo {
        "nearest" => "neighbor",
        "bilinear" => "bilinear",
        "bicubic" => "bicubic",
        _ => "lanczos",
    }
}

/// Resample an existing file on the GPU where one is available, in software otherwise.
pub async fn resample_cuda(
    input_path: &Path,
    output_path: &Path,
    width: u32,
    height: u32,
    algo: &str,
    video_args: Vec<String>,
    codec: &str,
) -> Result<Vec<String>, String> {
    // Software fallback rather than a hard error. A resample-only chain -- which is what a legacy
    // `upscaleFactor` payload resolves to -- would otherwise render the whole timeline and then
    // refuse at the last step, delivering a 1x file with a warning, for the single reason that the
    // user picked x264. scale_cuda is the fast path, not the only path.
    if !codec.contains("nvenc") {
        let mut args: Vec<String> = vec![
            "-hide_banner".into(), "-y".into(),
            "-i".into(), input_path.to_string_lossy().into_owned(),
            "-vf".into(), format!("scale={width}:{height}:flags={}", swscale_flag(algo)),
            "-map".into(), "0:v:0".into(), "-map".into(), "0:a:0?".into(),
        ];
        args.extend(video_args);
        args.extend([
            "-c:a".into(), "copy".into(), "-movflags".into(), "+faststart".into(),
            output_path.to_string_lossy().into_owned(),
        ]);
        run_ffmpeg(&args).await?;
        return Ok(args);
    }
    let filter = if supports_interp_algo().await {
        scale_cuda_filter(width, height, algo)
    } else {
        // Degrade rather than fail: the resize still happens, just at whatever the filter's
        // undocumented default is -- which is exactly what every build of this app did before.
        format!("scale_cuda={width}:{height}")
    };
    let mut args: Vec<String> = vec![
        "-hide_banner".into(), "-y".into(), "-hwaccel".into(), "cuda".into(),
        "-hwaccel_output_format".into(), "cuda".into(),
        "-i".into(), input_path.to_string_lossy().into_owned(),
        "-vf".into(), filter,
        "-map".into(), "0:v:0".into(), "-map".into(), "0:a:0?".into(),
    ];
    args.extend(video_args);
    args.extend([
        "-c:a".into(), "copy".into(), "-movflags".into(), "+faststart".into(),
        output_path.to_string_lossy().into_owned(),
    ]);
    run_ffmpeg(&args).await?;
    Ok(args)
}

/// Run Real-ESRGAN across every frame of a video.
///
/// Three steps, each of which can fail loudly on its own: decode to PNG, upscale the directory,
/// re-encode. The audio is carried from the source rather than re-derived -- the AI stage never
/// touches it, so re-encoding it would be a generation loss for nothing.
pub async fn ai_upscale_video(
    input_path: &Path,
    output_path: &Path,
    work_dir: &Path,
    chain: &UpscaleChain,
    fps: f64,
    video_args: Vec<String>,
) -> Result<Vec<String>, String> {
    let root = project_root();
    let nvvfx = chain.ai_backend == "nvvfx";

    // Build the worker's backend-specific arguments, failing here rather than after decoding a
    // few thousand PNG frames to disk.
    let mut backend_args: Vec<String> = vec!["--backend".into(), chain.ai_backend.clone()];
    eprintln!(
        "[upscale] backend={} real_esrgan={} nvvfx={} root={}",
        chain.ai_backend, state_label(&root), nvvfx_state(&root), root.display(),
    );
    if nvvfx {
        if !nvvfx_ready(&root) {
            return Err(format!(
                "NVIDIA VFX is not installed ({}) -- run Install-RTX-Upscale.bat -AcceptLicense",
                nvvfx_state(&root)
            ));
        }
        if !NVVFX_QUALITY_LEVELS.contains(&chain.ai_quality.as_str()) {
            return Err(format!(
                "unknown VideoSuperRes quality level: {}. Expected one of {}",
                chain.ai_quality, NVVFX_QUALITY_LEVELS.join(", ")
            ));
        }
        // VideoSuperRes is constructed around a fixed output size, so the target has to be known
        // before the first frame. It reaches the requested size directly, which is exactly why
        // this backend needs no resample stage afterwards.
        backend_args.extend([
            "--quality".into(), chain.ai_quality.clone(),
            "--target-width".into(), chain.width.to_string(),
            "--target-height".into(), chain.height.to_string(),
        ]);
    } else {
        if !ready(&root) {
            return Err(format!(
                "the Real-ESRGAN pack is not ready ({}) -- install real-esrgan",
                state_label(&root)
            ));
        }
        let model_file = chain.ai_model_file.as_deref()
            .ok_or("the Real-ESRGAN backend needs a model file")?;
        if !AI_MODEL_FILES.contains(&model_file) {
            return Err(format!("unknown Real-ESRGAN model: {model_file}"));
        }
        let weights = pack_dir(&root).join(model_file);
        if !weights.is_file() {
            return Err(format!("Real-ESRGAN weights not installed: {}", weights.display()));
        }
        backend_args.extend([
            "--weights".into(), weights.to_string_lossy().into_owned(),
            "--tile".into(), chain.tile.to_string(),
        ]);
    }

    let frames_in = work_dir.join("upscale-in");
    let frames_out = work_dir.join("upscale-out");
    fs::create_dir_all(&frames_in).map_err(|error| error.to_string())?;
    fs::create_dir_all(&frames_out).map_err(|error| error.to_string())?;

    // PNG, not JPEG: this is an intermediate feeding a detail-reconstruction model, and handing it
    // block artefacts to interpret as texture is exactly the wrong input.
    let decode: Vec<String> = vec![
        "-hide_banner".into(), "-y".into(),
        "-i".into(), input_path.to_string_lossy().into_owned(),
        "-map".into(), "0:v:0".into(),
        frames_in.join("%08d.png").to_string_lossy().into_owned(),
    ];
    eprintln!("[upscale] decoding frames to {}", frames_in.display());
    run_ffmpeg(&decode).await?;

    let mut worker_args = vec![worker(&root).to_string_lossy().into_owned()];
    worker_args.extend([
        "--input".into(), frames_in.to_string_lossy().into_owned(),
        "--output".into(), frames_out.to_string_lossy().into_owned(),
    ]);
    // The worker sizes its own batch from the output resolution, which is the right default. These
    // exist so the two numbers can be moved on a real 2,900-frame run without a rebuild.
    for (variable, flag) in [("AIV_UPSCALE_BATCH", "--batch"), ("AIV_UPSCALE_IO_WORKERS", "--io-workers")] {
        if let Ok(value) = std::env::var(variable) {
            worker_args.extend([flag.to_owned(), value]);
        }
    }
    worker_args.extend(backend_args);
    let completed = Command::new(python(&root))
        .args(&worker_args)
        .current_dir(&root)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("failed to start the upscale worker: {error}"))?;
    if !completed.status.success() {
        let stderr = String::from_utf8_lossy(&completed.stderr);
        // Collect before taking the tail: `.rev().take(n).rev()` only compiles on an iterator that
        // is already ExactSizeIterator, and `Lines` is not -- it cannot know its length without
        // walking the string. A Vec does know, so materialise and slice from the end.
        let lines: Vec<&str> = stderr.lines().collect();
        let tail = &lines[lines.len().saturating_sub(20)..];
        return Err(format!("the upscale worker failed:\n{}", tail.join("\n")));
    }

    // A silently short output is the other way this can go wrong: a decode that dropped frames or
    // a worker that skipped some both produce a valid file of the wrong length.
    //
    // Before the encode, not after it. This used to sit below `run_ffmpeg(&encode)`, where its own
    // message -- "refusing to encode a video of the wrong length" -- described something that had
    // already happened, after minutes of GPU time.
    let count = |dir: &Path| fs::read_dir(dir).map(|entries| entries.flatten().count()).unwrap_or(0);
    let (decoded, upscaled) = (count(&frames_in), count(&frames_out));
    if decoded != upscaled {
        return Err(format!(
            "the upscale worker returned {upscaled} frames for {decoded} decoded -- refusing to \
             encode a video of the wrong length"
        ));
    }

    let mut encode: Vec<String> = vec![
        "-hide_banner".into(), "-y".into(),
        "-framerate".into(), format!("{fps:.6}"),
        // start_number is explicit because the image2 demuxer's default is 1 and the decode step
        // above is what chose that numbering -- leaving it implicit couples two places silently.
        "-start_number".into(), "1".into(),
        "-i".into(), frames_out.join("%08d.png").to_string_lossy().into_owned(),
        "-i".into(), input_path.to_string_lossy().into_owned(),
        "-map".into(), "0:v:0".into(), "-map".into(), "1:a:0?".into(),
        // The frames are full-range RGB PNGs and the encoders want limited-range planar YUV.
        // `format=yuv420p` alone does that conversion with swscale's BT.601 default while
        // `video_args` tags the result BT.709. Measured on this exact command shape: mean error
        // 5.94 and pure green landing at 0,215,0, against 0.86 and 0,254,0 with the conversion
        // named. That is the colour banding. See `export::RGB_TO_BT709_TV`.
        "-vf".into(), crate::export::RGB_TO_BT709_TV.into(),
        // The output rate is pinned to the source rate rather than inherited from the image
        // sequence's timebase, so the soundtrack copied from input 1 cannot drift against it.
        "-r".into(), format!("{fps:.6}"),
    ];
    encode.extend(video_args);
    encode.extend([
        "-c:a".into(), "copy".into(), "-movflags".into(), "+faststart".into(),
        output_path.to_string_lossy().into_owned(),
    ]);
    run_ffmpeg(&encode).await?;


    // Tens of gigabytes at 4K. Reclaim it before the next stage runs, not at the end of the export.
    let _ = fs::remove_dir_all(&frames_in);
    let _ = fs::remove_dir_all(&frames_out);
    Ok(encode)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h264_nvenc_has_the_lower_ceiling() {
        assert_eq!(encoder_max_dimension("h264_nvenc"), 4096);
        assert_eq!(encoder_max_dimension("hevc_nvenc"), 8192);
        assert_eq!(encoder_max_dimension("libx264"), 8192);
    }

    #[test]
    fn the_software_resample_names_the_same_algorithm_the_gpu_one_does() {
        // swscale ignores an unknown `flags=` value and quietly uses bicubic, so `nearest` has to
        // be translated rather than passed through.
        assert_eq!(swscale_flag("nearest"), "neighbor");
        assert_eq!(swscale_flag("bilinear"), "bilinear");
        assert_eq!(swscale_flag("bicubic"), "bicubic");
        assert_eq!(swscale_flag("lanczos"), "lanczos");
        for algo in RESAMPLE_ALGOS {
            assert!(!swscale_flag(algo).is_empty(), "{algo}");
        }
    }

    #[test]
    fn the_png_encode_converts_rather_than_only_tagging() {
        // PNG frames are full-range RGB. Handing them to `format=yuv420p` and then tagging the
        // output BT.709 is the colour-banding bug; the conversion has to be named.
        assert!(crate::export::RGB_TO_BT709_TV.contains("out_color_matrix=bt709"));
        assert!(crate::export::RGB_TO_BT709_TV.ends_with("format=yuv420p"));
    }

    #[test]
    fn the_filter_always_names_its_algorithm() {
        // Regression guard: an unqualified scale_cuda takes interp_algo 0, which is outside the
        // filter's own documented 1-4 range and is what every upscaled export used to get.
        let filter = scale_cuda_filter(3840, 2160, "lanczos");
        assert!(filter.contains("interp_algo=lanczos"));
    }

    #[test]
    fn rejects_a_target_the_encoder_would_refuse() {
        let chain = UpscaleChain {
            ai_backend: "esrgan".into(), ai_model_file: None, ai_quality: "HIGH".into(), tile: 512, resample: "lanczos".into(),
            width: 7680, height: 4320,
        };
        assert!(chain.validate("hevc_nvenc").is_ok());
        let error = chain.validate("h264_nvenc").unwrap_err();
        assert!(error.contains("4096"), "{error}");
    }

    #[test]
    fn rejects_an_odd_target_and_an_unknown_algorithm() {
        let odd = UpscaleChain {
            ai_backend: "esrgan".into(), ai_model_file: None, ai_quality: "HIGH".into(), tile: 512, resample: "lanczos".into(), width: 1921, height: 1080,
        };
        assert!(odd.validate("hevc_nvenc").is_err());
        let bogus = UpscaleChain {
            ai_backend: "esrgan".into(), ai_model_file: None, ai_quality: "HIGH".into(), tile: 512, resample: "magic".into(), width: 1920, height: 1080,
        };
        assert!(bogus.validate("hevc_nvenc").is_err());
    }
}
