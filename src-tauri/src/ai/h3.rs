//! MiniMax-H3.
//!
//! A second generator alongside LTX, not a replacement. It lives in its own virtualenv
//! (`vendor/MiniMax-H3/.venv`) with its own weights (`models/MiniMax-H3`), so nothing here can
//! affect the LTX path — deliberately, because the LTX environment took three sessions to stabilise.
//! The adapter writes video and audio as SEPARATE files: an mp4 with no audio stream and a wav
//! beside it. The panel places them on V1 and A1 as a linked pair. Muxing them and also placing the
//! wav would double the audio in every export.
//! ---------------------------------------------------------------------------------------------

use super::*;

/// H3's geometry contract, mirrored from `ai-runtime/aiv_runtime/h3_geometry.py`. Duplicated here
/// so a bad request is refused in microseconds rather than after the worker has loaded 20 GiB.
const H3_CANVAS_MULTIPLE: u32 = 32;
const H3_MAX_PIXELS: u64 = 768 * 1344;
const H3_MIN_FRAMES: u32 = 124;
const H3_MAX_FRAMES: u32 = 345;
const H3_FRAMES_PER_CHUNK: u32 = 17;
const H3_LATENTS_PER_CHUNK: u32 = 5;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MiniMaxH3Request {
    /// Native modes: `text`, `image`, `first-last`, `last-only`.
    /// Editor workflows: `clip-extend`, `clip-to-clip`.
    ///
    /// Required, deliberately. It used to default to `text`, so a caller that omitted or
    /// misspelled it on an image request got a text generation that ignored the keyframes --
    /// silently, and at full cost. Width, height, frames, steps, seed and offload were always
    /// required; these two were the exceptions, and they were the two that change which model
    /// runs.
    pub mode: String,
    #[serde(default)]
    pub first_frame: Option<String>,
    #[serde(default)]
    pub last_frame: Option<String>,
    /// Selected outgoing clip. The wrapper extracts its exact last visible frame losslessly.
    #[serde(default)]
    pub video_path: Option<String>,
    #[serde(default)]
    pub source_in_seconds: Option<f64>,
    #[serde(default)]
    pub source_out_seconds: Option<f64>,
    /// Following clip for `clip-to-clip`; its exact first visible frame becomes the last condition.
    #[serde(default)]
    pub to_video_path: Option<String>,
    #[serde(default)]
    pub to_source_in_seconds: Option<f64>,
    pub prompt: String,
    /// Video, written without an audio stream.
    pub output_path: String,
    /// Audio. Defaults to the mp4 stem with a `.wav` extension when the frontend omits it.
    #[serde(default)]
    pub audio_path: Option<String>,
    pub width: u32,
    pub height: u32,
    pub num_frames: u32,
    pub steps: u32,
    pub seed: i64,
    /// `base`, `turbo-4` Draft, or `turbo-8`; each Turbo mode has its own adapter.
    ///
    /// Required for the same reason as `mode`: defaulting to `turbo-8` meant an omitted field
    /// quietly selected a different adapter, a different step count and a different pair of
    /// scheduler shifts. The TS side already sends it explicitly, as it does for the Ref path.
    pub sampling_mode: String,
    /// `native` or `sage`; the Python worker falls back to native when Sage is unavailable.
    #[serde(default = "default_h3_attention_backend")]
    pub attention_backend: String,
    /// `performance`, `balanced`, or `minimal`.
    pub offload: String,
}

/// Sage is the default because native SDPA was never a *choice* -- it was the value nobody
/// changed. sageattention 2.2.0 and triton are installed in the H3 venv and went unused on every
/// run. The dispatch guard checks each call against Sage's constraints and falls back to native
/// SDPA per call when they are not met, so the worst case here is the old behaviour plus a line
/// in the log saying why.
fn default_h3_attention_backend() -> String {
    "sage".to_owned()
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MiniMaxH3Result {
    pub output_path: String,
    pub audio_path: String,
    pub log: String,
}

fn h3_root(root: &Path) -> PathBuf {
    root.join("models/MiniMax-H3")
}

fn h3_python(root: &Path) -> PathBuf {
    root.join("vendor/MiniMax-H3/.venv/Scripts/python.exe")
}

/// The published Comfy-Org INT8 ConvRot denoiser. `t2va`, `i2va` and `fl2va` share this partition;
/// Ref2VA has its own checkpoint and worker.
fn h3_transformer(root: &Path) -> PathBuf {
    h3_root(root).join("diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors")
}

fn h3_turbo_lora(root: &Path, sampling_mode: &str) -> Option<PathBuf> {
    let name = match sampling_mode {
        "turbo-4" => "minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
        "turbo-8" => "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
        _ => return None,
    };
    Some(h3_root(root).join("loras").join(name))
}

pub fn turbo_ready(root: &Path, sampling_mode: &str) -> bool {
    h3_turbo_lora(root, sampling_mode).is_some_and(|path| path.is_file())
}

/// Comfy-Org's published INT8 ConvRot Qwen3-VL conditioner.  Production deliberately does not
/// fall back to local `.pt`/`.bin` conversions.
fn h3_text_encoder(root: &Path) -> Option<PathBuf> {
    let checkpoint = h3_root(root)
        .join("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors");
    checkpoint.is_file().then_some(checkpoint)
}

/// Why H3 is unavailable, or `ready`. One string beats five booleans the panel has to reassemble
/// into a sentence, and follows the `kiwi_edit_state` precedent.
pub fn state_label(root: &Path) -> &'static str {
    if !h3_python(root).is_file() {
        "venv-missing"
    } else if !root.join("ai-runtime/aiv_runtime/minimax_h3.py").is_file() {
        "adapter-missing"
    } else if !h3_root(root).join("modular_model_index.json").is_file() {
        "pack-missing"
    } else if !h3_transformer(root).is_file() {
        "checkpoint-missing"
    } else if h3_text_encoder(root).is_none() {
        "encoder-missing"
    } else {
        "ready"
    }
}

fn h3_aligned(num_frames: u32) -> bool {
    num_frames % H3_FRAMES_PER_CHUNK == H3_LATENTS_PER_CHUNK
}

pub async fn generate(request: MiniMaxH3Request) -> Result<MiniMaxH3Result, String> {
    match request.mode.as_str() {
        "clip-extend" => generate_clip_extension(request).await,
        "clip-to-clip" => generate_clip_bridge(request).await,
        _ => generate_base(request).await,
    }
}

async fn generate_base(request: MiniMaxH3Request) -> Result<MiniMaxH3Result, String> {
    let root = project_root();
    match state_label(&root) {
        "ready" => {}
        "venv-missing" => {
            return Err("the MiniMax-H3 environment is missing. Build it with:\n  \
             powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\setup-h3.ps1"
                .into())
        }
        "adapter-missing" => return Err("the MiniMax-H3 adapter is missing".into()),
        "encoder-missing" => {
            return Err(
                "the published MiniMax-H3 INT8 ConvRot conditioner safetensor is missing. Reinstall the H3 pack"
                    .into(),
            )
        }
        "pack-missing" => {
            return Err(
                "the MiniMax-H3 weights are not installed. Install them with:\n  \
             .venv\\Scripts\\python.exe scripts\\models.py install minimax-h3 --accept-licenses"
                    .into(),
            )
        }
        _ => {
            return Err(
                "the published MiniMax-H3 FL2VA INT8 ConvRot denoiser is missing. \
             Re-run the MiniMax-H3 model installer; the BF16 partition does not fit this 16 GiB profile."
                    .into(),
            )
        }
    }

    let required: &[(&str, &Option<String>)] = match request.mode.as_str() {
        "text" => &[],
        "image" => &[("first frame", &request.first_frame)],
        "first-last" => &[
            ("first frame", &request.first_frame),
            ("last frame", &request.last_frame),
        ],
        "last-only" => &[("last frame", &request.last_frame)],
        other => return Err(format!("unknown MiniMax-H3 mode: {other}")),
    };
    for (label, value) in required {
        match value.as_deref().filter(|path| !path.is_empty()) {
            Some(path) if Path::new(path).is_file() => {}
            Some(path) => return Err(format!("{label} not found: {path}")),
            None => return Err(format!("{} mode requires a {label}", request.mode)),
        }
    }
    if request.prompt.trim().is_empty() {
        return Err("prompt cannot be empty".into());
    }
    for (label, value) in [("width", request.width), ("height", request.height)] {
        if value == 0 || value % H3_CANVAS_MULTIPLE != 0 {
            return Err(format!(
                "{label} must be a positive multiple of {H3_CANVAS_MULTIPLE}"
            ));
        }
    }
    if request.width as u64 * request.height as u64 > H3_MAX_PIXELS {
        return Err(format!(
            "MiniMax-H3 is limited to {H3_MAX_PIXELS} pixels (1344x768)"
        ));
    }
    if !h3_aligned(request.num_frames) {
        return Err("frame count must be 17n+5".into());
    }
    if !(H3_MIN_FRAMES..=H3_MAX_FRAMES).contains(&request.num_frames) {
        return Err(format!(
            "MiniMax-H3 generates 5 to 15 seconds, so the frame count must be between \
             {H3_MIN_FRAMES} and {H3_MAX_FRAMES}"
        ));
    }
    if request.steps == 0 || request.steps > 60 {
        return Err("steps must be between 1 and 60".into());
    }
    if !["base", "turbo-4", "turbo-8"].contains(&request.sampling_mode.as_str()) {
        return Err("samplingMode must be base, turbo-4 or turbo-8".into());
    }
    if request.sampling_mode != "base" {
        let expected = if request.sampling_mode == "turbo-4" { 4 } else { 8 };
        if request.steps != expected {
            return Err(format!(
                "LightX2V {}-step Turbo is fixed at exactly {expected} transformer evaluations",
                expected
            ));
        }
        let lora = h3_turbo_lora(&root, &request.sampling_mode).expect("validated Turbo mode");
        if !lora.is_file() {
            return Err(
                format!(
                    "the LightX2V H3 {expected}-step Turbo LoRA is missing. Re-run the MiniMax-H3 \
                     model installer; expected {}",
                    lora.display()
                ),
            );
        }
    }
    if !["native", "sage", "sage2_patched", "sage1", "sage2_stock"].contains(&request.attention_backend.as_str()) {
        return Err("attentionBackend must be native, sage, sage1, or sage2_stock".into());
    }
    // Named profiles rather than LTX's cpu/disk: H3's levers are blocks-per-group, streaming and
    // whether the 5.2 GiB video VAE stays resident, which do not reduce to a single axis.
    if !["performance", "balanced", "minimal"].contains(&request.offload.as_str()) {
        return Err("offload must be performance, balanced or minimal".into());
    }
    let output_path = PathBuf::from(&request.output_path);
    if output_path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
        != Some("mp4")
    {
        return Err("MiniMax-H3 video output must be an .mp4 file".into());
    }
    let audio_path = request
        .audio_path
        .clone()
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| {
            output_path
                .with_extension("wav")
                .to_string_lossy()
                .into_owned()
        });

    let python = h3_python(&root);
    let adapter = root.join("ai-runtime/aiv_runtime/minimax_h3.py");
    let pack = h3_root(&root);
    let mut command = Command::new(python);
    command
        .arg(adapter)
        .args(["--repo-root", pack.to_string_lossy().as_ref()])
        .args([
            "--transformer-path",
            h3_transformer(&root).to_string_lossy().as_ref(),
        ])
        .args(["--mode", &request.mode])
        .args(["--prompt", &request.prompt])
        .args(["--output-path", &request.output_path])
        .args(["--audio-path", &audio_path])
        .args(["--width", &request.width.to_string()])
        .args(["--height", &request.height.to_string()])
        .args(["--num-frames", &request.num_frames.to_string()])
        .args(["--steps", &request.steps.to_string()])
        .args(["--seed", &request.seed.to_string()])
        .args(["--sampling-mode", &request.sampling_mode])
        .args(["--attention-backend", &request.attention_backend])
        .args(["--offload", &request.offload]);
    if request.sampling_mode != "base" {
        let lora = h3_turbo_lora(&root, &request.sampling_mode).expect("validated Turbo mode");
        command.args([
            "--turbo-lora-path",
            lora.to_string_lossy().as_ref(),
        ]);
    }
    if let Some(encoder) = h3_text_encoder(&root) {
        command.args(["--text-encoder-path", encoder.to_string_lossy().as_ref()]);
    }
    for (flag, value) in [
        ("--first-frame", &request.first_frame),
        ("--last-frame", &request.last_frame),
    ] {
        if let Some(path) = value.as_deref().filter(|item| !item.is_empty()) {
            command.args([flag, path]);
        }
    }
    let output = command
        .current_dir(&root)
        .env("PYTHONUNBUFFERED", "1")
        // Same reason as the LTX workers: Windows defaults these to cp1252 and a warning with an
        // emoji in it then raises UnicodeEncodeError from inside a logging call.
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("failed to start the MiniMax-H3 worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        return Err(concise_worker_error_with_status(
            &stdout,
            &stderr,
            output.status.code(),
        ));
    }
    if !output_path.is_file() {
        return Err("the MiniMax-H3 worker finished without creating its video".into());
    }
    if !Path::new(&audio_path).is_file() {
        return Err("the MiniMax-H3 worker finished without creating its soundtrack".into());
    }
    // Both streams, tqdm stripped, and a copy on disk: the H3 workers print their schedule
    // asserts and VRAM figures to stderr the same way the LTX ones do.
    let log = worker_diagnostics(&stdout, &stderr);
    record_worker_log(&project_root(), "minimax-h3", &log);
    Ok(MiniMaxH3Result {
        output_path: request.output_path,
        audio_path,
        log,
    })
}

fn resolved_audio_path(request: &MiniMaxH3Request) -> String {
    request.audio_path.clone().filter(|value| !value.is_empty()).unwrap_or_else(|| {
        PathBuf::from(&request.output_path).with_extension("wav").to_string_lossy().into_owned()
    })
}

fn required_clip_path(value: &Option<String>, label: &str) -> Result<String, String> {
    match value.as_deref().filter(|item| !item.is_empty()) {
        Some(path) if Path::new(path).is_file() => Ok(path.to_owned()),
        Some(path) => Err(format!("{label} not found: {path}")),
        None => Err(format!("{label} is required")),
    }
}

/// H3 clip extension is native first-frame conditioning fed by the selected clip's exact final
/// visible frame. Generate into scratch files so the duplicated guide frame and the same 1/24 s of
/// independently decoded audio can be removed before anything reaches the timeline.
async fn generate_clip_extension(mut request: MiniMaxH3Request) -> Result<MiniMaxH3Result, String> {
    let source = required_clip_path(&request.video_path, "selected clip source")?;
    let source_in = request.source_in_seconds.unwrap_or(-1.0);
    let source_out = request.source_out_seconds.unwrap_or(-1.0);
    if source_in < 0.0 || source_out <= source_in {
        return Err("clip extension requires a valid visible source range".into());
    }
    let final_video = PathBuf::from(&request.output_path);
    if final_video.extension().and_then(|item| item.to_str()).map(str::to_ascii_lowercase).as_deref() != Some("mp4") {
        return Err("MiniMax-H3 clip extension output must be an .mp4 file".into());
    }
    let final_audio_string = resolved_audio_path(&request);
    let final_audio = PathBuf::from(&final_audio_string);
    let scratch = std::env::temp_dir().join(format!("aiv-h3-clip-extend-{}", Uuid::new_v4()));
    fs::create_dir_all(&scratch).map_err(|error| error.to_string())?;
    let guide = scratch.join("last-visible-frame.png");
    let raw_video = scratch.join("raw-extension.mp4");
    let raw_audio = scratch.join("raw-extension.wav");
    let result = async {
        crate::media::extract_last_visible_frame(&source, source_in, source_out, &guide).await?;
        request.mode = "image".into();
        request.first_frame = Some(guide.to_string_lossy().into_owned());
        request.last_frame = None;
        request.output_path = raw_video.to_string_lossy().into_owned();
        request.audio_path = Some(raw_audio.to_string_lossy().into_owned());
        let generated = generate_base(request).await?;
        let frame = 1.0 / 24.0;
        crate::media::trim_extension_boundary(&raw_video, &final_video, frame).await?;
        if let Err(error) = crate::media::trim_audio_boundaries(&raw_audio, &final_audio, frame, 0.0).await {
            let _ = fs::remove_file(&final_video);
            return Err(error);
        }
        Ok(MiniMaxH3Result {
            output_path: final_video.to_string_lossy().into_owned(),
            audio_path: final_audio_string,
            log: generated.log,
        })
    }.await;
    let _ = fs::remove_dir_all(&scratch);
    result
}

/// H3 clip-to-clip uses its native first+last-frame task with frames extracted from the two adjacent
/// trimmed clips. The duplicated boundary frames and matching standalone audio durations are
/// removed before the generated A/V pair is ripple-inserted by the editor.
async fn generate_clip_bridge(mut request: MiniMaxH3Request) -> Result<MiniMaxH3Result, String> {
    let outgoing = required_clip_path(&request.video_path, "outgoing clip source")?;
    let incoming = required_clip_path(&request.to_video_path, "incoming clip source")?;
    let source_in = request.source_in_seconds.unwrap_or(-1.0);
    let source_out = request.source_out_seconds.unwrap_or(-1.0);
    let to_source_in = request.to_source_in_seconds.unwrap_or(-1.0);
    if source_in < 0.0 || source_out <= source_in {
        return Err("the outgoing clip has no valid visible source range".into());
    }
    if to_source_in < 0.0 {
        return Err("the incoming clip has an invalid source in point".into());
    }
    let final_video = PathBuf::from(&request.output_path);
    if final_video.extension().and_then(|item| item.to_str()).map(str::to_ascii_lowercase).as_deref() != Some("mp4") {
        return Err("MiniMax-H3 clip bridge output must be an .mp4 file".into());
    }
    let final_audio_string = resolved_audio_path(&request);
    let final_audio = PathBuf::from(&final_audio_string);
    let scratch = std::env::temp_dir().join(format!("aiv-h3-clip-bridge-{}", Uuid::new_v4()));
    fs::create_dir_all(&scratch).map_err(|error| error.to_string())?;
    let first = scratch.join("outgoing-last-frame.png");
    let last = scratch.join("incoming-first-frame.png");
    let raw_video = scratch.join("raw-bridge.mp4");
    let raw_audio = scratch.join("raw-bridge.wav");
    let result = async {
        crate::media::extract_last_visible_frame(&outgoing, source_in, source_out, &first).await?;
        crate::media::extract_first_visible_frame(&incoming, to_source_in, &last).await?;
        request.mode = "first-last".into();
        request.first_frame = Some(first.to_string_lossy().into_owned());
        request.last_frame = Some(last.to_string_lossy().into_owned());
        request.output_path = raw_video.to_string_lossy().into_owned();
        request.audio_path = Some(raw_audio.to_string_lossy().into_owned());
        let generated = generate_base(request).await?;
        let frame = 1.0 / 24.0;
        crate::media::trim_bridge_boundaries(&raw_video, &final_video, frame).await?;
        if let Err(error) = crate::media::trim_audio_boundaries(&raw_audio, &final_audio, frame, frame).await {
            let _ = fs::remove_file(&final_video);
            return Err(error);
        }
        Ok(MiniMaxH3Result {
            output_path: final_video.to_string_lossy().into_owned(),
            audio_path: final_audio_string,
            log: generated.log,
        })
    }.await;
    let _ = fs::remove_dir_all(&scratch);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_alignment_matches_the_python_contract() {
        // 17n + 5. These are the counts the panel's slider can produce.
        assert!(h3_aligned(124));
        assert!(h3_aligned(141));
        assert!(h3_aligned(345));
        // LTX's 8n+1 counts are not H3 counts; 121 is the LTX default and must be refused.
        assert!(!h3_aligned(121));
        assert!(!h3_aligned(120));
        assert!(!h3_aligned(240));
    }

    #[test]
    fn the_duration_window_excludes_the_next_aligned_count() {
        // 362 is aligned but 15.083 s, past the checkpoint's 15 s ceiling.
        assert!(h3_aligned(362));
        assert!(!(H3_MIN_FRAMES..=H3_MAX_FRAMES).contains(&362));
    }

    fn base_request_json() -> serde_json::Value {
        serde_json::json!({
            "mode": "image",
            "prompt": "a lighthouse",
            "outputPath": "out.mp4",
            "width": 960,
            "height": 544,
            "numFrames": 124,
            "steps": 8,
            "seed": 42,
            "samplingMode": "turbo-8",
            "offload": "balanced",
        })
    }

    #[test]
    fn a_complete_request_still_deserializes() {
        let request: MiniMaxH3Request = serde_json::from_value(base_request_json()).unwrap();
        assert_eq!(request.mode, "image");
        assert_eq!(request.sampling_mode, "turbo-8");
        // The genuinely optional fields keep their defaults.
        assert!(request.first_frame.is_none());
        assert_eq!(request.attention_backend, "sage");
    }

    #[test]
    fn omitting_the_mode_is_an_error_not_a_text_generation() {
        // It used to default to `text`, so a misspelled or dropped `mode` on an image request
        // silently ignored the keyframes and generated from the prompt alone.
        let mut json = base_request_json();
        json.as_object_mut().unwrap().remove("mode");
        assert!(serde_json::from_value::<MiniMaxH3Request>(json).is_err());
    }

    #[test]
    fn omitting_the_sampling_mode_is_an_error_not_a_different_adapter() {
        let mut json = base_request_json();
        json.as_object_mut().unwrap().remove("samplingMode");
        assert!(serde_json::from_value::<MiniMaxH3Request>(json).is_err());
    }

    #[test]
    fn the_low_vram_canvases_fit_the_budget() {
        for (width, height) in [(960u64, 544u64), (544, 960), (544, 544), (1280, 544)] {
            assert_eq!(width % H3_CANVAS_MULTIPLE as u64, 0);
            assert_eq!(height % H3_CANVAS_MULTIPLE as u64, 0);
            assert!(width * height <= H3_MAX_PIXELS);
        }
    }
}
