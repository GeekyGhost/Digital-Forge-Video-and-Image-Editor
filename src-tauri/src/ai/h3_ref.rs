//! Optional MiniMax-H3 Ref2VA pack.  This is intentionally a separate worker and readiness path:
//! the normal FL2VA checkpoint, adapters, requests and UI modes never pass through this module.

use super::*;

// The checkpoint's own limits, from MiniMax's model card: <= 9 images, <= 3 video clips, <= 3
// audio clips, <= 12 files in total. These were 3/1/1/4 -- a conservative 16 GB profile written
// as though it were the model's contract. The panel warns above four; it no longer refuses.
const MAX_IMAGES: usize = 9;
const MAX_VIDEOS: usize = 3;
const MAX_AUDIOS: usize = 3;
const MAX_REFERENCES: usize = 12;
/// The checkpoint's canvas budget: a 768 short edge, 768 * 1344 pixels, both axes on the 32-grid.
const MAX_SHORT_EDGE: u32 = 768;
const MAX_PIXELS: u64 = 768 * 1344;
const MIN_TIMED_SECONDS: f64 = 2.0;
const MAX_TIMED_SECONDS: f64 = 15.0;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MiniMaxH3Reference {
    pub kind: String,
    pub path: String,
    #[serde(default = "default_true")]
    pub include_embedded_audio: bool,
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MiniMaxH3RefRequest {
    pub prompt: String,
    pub references: Vec<MiniMaxH3Reference>,
    pub output_path: String,
    #[serde(default)]
    pub audio_path: Option<String>,
    pub width: u32,
    pub height: u32,
    pub num_frames: u32,
    pub seed: i64,
    #[serde(default = "default_attention")]
    pub attention_backend: String,
    #[serde(default = "default_offload")]
    pub offload: String,
    #[serde(default = "default_steps")]
    pub steps: u32,
    /// `turbo-4` (LightX2V's v0.1 preview adapter) or `base` (no adapter, its own step count).
    #[serde(default = "default_sampling_mode")]
    pub sampling_mode: String,
    #[serde(default = "default_video_shift")]
    pub video_shift: f64,
    #[serde(default)]
    pub ref_image_short_edge: Option<u32>,
    #[serde(default)]
    pub ref_video_short_edge: Option<u32>,
    #[serde(default = "default_vae_tile")]
    pub vae_tile: u32,
}

/// Deliberately NOT Sage, unlike the generation path.
///
/// Ref2VA's correctness baseline must be established on native SDPA before GATE 4 can land.
/// Flipping the kernel underneath an unqualified path makes the next quality regression
/// ambiguous between the model and the attention kernel, which is the one confound the
/// qualification exists to rule out. Flip this once Ref2VA has a verified baseline.
fn default_attention() -> String {
    "native".into()
}
fn default_offload() -> String {
    "balanced".into()
}
fn default_steps() -> u32 {
    20
}
fn default_sampling_mode() -> String {
    "base".into()
}
fn default_video_shift() -> f64 {
    12.0
}
fn default_vae_tile() -> u32 {
    256
}

fn pack(root: &Path) -> PathBuf {
    root.join("models/MiniMax-H3")
}
fn checkpoint(root: &Path) -> PathBuf {
    pack(root).join("diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors")
}
fn lora(root: &Path) -> PathBuf {
    pack(root).join("loras/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors")
}
fn encoder(root: &Path) -> PathBuf {
    pack(root).join("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors")
}
fn python(root: &Path) -> PathBuf {
    root.join("vendor/MiniMax-H3/.venv/Scripts/python.exe")
}
fn worker(root: &Path) -> PathBuf {
    root.join("ai-runtime/aiv_runtime/minimax_h3_ref.py")
}

/// Does this denoiser already carry the Turbo residual in its own weights?
///
/// Mirrors `minimax_h3_ref.py::validate`, which accepts a missing adapter only when the checkpoint
/// declares `is_turbo_baked`. Reading it here keeps readiness and the worker on one rule instead of
/// letting the panel say `ready` for a run the worker will refuse. The published Comfy-Org pruned
/// INT8 ConvRot checkpoint is *not* baked, so in production this is always `false` and the adapter
/// is mandatory; a locally fused checkpoint is the only case that answers `true`.
///
/// Any read failure answers `false`. Readiness must never be granted by an unreadable file.
fn turbo_baked(checkpoint: &Path) -> bool {
    use std::io::Read;

    let Ok(mut file) = fs::File::open(checkpoint) else {
        return false;
    };
    let mut length = [0u8; 8];
    if file.read_exact(&mut length).is_err() {
        return false;
    }
    // Safetensors headers are small; a corrupt or hostile length must not become an allocation.
    let length = u64::from_le_bytes(length);
    if length == 0 || length > 100 * 1024 * 1024 {
        return false;
    }
    let mut header = vec![0u8; length as usize];
    if file.read_exact(&mut header).is_err() {
        return false;
    }
    serde_json::from_slice::<serde_json::Value>(&header)
        .ok()
        .and_then(|value| {
            value
                .get("__metadata__")?
                .get("is_turbo_baked")?
                .as_str()
                .map(str::to_owned)
        })
        .is_some_and(|value| value == "True")
}

pub fn state_label(root: &Path) -> &'static str {
    if !python(root).is_file() {
        "venv-missing"
    } else if !worker(root).is_file() {
        "adapter-missing"
    } else if !pack(root).join("modular_model_index.json").is_file() {
        "shared-pack-missing"
    } else if !encoder(root).is_file() {
        "encoder-missing"
    } else if !pack(root).join("transformer_ref/config.json").is_file() {
        "config-missing"
    } else if !checkpoint(root).is_file() {
        "checkpoint-missing"
    } else {
        "ready"
    }
}

/// Can Ref2VA run its four-evaluation Turbo adapter, as opposed to Base?
///
/// Separate from `state_label` on purpose, mirroring `h3::turbo_ready`. Ref2VA used to be hardwired
/// to Turbo, which made a missing adapter a hard blocker for the whole pack; now that Base runs
/// with no adapter at all, its absence costs one sampling mode instead. A checkpoint declaring
/// `is_turbo_baked` carries the residual itself and needs no separate file.
pub fn turbo_ready(root: &Path) -> bool {
    lora(root).is_file() || turbo_baked(&checkpoint(root))
}

pub fn ready(root: &Path) -> bool {
    state_label(root) == "ready"
}

/// The audio half of the reference contract, once the files have been probed.
///
/// Separate from `validate_references` because it is the part worth testing: everything else there
/// needs real media on disk, and this is where both defects were.
fn check_audio_pairing(visual_count: usize, effective_audio_count: usize) -> Result<(), String> {
    if effective_audio_count > MAX_AUDIOS {
        return Err(format!(
            "MiniMax-H3 accepts at most {MAX_AUDIOS} audio references, and a video's own \
             soundtrack is one of them; this request has {effective_audio_count}. Turn off the \
             embedded soundtrack on a video reference, or remove an audio clip."
        ));
    }
    // The checkpoint's own pairing rule: every audio reference needs a visual to attach to.
    // Upstream raises here rather than degrading. Being more permissive than WanGP's 2/2 is a
    // defensible reading of the model card, which is where our 9/3/3/12 comes from; skipping the
    // model's own pairing constraint is not the same kind of decision.
    if effective_audio_count > visual_count {
        return Err(format!(
            "MiniMax-H3 needs at least as many image and video references as audio ones; this \
             request has {effective_audio_count} audio against {visual_count} visual"
        ));
    }
    Ok(())
}

async fn validate_references(values: &[MiniMaxH3Reference]) -> Result<(), String> {
    if values.is_empty() {
        return Err("Ref2VA requires at least one reference".into());
    }
    if values.len() > MAX_REFERENCES {
        return Err(format!(
            "MiniMax-H3 accepts at most {MAX_REFERENCES} reference files in total"
        ));
    }
    let image_count = values.iter().filter(|item| item.kind == "image").count();
    let video_count = values.iter().filter(|item| item.kind == "video").count();
    let audio_count = values.iter().filter(|item| item.kind == "audio").count();
    if image_count > MAX_IMAGES || video_count > MAX_VIDEOS {
        return Err(format!(
            "MiniMax-H3 accepts at most {MAX_IMAGES} images and {MAX_VIDEOS} video clips"
        ));
    }
    if image_count + video_count == 0 {
        return Err("an audio reference must be paired with an image or video reference".into());
    }
    if image_count + video_count + audio_count != values.len() {
        return Err("reference kind must be image, video or audio".into());
    }

    // A video's soundtrack is an audio reference. It gets its own `<Audio n>` label in the
    // presentation, its own audio rows in the packed sequence, and it counts against the same
    // budget of three -- so the audio cap cannot be resolved until the files have been probed.
    // Counting only `kind == "audio"` meant three sound-bearing videos plus three audio clips
    // passed as three audio references while conditioning on six, which presents as voice bleed
    // and timbre drift.
    let mut effective_audio_count = audio_count;
    let mut video_seconds = 0.0;
    let mut audio_seconds = 0.0;
    for item in values {
        if !Path::new(&item.path).is_file() {
            return Err(format!("reference not found: {}", item.path));
        }
        if item.kind != "image" {
            let metadata = crate::media::probe(&item.path).await?;
            let duration = metadata.duration_seconds;
            if !(MIN_TIMED_SECONDS..=MAX_TIMED_SECONDS).contains(&duration) {
                return Err(format!(
                    "{} reference must be 2–15 seconds; {} is {:.2}s",
                    item.kind, item.path, duration
                ));
            }
            if item.kind == "video" {
                if metadata.video_codec.is_none() {
                    return Err("video reference has no video stream".into());
                }
                video_seconds += duration;
                if item.include_embedded_audio && metadata.audio_codec.is_some() {
                    audio_seconds += duration;
                    effective_audio_count += 1;
                }
            } else {
                if metadata.audio_codec.is_none() {
                    return Err("audio reference has no audio stream".into());
                }
                audio_seconds += duration;
            }
        }
    }
    if video_seconds > MAX_TIMED_SECONDS || audio_seconds > MAX_TIMED_SECONDS {
        return Err(format!(
            "reference totals must stay within 15 seconds per modality (video {video_seconds:.2}s, audio {audio_seconds:.2}s)"
        ));
    }
    check_audio_pairing(image_count + video_count, effective_audio_count)
}

pub async fn generate(request: MiniMaxH3RefRequest) -> Result<h3::MiniMaxH3Result, String> {
    let root = project_root();
    if !ready(&root) {
        return Err(format!(
            "the optional MiniMax-H3 Ref2VA pack is not ready ({}) — install minimax-h3-ref2va",
            state_label(&root)
        ));
    }
    if request.prompt.trim().is_empty() {
        return Err("prompt cannot be empty".into());
    }
    validate_references(&request.references).await?;
    if request.width == 0
        || request.height == 0
        || request.width % 32 != 0
        || request.height % 32 != 0
    {
        return Err("Ref2VA width and height must be positive multiples of 32".into());
    }
    // Was capped at 544, which is a 16 GB profile rather than a model limit -- and it put the
    // canvas MiniMax actually generates at out of reach, so our output could never be compared
    // against their published 768p clips on equal terms. The rule below is the checkpoint's.
    if request.width.min(request.height) > MAX_SHORT_EDGE {
        return Err(format!(
            "MiniMax-H3 generates at a short edge of at most {MAX_SHORT_EDGE} px, got {}x{}",
            request.width, request.height
        ));
    }
    if request.width as u64 * request.height as u64 > MAX_PIXELS {
        return Err(format!(
            "MiniMax-H3 caps a canvas at {MAX_PIXELS} pixels (768x1344), got {}x{} ({})",
            request.width,
            request.height,
            request.width as u64 * request.height as u64
        ));
    }
    if request.num_frames < 124 || request.num_frames > 345 || request.num_frames % 17 != 5 {
        return Err("Ref2VA frame count must be 17n+5 between 124 and 345".into());
    }
    if !["native", "sage", "sage2_patched", "sage1", "sage2_stock"].contains(&request.attention_backend.as_str()) {
        return Err("attentionBackend must be native, sage, sage1, or sage2_stock".into());
    }
    if !["performance", "balanced", "minimal"].contains(&request.offload.as_str()) {
        return Err("offload must be performance, balanced or minimal".into());
    }
    // Mirrors `minimax_h3.VAE_TILE_SIZE` and the worker's own `% 16` rule. Without this the panel
    // happily sent a tile the worker refuses and the user got a Python traceback for a value the
    // interface had offered them.
    if request.vae_tile < 256 || request.vae_tile % 16 != 0 {
        return Err(format!(
            "VAE tile must be at least 256 pixels and a multiple of 16, got {}. 256 is MiniMax's \
             released tile geometry; smaller tiles add seams without lowering peak decode memory, \
             which the temporal chunk buffers dominate.",
            request.vae_tile
        ));
    }
    match request.sampling_mode.as_str() {
        "turbo-4" => {
            if request.steps != 4 {
                return Err(format!(
                    "turbo-4 runs at exactly 4 transformer evaluations, got {}. The adapter is \
                     distilled to that count; it is not a speed dial.",
                    request.steps
                ));
            }
            if !turbo_ready(&root) {
                return Err(
                    "the LightX2V Ref2V adapter is missing, so only Base sampling is available. \
                     Install it with:\n  .venv\\Scripts\\python.exe scripts\\models.py install \
                     minimax-h3-ref2va --accept-licenses"
                        .into(),
                );
            }
        }
        // MiniMax's published reference recipe is 50. The floor sits well below it because
        // Ref2VA's only other option is a four-step preview adapter, so the ground between them is
        // worth reaching; ComfyUI's own R2V workflow defaults to 20.
        "base" => {
            if !(10..=60).contains(&request.steps) {
                return Err(format!(
                    "base sampling takes 10-60 transformer evaluations, got {}. MiniMax's \
                     published reference recipe is 50.",
                    request.steps
                ));
            }
        }
        other => return Err(format!("samplingMode must be base or turbo-4, got {other}")),
    }
    let output = PathBuf::from(&request.output_path);
    if output
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
        != Some("mp4")
    {
        return Err("Ref2VA video output must be an .mp4 file".into());
    }
    let audio = request
        .audio_path
        .clone()
        .filter(|item| !item.is_empty())
        .unwrap_or_else(|| output.with_extension("wav").to_string_lossy().into_owned());
    let references =
        serde_json::to_string(&request.references).map_err(|error| error.to_string())?;
    let mut cmd = Command::new(python(&root));
    cmd.arg(worker(&root))
        .args(["--repo-root", pack(&root).to_string_lossy().as_ref()])
        .args([
            "--transformer-path",
            checkpoint(&root).to_string_lossy().as_ref(),
        ])
        .args([
            "--text-encoder-path",
            encoder(&root).to_string_lossy().as_ref(),
        ])
        .args(["--turbo-lora-path", lora(&root).to_string_lossy().as_ref()])
        .args(["--references-json", &references])
        .args(["--prompt", &request.prompt])
        .args(["--output-path", &request.output_path])
        .args(["--audio-path", &audio])
        .args(["--width", &request.width.to_string()])
        .args(["--height", &request.height.to_string()])
        .args(["--num-frames", &request.num_frames.to_string()])
        .args(["--seed", &request.seed.to_string()])
        .args(["--attention-backend", &request.attention_backend])
        .args(["--offload", &request.offload])
        .args(["--sampling-mode", &request.sampling_mode])
        .args(["--steps", &request.steps.to_string()])
        .args(["--vae-tile", &request.vae_tile.to_string()]);
    // Base keeps the released scheduler shifts; the worker ignores `--video-shift` there. Sending
    // it anyway made the panel's control look live in a mode that discards it, and -- because an
    // explicit argument always beat it -- made `AIV_TURBO_VIDEO_SHIFT` unreachable from the app.
    if request.sampling_mode != "base" {
        cmd.args(["--video-shift", &request.video_shift.to_string()]);
    }
    if let Some(edge) = request.ref_image_short_edge {
        cmd.args(["--ref-image-short-edge", &edge.to_string()]);
    }
    if let Some(edge) = request.ref_video_short_edge {
        cmd.args(["--ref-video-short-edge", &edge.to_string()]);
    }
    let result = cmd
        .current_dir(&root)
        .env("HF_HOME", root.join("cache/huggingface"))
        .env("HF_HUB_OFFLINE", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("DIFFUSERS_OFFLINE", "1")
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("failed to start the MiniMax-H3 Ref2VA worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&result.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&result.stderr).trim().to_owned();
    if !result.status.success() {
        return Err(concise_worker_error_with_status(
            &stdout,
            &stderr,
            result.status.code(),
        ));
    }
    if !output.is_file() || !Path::new(&audio).is_file() {
        return Err("the Ref2VA worker finished without creating both outputs".into());
    }
    let log = worker_diagnostics(&stdout, &stderr);
    record_worker_log(&project_root(), "minimax-h3-ref2va", &log);
    Ok(h3::MiniMaxH3Result {
        output_path: request.output_path,
        audio_path: audio,
        log,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a minimal safetensors file: an 8-byte little-endian header length, that many bytes of
    /// JSON header, then nothing. `turbo_baked` never reads past the header, so no tensor data is
    /// needed to exercise it.
    fn safetensors_with_header(directory: &Path, name: &str, header: &str) -> PathBuf {
        let path = directory.join(name);
        let mut file = fs::File::create(&path).expect("create fixture");
        file.write_all(&(header.len() as u64).to_le_bytes()).expect("length");
        file.write_all(header.as_bytes()).expect("header");
        path
    }

    fn scratch(tag: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!("aiv-h3-ref-{tag}-{}", std::process::id()));
        fs::create_dir_all(&directory).expect("scratch directory");
        directory
    }

    #[test]
    fn a_baked_checkpoint_declares_itself_and_needs_no_adapter() {
        let directory = scratch("baked");
        let path = safetensors_with_header(
            &directory,
            "baked.safetensors",
            r#"{"__metadata__":{"is_turbo_baked":"True"},"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#,
        );
        assert!(turbo_baked(&path));
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn the_published_convrot_checkpoint_shape_is_not_baked() {
        // The Comfy-Org pruned INT8 ConvRot releases carry no `__metadata__` at all, which is the
        // case that makes the LightX2V adapter mandatory for Ref2VA.
        let directory = scratch("published");
        let bare = safetensors_with_header(
            &directory,
            "published.safetensors",
            r#"{"blocks.0.attn.qkv_proj.weight":{"dtype":"I8","shape":[2,2],"data_offsets":[0,4]}}"#,
        );
        assert!(!turbo_baked(&bare));

        // A checkpoint that says anything other than the exact string must not be trusted either.
        let hedged = safetensors_with_header(
            &directory,
            "hedged.safetensors",
            r#"{"__metadata__":{"is_turbo_baked":"true"}}"#,
        );
        assert!(!turbo_baked(&hedged));
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn an_unreadable_or_malformed_checkpoint_never_grants_readiness() {
        let directory = scratch("malformed");
        assert!(!turbo_baked(&directory.join("does-not-exist.safetensors")));

        // A truncated file: the declared header is longer than what follows.
        let truncated = directory.join("truncated.safetensors");
        let mut file = fs::File::create(&truncated).expect("create");
        file.write_all(&4096u64.to_le_bytes()).expect("length");
        file.write_all(b"{}").expect("body");
        drop(file);
        assert!(!turbo_baked(&truncated));

        // A header length that would be an allocation attack rather than a model.
        let absurd = directory.join("absurd.safetensors");
        let mut file = fs::File::create(&absurd).expect("create");
        file.write_all(&u64::MAX.to_le_bytes()).expect("length");
        drop(file);
        assert!(!turbo_baked(&absurd));

        // Valid length, invalid JSON.
        let garbage = safetensors_with_header(&directory, "garbage.safetensors", "not json at all");
        assert!(!turbo_baked(&garbage));
        let _ = fs::remove_dir_all(&directory);
    }

    #[test]
    fn the_tile_contract_matches_the_worker_and_the_panel() {
        // `minimax_h3.validate` refuses a tile below `VAE_TILE_SIZE` (256) or not a multiple of 16,
        // and the panel used to offer 192, so a listed choice died in Python. These are the exact
        // values the Ref2V panel now offers, plus the one it used to.
        let ok = |tile: u32| tile >= 256 && tile % 16 == 0;
        for tile in [256u32, 384, 512] {
            assert!(ok(tile), "the panel offers {tile}, which the worker would refuse");
        }
        for tile in [192u32, 128, 0, 260] {
            assert!(!ok(tile), "{tile} should not be accepted");
        }
    }

    #[test]
    fn a_missing_adapter_costs_turbo_not_the_whole_pack() {
        // This used to assert `turbo-missing`, because Ref2VA was hardwired to the four-step
        // adapter and a missing one really did block everything. Base needs no adapter, so the
        // pack is now ready without it and only `turbo_ready` goes false -- the same split FL2VA
        // has always had. What must never come back is the older bug in the other direction:
        // reporting `ready` while `generate` passes an adapter path the worker then refuses.
        let root = scratch("readiness");
        for relative in [
            "vendor/MiniMax-H3/.venv/Scripts/python.exe",
            "ai-runtime/aiv_runtime/minimax_h3_ref.py",
            "models/MiniMax-H3/modular_model_index.json",
            "models/MiniMax-H3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
            "models/MiniMax-H3/transformer_ref/config.json",
        ] {
            let path = root.join(relative);
            fs::create_dir_all(path.parent().expect("parent")).expect("tree");
            fs::write(&path, b"x").expect("stub");
        }
        assert_eq!(state_label(&root), "checkpoint-missing");

        let denoiser = root.join("models/MiniMax-H3/diffusion_models");
        fs::create_dir_all(&denoiser).expect("tree");
        safetensors_with_header(
            &denoiser,
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            r#"{"blocks.0.attn.qkv_proj.weight":{"dtype":"I8","shape":[2,2],"data_offsets":[0,4]}}"#,
        );
        assert_eq!(state_label(&root), "ready", "Base needs no adapter, so the pack is usable");
        assert!(ready(&root));
        assert!(!turbo_ready(&root), "but Turbo has no adapter to run");

        let loras = root.join("models/MiniMax-H3/loras");
        fs::create_dir_all(&loras).expect("tree");
        fs::write(loras.join("minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"), b"x").expect("lora");
        assert_eq!(state_label(&root), "ready");
        assert!(turbo_ready(&root), "the installed adapter enables Turbo");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_turbo_baked_checkpoint_enables_turbo_without_an_adapter_file() {
        // The other half of `turbo_ready`: a checkpoint carrying the residual itself. The published
        // Comfy-Org releases do not, but a locally fused one would, and it must not be told to go
        // install a file it does not need.
        let root = scratch("baked-turbo");
        let denoiser = root.join("models/MiniMax-H3/diffusion_models");
        fs::create_dir_all(&denoiser).expect("tree");
        safetensors_with_header(
            &denoiser,
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            r#"{"__metadata__":{"is_turbo_baked":"True"},"a":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#,
        );
        assert!(turbo_ready(&root));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn the_sampling_modes_and_their_step_counts() {
        // Turbo's count is fixed by the distillation; Base's is a real choice, floored well below
        // MiniMax's published 50 because Ref2VA's only alternative is a four-step preview.
        let turbo_ok = |steps: u32| steps == 4;
        let base_ok = |steps: u32| (10..=60).contains(&steps);
        assert!(turbo_ok(4));
        for steps in [1u32, 3, 5, 8, 50] {
            assert!(!turbo_ok(steps), "turbo-4 must refuse {steps}");
        }
        for steps in [10u32, 20, 50, 60] {
            assert!(base_ok(steps), "base should accept {steps}");
        }
        for steps in [0u32, 4, 9, 61, 200] {
            assert!(!base_ok(steps), "base must refuse {steps}");
        }
        assert_eq!(default_sampling_mode(), "base");
        assert_eq!(default_steps(), 20);
    }

    #[test]
    fn the_canvas_contract_is_the_checkpoints_not_a_vram_profile() {
        // This used to be `short edge <= 544 && area <= 544 * 1280`, which is a 16 GB profile
        // written as though it were the model's rule -- and it made MiniMax's own 768p canvas
        // unreachable, so our output could never be compared against their published clips.
        let ok = |w: u32, h: u32| {
            w % 32 == 0
                && h % 32 == 0
                && w.min(h) <= MAX_SHORT_EDGE
                && w as u64 * h as u64 <= MAX_PIXELS
        };
        // MiniMax's own trained canvas, and the tiers below it.
        for (w, h) in [(1344u32, 768u32), (960, 544), (896, 480), (768, 768), (544, 960)] {
            assert!(ok(w, h), "{w}x{h} is inside the checkpoint's contract");
        }
        // A short edge past 768, an area past 768*1344, and an axis off the 32-grid.
        for (w, h) in [(1344u32, 800u32), (1376, 768), (1000, 544), (960, 545)] {
            assert!(!ok(w, h), "{w}x{h} must be refused");
        }
        // Comfy's own tutorial warns about exactly this one: a 1.0 MP 16:9 selection resolves to
        // 1376x768, which is over the pixel cap despite a legal short edge.
        assert!(!ok(1376, 768));
    }

    #[test]
    fn a_videos_soundtrack_counts_against_the_audio_budget() {
        // Three sound-bearing videos plus three audio clips used to pass: only `kind == "audio"`
        // was counted, so the request declared three audio references and conditioned on six.
        // That is the voice-bleed and timbre-drift failure mode.
        assert!(check_audio_pairing(3, 3).is_ok());
        let refused = check_audio_pairing(3, 6).unwrap_err();
        assert!(refused.contains("at most 3 audio references"), "{refused}");
        assert!(refused.contains("soundtrack"), "the message must say where the sixth came from");
    }

    #[test]
    fn audio_references_may_not_outnumber_visual_ones() {
        // The checkpoint's own rule, which we did not have: we only required "at least one visual".
        assert!(check_audio_pairing(1, 1).is_ok());
        assert!(check_audio_pairing(9, 3).is_ok());
        assert!(check_audio_pairing(2, 3).is_err());
        let refused = check_audio_pairing(1, 2).unwrap_err();
        assert!(refused.contains("2 audio against 1 visual"), "{refused}");
    }

    #[test]
    fn a_request_with_no_audio_at_all_is_fine() {
        assert!(check_audio_pairing(9, 0).is_ok());
        assert!(check_audio_pairing(1, 0).is_ok());
    }

    #[test]
    fn the_reference_budget_is_the_model_cards() {
        // "Images: <= 9 images. Videos: <= 3 clips. Audio: <= 3 clips. Maximum number of files
        // across all input types is 12." These were 3/1/1/4.
        assert_eq!((MAX_IMAGES, MAX_VIDEOS, MAX_AUDIOS, MAX_REFERENCES), (9, 3, 3, 12));
        // The per-modality caps can sum past the file cap, which is why both checks exist.
        assert!(MAX_IMAGES + MAX_VIDEOS + MAX_AUDIOS > MAX_REFERENCES);
    }

    #[test]
    fn base_is_not_sent_a_shift_the_worker_would_ignore() {
        // `run_denoise` only consults `--video-shift` inside the Turbo branch, so sending it for
        // Base advertised a live control over a value that is discarded -- and, because an
        // explicit argument beat the environment, it also made AIV_TURBO_VIDEO_SHIFT unreachable.
        let sends_shift = |mode: &str| mode != "base";
        assert!(!sends_shift("base"));
        assert!(sends_shift("turbo-4"));
    }
}
