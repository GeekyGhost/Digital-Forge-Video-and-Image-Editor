use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{env, fs, path::{Path, PathBuf}, process::Stdio, sync::OnceLock};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{Mutex, OwnedSemaphorePermit, Semaphore},
};
use uuid::Uuid;

/// MiniMax-H3 lives in its own module. It shares `project_root`, the worker-error formatter and
/// the process plumbing here, but nothing else -- its venv, weights and adapter are all separate,
/// so it cannot regress the LTX path.
pub mod h3;
pub mod h3_ref;
pub mod image;
pub mod lora;
pub mod photo;
pub mod resource_guard;
pub mod upscale;
pub mod elements;
pub mod wan_alpha;
pub mod motion;
pub mod render;
pub mod frame_channel;
pub mod wan_animate;
pub mod clean_plate;
pub mod spritesheet;
pub mod clip_matte;
pub mod prompt_rewrite;

pub use resource_guard::JobKind;

/// One GPU-heavy AI operation at a time, enforced below the React `busy` flag, with the resource
/// guard's pre-flight check folded into acquisition (VFX Phase 0, task 0.1).
#[derive(Clone)]
pub struct AiExecutionGate {
    semaphore: std::sync::Arc<Semaphore>,
    /// Pre-flight prompts waiting for the user's answer, by prompt id.
    pending: std::sync::Arc<parking_lot::Mutex<std::collections::HashMap<String, tokio::sync::oneshot::Sender<PreflightDecision>>>>,
    /// Set once the frontend's warning dialog has mounted. Before that a failed check cannot be
    /// shown, so it fails fast instead of waiting on a prompt nobody will see.
    ui_ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Default for AiExecutionGate {
    fn default() -> Self {
        Self {
            semaphore: std::sync::Arc::new(Semaphore::new(1)),
            pending: Default::default(),
            ui_ready: Default::default(),
        }
    }
}

/// What the user chose in the resource warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PreflightDecision {
    Cancel,
    Continue,
    /// Stop this app's resident model worker (never anything else), then continue.
    ReleaseAppWorkers,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PreflightPrompt {
    pub id: String,
    pub verdict: resource_guard::Verdict,
}

/// Holding this is holding the gate. It also carries the job monitor, which logs the job's real
/// peak usage to `.aiv/resource-log.jsonl` when the permit is dropped.
pub struct AiPermit {
    _permit: OwnedSemaphorePermit,
    _monitor: resource_guard::JobMonitor,
}

/// A prompt left unanswered this long is treated as Cancel, so the gate cannot be held forever.
const PREFLIGHT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

impl AiExecutionGate {
    /// Acquire the gate for `job`, then check resources. The check runs *after* acquisition on
    /// purpose: measured before, it would count the memory of the job still running ahead of us.
    /// If the check fails the user is asked (event `ai-preflight`); the gate is held while they
    /// decide, so nothing else can start in between. Nothing is ever killed automatically.
    pub async fn acquire_for(&self, job: JobKind, app: &tauri::AppHandle) -> Result<AiPermit, String> {
        use tauri::Emitter;
        let permit = self.semaphore.clone().acquire_owned().await
            .map_err(|_| "the AI execution gate is closed".to_owned())?;
        let reclaimable = reclaimable_worker_pids(job);
        let snapshot = tokio::task::spawn_blocking(resource_guard::measure).await
            .map_err(|error| format!("resource check failed: {error}"))?;
        let verdict = resource_guard::evaluate(&resource_guard::requirement(job), &snapshot, &reclaimable);
        let mut overridden = false;
        if !verdict.ok {
            if !self.ui_ready.load(std::sync::atomic::Ordering::SeqCst) {
                return Err(format!("Not started: {}", verdict.summary()));
            }
            let id = Uuid::new_v4().simple().to_string();
            let (sender, receiver) = tokio::sync::oneshot::channel();
            self.pending.lock().insert(id.clone(), sender);
            if let Err(error) = app.emit("ai-preflight", PreflightPrompt { id: id.clone(), verdict: verdict.clone() }) {
                self.pending.lock().remove(&id);
                return Err(format!("Not started: {} (could not show the warning: {error})", verdict.summary()));
            }
            let decision = tokio::time::timeout(PREFLIGHT_TIMEOUT, receiver).await;
            self.pending.lock().remove(&id);
            match decision {
                Ok(Ok(PreflightDecision::Continue)) => overridden = true,
                Ok(Ok(PreflightDecision::ReleaseAppWorkers)) => {
                    release_resident_workers().await;
                    overridden = true;
                }
                _ => return Err(format!("Cancelled: {}", verdict.summary())),
            }
        }
        let monitor = resource_guard::JobMonitor::start(job, project_root().join(".aiv/resource-log.jsonl"), &verdict, overridden);
        Ok(AiPermit { _permit: permit, _monitor: monitor })
    }

    /// The frontend's answer to an `ai-preflight` prompt.
    pub fn decide(&self, id: &str, decision: PreflightDecision) -> Result<(), String> {
        let sender = self.pending.lock().remove(id).ok_or("that resource warning is no longer waiting")?;
        sender.send(decision).map_err(|_| "the job stopped waiting for this answer".to_owned())
    }

    pub fn set_ui_ready(&self, ready: bool) {
        self.ui_ready.store(ready, std::sync::atomic::Ordering::SeqCst);
    }
}

/// A dry run for diagnostics: measure and evaluate without taking the gate.
pub async fn resource_check(job: JobKind) -> Result<resource_guard::Verdict, String> {
    let reclaimable = reclaimable_worker_pids(job);
    let snapshot = tokio::task::spawn_blocking(resource_guard::measure).await
        .map_err(|error| format!("resource check failed: {error}"))?;
    Ok(resource_guard::evaluate(&resource_guard::requirement(job), &snapshot, &reclaimable))
}

/// App workers whose memory `job` will take over. Only the resident LTX retake worker exists
/// today: a retake reuses it or kills it before loading, so its commit is effectively free.
fn reclaimable_worker_pids(job: JobKind) -> Vec<u32> {
    if job != JobKind::LtxRetake { return Vec::new(); }
    RETAKE_WORKER.get()
        .and_then(|slot| slot.try_lock().ok())
        .and_then(|guard| guard.as_ref().and_then(|worker| worker.child.id()))
        .into_iter()
        .collect()
}

/// Stops this app's resident model workers. Only ever called on an explicit user choice.
async fn release_resident_workers() {
    if let Some(slot) = RETAKE_WORKER.get() {
        let mut guard = slot.lock().await;
        if let Some(mut worker) = guard.take() {
            let _ = worker.child.kill().await;
            let _ = worker.child.wait().await;
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AiRuntimeStatus {
    pub root: String,
    pub python_ready: bool,
    pub ltx_source_ready: bool,
    pub checkpoint_ready: bool,
    pub gemma_ready: bool,
    pub single_pass_ready: bool,
    pub retake_ready: bool,
    /// The Foley V2A IC-LoRA is what makes generated audio actually track the picture.
    pub foley_lora_ready: bool,
    /// MOSS has an app-owned Python runtime because its Python/Torch pins conflict with LTX.
    pub moss_sfx_state: String,
    pub moss_sfx_ready: bool,
    /// SAM 2 needs its own package plus a checkpoint.
    pub sam2_ready: bool,
    /// `ready`, `model-missing`, `runtime-missing`, `dependencies-missing`, or `adapter-missing`.
    pub sam2_state: String,
    pub kiwi_edit_ready: bool,
    /// `ready`, `pack-missing`, `deps-missing`, `torch-cpu`, `adapter-missing`, or `venv-missing`.
    pub kiwi_edit_state: String,
    pub whisper_ready: bool,
    /// `ready`, `encoder-missing`, `checkpoint-missing`, `pack-missing`, `adapter-missing` or
    /// `venv-missing`.
    pub minimax_h3_state: String,
    pub minimax_h3_ready: bool,
    pub minimax_h3_draft_ready: bool,
    pub minimax_h3_turbo_ready: bool,
    /// Optional Ref2VA partition; never gates or alters the FL2VA modes above.
    /// `ready`, `checkpoint-missing`, `config-missing`, `encoder-missing`,
    /// `shared-pack-missing`, `adapter-missing` or `venv-missing`.
    pub minimax_h3_ref_state: String,
    pub minimax_h3_ref_ready: bool,
    /// Dedicated still-image runtime. It never imports from the H3 or LTX environments.
    pub image_studio_state: String,
    pub zit_image_ready: bool,
    pub zimage_base_ready: bool,
    pub qwen_image_edit_ready: bool,
    pub zimage_trainer_ready: bool,
    pub zimage_captioner_ready: bool,
    /// The Qwen edit prompt rewriter (same runtime and model files as the captioner).
    pub qwen_edit_rewriter_ready: bool,
    /// Photo Edit's optional model steps. Each is independent: the workspace is fully usable with
    /// all three missing, which is why they are plain booleans rather than a blocking state string.
    pub photo_depth_ready: bool,
    pub photo_inpaint_ready: bool,
    pub photo_select_ready: bool,
    pub birefnet_ready: bool,
    /// Real-ESRGAN weights for the AI stage of the upscale chain. `ready`, `weights-missing`,
    /// `pack-missing`, `adapter-missing` or `venv-missing`.
    pub real_esrgan_state: String,
    pub real_esrgan_ready: bool,
    /// Weight files actually on disk, so the panel offers only models that will load.
    pub real_esrgan_models: Vec<String>,
    /// NVIDIA VideoSuperRes, installed as a pip wheel rather than a model pack. `ready`,
    /// `not-installed`, `incomplete`, `adapter-missing` or `venv-missing`.
    pub nvvfx_state: String,
    pub nvvfx_ready: bool,
    pub profile: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct KiwiEditRequest {
    pub video_path: String,
    pub output_path: String,
    pub instruction: String,
    #[serde(default)]
    pub reference_image: Option<String>,
    pub start_time: f64,
    pub end_time: f64,
    pub seed: i64,
    /// `instruction` or `reference`. The bundled pipeline exposes no mask parameter.
    pub mode: String,
    /// `stable` is one supported 81-frame pass; `long` uses 25-frame overlap and seam analysis.
    #[serde(default = "default_kiwi_strategy")]
    pub strategy: String,
    pub steps: u32,
    /// Classifier-free guidance. 1.0 reproduces the pack's shipped behaviour, in which the
    /// instruction conditions the model but nothing steers toward it. Defaulted so an older
    /// frontend bundle keeps working against a newer binary.
    #[serde(default = "default_kiwi_guidance")]
    pub guidance_scale: f64,
    #[serde(default)]
    pub negative_prompt: String,
    /// `balanced` stages the encoder and the denoiser through the GPU one at a time (16 GB),
    /// `performance` keeps everything resident (24 GB), `minimal` adds transformer block
    /// streaming for cards below 16 GB.
    pub offload: String,
}

fn default_kiwi_strategy() -> String { "stable".to_owned() }
fn default_kiwi_guidance() -> f64 { 1.0 }

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscribeRequest {
    pub path: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub translate: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptCue {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LtxBridgeRequest {
    /// `first-last` (the original single-pass path), `text`, `image`, or `video-audio`.
    /// The first three reach `ltx_single_stage.py`; `video-audio` returns earlier through the
    /// picture-synced foley adapter and never touches the single-pass worker.
    #[serde(default = "default_ltx_mode")]
    pub mode: String,
    #[serde(default)]
    pub first_frame: Option<String>,
    #[serde(default)]
    pub last_frame: Option<String>,
    /// Source clip for `video-audio`, which generates only a soundtrack and copies the picture.
    #[serde(default)]
    pub video_path: Option<String>,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    pub output_path: String,
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub num_frames: u32,
    pub frame_rate: f64,
    pub seed: i64,
    pub guide_strength: f64,
    /// Classifier-free guidance. 1.0 is unguided (the distilled default); above it the model
    /// runs a second unconditional pass per step. Audio control begins around 1.5.
    #[serde(default = "default_cfg_scale")]
    pub cfg_scale: f64,
    #[serde(default = "default_cfg_scale")]
    pub audio_cfg_scale: f64,
    /// Foley only. `lora_strength` scales the Foley adapter itself; `reference_strength` scales how
    /// hard the generated audio is pinned to the reference footage.
    #[serde(default = "default_strength")]
    pub lora_strength: f64,
    #[serde(default = "default_strength")]
    pub reference_strength: f64,
    pub offload: String,
    #[serde(default = "default_ltx_attention")]
    pub attention_backend: String,
}

fn default_cfg_scale() -> f64 { 1.0 }
fn default_strength() -> f64 { 1.0 }

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LtxBridgeResult {
    pub output_path: String,
    pub log: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LtxRetakeResult {
    pub output_path: String,
    pub preview_path: String,
    pub log: String,
    pub effective_start_time: f64,
    pub effective_end_time: f64,
    pub start_frame: u64,
    pub end_frame: u64,
    pub fps_numerator: u32,
    pub fps_denominator: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SfxGenerationResult {
    pub output_path: String,
    pub sample_rate: u32,
    pub channels: u32,
    pub duration_seconds: f64,
    pub seed: i64,
    pub log: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MossSfxRequest {
    pub project_id: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub duration_seconds: f64,
    pub seed: i64,
    /// `draft` (30 steps) or `final` (100 steps).
    pub quality: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LtxFoleyRequest {
    pub project_id: String,
    pub video_path: String,
    pub source_in_seconds: f64,
    pub source_out_seconds: f64,
    #[serde(default = "default_playback_rate")]
    pub playback_rate: f64,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub seed: i64,
    /// `minimal`, `normal`, or `detailed`.
    pub density: String,
    pub offload: String,
    #[serde(default = "default_ltx_attention")]
    pub attention_backend: String,
}

fn default_playback_rate() -> f64 { 1.0 }

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LtxClipExtendRequest {
    pub video_path: String,
    pub source_in_seconds: f64,
    pub source_out_seconds: f64,
    pub output_path: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    pub num_frames: u32,
    pub frame_rate: f64,
    pub seed: i64,
    pub guide_strength: f64,
    #[serde(default = "default_cfg_scale")]
    pub cfg_scale: f64,
    #[serde(default = "default_cfg_scale")]
    pub audio_cfg_scale: f64,
    pub offload: String,
    /// Threaded through to `generate_bridge` alongside `offload`: both are choices about how this
    /// machine runs the transformer, and a clip extension has no reason to differ from a plain
    /// generation.
    #[serde(default = "default_ltx_attention")]
    pub attention_backend: String,
}

/// Clip-to-clip: generate the motion between two adjacent V1 clips. The outgoing clip supplies the
/// first conditioning frame and the incoming clip supplies the last, both taken from their
/// *trimmed* visible ranges rather than the raw files.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LtxClipBridgeRequest {
    /// Outgoing clip: its last visible frame becomes the bridge's first frame.
    pub from_video_path: String,
    pub from_source_in_seconds: f64,
    pub from_source_out_seconds: f64,
    /// Incoming clip: its first visible frame becomes the bridge's last frame.
    pub to_video_path: String,
    pub to_source_in_seconds: f64,
    pub output_path: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    pub width: u32,
    pub height: u32,
    pub num_frames: u32,
    pub frame_rate: f64,
    pub seed: i64,
    pub guide_strength: f64,
    #[serde(default = "default_cfg_scale")]
    pub cfg_scale: f64,
    #[serde(default = "default_cfg_scale")]
    pub audio_cfg_scale: f64,
    pub offload: String,
    /// Threaded through to `generate_bridge` alongside `offload`: both are choices about how this
    /// machine runs the transformer, and a clip extension has no reason to differ from a plain
    /// generation.
    #[serde(default = "default_ltx_attention")]
    pub attention_backend: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LtxRetakeRequest {
    pub video_path: String,
    pub output_path: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub start_time: f64,
    pub end_time: f64,
    pub seed: i64,
    pub mode: String,
    pub offload: String,
    /// `fast` or `experimental-guided-distilled`, resolved into the five guidance scales by the worker's
    /// `GUIDANCE_PRESETS`.
    ///
    /// `fast` leaves every axis inert, so the pipeline takes `SimpleDenoiser` and runs one
    /// transformer pass per step. The experimental mode runs a documented distilled checkpoint
    /// through an undocumented guided-denoiser/eight-sigma hybrid, so it is explicitly not called
    /// a reference mode.
    #[serde(default = "default_retake_guidance")]
    pub guidance: String,
    #[serde(default = "default_ltx_attention")]
    pub attention_backend: String,
    /// Classifier-free guidance, when the caller wants to override the preset. `None` means
    /// "whatever `guidance` says", which is the normal case -- the panel only sends a number
    /// when it is showing the slider, and the slider only appears under the experimental mode.
    ///
    /// Without guidance the pipeline runs `SimpleDenoiser`, where the prompt conditions but
    /// never steers -- so a masked region surrounded by pinned original latents can reconstruct
    /// itself and a partial retake appear to do nothing. The worker's `[latent]` diagnostic
    /// measures exactly that and reports the failure mode when it happens.
    #[serde(default)]
    pub cfg_scale: Option<f64>,
    #[serde(default)]
    pub audio_cfg_scale: Option<f64>,
    #[serde(default)]
    pub enhance_prompt: bool,
}

fn default_retake_guidance() -> String { "fast".to_owned() }

struct ResidentRetakeWorker {
    key: String,
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    stderr_path: PathBuf,
}

static RETAKE_WORKER: OnceLock<Mutex<Option<ResidentRetakeWorker>>> = OnceLock::new();

fn retake_worker_key(request: &LtxRetakeRequest) -> String {
    format!(
        "{}|{}|{}|{:?}|{:?}",
        request.guidance, request.attention_backend, request.offload,
        request.cfg_scale, request.audio_cfg_scale,
    )
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sam2Request {
    pub video_path: String,
    pub output_path: String,
    pub start_time: f64,
    pub end_time: f64,
    /// Normalised 0..1 monitor coordinates. `include` false carves the region away.
    pub points: Vec<Sam2Point>,
    /// `matte`, `cutout` or `removed`.
    pub output_kind: String,
    #[serde(default)]
    pub feather: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Sam2Point {
    pub x: f64,
    pub y: f64,
    pub include: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskEffectGradientStop {
    pub at: f64,
    pub color: String,
    pub opacity: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskEffectGradient {
    pub stops: Vec<MaskEffectGradientStop>,
    pub kind: String,
    pub angle: f64,
    pub cx: f64,
    pub cy: f64,
    pub scale: f64,
    pub offset: f64,
    pub repeat: f64,
    pub mirror: bool,
    pub space: String,
    pub ease: f64,
    pub dither: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskVfxAnimatedValues {
    pub effect_x: f64,
    pub effect_y: f64,
    pub effect_scale: f64,
    pub effect_rotation: f64,
    pub effect_opacity: f64,
    pub intensity: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskVfxKeyframe {
    pub id: String,
    pub at: f64,
    pub easing: String,
    pub values: MaskVfxAnimatedValues,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskTrackRequest {
    pub video_path: String,
    pub start_time: f64,
    pub end_time: f64,
    pub model: String,
    #[serde(default)]
    pub points: Vec<Sam2Point>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskTrackResult {
    pub cache_id: String,
    pub cache_path: String,
    pub manifest_path: String,
    pub frame_count: u32,
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub model: String,
    pub created_at: String,
    pub reused: bool,
    #[serde(default)]
    pub log: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MaskEffectRequest {
    pub video_path: String,
    pub output_path: String,
    pub start_time: f64,
    pub end_time: f64,
    /// `sam2` for an interactive tracked subject, or `birefnet` for automatic foreground.
    pub model: String,
    #[serde(default)]
    pub points: Vec<Sam2Point>,
    pub effect: String,
    pub color: String,
    #[serde(default = "default_mask_secondary_color")]
    pub secondary_color: String,
    pub intensity: f64,
    pub feather: u32,
    pub expand: i32,
    #[serde(default)]
    pub temporal_smooth: u32,
    #[serde(default)]
    pub cleanup: u32,
    #[serde(default = "default_mask_effect_speed")]
    pub effect_speed: f64,
    #[serde(default = "default_mask_particle_amount")]
    pub particle_amount: f64,
    #[serde(default = "default_mask_transition_in")]
    pub transition_in: String,
    #[serde(default = "default_mask_transition_out")]
    pub transition_out: String,
    #[serde(default = "default_mask_transition_in_duration")]
    pub transition_in_duration: f64,
    #[serde(default = "default_mask_transition_out_duration")]
    pub transition_out_duration: f64,
    #[serde(default)]
    pub invert: bool,
    #[serde(default)]
    pub audio_reactive: String,
    #[serde(default)]
    pub reactive_amount: f64,
    #[serde(default)]
    pub reactive_smoothing: f64,
    #[serde(default = "default_mask_placement")]
    pub placement: String,
    #[serde(default = "default_mask_matte_style")]
    pub matte_style: String,
    #[serde(default)]
    pub effect_clip_path: String,
    #[serde(default = "default_mask_effect_clip_color")]
    pub effect_clip_color: String,
    #[serde(default = "default_mask_effect_opacity")]
    pub effect_opacity: f64,
    #[serde(default = "default_mask_effect_center")]
    pub effect_x: f64,
    #[serde(default = "default_mask_effect_center")]
    pub effect_y: f64,
    #[serde(default = "default_mask_effect_scale")]
    pub effect_scale: f64,
    #[serde(default)]
    pub effect_rotation: f64,
    #[serde(default = "default_true", rename = "loop")]
    pub loop_effect: bool,
    #[serde(default = "default_mask_pulse_amount")]
    pub pulse_amount: f64,
    #[serde(default = "default_mask_pulse_rate")]
    pub pulse_rate: f64,
    #[serde(default)]
    pub gradient_enabled: bool,
    #[serde(default = "default_mask_gradient")]
    pub gradient: MaskEffectGradient,
    #[serde(default)]
    pub matte_cache_path: String,
    #[serde(default = "default_mask_occlusion_feather")]
    pub occlusion_feather: f64,
    #[serde(default)]
    pub occlusion_expand: i32,
    #[serde(default = "default_mask_transition_origin")]
    pub transition_origin: String,
    #[serde(default = "default_mask_blend_mode")]
    pub blend_mode: String,
    #[serde(default = "default_mask_key_mode")]
    pub key_mode: String,
    #[serde(default = "default_mask_key_threshold")]
    pub key_threshold: f64,
    #[serde(default = "default_mask_key_softness")]
    pub key_softness: f64,
    #[serde(default = "default_mask_key_color")]
    pub key_color: String,
    #[serde(default = "default_one")]
    pub rear_amount: f64,
    #[serde(default)]
    pub subject_amount: f64,
    #[serde(default)]
    pub front_amount: f64,
    #[serde(default)]
    pub motion_blur: f64,
    #[serde(default)]
    pub light_wrap: f64,
    #[serde(default)]
    pub color_spill: f64,
    #[serde(default)]
    pub grain: f64,
    #[serde(default)]
    pub displacement: f64,
    #[serde(default)]
    pub keyframes: Vec<MaskVfxKeyframe>,
    #[serde(default)]
    pub preview_width: u32,
}

fn default_mask_secondary_color() -> String { "#ff6b1a".to_owned() }
fn default_mask_effect_speed() -> f64 { 1.0 }
fn default_mask_particle_amount() -> f64 { 1.0 }
fn default_mask_transition_in() -> String { "none".to_owned() }
fn default_mask_transition_out() -> String { "none".to_owned() }
fn default_mask_transition_in_duration() -> f64 { 0.6 }
fn default_mask_transition_out_duration() -> f64 { 0.45 }
fn default_mask_placement() -> String { "behind".to_owned() }
fn default_mask_matte_style() -> String { "subject".to_owned() }
fn default_mask_effect_clip_color() -> String { "original".to_owned() }
fn default_mask_effect_opacity() -> f64 { 0.9 }
fn default_mask_effect_center() -> f64 { 0.5 }
fn default_mask_effect_scale() -> f64 { 100.0 }
fn default_true() -> bool { true }
fn default_mask_pulse_amount() -> f64 { 0.18 }
fn default_mask_pulse_rate() -> f64 { 1.25 }
fn default_mask_occlusion_feather() -> f64 { 1.5 }
fn default_mask_transition_origin() -> String { "subject".to_owned() }
fn default_mask_blend_mode() -> String { "screen".to_owned() }
fn default_mask_key_mode() -> String { "luma".to_owned() }
fn default_mask_key_threshold() -> f64 { 0.08 }
fn default_mask_key_softness() -> f64 { 0.12 }
fn default_mask_key_color() -> String { "#00ff00".to_owned() }
fn default_one() -> f64 { 1.0 }
fn default_mask_gradient() -> MaskEffectGradient {
    MaskEffectGradient {
        stops: vec![
            MaskEffectGradientStop { at: 0.0, color: "#ffffff".to_owned(), opacity: 1.0 },
            MaskEffectGradientStop { at: 0.28, color: "#00f0ff".to_owned(), opacity: 1.0 },
            MaskEffectGradientStop { at: 0.72, color: "#3166ff".to_owned(), opacity: 0.82 },
            MaskEffectGradientStop { at: 1.0, color: "#1010a0".to_owned(), opacity: 0.0 },
        ],
        kind: "radial".to_owned(), angle: 270.0, cx: 0.5, cy: 0.5, scale: 0.72,
        offset: 0.0, repeat: 1.0, mirror: false, space: "oklab".to_owned(), ease: 0.8, dither: 0.7,
    }
}

/// Decode a process exit code into something a human can act on. A worker that dies without
/// writing to stderr is the hard case: the code is then the *only* evidence of what happened, and
/// "no error message" is exactly what a hard crash or an OS kill looks like from here.
fn exit_code_meaning(code: Option<i32>) -> String {
    let Some(code) = code else {
        return "the worker was terminated by a signal before it could report anything".to_owned();
    };
    // Windows surfaces NTSTATUS values as the exit code; Rust hands them back as a signed i32.
    let status = code as u32;
    let note = match status {
        0xC000_0005 => Some("access violation — a native crash inside a model library"),
        0xC000_0017 | 0xC000_009A => Some("the process ran out of memory (system commit limit)"),
        0xC000_0409 => Some("stack buffer overrun detected by the CRT"),
        0xC000_0135 | 0xC000_0142 => Some("a dependency DLL failed to load or initialise"),
        0xC000_013A => Some("the worker was interrupted (Ctrl+C or a console close)"),
        1 => Some("an unhandled Python exception"),
        _ => None,
    };
    match note {
        Some(text) => format!("exit code 0x{status:08X} — {text}"),
        None => format!("exit code {code} (0x{status:08X})"),
    }
}

/// Worker stderr is dominated by tqdm progress bars that redraw with carriage returns, which
/// turns a failure into an unreadable wall of text. Keep the last few meaningful lines so the
/// UI shows the actual error instead of a loading animation's corpse.
///
/// The exit code is always appended. A worker killed by the OS, or one that dies inside a native
/// library, writes nothing at all — and a bare "the worker exited without an error message" is
/// indistinguishable from a dozen different faults without it.
fn concise_worker_error_with_status(stdout: &str, stderr: &str, code: Option<i32>) -> String {
    let raw = if stderr.trim().is_empty() { stdout } else { stderr };
    let lines: Vec<&str> = raw
        .split(['\n', '\r'])
        .map(str::trim)
        .filter(|line| !line.is_empty())
        // "%|" is tqdm's signature; "it/s" and "s/it" catch its rate suffixes.
        .filter(|line| !line.contains("%|") && !line.contains("it/s") && !line.contains("s/it"))
        .collect();
    // 14 lines was too tight for a two-process worker: a Python traceback that crosses a
    // subprocess boundary spends most of that budget on frame/source pairs, and the exception
    // line itself -- the only line that says what went wrong -- falls off the top.
    let tail = lines.iter().rev().take(28).rev().copied().collect::<Vec<_>>();
    let detail = exit_code_meaning(code);
    if tail.is_empty() {
        // Nothing readable survived the filter, so keep whatever raw tail exists rather than
        // discarding the only evidence there is.
        // Collect before taking a tail: `.rev().take(n).rev()` only compiles on an iterator that
        // is already ExactSizeIterator, and `Filter` never is — it cannot know its length without
        // running the predicate. A slice does know, so materialise first and take the tail from
        // the end, which is what the line-14 tail above does via `lines.iter()`.
        let salvaged: Vec<&str> = raw
            .split(['\n', '\r'])
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect();
        let salvage = &salvaged[salvaged.len().saturating_sub(4)..];
        if salvage.is_empty() {
            format!("The worker exited without writing an error message.\n{detail}")
        } else {
            format!("{}\n\n{detail}", salvage.join("\n"))
        }
    } else {
        format!("{}\n\n{detail}", tail.join("\n"))
    }
}

/// A worker that finished still has something to say, and until now nobody was listening.
///
/// Every LTX worker prints its measurements to *stderr*: the retake window plan, the offload
/// profile, the checkpoint backend, peak VRAM, and the calibration line that exists for no other
/// purpose than to set `DEFAULT_MAX_PIXEL_FRAMES`. `LtxBridgeResult::log` carried stdout alone,
/// and stdout on these paths is a single line of JSON. So a run that *succeeded* discarded the
/// only evidence of what it cost -- the one number a seven-minute GPU run was performed to
/// obtain was dropped by the code collecting its result. The H3 module has always kept both
/// streams; this brings LTX in line with it.
///
/// tqdm's redraws are filtered exactly as `concise_worker_error_with_status` filters them: a
/// progress bar frozen mid-sweep is noise in the log of a run that reached the end.
fn worker_diagnostics(stdout: &str, stderr: &str) -> String {
    fn meaningful(raw: &str) -> Vec<&str> {
        raw.split(['\n', '\r'])
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .filter(|line| !line.contains("%|") && !line.contains("it/s") && !line.contains("s/it"))
            .collect()
    }
    let stderr_lines = meaningful(stderr);
    // A cold load prints hundreds of library warnings before the run starts; the diagnostics
    // worth keeping are printed last, after the work is done. So the tail is the useful end.
    let kept = &stderr_lines[stderr_lines.len().saturating_sub(MAX_DIAGNOSTIC_LINES)..];
    let mut lines: Vec<&str> = kept.to_vec();
    lines.extend(meaningful(stdout));
    lines.join("\n")
}

/// How many stderr lines survive into a kept log. Generous, because the failure path's 28 exists
/// to keep an error message readable and this exists to keep a measurement retrievable.
const MAX_DIAGNOSTIC_LINES: usize = 400;

/// Append a finished worker's diagnostics to `.aiv/worker.log`.
///
/// The panel shows the same text, but a panel is cleared by the next run and by closing the
/// dialog. A measurement that costs minutes of GPU time to produce should not depend on someone
/// reading it before they click anything, so it also goes somewhere that outlives the window.
///
/// Best effort by construction: a log that cannot be written must never fail the run that
/// produced the output the user actually asked for.
fn record_worker_log(root: &Path, worker: &str, diagnostics: &str) {
    use std::io::Write;
    if diagnostics.trim().is_empty() {
        return;
    }
    let directory = root.join(".aiv");
    if fs::create_dir_all(&directory).is_err() {
        return;
    }
    let stamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let entry = format!("\n===== {worker} at {stamp} =====\n{diagnostics}\n");
    if let Ok(mut file) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(directory.join("worker.log"))
    {
        let _ = file.write_all(entry.as_bytes());
    }
}

fn default_ltx_mode() -> String { "first-last".to_owned() }

pub(crate) fn project_root() -> PathBuf {
    env::var_os("AIV_ROOT").map(PathBuf::from).unwrap_or_else(|| {
        env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
    })
}

fn paths(root: &Path) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let python = root.join("vendor/LTX-2/.venv/Scripts/python.exe");
    let source = root.join("vendor/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/__init__.py");
    let fp8_checkpoint = root.join("models/LTX-2.3/ltx-2.3-22b-distilled-fp8.safetensors");
    let checkpoint = if fp8_checkpoint.is_file() { fp8_checkpoint } else { root.join("models/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors") };
    let gemma = root.join("models/gemma-3-12b");
    (python, source, checkpoint, gemma)
}

/// The Kiwi-Edit pack is a diffusers snapshot with a custom remote-code pipeline. These are the
/// component folders `model_index.json` actually declares, plus the pipeline module itself.
const KIWI_EDIT_COMPONENTS: &[&str] = &[
    "transformer", "vae", "scheduler", "mllm_encoder", "processor", "source_embedder", "ref_embedder",
];

fn kiwi_edit_root(root: &Path) -> PathBuf {
    root.join("models/kiwi-edit-5b")
}

fn kiwi_edit_installed(root: &Path) -> bool {
    let pack = kiwi_edit_root(root);
    pack.join("model_index.json").is_file()
        && pack.join("pipeline_kiwi_edit.py").is_file()
        && pack.join("transformer/diffusion_pytorch_model.safetensors").is_file()
        && KIWI_EDIT_COMPONENTS.iter().all(|part| pack.join(part).is_dir())
}

/// The bundled pipeline, its Wan VAE, and the vendored Qwen2.5-VL encoder import these at load
/// time. Checking for the packages separately lets the UI say "dependencies" instead of "pack".
const INFERENCE_PACKAGES: &[&str] = &["diffusers", "transformers", "einops", "torchvision", "torch"];

fn inference_packages_installed(root: &Path) -> bool {
    let site_packages = root.join(".venv/Lib/site-packages");
    INFERENCE_PACKAGES.iter().all(|name| site_packages.join(name).is_dir())
}

/// A CPU-only PyTorch imports cleanly and then fails the moment a model reaches for the GPU, so
/// read the build tag rather than trusting that the package is present. Torch writes
/// `__version__ = '<n>+cpu'` and `cuda: Optional[str] = None` for CPU-only wheels.
fn torch_cuda_ready(root: &Path) -> bool {
    let version_file = root.join(".venv/Lib/site-packages/torch/version.py");
    match std::fs::read_to_string(version_file) {
        Ok(text) => !text.contains("+cpu") && !text.contains("cuda: Optional[str] = None"),
        Err(_) => false,
    }
}

/// Where LTX adapter LoRAs may live. `models/loras/ltx` is the canonical location; the others are
/// accepted so a hand-placed file is found without being moved.
const LORA_SEARCH_DIRS: &[&str] = &["models/loras/ltx", "models/loras", "models/LTX-2.3/loras"];

// LTX geometry, mirroring `ai-runtime/aiv_runtime/ltx_geometry.py` and
// `src/features/generation/ltxGeometry.ts`. Change one and change all three.
//
// The axis multiple is 32, not 64: `assert_resolution` in the pinned tree computes
// `64 if is_two_stage else 32`, and there is no stage 2 in this project. 832x480 used to be
// retakeable but not generatable purely because this check said 64.
const LTX_AXIS_MULTIPLE: u32 = 32;
const LTX_FRAMES_PER_LATENT: u32 = 8;
const LTX_MIN_FRAMES: u32 = 9;
/// The longest count the 16 GB profile has been run at. Generation had no cap at all, which is how
/// `--num-frames 481` reached a 27 GB load before failing.
const LTX_MAX_FRAMES: u32 = 193;
const LTX_MAX_PIXELS: u64 = 768 * 768;

/// The LTX block streamer's named offload profiles, mirroring
/// `ai-runtime/aiv_runtime/ltx_offload.py`. `disk-cached` is a disk-backed source with a third of
/// the 48 transformer blocks kept in pinned RAM -- roughly 6.4 GiB against `cpu`'s 19.1 GiB, and
/// the honest answer to "disk crashes here, so cpu is the only option".
const LTX_OFFLOAD_PROFILES: &[&str] = &["cpu", "disk-cached", "disk"];

/// The `--attention-backend` choices, mirroring `aiv_runtime.ltx_attention.ATTENTION_BACKENDS`.
///
/// Native leads, and is the default, because nothing is installed in the LTX venv until
/// `scripts\install-ltx-attention.ps1` has been run -- unlike H3, where the Sage kernels were
/// already on disk and merely unselected. Asking for a backend that is not there costs a line in
/// the worker log, not the run.
///
/// Flash is deliberately absent. The pinned selector reaches FlashAttention only on compute
/// capability major 9 and 10; this machine is Ada, major 8, so no Flash install could change
/// which kernel LTX runs. `ai-runtime/tests/test_ltx_attention.py` reads that guard out of the
/// pinned source so a torch bump that moves it fails loudly.
const LTX_ATTENTION_BACKENDS: &[&str] = &["native", "sage"];

fn default_ltx_attention() -> String { "native".to_owned() }

/// The LTX geometry contract as one message or nothing, so the three call sites cannot drift.
fn ltx_geometry_error(width: u32, height: u32, num_frames: u32) -> Option<String> {
    for (label, value) in [("width", width), ("height", height)] {
        if value == 0 || value % LTX_AXIS_MULTIPLE != 0 {
            return Some(format!(
                "{label} must be a positive multiple of {LTX_AXIS_MULTIPLE}; got {value}"
            ));
        }
    }
    if width as u64 * height as u64 > LTX_MAX_PIXELS {
        return Some(format!(
            "the 16 GB single-pass profile is limited to {LTX_MAX_PIXELS} pixels; {width}x{height} is {}",
            width as u64 * height as u64
        ));
    }
    if num_frames < LTX_MIN_FRAMES {
        return Some(format!("frame count must be at least {LTX_MIN_FRAMES}; got {num_frames}"));
    }
    if num_frames % LTX_FRAMES_PER_LATENT != 1 {
        return Some(format!("frame count must be 8n+1; got {num_frames}"));
    }
    if num_frames > LTX_MAX_FRAMES {
        return Some(format!(
            "{num_frames} frames is over the {LTX_MAX_FRAMES} this 16 GB profile has been run at; \
             shorten the request"
        ));
    }
    None
}

fn find_lora(root: &Path, filename: &str) -> Option<PathBuf> {
    LORA_SEARCH_DIRS
        .iter()
        .map(|dir| root.join(dir).join(filename))
        .find(|candidate| candidate.is_file())
}

fn foley_lora_path(root: &Path) -> Option<PathBuf> {
    find_lora(root, "ltx-2.3-22b-lora-foley-v2a-1.0.safetensors")
}

/// LoRA fusion is unsupported on the scaled-FP8 streaming path, so the foley adapter needs the
/// plain distilled checkpoint rather than the FP8 one the other modes prefer.
fn lora_capable_checkpoint(root: &Path) -> PathBuf {
    root.join("models/LTX-2.3/ltx-2.3-22b-distilled-1.1.safetensors")
}

fn sam2_root(root: &Path) -> PathBuf {
    root.join("models/sam2.1-hiera-large")
}

fn sam2_model_installed(root: &Path) -> bool {
    let pack = sam2_root(root);
    // The checkpoint is a .pt; the Hydra config sits beside it.
    pack.is_dir()
        && std::fs::read_dir(&pack)
            .map(|entries| {
                entries.filter_map(Result::ok).any(|entry| {
                    entry.path().extension().and_then(|ext| ext.to_str()) == Some("pt")
                })
            })
            .unwrap_or(false)
}

fn venv_python_usable(python: &Path) -> bool {
    if !python.is_file() { return false; }
    let Some(venv) = python.parent().and_then(Path::parent) else { return true; };
    let config = venv.join("pyvenv.cfg");
    let Ok(contents) = fs::read_to_string(config) else { return true; };
    let Some(home) = contents.lines().find_map(|line| line.strip_prefix("home = ")) else { return true; };
    Path::new(home.trim()).join("python.exe").is_file()
}

fn python_has_sam2(python: &Path) -> bool {
    let Some(venv) = python.parent().and_then(Path::parent) else { return false; };
    venv.join("Lib/site-packages/sam2/__init__.py").is_file()
}

/// SAM 2 can live in the small utility environment or the isolated Image Studio environment.
/// Prefer the utility venv when healthy, but never report a stale python.exe trampoline as ready.
fn sam2_python(root: &Path) -> Option<PathBuf> {
    [utility_python(root), image::python(root)].into_iter()
        .find(|python| venv_python_usable(python) && python_has_sam2(python))
}

fn sam2_state(root: &Path) -> &'static str {
    if !sam2_model_installed(root) { "model-missing" }
    else if !root.join("ai-runtime/aiv_runtime/sam2_track.py").is_file() { "adapter-missing" }
    else if ![utility_python(root), image::python(root)].iter().any(|python| venv_python_usable(python)) { "runtime-missing" }
    else if sam2_python(root).is_none() { "dependencies-missing" }
    else { "ready" }
}

fn whisper_model(root: &Path) -> PathBuf {
    root.join("models/whisper-large-v3-turbo")
}

fn whisper_installed(root: &Path) -> bool {
    let pack = whisper_model(root);
    pack.join("config.json").is_file()
        && (pack.join("model.safetensors").is_file() || pack.join("model.safetensors.index.json").is_file())
}

/// The utility models run in the project's own venv, not the LTX vendor venv.
fn utility_python(root: &Path) -> PathBuf {
    root.join(".venv/Scripts/python.exe")
}

fn moss_python(root: &Path) -> PathBuf {
    let embedded = root.join("runtimes/moss-sfx/python.exe");
    if embedded.is_file() { embedded } else { root.join("runtimes/moss-sfx/Scripts/python.exe") }
}

fn moss_model_root(root: &Path) -> PathBuf {
    root.join("models/moss-sfx-v2")
}

fn moss_sfx_state(root: &Path) -> &'static str {
    let model = moss_model_root(root);
    if !model.join("model_index.json").is_file()
        || !model.join("transformer/diffusion_pytorch_model.safetensors").is_file()
        || !model.join("vae/vae_128d_48k.pth").is_file()
    {
        "model-missing"
    } else if !root.join("ai-runtime/aiv_runtime/moss_sfx.py").is_file()
        || !root.join("ai-runtime/moss_soundeffect_v2/pipeline_moss_soundeffect.py").is_file()
    {
        "adapter-missing"
    } else if !moss_python(root).is_file() {
        "runtime-missing"
    } else if !root.join("runtimes/moss-sfx/.moss-sfx-ready").is_file() {
        "dependencies-missing"
    } else {
        "ready"
    }
}

pub fn status() -> AiRuntimeStatus {
    let root = project_root();
    let (python, source, checkpoint, gemma) = paths(&root);
    let python_ready = python.is_file();
    let ltx_source_ready = source.is_file();
    let checkpoint_ready = checkpoint.file_name().and_then(|name| name.to_str()) == Some("ltx-2.3-22b-distilled-fp8.safetensors") && checkpoint.is_file();
    let gemma_ready = gemma.join("model.safetensors.index.json").is_file();
    let utility_ready = utility_python(&root).is_file();
    let packages_ready = inference_packages_installed(&root);
    let kiwi_pack = kiwi_edit_installed(&root);
    let kiwi_adapter = root.join("ai-runtime/aiv_runtime/kiwi_edit_16gb.py").is_file();
    // Report *why* the clip editor is unavailable; "pack missing" and "dependencies missing" need
    // completely different fixes and used to be indistinguishable.
    let kiwi_edit_state = if !utility_ready {
        "venv-missing"
    } else if !kiwi_adapter {
        "adapter-missing"
    } else if !kiwi_pack {
        "pack-missing"
    } else if !packages_ready {
        "deps-missing"
    } else if !torch_cuda_ready(&root) {
        "torch-cpu"
    } else {
        "ready"
    };
    let moss_state = moss_sfx_state(&root);
    AiRuntimeStatus {
        root: root.to_string_lossy().into_owned(),
        python_ready,
        ltx_source_ready,
        checkpoint_ready,
        gemma_ready,
        single_pass_ready: python_ready && ltx_source_ready && checkpoint_ready && gemma_ready,
        retake_ready: python_ready && ltx_source_ready && checkpoint_ready && gemma_ready,
        // A complete readiness signal: the UI gates the video-to-audio button on this alone, since
        // that mode never touches the single-pass FP8 path the other modes require.
        foley_lora_ready: python_ready
            && ltx_source_ready
            && gemma_ready
            && foley_lora_path(&root).is_some()
            // Scaled-FP8 fusion landed in `scaled_fp8_lora`, so the FP8 checkpoint now satisfies
            // foley on its own; the BF16 file is only a fallback for a BF16-only install.
            && (checkpoint_ready || lora_capable_checkpoint(&root).is_file())
            && root.join("ai-runtime/aiv_runtime/ltx_foley_v2a.py").is_file(),
        moss_sfx_state: moss_state.to_owned(),
        moss_sfx_ready: moss_state == "ready",
        sam2_ready: sam2_state(&root) == "ready",
        sam2_state: sam2_state(&root).to_owned(),
        photo_depth_ready: photo::depth_ready(&root),
        photo_inpaint_ready: photo::inpaint_ready(&root),
        photo_select_ready: photo::select_ready(&root),
        kiwi_edit_ready: kiwi_edit_state == "ready",
        kiwi_edit_state: kiwi_edit_state.to_owned(),
        whisper_ready: utility_ready
            && packages_ready
            && whisper_installed(&root)
            && root.join("ai-runtime/aiv_runtime/transcribe.py").is_file(),
        minimax_h3_state: h3::state_label(&root).to_owned(),
        minimax_h3_ready: h3::state_label(&root) == "ready",
        minimax_h3_draft_ready: h3::turbo_ready(&root, "turbo-4"),
        minimax_h3_turbo_ready: h3::turbo_ready(&root, "turbo-8"),
        minimax_h3_ref_state: h3_ref::state_label(&root).to_owned(),
        minimax_h3_ref_ready: h3_ref::ready(&root),
        image_studio_state: image::state_label(&root).to_owned(),
        zit_image_ready: image::zit_ready(&root),
        zimage_base_ready: image::zimage_base_ready(&root),
        qwen_image_edit_ready: image::qwen_ready(&root),
        zimage_trainer_ready: lora::trainer_ready(&root),
        zimage_captioner_ready: lora::captioner_ready(&root),
        qwen_edit_rewriter_ready: prompt_rewrite::ready(&root),
        birefnet_ready: image::birefnet_ready(&root),
        real_esrgan_state: upscale::state_label(&root).to_owned(),
        real_esrgan_ready: upscale::ready(&root),
        real_esrgan_models: upscale::installed_models(&root),
        nvvfx_state: upscale::nvvfx_state(&root).to_owned(),
        nvvfx_ready: upscale::nvvfx_ready(&root),
        profile: "16 GB · official distilled FP8 · one-slot block streaming (disk default, CPU optional) · native upscalers disabled".into(),
    }
}

fn sfx_output_path(root: &Path, project_id: &str, prefix: &str) -> Result<PathBuf, String> {
    let project_id = Uuid::parse_str(project_id).map_err(|_| "invalid project id".to_owned())?;
    let directory = root.join("projects").join(project_id.to_string()).join("media/audio/sfx");
    fs::create_dir_all(&directory).map_err(|error| format!("could not create the SFX media directory: {error}"))?;
    Ok(directory.join(format!("{prefix}-{}.wav", Uuid::new_v4().simple())))
}

/// Prompt-designed sound. MOSS is intentionally a one-shot process: when it exits all of its
/// weights leave VRAM before LTX can acquire the same execution gate.
pub async fn generate_moss_sfx(request: MossSfxRequest) -> Result<SfxGenerationResult, String> {
    let root = project_root();
    let state = moss_sfx_state(&root);
    if state != "ready" {
        return Err(format!(
            "MOSS-SoundEffect is not ready ({state}). Run scripts\\install-moss-sfx.ps1 after installing the app-owned model and runtime."
        ));
    }
    if request.prompt.trim().is_empty() { return Err("describe the sound to generate".into()); }
    if !(0.1..=30.0).contains(&request.duration_seconds) || !request.duration_seconds.is_finite() {
        return Err("MOSS duration must be between 0.1 and 30 seconds".into());
    }
    let steps = match request.quality.as_str() {
        "draft" => 30,
        "final" => 100,
        _ => return Err("MOSS quality must be draft or final".into()),
    };
    let output_path = sfx_output_path(&root, &request.project_id, "moss")?;
    let output = Command::new(moss_python(&root))
        .arg(root.join("ai-runtime/aiv_runtime/moss_sfx.py"))
        .args(["--model-root", moss_model_root(&root).to_string_lossy().as_ref()])
        .args(["--output-path", output_path.to_string_lossy().as_ref()])
        .args(["--prompt", &request.prompt])
        .args(["--negative-prompt", &request.negative_prompt])
        .args(["--seconds", &request.duration_seconds.to_string()])
        .args(["--steps", &steps.to_string()])
        .args(["--cfg-scale", "4.0"])
        .args(["--sigma-shift", "5.0"])
        .args(["--seed", &request.seed.to_string()])
        .current_dir(&root)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        // Match the proven Digital Forge launch contract: eager execution avoids a first-run
        // Triton compile and makes the transferred pipeline reliable on Windows.
        .env("TORCHDYNAMO_DISABLE", "1")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1")
        .env("TOKENIZERS_PARALLELISM", "false")
        .stdin(Stdio::null())
        .output().await
        .map_err(|error| format!("failed to start the MOSS SFX worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        let _ = fs::remove_file(&output_path);
        return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code()));
    }
    if !output_path.is_file() { return Err("MOSS completed without creating a WAV".into()); }
    let probe = crate::media::probe(output_path.to_string_lossy().as_ref()).await?;
    let log = worker_diagnostics(&stdout, &stderr);
    record_worker_log(&root, "moss-sfx", &log);
    Ok(SfxGenerationResult {
        output_path: output_path.to_string_lossy().into_owned(), sample_rate: 48_000, channels: 1,
        duration_seconds: probe.duration_seconds, seed: request.seed, log,
    })
}

/// Picture-conditioned Foley. The existing LTX adapter writes an MP4, so this command first
/// materialises exactly the selected source range, runs that proven adapter, then extracts a
/// standalone mono WAV. From the UI onward MOSS and Foley are therefore the same kind of take.
pub async fn generate_ltx_foley(request: LtxFoleyRequest) -> Result<SfxGenerationResult, String> {
    let root = project_root();
    if !status().foley_lora_ready { return Err("LTX Picture Foley is not ready".into()); }
    if request.prompt.trim().is_empty() { return Err("describe the Foley to generate".into()); }
    if !Path::new(&request.video_path).is_file() { return Err("the Foley source video is missing".into()); }
    if request.source_in_seconds < 0.0 || request.source_out_seconds <= request.source_in_seconds {
        return Err("Picture Foley needs a valid selected range".into());
    }
    if !request.playback_rate.is_finite() || !(0.25..=4.0).contains(&request.playback_rate) {
        return Err("Foley playback rate must be between 0.25 and 4".into());
    }
    let reference_strength = match request.density.as_str() {
        "minimal" => 0.8,
        "normal" => 1.0,
        "detailed" => 1.2,
        _ => return Err("Foley density must be minimal, normal, or detailed".into()),
    };
    let work = root.join("cache/sfx").join(Uuid::new_v4().simple().to_string());
    fs::create_dir_all(&work).map_err(|error| format!("could not create the Foley work directory: {error}"))?;
    let selected_video = work.join("selected-range.mp4");
    let generated_video = work.join("foley.mp4");
    let output_path = sfx_output_path(&root, &request.project_id, "ltx-foley")?;
    let result = async {
        crate::media::materialize_clip(crate::media::MaterializeClipRequest {
            path: request.video_path.clone(),
            output_path: selected_video.to_string_lossy().into_owned(),
            source_in_seconds: request.source_in_seconds,
            source_out_seconds: request.source_out_seconds,
            playback_rate: request.playback_rate,
            kind: "video".into(),
        }).await?;
        let bridge = generate_bridge(LtxBridgeRequest {
            mode: "video-audio".into(), first_frame: None, last_frame: None,
            video_path: Some(selected_video.to_string_lossy().into_owned()),
            negative_prompt: Some(request.negative_prompt.clone()),
            output_path: generated_video.to_string_lossy().into_owned(), prompt: request.prompt.clone(),
            width: 0, height: 0, num_frames: 0, frame_rate: 24.0, seed: request.seed,
            guide_strength: 1.0, cfg_scale: 1.0, audio_cfg_scale: 1.5,
            lora_strength: 1.0, reference_strength, offload: request.offload.clone(),
            attention_backend: request.attention_backend.clone(),
        }).await?;
        let extraction = Command::new("ffmpeg")
            .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i"])
            .arg(&generated_video)
            .args(["-map", "0:a:0", "-vn", "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le"])
            .arg(&output_path)
            .stdin(Stdio::null()).output().await
            .map_err(|error| format!("failed to extract the generated Foley WAV: {error}"))?;
        if !extraction.status.success() {
            return Err(String::from_utf8_lossy(&extraction.stderr).trim().to_owned());
        }
        if !output_path.is_file() { return Err("LTX Foley completed without creating a WAV".into()); }
        let probe = crate::media::probe(output_path.to_string_lossy().as_ref()).await?;
        Ok(SfxGenerationResult {
            output_path: output_path.to_string_lossy().into_owned(), sample_rate: 48_000, channels: 1,
            duration_seconds: probe.duration_seconds, seed: request.seed, log: bridge.log,
        })
    }.await;
    let _ = fs::remove_dir_all(&work);
    if result.is_err() { let _ = fs::remove_file(&output_path); }
    result
}

/// Instruction and reference-guided clip editing through the Kiwi-Edit 5B pack.
pub async fn edit_clip(request: KiwiEditRequest) -> Result<LtxBridgeResult, String> {
    let root = project_root();
    if !status().kiwi_edit_ready {
        return Err("the Kiwi-Edit pack is not installed; run models.py install kiwi-edit --accept-licenses".into());
    }
    if !Path::new(&request.video_path).is_file() { return Err("clip source video is missing".into()); }
    if request.instruction.trim().is_empty() { return Err("an edit instruction is required".into()); }
    if request.start_time < 0.0 || request.end_time <= request.start_time {
        return Err("the clip editor requires a valid time range".into());
    }
    if !["instruction", "reference"].contains(&request.mode.as_str()) {
        return Err("invalid clip edit mode".into());
    }
    if !["stable", "long"].contains(&request.strategy.as_str()) {
        return Err("Kiwi strategy must be stable or long".into());
    }
    if !(1.0..=12.0).contains(&request.guidance_scale) {
        return Err("Kiwi guidance scale must be between 1.0 (off) and 12.0".into());
    }
    if request.mode == "reference" && request.reference_image.as_deref().unwrap_or("").is_empty() {
        return Err("reference-guided editing needs a reference image".into());
    }
    if let Some(path) = request.reference_image.as_deref().filter(|value| !value.is_empty()) {
        if !Path::new(path).is_file() {
            return Err(format!("reference image not found: {path}"));
        }
    }
    // `disk`/`cpu` are the pre-explicit-placement spellings. Translating them keeps an older
    // frontend bundle working against a newer binary instead of failing on a value we understand.
    let placement = match request.offload.as_str() {
        "balanced" | "disk" => "balanced",
        "performance" | "cpu" => "performance",
        "minimal" => "minimal",
        other => return Err(format!("unknown clip editor placement: {other}")),
    };
    let output_path = PathBuf::from(&request.output_path);
    if output_path.extension().and_then(|value| value.to_str()).map(str::to_ascii_lowercase).as_deref() != Some("mp4") {
        return Err("clip editor output must be an .mp4 file".into());
    }
    let adapter = root.join("ai-runtime/aiv_runtime/kiwi_edit_16gb.py");
    let mut command = Command::new(utility_python(&root));
    command
        .arg(adapter)
        .args(["--model-root", kiwi_edit_root(&root).to_string_lossy().as_ref()])
        .args(["--video-path", &request.video_path])
        .args(["--output-path", &request.output_path])
        .args(["--instruction", &request.instruction])
        .args(["--start-time", &request.start_time.to_string()])
        .args(["--end-time", &request.end_time.to_string()])
        .args(["--seed", &request.seed.to_string()])
        .args(["--mode", &request.mode])
        .args(["--strategy", &request.strategy])
        .args(["--steps", &request.steps.clamp(4, 60).to_string()])
        .args(["--guidance-scale", &request.guidance_scale.clamp(1.0, 12.0).to_string()])
        .args(["--negative-prompt", &request.negative_prompt])
        .args(["--offload", placement]);
    if let Some(path) = request.reference_image.as_deref().filter(|value| !value.is_empty()) {
        command.args(["--reference-image", path]);
    }
    let output = command
        .current_dir(&root).env("PYTHONUNBUFFERED", "1")
        // Windows defaults these workers to cp1252. diffusers and transformers print
        // emoji in their warnings, which then raises UnicodeEncodeError from inside a
        // logging call and takes the whole load down with it.
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1").stdin(Stdio::null()).output().await
        .map_err(|error| format!("failed to start the clip editor worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() { return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code())); }
    if !output_path.is_file() { return Err("the clip editor finished without creating output".into()); }
    let log = worker_diagnostics(&stdout, &stderr);
    record_worker_log(&root, "kiwi-edit", &log);
    Ok(LtxBridgeResult { output_path: request.output_path, log })
}

/// Whisper large-v3-turbo transcription, returned as timed cues ready for the C1 caption track.
pub async fn transcribe(request: TranscribeRequest) -> Result<Vec<TranscriptCue>, String> {
    let root = project_root();
    if !status().whisper_ready {
        return Err("Whisper is not installed; run models.py install utilities --accept-licenses".into());
    }
    if !Path::new(&request.path).is_file() { return Err("the media file for transcription is missing".into()); }
    let adapter = root.join("ai-runtime/aiv_runtime/transcribe.py");
    let mut command = Command::new(utility_python(&root));
    command
        .arg(adapter)
        .args(["--model-root", whisper_model(&root).to_string_lossy().as_ref()])
        .args(["--media-path", &request.path]);
    if let Some(language) = request.language.as_deref().filter(|value| !value.is_empty()) {
        command.args(["--language", language]);
    }
    if request.translate {
        command.arg("--translate");
    }
    let output = command
        .current_dir(&root).env("PYTHONUNBUFFERED", "1")
        // Windows defaults these workers to cp1252. diffusers and transformers print
        // emoji in their warnings, which then raises UnicodeEncodeError from inside a
        // logging call and takes the whole load down with it.
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1").stdin(Stdio::null()).output().await
        .map_err(|error| format!("failed to start the transcription worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() { return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code())); }
    // The adapter prints progress lines and then a single JSON document on the last line.
    let payload = stdout.lines().rev().find(|line| line.trim_start().starts_with('{'))
        .ok_or("the transcription worker returned no JSON result")?;
    let parsed: serde_json::Value = serde_json::from_str(payload)
        .map_err(|error| format!("invalid transcription output: {error}"))?;
    let cues = parsed["cues"].as_array().cloned().unwrap_or_default();
    Ok(cues.iter().filter_map(|cue| {
        let text = cue["text"].as_str()?.trim().to_owned();
        let start = cue["start"].as_f64()?;
        let end = cue["end"].as_f64()?;
        (end > start && !text.is_empty()).then_some(TranscriptCue { start_seconds: start, end_seconds: end, text })
    }).collect())
}

pub async fn retake_bridge(request: LtxRetakeRequest) -> Result<LtxRetakeResult, String> {
    let runtime = status();
    if !runtime.retake_ready { return Err("LTX retake runtime is incomplete".into()); }
    if !Path::new(&request.video_path).is_file() { return Err("retake source video is missing".into()); }
    if request.prompt.trim().is_empty() { return Err("retake prompt cannot be empty".into()); }
    if request.start_time < 0.0 || request.end_time <= request.start_time { return Err("retake requires a valid time range".into()); }
    if !["video-audio", "video", "audio"].contains(&request.mode.as_str()) { return Err("invalid retake mode".into()); }
    if !["fast", "experimental-guided-distilled"].contains(&request.guidance.as_str()) {
        return Err("guidance must be fast or experimental-guided-distilled".into());
    }
    if !LTX_ATTENTION_BACKENDS.contains(&request.attention_backend.as_str()) {
        return Err(format!("attention backend must be one of {}", LTX_ATTENTION_BACKENDS.join(", ")));
    }
    for (label, value) in [("cfg scale", request.cfg_scale), ("audio cfg scale", request.audio_cfg_scale)] {
        if let Some(value) = value {
            if !(1.0..=10.0).contains(&value) {
                return Err(format!("{label} must be between 1.0 (unguided) and 10.0"));
            }
        }
    }
    if !LTX_OFFLOAD_PROFILES.contains(&request.offload.as_str()) {
        return Err(format!("offload must be one of {}", LTX_OFFLOAD_PROFILES.join(", ")));
    }
    let output_path = PathBuf::from(&request.output_path);
    if output_path.extension().and_then(|value| value.to_str()).map(str::to_ascii_lowercase).as_deref() != Some("mp4") {
        return Err("LTX retake output must be an .mp4 file".into());
    }
    let root = project_root();
    let (python, _, checkpoint, gemma) = paths(&root);
    let adapter = root.join("ai-runtime/aiv_runtime/ltx_retake_16gb.py");
    if !adapter.is_file() { return Err("16 GB LTX retake adapter is missing".into()); }
    let key = retake_worker_key(&request);
    let request_line = serde_json::to_string(&request)
        .map_err(|error| format!("failed to serialize retake request: {error}"))?;
    let worker_slot = RETAKE_WORKER.get_or_init(|| Mutex::new(None));
    let mut slot = worker_slot.lock().await;
    let restart = slot.as_ref().map(|worker| worker.key != key).unwrap_or(false);
    if restart {
        if let Some(mut previous) = slot.take() {
            let _ = previous.child.kill().await;
            let _ = previous.child.wait().await;
        }
    }
    if slot.is_none() {
        let log_dir = root.join(".aiv");
        fs::create_dir_all(&log_dir).map_err(|error| format!("failed to create worker log directory: {error}"))?;
        let stderr_path = log_dir.join("ltx-retake-resident.stderr.log");
        let stderr_file = fs::OpenOptions::new().create(true).append(true).open(&stderr_path)
            .map_err(|error| format!("failed to open resident retake log: {error}"))?;
        let mut command = Command::new(python);
        command
            .arg(adapter)
            .args(["--checkpoint-path", checkpoint.to_string_lossy().as_ref()])
            .args(["--gemma-root", gemma.to_string_lossy().as_ref()])
            // The parser requires request fields even in server mode; the first JSON request
            // replaces all of these placeholders before validation or inference.
            .args(["--video-path", &request.video_path])
            .args(["--output-path", &request.output_path])
            .args(["--prompt", &request.prompt])
            .args(["--start-time", &request.start_time.to_string()])
            .args(["--end-time", &request.end_time.to_string()])
            .args(["--seed", &request.seed.to_string()])
            .args(["--negative-prompt", &request.negative_prompt])
            .args(["--guidance", &request.guidance])
            .args(["--attention-backend", &request.attention_backend])
            .args(["--mode", &request.mode])
            .args(["--offload", &request.offload])
            .arg("--server")
            .current_dir(&root)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUTF8", "1")
            .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::from(stderr_file));
        for (flag, value) in [
            ("--cfg-scale", request.cfg_scale),
            ("--audio-cfg-scale", request.audio_cfg_scale),
        ] {
            if let Some(value) = value {
                let rendered = value.to_string();
                command.args([flag, rendered.as_str()]);
            }
        }
        let mut child = command.spawn().map_err(|error| format!("failed to start resident LTX retake worker: {error}"))?;
        let stdin = child.stdin.take().ok_or("resident LTX retake worker has no stdin")?;
        let stdout = child.stdout.take().ok_or("resident LTX retake worker has no stdout")?;
        let mut worker = ResidentRetakeWorker { key: key.clone(), child, stdin, stdout: BufReader::new(stdout), stderr_path };
        let mut ready = String::new();
        worker.stdout.read_line(&mut ready).await
            .map_err(|error| format!("resident LTX retake worker did not start: {error}"))?;
        let ready_json: serde_json::Value = serde_json::from_str(ready.trim())
            .map_err(|error| format!("resident LTX retake worker returned an invalid handshake: {error}"))?;
        if ready_json["ready"].as_bool() != Some(true) {
            return Err("resident LTX retake worker did not become ready".into());
        }
        *slot = Some(worker);
    }
    let worker = slot.as_mut().expect("resident worker was just initialized");
    let log_start = fs::metadata(&worker.stderr_path).map(|value| value.len()).unwrap_or(0) as usize;
    worker.stdin.write_all(request_line.as_bytes()).await
        .map_err(|error| format!("failed to send request to resident LTX retake worker: {error}"))?;
    worker.stdin.write_all(b"\n").await
        .map_err(|error| format!("failed to send request to resident LTX retake worker: {error}"))?;
    worker.stdin.flush().await
        .map_err(|error| format!("failed to send request to resident LTX retake worker: {error}"))?;
    let parsed = loop {
        let mut line = String::new();
        let read = worker.stdout.read_line(&mut line).await
            .map_err(|error| format!("failed to read resident LTX retake result: {error}"))?;
        if read == 0 {
            *slot = None;
            return Err("resident LTX retake worker exited before returning a result".into());
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line.trim()) {
            if value.get("ok").is_some() { break value; }
        }
    };
    let stderr_bytes = fs::read(&worker.stderr_path).unwrap_or_default();
    let stderr = String::from_utf8_lossy(&stderr_bytes[log_start.min(stderr_bytes.len())..]).into_owned();
    let log = worker_diagnostics("", &stderr);
    if parsed["ok"].as_bool() != Some(true) {
        let message = parsed["error"].as_str().unwrap_or("LTX retake worker failed");
        record_worker_log(&root, "ltx-retake", &log);
        return Err(if log.is_empty() { message.to_owned() } else { format!("{message}\n{log}") });
    }
    if !output_path.is_file() { return Err("LTX retake completed without creating output".into()); }
    record_worker_log(&root, "ltx-retake", &log);
    let number = |name: &str| parsed[name].as_f64().ok_or_else(|| format!("retake result omitted {name}"));
    let integer = |name: &str| parsed[name].as_u64().ok_or_else(|| format!("retake result omitted {name}"));
    let preview_path = parsed["previewPath"].as_str().ok_or("retake result omitted previewPath")?.to_owned();
    if !Path::new(&preview_path).is_file() { return Err("LTX retake completed without creating its seam preview".into()); }
    Ok(LtxRetakeResult {
        output_path: request.output_path,
        preview_path,
        log,
        effective_start_time: number("effectiveStartTime")?,
        effective_end_time: number("effectiveEndTime")?,
        start_frame: integer("startFrame")?,
        end_frame: integer("endFrame")?,
        fps_numerator: integer("fpsNumerator")? as u32,
        fps_denominator: integer("fpsDenominator")? as u32,
    })
}

pub async fn generate_bridge(request: LtxBridgeRequest) -> Result<LtxBridgeResult, String> {
    let runtime = status();
    if !runtime.single_pass_ready {
        return Err("LTX runtime is incomplete; open the AI panel for component status".into());
    }
    // Each mode needs different conditioning; check exactly what this one uses.
    let required: &[(&str, &Option<String>)] = match request.mode.as_str() {
        "first-last" => &[("first frame", &request.first_frame), ("last frame", &request.last_frame)],
        "image" => &[("first frame", &request.first_frame)],
        "text" => &[],
        "video-audio" => &[("source video", &request.video_path)],
        other => return Err(format!("unknown LTX mode: {other}")),
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
    for (label, value) in [("CFG", request.cfg_scale), ("audio CFG", request.audio_cfg_scale)] {
        if !(1.0..=10.0).contains(&value) {
            return Err(format!("{label} scale must be between 1.0 (unguided) and 10.0"));
        }
    }
    for (label, value) in [("LoRA", request.lora_strength), ("reference", request.reference_strength)] {
        if !(0.0..=2.0).contains(&value) || value <= 0.0 {
            return Err(format!("{label} strength must be greater than 0 and at most 2.0"));
        }
    }
    let generates_video = request.mode != "video-audio";
    if generates_video {
        if let Some(message) = ltx_geometry_error(request.width, request.height, request.num_frames) {
            return Err(message);
        }
    }
    if !LTX_OFFLOAD_PROFILES.contains(&request.offload.as_str()) {
        return Err(format!("offload must be one of {}", LTX_OFFLOAD_PROFILES.join(", ")));
    }
    if !LTX_ATTENTION_BACKENDS.contains(&request.attention_backend.as_str()) {
        return Err(format!("attention backend must be one of {}", LTX_ATTENTION_BACKENDS.join(", ")));
    }
    let output_path = PathBuf::from(&request.output_path);
    if output_path.extension().and_then(|value| value.to_str()).map(str::to_ascii_lowercase).as_deref() != Some("mp4") {
        return Err("LTX output must be an .mp4 file".into());
    }

    let root = project_root();
    let (python, _, checkpoint, gemma) = paths(&root);

    // Video-conditioned audio is a different pipeline with different weight requirements: it needs
    // the Foley IC-LoRA for the conditioning behaviour, and the plain distilled checkpoint because
    // the scaled-FP8 streaming path cannot fuse LoRAs.
    if request.mode == "video-audio" {
        let adapter = root.join("ai-runtime/aiv_runtime/ltx_foley_v2a.py");
        if !adapter.is_file() {
            return Err("the foley adapter is missing".into());
        }
        let Some(lora) = foley_lora_path(&root) else {
            return Err(format!(
                "ltx-2.3-22b-lora-foley-v2a-1.0.safetensors was not found in any of: {}. \
                 Without it the model cannot see the picture and the audio will not sync.",
                LORA_SEARCH_DIRS.join(", ")
            ));
        };
        // The adapter detects a scaled-FP8 checkpoint and routes the IC-LoRA through
        // `scaled_fp8_lora`, which supplies the companion-key write-back the stock streaming
        // provider lacks. So take whatever `paths()` chose and only fall back to the BF16 file.
        let lora_checkpoint = if checkpoint.is_file() { checkpoint.clone() } else { lora_capable_checkpoint(&root) };
        if !lora_checkpoint.is_file() {
            return Err("foley needs an LTX checkpoint in models/LTX-2.3".into());
        }
        let video_path = request.video_path.as_deref().unwrap_or_default();
        let output = Command::new(python)
            .arg(adapter)
            .args(["--checkpoint-path", lora_checkpoint.to_string_lossy().as_ref()])
            .args(["--gemma-root", gemma.to_string_lossy().as_ref()])
            .args(["--lora-path", lora.to_string_lossy().as_ref()])
            .args(["--video-path", video_path])
            .args(["--output-path", &request.output_path])
            .args(["--prompt", &request.prompt])
            .args(["--negative-prompt", request.negative_prompt.as_deref().unwrap_or("")])
            .args(["--seed", &request.seed.to_string()])
            .args(["--cfg-scale", &request.cfg_scale.to_string()])
            .args(["--audio-cfg-scale", &request.audio_cfg_scale.to_string()])
            .args(["--lora-strength", &request.lora_strength.to_string()])
            .args(["--reference-strength", &request.reference_strength.to_string()])
            .args(["--offload", &request.offload])
            .args(["--attention-backend", &request.attention_backend])
            .current_dir(&root)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUTF8", "1")
            .stdin(Stdio::null())
            .output()
            .await
            .map_err(|error| format!("failed to start the foley worker: {error}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        if !output.status.success() { return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code())); }
        if !PathBuf::from(&request.output_path).is_file() {
            return Err("the foley worker finished without creating output".into());
        }
        let log = worker_diagnostics(&stdout, &stderr);
        record_worker_log(&root, "ltx-foley", &log);
        return Ok(LtxBridgeResult { output_path: request.output_path, log });
    }

    let adapter = root.join("ai-runtime/aiv_runtime/ltx_single_stage.py");
    if !adapter.is_file() {
        return Err("single-pass LTX adapter is missing".into());
    }
    let mut command = Command::new(python);
    command
        .arg(adapter)
        .args(["--checkpoint-path", checkpoint.to_string_lossy().as_ref()])
        .args(["--gemma-root", gemma.to_string_lossy().as_ref()])
        .args(["--mode", &request.mode])
        .args(["--output-path", &request.output_path])
        .args(["--prompt", &request.prompt])
        .args(["--width", &request.width.to_string()])
        .args(["--height", &request.height.to_string()])
        .args(["--num-frames", &request.num_frames.to_string()])
        .args(["--frame-rate", &request.frame_rate.to_string()])
        .args(["--seed", &request.seed.to_string()])
        .args(["--guide-strength", &request.guide_strength.to_string()])
        .args(["--cfg-scale", &request.cfg_scale.to_string()])
        .args(["--audio-cfg-scale", &request.audio_cfg_scale.to_string()])
        .args(["--offload", &request.offload])
        .args(["--attention-backend", &request.attention_backend]);
    for (flag, value) in [
        ("--first-frame", &request.first_frame),
        ("--last-frame", &request.last_frame),
        // No `--video-path`: the only mode that carried one was `video-audio`, and that route
        // returns above through the foley adapter. The single-pass worker no longer accepts it.
        ("--negative-prompt", &request.negative_prompt),
    ] {
        if let Some(path) = value.as_deref().filter(|item| !item.is_empty()) {
            command.args([flag, path]);
        }
    }
    let output = command
        .current_dir(&root)
        .env("PYTHONUNBUFFERED", "1")
        // Windows defaults these workers to cp1252. diffusers and transformers print
        // emoji in their warnings, which then raises UnicodeEncodeError from inside a
        // logging call and takes the whole load down with it.
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("failed to start LTX worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code()));
    }
    if !output_path.is_file() {
        return Err("LTX worker completed without creating its output".into());
    }
    let log = worker_diagnostics(&stdout, &stderr);
    record_worker_log(&root, "ltx-generate", &log);
    Ok(LtxBridgeResult { output_path: request.output_path, log })
}


/// Last-frame image-to-video extension. Frame extraction and boundary trimming stay native so the
/// UI never conditions LTX with a reduced thumbnail and never inserts the duplicated guide frame.
pub async fn extend_clip(request: LtxClipExtendRequest) -> Result<LtxBridgeResult, String> {
    if !Path::new(&request.video_path).is_file() { return Err("selected clip source is missing".into()); }
    if request.source_in_seconds < 0.0 || request.source_out_seconds <= request.source_in_seconds {
        return Err("clip extension requires a valid visible source range".into());
    }
    if request.frame_rate <= 0.0 { return Err("clip extension requires a positive output frame rate".into()); }
    let final_output = PathBuf::from(&request.output_path);
    if final_output.extension().and_then(|item| item.to_str()).map(str::to_ascii_lowercase).as_deref() != Some("mp4") {
        return Err("clip extension output must be an MP4 file".into());
    }
    let scratch = std::env::temp_dir().join(format!("aiv-clip-extend-{}", Uuid::new_v4()));
    fs::create_dir_all(&scratch).map_err(|error| error.to_string())?;
    let guide = scratch.join("last-visible-frame.png");
    let raw = scratch.join("raw-extension.mp4");
    let result = async {
        crate::media::extract_last_visible_frame(
            &request.video_path,
            request.source_in_seconds,
            request.source_out_seconds,
            &guide,
        ).await?;
        let generated = generate_bridge(LtxBridgeRequest {
            mode: "image".into(),
            first_frame: Some(guide.to_string_lossy().into_owned()),
            last_frame: None,
            video_path: None,
            negative_prompt: request.negative_prompt.clone(),
            output_path: raw.to_string_lossy().into_owned(),
            prompt: request.prompt.clone(),
            width: request.width,
            height: request.height,
            num_frames: request.num_frames,
            frame_rate: request.frame_rate,
            seed: request.seed,
            guide_strength: request.guide_strength,
            cfg_scale: request.cfg_scale,
            audio_cfg_scale: request.audio_cfg_scale,
            lora_strength: 1.0,
            reference_strength: 1.0,
            offload: request.offload.clone(),
            attention_backend: request.attention_backend.clone(),
        }).await?;
        crate::media::trim_extension_boundary(&raw, &final_output, 1.0 / request.frame_rate).await?;
        Ok(LtxBridgeResult { output_path: request.output_path.clone(), log: generated.log })
    }.await;
    let _ = fs::remove_dir_all(&scratch);
    result
}

/// Clip-to-clip bridging. Both conditioning frames are extracted natively and losslessly from the
/// clips' visible ranges, so LTX is pinned to the exact frames the timeline shows. The generated
/// bridge starts and ends on duplicates of frames already in the edit, so both boundaries are
/// trimmed before the caller ripple-inserts it between the pair.
pub async fn bridge_clips(request: LtxClipBridgeRequest) -> Result<LtxBridgeResult, String> {
    if !Path::new(&request.from_video_path).is_file() { return Err("the outgoing clip's source is missing".into()); }
    if !Path::new(&request.to_video_path).is_file() { return Err("the incoming clip's source is missing".into()); }
    if request.from_source_in_seconds < 0.0 || request.from_source_out_seconds <= request.from_source_in_seconds {
        return Err("the outgoing clip has no visible source range".into());
    }
    if request.to_source_in_seconds < 0.0 {
        return Err("the incoming clip has an invalid source in point".into());
    }
    if request.frame_rate <= 0.0 { return Err("clip bridging requires a positive output frame rate".into()); }
    let final_output = PathBuf::from(&request.output_path);
    if final_output.extension().and_then(|item| item.to_str()).map(str::to_ascii_lowercase).as_deref() != Some("mp4") {
        return Err("clip bridge output must be an MP4 file".into());
    }
    let scratch = std::env::temp_dir().join(format!("aiv-clip-bridge-{}", Uuid::new_v4()));
    fs::create_dir_all(&scratch).map_err(|error| error.to_string())?;
    let first = scratch.join("outgoing-last-frame.png");
    let last = scratch.join("incoming-first-frame.png");
    let raw = scratch.join("raw-bridge.mp4");
    let result = async {
        crate::media::extract_last_visible_frame(
            &request.from_video_path,
            request.from_source_in_seconds,
            request.from_source_out_seconds,
            &first,
        ).await?;
        crate::media::extract_first_visible_frame(
            &request.to_video_path,
            request.to_source_in_seconds,
            &last,
        ).await?;
        let generated = generate_bridge(LtxBridgeRequest {
            mode: "first-last".into(),
            first_frame: Some(first.to_string_lossy().into_owned()),
            last_frame: Some(last.to_string_lossy().into_owned()),
            video_path: None,
            negative_prompt: request.negative_prompt.clone(),
            output_path: raw.to_string_lossy().into_owned(),
            prompt: request.prompt.clone(),
            width: request.width,
            height: request.height,
            num_frames: request.num_frames,
            frame_rate: request.frame_rate,
            seed: request.seed,
            guide_strength: request.guide_strength,
            cfg_scale: request.cfg_scale,
            audio_cfg_scale: request.audio_cfg_scale,
            lora_strength: 1.0,
            reference_strength: 1.0,
            offload: request.offload.clone(),
            attention_backend: request.attention_backend.clone(),
        }).await?;
        crate::media::trim_bridge_boundaries(&raw, &final_output, 1.0 / request.frame_rate).await?;
        Ok(LtxBridgeResult { output_path: request.output_path.clone(), log: generated.log })
    }.await;
    let _ = fs::remove_dir_all(&scratch);
    result
}

/// SAM 2 click-to-track masking over a clip range.
pub async fn track_sam2(request: Sam2Request) -> Result<LtxBridgeResult, String> {
    let root = project_root();
    if !status().sam2_ready {
        return Err(format!("SAM 2 is unavailable ({})", sam2_state(&root)));
    }
    if !Path::new(&request.video_path).is_file() { return Err("source clip is missing".into()); }
    if request.start_time < 0.0 || request.end_time <= request.start_time {
        return Err("tracking requires a valid time range".into());
    }
    if request.points.is_empty() { return Err("click the subject to track first".into()); }
    if !request.points.iter().any(|point| point.include) {
        return Err("at least one point must be an include point".into());
    }
    if !["matte", "cutout", "removed"].contains(&request.output_kind.as_str()) {
        return Err("output kind must be matte, cutout or removed".into());
    }
    let output_path = PathBuf::from(&request.output_path);
    if output_path.extension().and_then(|value| value.to_str()).map(str::to_ascii_lowercase).as_deref() != Some("mp4") {
        return Err("SAM 2 output must be an .mp4 file".into());
    }
    let adapter = root.join("ai-runtime/aiv_runtime/sam2_track.py");
    let python = sam2_python(&root).ok_or_else(|| format!("SAM 2 is unavailable ({})", sam2_state(&root)))?;
    let mut command = Command::new(python);
    command
        .arg(adapter)
        .args(["--model-root", sam2_root(&root).to_string_lossy().as_ref()])
        .args(["--video-path", &request.video_path])
        .args(["--output-path", &request.output_path])
        .args(["--start-time", &request.start_time.to_string()])
        .args(["--end-time", &request.end_time.to_string()])
        .args(["--output-kind", &request.output_kind])
        .args(["--feather", &request.feather.min(64).to_string()]);
    for point in &request.points {
        command.args(["--point", &format!("{},{},{}", point.x, point.y, if point.include { 1 } else { 0 })]);
    }
    let output = command
        .current_dir(&root)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("failed to start the SAM 2 worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() { return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code())); }
    if !output_path.is_file() { return Err("the SAM 2 worker finished without creating output".into()); }
    let log = worker_diagnostics(&stdout, &stderr);
    record_worker_log(&root, "sam2", &log);
    Ok(LtxBridgeResult { output_path: request.output_path, log })
}

/// Track once, keep the lossless matte sequence, and let every look/preview reuse it.
pub async fn cache_mask_track(request: MaskTrackRequest) -> Result<MaskTrackResult, String> {
    let root = project_root();
    if request.model != "sam2" { return Err("cached interactive tracking currently uses SAM 2".into()); }
    let source = Path::new(&request.video_path);
    if !source.is_file() { return Err("source clip is missing".into()); }
    if request.start_time < 0.0 || request.end_time <= request.start_time {
        return Err("mask tracking requires a valid time range".into());
    }
    if !request.points.iter().any(|point| point.include) {
        return Err("SAM 2 needs at least one include point".into());
    }
    let metadata = fs::metadata(source).map_err(|error| format!("cannot inspect source clip: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(source.to_string_lossy().as_bytes());
    hasher.update(metadata.len().to_le_bytes());
    if let Ok(modified) = metadata.modified() {
        if let Ok(since_epoch) = modified.duration_since(std::time::UNIX_EPOCH) {
            hasher.update(since_epoch.as_nanos().to_le_bytes());
        }
    }
    hasher.update(request.start_time.to_le_bytes());
    hasher.update(request.end_time.to_le_bytes());
    hasher.update(serde_json::to_vec(&request.points).map_err(|error| error.to_string())?);
    let cache_id = format!("{:x}", hasher.finalize());
    let cache_dir = root.join(".aiv/cache/masks").join(&cache_id);
    let manifest_path = cache_dir.join("manifest.json");
    if manifest_path.is_file() {
        let mut cached: MaskTrackResult = serde_json::from_str(
            &fs::read_to_string(&manifest_path).map_err(|error| format!("cannot read mask cache: {error}"))?,
        ).map_err(|error| format!("mask cache manifest is invalid: {error}"))?;
        cached.reused = true;
        cached.log = "Reused the existing SAM 2 matte cache.".to_owned();
        return Ok(cached);
    }
    if !status().sam2_ready { return Err(format!("SAM 2 is unavailable ({})", sam2_state(&root))); }
    fs::create_dir_all(&cache_dir).map_err(|error| format!("cannot create mask cache: {error}"))?;
    let adapter = root.join("ai-runtime/aiv_runtime/mask_cache.py");
    if !adapter.is_file() { return Err("the reusable mask-cache worker is missing".into()); }
    let python = sam2_python(&root).ok_or_else(|| format!("SAM 2 is unavailable ({})", sam2_state(&root)))?;
    let mut command = Command::new(python);
    command.arg(adapter)
        .args(["--model-root", sam2_root(&root).to_string_lossy().as_ref()])
        .args(["--video-path", &request.video_path])
        .args(["--cache-dir", cache_dir.to_string_lossy().as_ref()])
        .args(["--cache-id", &cache_id])
        .args(["--start-time", &request.start_time.to_string()])
        .args(["--end-time", &request.end_time.to_string()]);
    for point in &request.points {
        command.args(["--point", &format!("{},{},{}", point.x, point.y, if point.include { 1 } else { 0 })]);
    }
    let output = command.current_dir(&root)
        .env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8").env("PYTHONUTF8", "1")
        .stdin(Stdio::null()).output().await
        .map_err(|error| format!("failed to start the SAM 2 cache worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code()));
    }
    let json_line = stdout.lines().rev().find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| "the SAM 2 cache worker returned no manifest".to_owned())?;
    let mut result: MaskTrackResult = serde_json::from_str(json_line)
        .map_err(|error| format!("the SAM 2 cache manifest was unreadable: {error}"))?;
    result.log = worker_diagnostics(&stdout, &stderr);
    record_worker_log(&root, "sam2-cache", &result.log);
    Ok(result)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatteStabilizeRequest {
    pub cache_path: String,
    pub temporal_smooth: u32,
    pub cleanup: u32,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatteStabilizeResult {
    pub cache_path: String,
    pub temporal_smooth: u32,
    pub cleanup: u32,
    #[serde(default)]
    pub reused: bool,
}

/// VFX Phase 7: bake a cached matte's temporal smoothing and island cleanup once (the old renderer
/// redid it on every render), into `<cache>/stable-s<N>-c<M>/masks/`, so the compositor can read any
/// frame of it directly. The folder is reused when it exists.
pub async fn stabilize_mask_matte(request: MatteStabilizeRequest) -> Result<MatteStabilizeResult, String> {
    let root = project_root();
    let cache = PathBuf::from(&request.cache_path);
    let caches = root.join(".aiv/cache/masks");
    let inside = cache.canonicalize().ok().zip(caches.canonicalize().ok()).is_some_and(|(cache, caches)| cache.starts_with(caches));
    if !inside { return Err("the matte cache is not in this project's mask cache".into()); }
    if !cache.join("manifest.json").is_file() { return Err("the matte cache is missing or incomplete".into()); }
    let smooth = request.temporal_smooth.min(5);
    let cleanup = request.cleanup.min(2000);
    let stable = cache.join(format!("stable-s{smooth}-c{cleanup}"));
    if stable.join("manifest.json").is_file() {
        return Ok(MatteStabilizeResult { cache_path: stable.to_string_lossy().into_owned(), temporal_smooth: smooth, cleanup, reused: true });
    }
    let adapter = root.join("ai-runtime/aiv_runtime/mask_cache.py");
    let python = sam2_python(&root).ok_or_else(|| format!("SAM 2 is unavailable ({})", sam2_state(&root)))?;
    let output = Command::new(python)
        .arg(adapter)
        .args(["--stabilize", "--cache-dir", cache.to_string_lossy().as_ref()])
        .args(["--temporal-smooth", &smooth.to_string(), "--cleanup", &cleanup.to_string()])
        .current_dir(&root)
        .env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8").env("PYTHONUTF8", "1")
        .stdin(Stdio::null()).output().await
        .map_err(|error| format!("failed to start the matte worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() { return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code())); }
    if !stable.join("manifest.json").is_file() { return Err("the matte worker finished without a stabilised matte".into()); }
    record_worker_log(&root, "sam2-stabilize", &worker_diagnostics(&stdout, &stderr));
    Ok(MatteStabilizeResult { cache_path: stable.to_string_lossy().into_owned(), temporal_smooth: smooth, cleanup, reused: false })
}

/// Acquire a BiRefNet or SAM 2 mask, composite one focused VFX recipe, and return a reviewable
/// range clip. The selected model decides which isolated Python environment starts; they are never
/// loaded together.
pub async fn render_mask_effect(request: MaskEffectRequest) -> Result<LtxBridgeResult, String> {
    const EFFECTS: &[&str] = &[
        "glow", "outline", "aura", "neon-edge", "electric", "energy-shield",
        "spotlight", "background-blur", "background-pixelate", "silhouette", "hologram",
        "heat-vision", "rgb-split", "subject-echo", "fireball", "disintegrate", "comic-pop",
        "power-up", "animated-matte",
    ];
    const REACTIVE: &[&str] = &["off", "amplitude", "bass", "transients"];
    const TRANSITIONS: &[&str] = &[
        "none", "fade", "edges-to-center", "center-to-edges", "top-to-bottom",
        "bottom-to-top", "explosion", "power-up",
    ];
    const BLENDS: &[&str] = &["normal", "screen", "add", "multiply", "overlay", "soft-light"];
    const KEYS: &[&str] = &["alpha", "luma", "chroma"];
    let root = project_root();
    if !Path::new(&request.video_path).is_file() { return Err("source clip is missing".into()); }
    if request.start_time < 0.0 || request.end_time <= request.start_time {
        return Err("mask effects require a valid time range".into());
    }
    if !["sam2", "birefnet"].contains(&request.model.as_str()) {
        return Err("mask model must be sam2 or birefnet".into());
    }
    if !EFFECTS.contains(&request.effect.as_str()) { return Err("unknown mask effect preset".into()); }
    let reactive = if request.audio_reactive.is_empty() { "off" } else { request.audio_reactive.as_str() };
    if !REACTIVE.contains(&reactive) { return Err("unknown audio-reactive mode".into()); }
    if !TRANSITIONS.contains(&request.transition_in.as_str())
        || !TRANSITIONS.contains(&request.transition_out.as_str())
    {
        return Err("unknown mask-effect transition".into());
    }
    if !BLENDS.contains(&request.blend_mode.as_str()) { return Err("unknown VFX blend mode".into()); }
    if !KEYS.contains(&request.key_mode.as_str()) { return Err("unknown VFX key mode".into()); }
    if !["subject", "effect"].contains(&request.transition_origin.as_str()) {
        return Err("transition origin must be subject or effect".into());
    }
    if !["behind", "inside", "front"].contains(&request.placement.as_str()) {
        return Err("mask effect placement must be behind, inside or front".into());
    }
    if !["subject", "flame", "rings", "burst", "clip"].contains(&request.matte_style.as_str()) {
        return Err("unknown effect matte style".into());
    }
    if !["original", "gradient"].contains(&request.effect_clip_color.as_str()) {
        return Err("effect clip color mode must be original or gradient".into());
    }
    if request.matte_style == "clip"
        && (request.effect_clip_path.is_empty() || !Path::new(&request.effect_clip_path).is_file())
    {
        return Err("the imported effect clip is missing".into());
    }
    if !(0.0..=2.0).contains(&request.intensity)
        || !(0.0..=2.0).contains(&request.reactive_amount)
        || !(0.0..=1.0).contains(&request.reactive_smoothing)
        || !(0.1..=3.0).contains(&request.effect_speed)
        || !(0.0..=2.0).contains(&request.particle_amount)
        || request.temporal_smooth > 5
        || request.cleanup > 2_000
        || !(0.05..=3.0).contains(&request.transition_in_duration)
        || !(0.05..=3.0).contains(&request.transition_out_duration)
        || !(0.0..=1.0).contains(&request.effect_opacity)
        || !(0.0..=1.0).contains(&request.effect_x)
        || !(0.0..=1.0).contains(&request.effect_y)
        || !(10.0..=300.0).contains(&request.effect_scale)
        || !(-180.0..=180.0).contains(&request.effect_rotation)
        || !(0.0..=1.0).contains(&request.pulse_amount)
        || !(0.1..=8.0).contains(&request.pulse_rate)
        || request.gradient.stops.len() < 2
        || !(0.01..=4.0).contains(&request.gradient.scale)
        || !(1.0..=12.0).contains(&request.gradient.repeat)
        || !(-0.5..=1.5).contains(&request.gradient.cx)
        || !(-0.5..=1.5).contains(&request.gradient.cy)
        || !(-1.0..=1.0).contains(&request.gradient.offset)
        || !(0.25..=4.0).contains(&request.gradient.ease)
        || !(0.0..=1.0).contains(&request.gradient.dither)
        || !(0.0..=1.0).contains(&request.key_threshold)
        || !(0.0..=1.0).contains(&request.key_softness)
        || !(0.0..=1.0).contains(&request.rear_amount)
        || !(0.0..=1.0).contains(&request.subject_amount)
        || !(0.0..=1.0).contains(&request.front_amount)
        || !(0.0..=24.0).contains(&request.motion_blur)
        || !(0.0..=1.0).contains(&request.light_wrap)
        || !(0.0..=1.0).contains(&request.color_spill)
        || !(0.0..=1.0).contains(&request.grain)
        || !(0.0..=24.0).contains(&request.displacement)
        || !(0.0..=16.0).contains(&request.occlusion_feather)
        || !(-8..=8).contains(&request.occlusion_expand)
        || (request.preview_width != 0 && !(320..=1280).contains(&request.preview_width))
        || request.keyframes.iter().any(|keyframe| !(0.0..=1.0).contains(&keyframe.at))
        || request.keyframes.iter().any(|keyframe|
            !["linear", "smooth", "ease-in", "ease-out"].contains(&keyframe.easing.as_str())
            || !(0.0..=1.0).contains(&keyframe.values.effect_x)
            || !(0.0..=1.0).contains(&keyframe.values.effect_y)
            || !(10.0..=300.0).contains(&keyframe.values.effect_scale)
            || !(-180.0..=180.0).contains(&keyframe.values.effect_rotation)
            || !(0.0..=1.0).contains(&keyframe.values.effect_opacity)
            || !(0.0..=2.0).contains(&keyframe.values.intensity))
    {
        return Err("mask effect controls are outside the supported range".into());
    }
    let cached_matte = !request.matte_cache_path.is_empty()
        && Path::new(&request.matte_cache_path).join("manifest.json").is_file()
        && Path::new(&request.matte_cache_path).join("masks").is_dir();
    if !request.matte_cache_path.is_empty() && !cached_matte {
        return Err("the cached matte is missing or incomplete; track the subject again".into());
    }
    if request.model == "sam2" {
        if !cached_matte {
            if !status().sam2_ready { return Err(format!("SAM 2 is unavailable ({})", sam2_state(&root))); }
            if !request.points.iter().any(|point| point.include) {
                return Err("SAM 2 needs at least one include point".into());
            }
        }
    } else if !image::birefnet_ready(&root) {
        return Err("the BiRefNet model or Image Studio runtime is missing".into());
    }
    let output_path = PathBuf::from(&request.output_path);
    if output_path.extension().and_then(|value| value.to_str()).map(str::to_ascii_lowercase).as_deref() != Some("mp4") {
        return Err("mask effects output must be an .mp4 file".into());
    }
    let adapter = root.join("ai-runtime/aiv_runtime/mask_effects.py");
    if !adapter.is_file() { return Err("the Mask Effects adapter is missing".into()); }
    let python = if request.model == "sam2" {
        sam2_python(&root).ok_or_else(|| format!("SAM 2 is unavailable ({})", sam2_state(&root)))?
    } else { image::python(&root) };
    let model_root = if request.model == "sam2" { sam2_root(&root) } else { root.join("models/BiRefNet") };
    let gradient_json = serde_json::to_string(&request.gradient)
        .map_err(|error| format!("failed to serialize the effect gradient: {error}"))?;
    let keyframes_json = serde_json::to_string(&request.keyframes)
        .map_err(|error| format!("failed to serialize VFX keyframes: {error}"))?;
    let mut command = Command::new(python);
    command
        .arg(adapter)
        .args(["--model", &request.model])
        .args(["--model-root", model_root.to_string_lossy().as_ref()])
        .args(["--video-path", &request.video_path])
        .args(["--output-path", &request.output_path])
        .args(["--start-time", &request.start_time.to_string()])
        .args(["--end-time", &request.end_time.to_string()])
        .args(["--effect", &request.effect])
        .args(["--color", &request.color])
        .args(["--secondary-color", &request.secondary_color])
        .args(["--intensity", &request.intensity.to_string()])
        .args(["--feather", &request.feather.min(64).to_string()])
        .args(["--expand", &request.expand.clamp(-32, 64).to_string()])
        .args(["--temporal-smooth", &request.temporal_smooth.min(5).to_string()])
        .args(["--cleanup", &request.cleanup.min(2_000).to_string()])
        .args(["--effect-speed", &request.effect_speed.to_string()])
        .args(["--particle-amount", &request.particle_amount.to_string()])
        .args(["--transition-in", &request.transition_in])
        .args(["--transition-out", &request.transition_out])
        .args(["--transition-in-duration", &request.transition_in_duration.to_string()])
        .args(["--transition-out-duration", &request.transition_out_duration.to_string()])
        .args(["--audio-reactive", reactive])
        .args(["--reactive-amount", &request.reactive_amount.to_string()])
        .args(["--reactive-smoothing", &request.reactive_smoothing.to_string()])
        .args(["--placement", &request.placement])
        .args(["--matte-style", &request.matte_style])
        .args(["--effect-clip-color", &request.effect_clip_color])
        .args(["--effect-opacity", &request.effect_opacity.to_string()])
        .args(["--effect-x", &request.effect_x.to_string()])
        .args(["--effect-y", &request.effect_y.to_string()])
        .args(["--effect-scale", &request.effect_scale.to_string()])
        .args(["--effect-rotation", &request.effect_rotation.to_string()])
        .args(["--pulse-amount", &request.pulse_amount.to_string()])
        .args(["--pulse-rate", &request.pulse_rate.to_string()])
        .args(["--gradient-json", &gradient_json])
        .args(["--occlusion-feather", &request.occlusion_feather.to_string()])
        .args(["--occlusion-expand", &request.occlusion_expand.to_string()])
        .args(["--transition-origin", &request.transition_origin])
        .args(["--blend-mode", &request.blend_mode])
        .args(["--key-mode", &request.key_mode])
        .args(["--key-threshold", &request.key_threshold.to_string()])
        .args(["--key-softness", &request.key_softness.to_string()])
        .args(["--key-color", &request.key_color])
        .args(["--rear-amount", &request.rear_amount.to_string()])
        .args(["--subject-amount", &request.subject_amount.to_string()])
        .args(["--front-amount", &request.front_amount.to_string()])
        .args(["--motion-blur", &request.motion_blur.to_string()])
        .args(["--light-wrap", &request.light_wrap.to_string()])
        .args(["--color-spill", &request.color_spill.to_string()])
        .args(["--grain", &request.grain.to_string()])
        .args(["--displacement", &request.displacement.to_string()])
        .args(["--keyframes-json", &keyframes_json])
        .args(["--preview-width", &request.preview_width.to_string()]);
    if request.invert { command.arg("--invert"); }
    if request.loop_effect { command.arg("--loop"); }
    if request.gradient_enabled { command.arg("--gradient-enabled"); }
    if !request.effect_clip_path.is_empty() {
        command.args(["--effect-clip-path", &request.effect_clip_path]);
    }
    if cached_matte { command.args(["--matte-cache-path", &request.matte_cache_path]); }
    for point in &request.points {
        command.args(["--point", &format!("{},{},{}", point.x, point.y, if point.include { 1 } else { 0 })]);
    }
    let output = command
        .current_dir(&root)
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("failed to start the Mask Effects worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() { return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code())); }
    if !output_path.is_file() { return Err("the Mask Effects worker finished without creating output".into()); }
    let log = worker_diagnostics(&stdout, &stderr);
    record_worker_log(&root, "mask-effects", &log);
    Ok(LtxBridgeResult { output_path: request.output_path, log })
}

/// The proxy is not a second renderer. It invokes the final renderer with an even reduced width,
/// keyed by the complete recipe and matte cache so an unchanged preview is reused byte-for-byte.
pub async fn preview_mask_effect(mut request: MaskEffectRequest) -> Result<LtxBridgeResult, String> {
    let root = project_root();
    request.preview_width = if request.preview_width == 0 { 720 } else { request.preview_width };
    request.output_path.clear();
    let encoded = serde_json::to_vec(&request)
        .map_err(|error| format!("failed to identify the VFX preview: {error}"))?;
    let mut hasher = Sha256::new();
    hasher.update(encoded);
    let preview_id = format!("{:x}", hasher.finalize());
    let preview_dir = root.join(".aiv/cache/vfx-previews");
    fs::create_dir_all(&preview_dir).map_err(|error| format!("cannot create VFX preview cache: {error}"))?;
    let output = preview_dir.join(format!("{preview_id}.mp4"));
    if output.is_file() {
        return Ok(LtxBridgeResult {
            output_path: output.to_string_lossy().into_owned(),
            log: "Reused the matching reduced-resolution VFX preview.".to_owned(),
        });
    }
    request.output_path = output.to_string_lossy().into_owned();
    render_mask_effect(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Rust third of the LTX geometry contract. `ai-runtime/tests/test_ltx_geometry.py` and
    /// `src/features/generation/ltxGeometry.test.ts` assert the same cases; the point of all three
    /// is that a request the panel builds is one this validator accepts and the worker can run.
    #[test]
    fn the_axis_multiple_is_the_one_stage_rule() {
        // 64 is the TWO-stage divisor and there is no stage 2 here. 832x480 was retakeable but not
        // generatable purely because this check said 64.
        assert_eq!(LTX_AXIS_MULTIPLE, 32);
        assert!(ltx_geometry_error(832, 480, 121).is_none());
        assert!(ltx_geometry_error(768, 512, 121).is_none());
    }

    #[test]
    fn an_off_grid_axis_is_named_in_the_message() {
        let message = ltx_geometry_error(770, 512, 121).unwrap();
        assert!(message.contains("width"), "{message}");
        assert!(message.contains("multiple of 32"), "{message}");
        assert!(ltx_geometry_error(768, 500, 121).unwrap().contains("height"));
    }

    #[test]
    fn generation_now_refuses_a_frame_count_it_used_to_accept() {
        // `--num-frames 481` was accepted here and OOMed after loading 27 GB. It is on the 8n+1
        // grid, so only the cap catches it.
        assert_eq!(481 % LTX_FRAMES_PER_LATENT, 1);
        assert!(ltx_geometry_error(768, 512, 481).unwrap().contains("over the 193"));
        assert!(ltx_geometry_error(768, 512, 193).is_none());
    }

    #[test]
    fn the_offload_profiles_include_the_intermediate_one() {
        // The two extremes were all we had: pin all 48 blocks (19.1 GiB) or pin 2 (0.8 GiB), and
        // the second is the configuration that crashes this machine.
        assert_eq!(LTX_OFFLOAD_PROFILES, &["cpu", "disk-cached", "disk"]);
    }

    #[test]
    fn the_rest_of_the_contract_still_holds() {
        assert!(ltx_geometry_error(1024, 1024, 121).unwrap().contains("limited to"));
        assert!(ltx_geometry_error(768, 512, 120).unwrap().contains("8n+1"));
        assert!(ltx_geometry_error(768, 512, 1).unwrap().contains("at least 9"));
        assert!(ltx_geometry_error(0, 512, 121).unwrap().contains("width"));
    }

    /// The calibration line is the whole reason this function exists: it is printed to stderr,
    /// after the run has finished, by a worker that then exits 0.
    /// The reason the LTX side gets Sage rather than Flash: this machine cannot reach Flash.
    ///
    /// The vendored selector gates FlashAttention 3 and 4 on compute-capability major 9 (Hopper)
    /// and 10 (datacenter Blackwell). An RTX 4090 Laptop is Ada, major 8, so no Flash install
    /// could change which kernel LTX runs. `test_flash_is_unreachable_on_this_architecture`
    /// reads that guard out of the pinned source; this end holds the choices it produced.
    #[test]
    fn native_leads_and_is_the_default() {
        assert_eq!(LTX_ATTENTION_BACKENDS[0], "native");
        assert_eq!(default_ltx_attention(), "native");
        assert!(LTX_ATTENTION_BACKENDS.contains(&"sage"));
    }

    #[test]
    fn the_backends_mirror_the_python_tuple() {
        // aiv_runtime.ltx_attention.ATTENTION_BACKENDS
        assert_eq!(LTX_ATTENTION_BACKENDS, &["native", "sage"][..]);
    }

    #[test]
    fn a_successful_run_keeps_what_it_measured() {
        let stderr = "[ltx-retake] peak VRAM 9,102 MiB allocated / 11,000 MiB reserved\n\
                      [ltx-retake] calibration: 13,056,000 pixel-frames -> 9,102 MiB peak";
        let stdout = "{\"ok\": true, \"outputPath\": \"out.mp4\"}";
        let log = worker_diagnostics(stdout, stderr);
        assert!(log.contains("calibration: 13,056,000 pixel-frames"));
        assert!(log.contains("peak VRAM"));
        // stdout is still there, and last, because it is the terminal payload.
        assert!(log.trim_end().ends_with("{\"ok\": true, \"outputPath\": \"out.mp4\"}"));
    }

    #[test]
    fn progress_bars_do_not_reach_the_log() {
        let stderr = "  8%|###       | 4/50 [00:31<06:02,  7.88s/it]\r\
                      100%|##########| 50/50 [06:33<00:00,  7.87s/it]\n\
                      [ltx] disk-cached offload, 16 CPU slots";
        let log = worker_diagnostics("", stderr);
        assert_eq!(log, "[ltx] disk-cached offload, 16 CPU slots");
    }

    #[test]
    fn a_flood_of_load_warnings_cannot_push_the_measurement_out() {
        let mut stderr: String = (0..MAX_DIAGNOSTIC_LINES * 2)
            .map(|index| format!("warning {index}\n"))
            .collect();
        stderr.push_str("[ltx-retake] calibration: 13,056,000 pixel-frames -> 9,102 MiB peak");
        let log = worker_diagnostics("", &stderr);
        // The tail is kept, so the last thing printed survives however noisy the load was.
        assert!(log.contains("calibration: 13,056,000"));
        assert_eq!(log.lines().count(), MAX_DIAGNOSTIC_LINES);
        assert!(!log.contains("warning 0\n"));
    }

    #[test]
    fn an_empty_pair_records_nothing() {
        assert_eq!(worker_diagnostics("", ""), "");
        assert_eq!(worker_diagnostics("  \n \r ", ""), "");
    }
}
