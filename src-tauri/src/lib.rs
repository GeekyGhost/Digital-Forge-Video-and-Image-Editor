mod ai;
mod analysis;
mod domain;
mod export;
mod fonts;
mod jobs;
mod media;
mod programme_export;
// VFX Phase 0 spikes. Throwaway: see spikes/rust/spikes.rs for how to remove.
#[path = "../../spikes/rust/spikes.rs"]
mod spikes;

use domain::ProjectEnvelope;
use export::{ExportRequest, ExportResult, TimelineExportRequest};
use jobs::{JobRegistry, NativeJob};
use media::MediaProbe;
use sha2::{Digest, Sha256};
use std::{fs, path::PathBuf};
use tauri::State;
use uuid::Uuid;

#[tauri::command]
async fn probe_media(path: String) -> Result<MediaProbe, String> {
    media::probe(&path).await
}

#[tauri::command]
fn ai_runtime_status() -> ai::AiRuntimeStatus {
    ai::status()
}

/// The resource guard's answer channel: the warning dialog reports the user's choice here.
#[tauri::command]
fn ai_preflight_decide(id: String, decision: ai::PreflightDecision, gate: State<'_, ai::AiExecutionGate>) -> Result<(), String> {
    gate.decide(&id, decision)
}

/// Called when the warning dialog mounts (true) and unmounts (false). Until then a failed check
/// fails fast instead of waiting for an answer nobody can give.
#[tauri::command]
fn ai_preflight_ready(ready: bool, gate: State<'_, ai::AiExecutionGate>) {
    gate.set_ui_ready(ready);
}

/// Dry run of the resource check for `job`, without taking the gate.
#[tauri::command]
async fn ai_resource_check(job: ai::JobKind) -> Result<ai::resource_guard::Verdict, String> {
    ai::resource_check(job).await
}

#[tauri::command]
fn ai_resource_requirements() -> Vec<ai::resource_guard::Requirement> {
    ai::resource_guard::REQUIREMENTS.to_vec()
}

#[tauri::command]
async fn generate_ltx_bridge(request: ai::LtxBridgeRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::LtxBridgeResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::Ltx, &app).await?;
    ai::generate_bridge(request).await
}

#[tauri::command]
async fn extend_clip_ltx(request: ai::LtxClipExtendRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::LtxBridgeResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::Ltx, &app).await?;
    ai::extend_clip(request).await
}

#[tauri::command]
async fn bridge_clips_ltx(request: ai::LtxClipBridgeRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::LtxBridgeResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::Ltx, &app).await?;
    ai::bridge_clips(request).await
}

#[tauri::command]
async fn retake_ltx_bridge(request: ai::LtxRetakeRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::LtxRetakeResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::LtxRetake, &app).await?;
    ai::retake_bridge(request).await
}

#[tauri::command]
async fn generate_moss_sfx(request: ai::MossSfxRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::SfxGenerationResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::MossSfx, &app).await?;
    ai::generate_moss_sfx(request).await
}

#[tauri::command]
async fn generate_ltx_foley(request: ai::LtxFoleyRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::SfxGenerationResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::LtxFoley, &app).await?;
    ai::generate_ltx_foley(request).await
}

#[tauri::command]
async fn edit_clip_kiwi(request: ai::KiwiEditRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::LtxBridgeResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::KiwiEdit, &app).await?;
    ai::edit_clip(request).await
}

#[tauri::command]
async fn track_sam2(request: ai::Sam2Request, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::LtxBridgeResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::Sam2Track, &app).await?;
    ai::track_sam2(request).await
}

#[tauri::command]
async fn render_mask_effect(request: ai::MaskEffectRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::LtxBridgeResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::MaskEffect, &app).await?;
    ai::render_mask_effect(request).await
}

#[tauri::command]
async fn cache_mask_track(request: ai::MaskTrackRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::MaskTrackResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::Sam2Track, &app).await?;
    ai::cache_mask_track(request).await
}

/// Background removal for a timeline clip (`ai/clip_matte.rs`): BiRefNet, SAM 2, or SAM 2 refined by
/// BiRefNet. Each step holds its own job's gate; a matte made before returns without starting a job.
#[tauri::command]
async fn clip_matte(request: ai::clip_matte::ClipMatteRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::MaskTrackResult, String> {
    ai::clip_matte::validate(&request)?;
    let root = ai::project_root();
    if let Some(done) = ai::clip_matte::cached(&root, &request)? { return Ok(done); }
    let guide = if request.uses_sam2() {
        let _permit = gate.acquire_for(ai::JobKind::Sam2Track, &app).await?;
        Some(ai::clip_matte::sam2_step(&root, &request).await?)
    } else { None };
    if !request.uses_birefnet() { return guide.ok_or_else(|| "SAM 2 produced no matte".to_owned()); }
    let _permit = gate.acquire_for(ai::JobKind::BackgroundRemoval, &app).await?;
    ai::clip_matte::birefnet_step(&root, &request, guide.as_ref()).await
}

#[tauri::command]
async fn preview_mask_effect(request: ai::MaskEffectRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::LtxBridgeResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::MaskEffect, &app).await?;
    ai::preview_mask_effect(request).await
}

#[tauri::command]
async fn generate_minimax_h3(request: ai::h3::MiniMaxH3Request, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::h3::MiniMaxH3Result, String> {
    let _permit = gate.acquire_for(ai::JobKind::MinimaxH3, &app).await?;
    ai::h3::generate(request).await
}

#[tauri::command]
async fn generate_minimax_h3_ref(request: ai::h3_ref::MiniMaxH3RefRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::h3::MiniMaxH3Result, String> {
    let _permit = gate.acquire_for(ai::JobKind::MinimaxH3Ref, &app).await?;
    ai::h3_ref::generate(request).await
}

#[tauri::command]
async fn transcribe_media(request: ai::TranscribeRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<Vec<ai::TranscriptCue>, String> {
    let _permit = gate.acquire_for(ai::JobKind::Whisper, &app).await?;
    ai::transcribe(request).await
}

#[tauri::command]
async fn generate_zit_image(request: ai::image::ZitRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::image::ImageResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::ZImageTurbo, &app).await?;
    ai::image::generate(request).await
}

#[tauri::command]
async fn generate_zimage_images(request: ai::image::ZImageRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::image::ImageBatchResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::ZImageBase, &app).await?;
    ai::image::generate_zimage(request).await
}

#[tauri::command]
fn inspect_zimage_lora(path: String) -> Result<ai::image::ZImageLoraInfo, String> {
    ai::image::inspect_zimage_lora(&path)
}

#[tauri::command]
fn list_zimage_loras() -> Vec<ai::image::ZImageLoraInfo> {
    ai::image::list_zimage_loras()
}

#[tauri::command]
async fn rewrite_qwen_edit_prompt(request: ai::prompt_rewrite::RewriteRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::prompt_rewrite::RewriteResult, String> {
    ai::prompt_rewrite::validate(&request)?;
    let _permit = gate.acquire_for(ai::JobKind::PromptRewrite, &app).await?;
    ai::prompt_rewrite::rewrite(request).await
}

#[tauri::command]
async fn caption_zimage_lora_images(request: ai::lora::CaptionRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::lora::CaptionResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::ZImageCaption, &app).await?;
    ai::lora::caption(request).await
}

#[tauri::command]
async fn start_zimage_lora_training(
    request: ai::lora::TrainingRequest,
    gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle,
    manager: State<'_, ai::lora::LoraJobManager>,
) -> Result<ai::lora::TrainingJob, String> {
    let permit = gate.acquire_for(ai::JobKind::ZImageTraining, &app).await?;
    ai::lora::start(request, manager.inner().clone(), permit).await
}

#[tauri::command]
fn zimage_lora_training_status(id: String, manager: State<'_, ai::lora::LoraJobManager>) -> Result<ai::lora::TrainingJob, String> {
    manager.get(&id).ok_or("training job not found".into())
}

#[tauri::command]
fn pause_zimage_lora_training(id: String, manager: State<'_, ai::lora::LoraJobManager>) -> Result<ai::lora::TrainingJob, String> {
    ai::lora::pause(&id, manager.inner())
}

#[tauri::command]
async fn resume_zimage_lora_training(
    id: String,
    target_steps: u32,
    gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle,
    manager: State<'_, ai::lora::LoraJobManager>,
) -> Result<ai::lora::TrainingJob, String> {
    let permit = gate.acquire_for(ai::JobKind::ZImageTraining, &app).await?;
    ai::lora::resume(&id, target_steps, manager.inner().clone(), permit).await
}

#[tauri::command]
async fn cancel_zimage_lora_training(id: String, manager: State<'_, ai::lora::LoraJobManager>) -> Result<ai::lora::TrainingJob, String> {
    ai::lora::cancel(&id, manager.inner()).await
}

#[tauri::command]
async fn edit_qwen_image(request: ai::image::QwenEditRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::image::ImageResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::QwenEdit, &app).await?;
    ai::image::edit(request).await
}

#[tauri::command]
async fn remove_image_background(request: ai::image::BackgroundRemovalRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::image::ImageResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::BackgroundRemoval, &app).await?;
    ai::image::remove_background(request).await
}

#[tauri::command]
async fn flatten_image_canvas(request: ai::image::CanvasFlattenRequest) -> Result<ai::image::ImageResult, String> {
    ai::image::flatten_canvas(request).await
}

#[tauri::command]
async fn export_image(request: ai::image::ImageExportRequest) -> Result<ai::image::ImageResult, String> {
    ai::image::export_image(request).await
}

/// No AI gate on either of these: they only move bytes, so they must not queue behind a model run.
#[tauri::command]
async fn read_image_base64(request: ai::image::ImageBytesRequest) -> Result<String, String> {
    ai::image::read_image_base64(request).await
}

#[tauri::command]
async fn write_edited_image(request: ai::image::EditedImageRequest) -> Result<ai::image::ImageResult, String> {
    ai::image::write_edited_image(request).await
}

/// The three Photo Edit model steps. All take the AI gate: each is a GPU model run, and the app's
/// rule is one at a time.
#[tauri::command]
async fn photo_depth_map(request: ai::photo::PhotoDepthRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::image::ImageResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::PhotoModel, &app).await?;
    ai::photo::depth(request).await
}

#[tauri::command]
async fn photo_inpaint(request: ai::photo::PhotoInpaintRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::image::ImageResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::PhotoModel, &app).await?;
    ai::photo::inpaint(request).await
}

#[tauri::command]
async fn photo_select(request: ai::photo::PhotoSelectRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::image::ImageResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::PhotoModel, &app).await?;
    ai::photo::select(request).await
}

#[tauri::command]
async fn upscale_image(request: ai::image::ImageUpscaleRequest) -> Result<ai::image::ImageResult, String> {
    ai::image::upscale_image(request).await
}

#[tauri::command]
async fn clip_waveform(request: media::WaveformRequest) -> Result<Vec<f32>, String> {
    media::waveform(request).await
}

#[tauri::command]
async fn mask_audio_envelope(request: media::AudioEnvelopeRequest) -> Result<Vec<f32>, String> {
    media::audio_envelope(request).await
}

#[tauri::command]
async fn stabilize_mask_matte(request: ai::MatteStabilizeRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle) -> Result<ai::MatteStabilizeResult, String> {
    let _permit = gate.acquire_for(ai::JobKind::Sam2Track, &app).await?;
    ai::stabilize_mask_matte(request).await
}

#[tauri::command]
async fn clip_thumbnail(request: media::ThumbnailRequest) -> Result<String, String> {
    media::thumbnail(request).await
}

#[tauri::command]
async fn extract_video_frames(request: media::ExtractFramesRequest) -> Result<Vec<media::ExtractedFrame>, String> {
    media::extract_frames(request).await
}

#[tauri::command]
async fn denoise_clip_audio(request: media::DenoiseRequest) -> Result<media::DenoiseResult, String> {
    media::denoise_audio(request).await
}

#[tauri::command]
async fn materialize_trimmed_clip(request: media::MaterializeClipRequest) -> Result<String, String> {
    media::materialize_clip(request).await
}

#[tauri::command]
async fn analyze_clip_pauses(request: analysis::PauseAnalysisRequest) -> Result<analysis::PauseAnalysisResult, String> {
    analysis::analyze_pauses(request).await
}

#[tauri::command]
fn save_project(path: String, project: serde_json::Value) -> Result<String, String> {
    let envelope: ProjectEnvelope = serde_json::from_value(project.clone())
        .map_err(|error| format!("invalid project: {error}"))?;
    envelope.validate()?;
    let destination = PathBuf::from(path);
    if destination.extension().and_then(|part| part.to_str()) != Some("aivproj") {
        return Err("project files must use the .aivproj extension".into());
    }
    let parent = destination.parent().ok_or("project destination has no parent")?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let bytes = serde_json::to_vec_pretty(&project).map_err(|error| error.to_string())?;
    let temporary = destination.with_extension("aivproj.tmp");
    let backup = destination.with_extension("aivproj.bak");
    fs::write(&temporary, &bytes).map_err(|error| error.to_string())?;
    if destination.exists() {
        if backup.exists() {
            fs::remove_file(&backup).map_err(|error| error.to_string())?;
        }
        fs::rename(&destination, &backup).map_err(|error| error.to_string())?;
    }
    if let Err(error) = fs::rename(&temporary, &destination) {
        if backup.exists() {
            let _ = fs::rename(&backup, &destination);
        }
        return Err(error.to_string());
    }
    if backup.exists() {
        fs::remove_file(&backup).map_err(|error| error.to_string())?;
    }
    let digest = format!("{:x}", Sha256::digest(&bytes));
    Ok(digest)
}

#[tauri::command]
fn load_project(path: String) -> Result<serde_json::Value, String> {
    let bytes = fs::read(path).map_err(|error| error.to_string())?;
    let value: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| error.to_string())?;
    let envelope: ProjectEnvelope = serde_json::from_value(value.clone()).map_err(|error| error.to_string())?;
    envelope.validate()?;
    Ok(value)
}

#[tauri::command]
fn read_subtitle_file(path: String) -> Result<String, String> {
    let source = PathBuf::from(path);
    let extension = source.extension().and_then(|part| part.to_str()).unwrap_or("").to_ascii_lowercase();
    if !["srt", "vtt"].contains(&extension.as_str()) {
        return Err("subtitle import supports .srt and .vtt files".into());
    }
    let metadata = fs::metadata(&source).map_err(|error| error.to_string())?;
    if metadata.len() > 10 * 1024 * 1024 {
        return Err("subtitle file is larger than 10 MB".into());
    }
    fs::read_to_string(source).map_err(|error| format!("subtitle file is not valid UTF-8: {error}"))
}

#[tauri::command]
fn queue_ai_job(kind: String, registry: State<'_, JobRegistry>) -> NativeJob {
    registry.queue(kind)
}

#[tauri::command]
fn list_jobs(registry: State<'_, JobRegistry>) -> Vec<NativeJob> {
    registry.list()
}

#[tauri::command]
async fn export_sequence(request: ExportRequest) -> Result<ExportResult, String> {
    export::export(request).await
}

#[tauri::command]
async fn export_timeline(request: TimelineExportRequest) -> Result<ExportResult, String> {
    export::export_timeline(request).await
}

#[tauri::command]
fn timeline_preview_path(kind: String, extension: String) -> Result<String, String> {
    if !["quick", "full"].contains(&kind.as_str()) {
        return Err("preview kind must be quick or full".into());
    }
    if !["mp4", "mov", "mkv"].contains(&extension.as_str()) {
        return Err("preview extension must be mp4, mov, or mkv".into());
    }
    let directory = std::env::temp_dir().join("aiv-studio-previews");
    fs::create_dir_all(&directory).map_err(|error| format!("could not create preview cache: {error}"))?;
    // Keep previews temporary without trying to delete the file WebView2 may currently have open.
    // The newest few survive for replay; older ones are pruned on the next render.
    let mut cached = fs::read_dir(&directory).map_err(|error| error.to_string())?
        .filter_map(Result::ok)
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("timeline-"))
        .collect::<Vec<_>>();
    cached.sort_by_key(|entry| std::cmp::Reverse(entry.metadata().and_then(|item| item.modified()).ok()));
    for entry in cached.into_iter().skip(8) { let _ = fs::remove_file(entry.path()); }
    Ok(directory.join(format!("timeline-{kind}-{}.{}", Uuid::new_v4().simple(), extension))
        .to_string_lossy().into_owned())
}

// ---- VFX Phase 3: the element library and media bytes

/// Every element in `elements/`, newest first.
#[tauri::command]
fn element_list() -> Vec<ai::elements::ElementEntry> {
    ai::elements::list(&ai::project_root())
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ElementImportRequest {
    folder: String,
    name: String,
    #[serde(default)]
    fps: f64,
}

/// Copy a folder of RGBA PNGs into the library, then build its stacked proxy. A proxy failure (no
/// ffmpeg) is returned as `proxyError` in the manifest's extra fields, not as a failed import: the
/// element works from its master without one.
#[tauri::command]
async fn element_import(request: ElementImportRequest) -> Result<ai::elements::ElementEntry, String> {
    let project = ai::project_root();
    let folder = PathBuf::from(&request.folder);
    let (name, fps) = (request.name.clone(), request.fps);
    let made = tokio::task::spawn_blocking(move || ai::elements::import_folder(&project, &folder, &name, fps))
        .await.map_err(|error| error.to_string())??;
    Ok(with_proxy(made).await)
}

async fn with_proxy(mut made: ai::elements::ElementEntry) -> ai::elements::ElementEntry {
    match ai::elements::build_proxy(std::path::Path::new(&made.dir)).await {
        Ok(manifest) => made.manifest = manifest,
        Err(error) => { made.manifest.extra.insert("proxyError".into(), serde_json::Value::String(error)); }
    }
    made
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct ElementGenerateRequest {
    name: String,
    prompt: String,
    frames: u32,
    seed: u64,
}

/// One Wan Alpha generation straight into a new element folder, through the resource guard
/// (`JobKind::WanAlpha`). The permit is released before the proxy is built: that step is ffmpeg on
/// the CPU and must not hold the GPU gate.
#[tauri::command]
async fn element_generate(
    request: ElementGenerateRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle,
) -> Result<ai::elements::ElementEntry, String> {
    ai::wan_alpha::validate_frames(request.frames)?;
    let project = ai::project_root();
    let id = ai::elements::new_id(&request.name);
    let dir = ai::elements::root(&project).join(&id);
    let stats = {
        let _permit = gate.acquire_for(ai::JobKind::WanAlpha, &app).await?;
        ai::wan_alpha::generate(&project, &request.prompt, request.frames, request.seed, &dir).await?
    };
    if stats.get("ok").and_then(|value| value.as_bool()) != Some(true) {
        let reason = stats.get("error").and_then(|value| value.as_str()).unwrap_or("the worker did not finish");
        return Err(format!("Wan Alpha failed: {reason}. See {}", dir.join("worker.log").display()));
    }
    let made = ai::elements::register_generation(&dir, &id, &request.name, &request.prompt, request.seed, &stats)?;
    Ok(with_proxy(made).await)
}

#[tauri::command]
fn element_update(request: ai::elements::ElementUpdate) -> Result<ai::elements::ElementEntry, String> {
    ai::elements::update(&ai::project_root(), request)
}

#[tauri::command]
async fn element_build_proxy(id: String) -> Result<ai::elements::ElementEntry, String> {
    let (dir, _) = ai::elements::find(&ai::project_root(), &id)?;
    let manifest = ai::elements::build_proxy(&dir).await?;
    Ok(ai::elements::ElementEntry { dir: dir.to_string_lossy().into_owned(), manifest })
}

// ---- VFX Phase 5: Wan Animate 2 character insertion

#[tauri::command]
fn wan_animate_status() -> ai::wan_animate::AnimateStatus {
    ai::wan_animate::status(&ai::project_root())
}

/// Animate a character image with a driving clip (Wan Animate 2), cut it out (SAM 2 track seeded with
/// the request's points, plus BiRefNet's soft edges) and register it as an element. Three gated steps,
/// each holding its own permit, so the guard checks each job's own requirement and logs its own peak.
/// Progress goes out as `character-progress` events for the status line.
#[tauri::command]
async fn character_generate(
    request: ai::wan_animate::CharacterRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle,
) -> Result<ai::elements::ElementEntry, String> {
    use tauri::Emitter;
    ai::wan_animate::validate(&request)?;
    let project = ai::project_root();
    let status = ai::wan_animate::status(&project);
    if !status.ready { return Err(status.detail); }
    let id = ai::elements::new_id(&request.name);
    let dir = ai::elements::root(&project).join(&id);
    let work = dir.join("animate");
    let progress = |value: serde_json::Value| { let _ = app.emit("character-progress", value); };
    let stats = {
        let _permit = gate.acquire_for(ai::JobKind::WanAnimate, &app).await?;
        ai::wan_animate::generate(&project, &request, &work, &progress).await?
    };
    if stats.get("ok").and_then(|value| value.as_bool()) != Some(true) {
        let reason = stats.get("error").and_then(|value| value.as_str()).unwrap_or("the worker did not finish");
        return Err(format!("Wan Animate 2 failed: {reason}. See {}", work.join("worker.log").display()));
    }
    let animate = stats.get("stages").and_then(|stages| stages.as_array())
        .and_then(|stages| stages.iter().find(|stage| stage.get("stage").and_then(|v| v.as_str()) == Some("animate")))
        .cloned().unwrap_or_default();
    let frames = animate.get("frames").and_then(|v| v.as_u64()).unwrap_or(request.frame_count as u64) as f64;
    let points = ai::wan_animate::reported_crop(&stats)
        .map(|crop| ai::wan_animate::output_points(&request.actor_points, crop, request.plate_width, request.plate_height))
        .unwrap_or_default();
    let mut sam2_cache = None;
    if points.iter().any(|point| point.include) {
        progress(serde_json::json!({ "event": "stage-start", "stage": "sam2" }));
        let _permit = gate.acquire_for(ai::JobKind::Sam2Track, &app).await?;
        let tracked = ai::cache_mask_track(ai::MaskTrackRequest {
            video_path: work.join("animate.mp4").to_string_lossy().into_owned(),
            start_time: 0.0, end_time: frames / request.fps, model: "sam2".into(), points,
        }).await?;
        sam2_cache = Some(std::path::PathBuf::from(tracked.cache_path));
    }
    progress(serde_json::json!({ "event": "stage-start", "stage": "matte" }));
    let matte = {
        let _permit = gate.acquire_for(ai::JobKind::BackgroundRemoval, &app).await?;
        ai::wan_animate::matte(&project, &work.join("frames"), sam2_cache.as_deref(), &dir).await?
    };
    let mut summary = animate;
    summary["crop"] = stats.get("crop").cloned().unwrap_or_default();
    summary["width"] = stats.get("width").cloned().unwrap_or_default();
    summary["height"] = stats.get("height").cloned().unwrap_or_default();
    summary["seconds"] = stats.get("seconds").cloned().unwrap_or_default();
    summary["peakProcessCommitGb"] = stats.get("peakProcessCommitGb").cloned().unwrap_or_default();
    summary["peakVramReservedGb"] = stats.get("peakVramReservedGb").cloned().unwrap_or_default();
    let made = ai::wan_animate::register(&dir, &id, &request, &summary, &matte)?;
    progress(serde_json::json!({ "event": "stage-start", "stage": "proxy" }));
    Ok(with_proxy(made).await)
}

/// VFX Phase 5: a clean plate patch for a locked-off shot (ai/clean_plate.rs): the background behind
/// the actor, per frame where the actor matte is, as an element placed like a character. Holes no
/// frame shows are LaMa-filled when LaMa is installed.
#[tauri::command]
async fn clean_plate_build(
    request: ai::clean_plate::CleanPlateRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle,
) -> Result<ai::elements::ElementEntry, String> {
    ai::clean_plate::validate(&request)?;
    let project = ai::project_root();
    let id = ai::elements::new_id(&request.name);
    let dir = ai::elements::root(&project).join(&id);
    let mut stats = {
        let _permit = gate.acquire_for(ai::JobKind::PhotoModel, &app).await?;
        let mut stats = ai::clean_plate::run(&project, &request, &dir, None).await?;
        let holes = stats.get("holes").and_then(|value| value.as_u64()).unwrap_or(0);
        if holes > 0 && ai::photo::inpaint_ready(&project) {
            let filled = dir.join("clean-filled.png");
            match ai::photo::lama_fill(&dir.join("clean.png"), &dir.join("holes.png"), &filled).await {
                // The patch-only run checks the fill against clean.png outside the holes and writes no
                // frame when it refuses, so the first run's (Telea-filled) frames stand.
                Ok(()) => match ai::clean_plate::run(&project, &request, &dir, Some(&filled)).await {
                    Ok(_) => stats["fill"] = "lama".into(),
                    Err(error) => { stats["fill"] = "telea".into(); stats["lamaError"] = error.into(); }
                },
                Err(error) => { stats["fill"] = "telea".into(); stats["lamaError"] = error.into(); }
            }
        } else if holes > 0 {
            stats["fill"] = "telea".into();
        }
        stats
    };
    stats["fps"] = request.fps.into();
    let made = ai::clean_plate::register(&dir, &id, &request, &stats)?;
    Ok(with_proxy(made).await)
}

/// A new export folder under `exports/spritesheets/`; returns its name for `spritesheet_write`.
#[tauri::command]
fn spritesheet_folder(name: String) -> Result<String, String> {
    ai::spritesheet::new_folder(&ai::project_root(), &name)
}

/// One file of a spritesheet export, sent as a raw body with `x-sheet-folder` and `x-sheet-file`.
#[tauri::command]
fn spritesheet_write(request: tauri::ipc::Request<'_>) -> Result<String, String> {
    let tauri::ipc::InvokeBody::Raw(bytes) = request.body() else { return Err("expected a raw binary body".into()) };
    let header = |name: &str| request.headers().get(name).and_then(|value| value.to_str().ok()).map(str::to_owned)
        .ok_or_else(|| format!("missing {name} header"));
    let path = ai::spritesheet::write(&ai::project_root(), &header("x-sheet-folder")?, &header("x-sheet-file")?, bytes)?;
    Ok(path.to_string_lossy().into_owned())
}

// ---- VFX Phase 4: the motion runtime (CoTracker3 point tracks, Video Depth Anything depth)

#[tauri::command]
fn motion_runtime_status() -> ai::motion::MotionStatus {
    ai::motion::status(&ai::project_root())
}

/// Track points through a clip with CoTracker3. A cached result for the same clip and request is
/// returned without taking the GPU gate.
#[tauri::command]
async fn track_points(
    request: ai::motion::TrackPointsRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle,
) -> Result<serde_json::Value, String> {
    let project = ai::project_root();
    ai::motion::validate_track(&request)?;
    if let Some(cached) = ai::motion::cached_track(&project, &request)? { return Ok(cached); }
    let _permit = gate.acquire_for(ai::JobKind::PointTrack, &app).await?;
    ai::motion::track_points(&project, &request).await
}

/// A depth sequence for a whole clip (Video Depth Anything Small), cached per clip.
#[tauri::command]
async fn video_depth(
    request: ai::motion::VideoDepthRequest, gate: State<'_, ai::AiExecutionGate>, app: tauri::AppHandle,
) -> Result<serde_json::Value, String> {
    let project = ai::project_root();
    if let Some(cached) = ai::motion::cached_depth(&project, &request)? { return Ok(cached); }
    let _permit = gate.acquire_for(ai::JobKind::VideoDepth, &app).await?;
    ai::motion::video_depth(&project, &request).await
}

// ---- Clip Editor VFX render (Phase 4b): the page streams composed RGBA frames into ffmpeg

#[tauri::command]
#[allow(clippy::too_many_arguments)]
async fn render_video_start(
    project_id: String, name: String, width: u32, height: u32, fps: f64, audio: Option<ai::render::AudioSpan>,
    pixel_format: Option<String>, http: Option<bool>,
    sessions: State<'_, ai::render::RenderSessions>,
) -> Result<ai::render::RenderStarted, String> {
    // Phase 6: `pixelFormat: "yuv420"` (packed on the page) and `http: true` (the localhost channel).
    // Older callers send neither and get the Phase 4b behaviour: RGBA over IPC.
    let format = ai::render::PixelFormat::parse(pixel_format.as_deref())?;
    ai::render::start(&sessions, &ai::project_root(), &project_id, &name, width, height, fps, audio, format, http.unwrap_or(false)).await
}

/// One frame, as the raw request body (RGBA, or I420 when the render was started with `yuv420`), for the
/// session named in the `x-render-session` header. The IPC fallback of the localhost channel. Returns how many frames have been written.
#[tauri::command]
async fn render_video_frame(request: tauri::ipc::Request<'_>, sessions: State<'_, ai::render::RenderSessions>) -> Result<u32, String> {
    let tauri::ipc::InvokeBody::Raw(bytes) = request.body() else { return Err("expected a raw binary body".into()) };
    let id = request.headers().get("x-render-session").and_then(|value| value.to_str().ok()).ok_or("missing x-render-session header")?.to_owned();
    ai::render::frame(&sessions, &id, bytes).await
}

/// VFX Phase 6: glyph outlines of `text`'s characters in an installed font (see fonts.rs), for
/// "Convert text to outlines". The first call reads the font index (once per launch).
#[tauri::command]
async fn text_outlines(family: String, generic: String, weight: u16, italic: bool, text: String) -> Result<fonts::FontOutlines, String> {
    tauri::async_runtime::spawn_blocking(move || fonts::outlines(&family, &generic, weight, italic, &text))
        .await.map_err(|error| error.to_string())?
}

#[tauri::command]
async fn render_video_finish(id: String, sessions: State<'_, ai::render::RenderSessions>) -> Result<ai::render::RenderFinished, String> {
    ai::render::finish(&sessions, &id).await
}

/// VFX Phase 7: the timeline export on the compositor (see `programme_export.rs`). Frames go to the
/// returned session exactly as a Clip Editor render's do.
#[tauri::command]
async fn programme_export_start(
    request: programme_export::ProgrammeExportRequest,
    jobs: State<'_, programme_export::ExportJobs>,
    sessions: State<'_, ai::render::RenderSessions>,
) -> Result<ai::render::RenderStarted, String> {
    programme_export::start(&jobs, &sessions, request).await
}

#[tauri::command]
async fn programme_export_finish(id: String, jobs: State<'_, programme_export::ExportJobs>, sessions: State<'_, ai::render::RenderSessions>) -> Result<ExportResult, String> {
    programme_export::finish(&jobs, &sessions, &id).await
}

#[tauri::command]
async fn programme_export_cancel(id: String, jobs: State<'_, programme_export::ExportJobs>, sessions: State<'_, ai::render::RenderSessions>) -> Result<(), String> {
    programme_export::cancel(&jobs, &sessions, &id).await
}

#[tauri::command]
async fn render_video_cancel(id: String, sessions: State<'_, ai::render::RenderSessions>) -> Result<(), String> {
    ai::render::cancel(&sessions, &id).await
}

/// A media file's bytes for the WebCodecs demuxer, when the WebView cannot fetch the asset URL. No
/// wider than the asset protocol's scope, which already serves any file to the <video> monitor.
#[tauri::command]
async fn media_read_bytes(path: String) -> Result<tauri::ipc::Response, String> {
    const MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
    let meta = tokio::fs::metadata(&path).await.map_err(|error| format!("{path}: {error}"))?;
    if !meta.is_file() { return Err(format!("not a file: {path}")); }
    if meta.len() > MAX_BYTES { return Err(format!("{path} is larger than 4 GB; use a proxy")); }
    let bytes = tokio::fs::read(&path).await.map_err(|error| format!("{path}: {error}"))?;
    Ok(tauri::ipc::Response::new(bytes))
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(JobRegistry::default())
        .manage(ai::AiExecutionGate::default())
        .manage(ai::render::RenderSessions::default())
        .manage(programme_export::ExportJobs::default())
        .manage(ai::lora::LoraJobManager::default())
        .manage(spikes::Sessions::default())
        .setup(|app| {
            spikes::open_if_requested(app);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            probe_media, clip_waveform, mask_audio_envelope, stabilize_mask_matte, clip_matte, clip_thumbnail, extract_video_frames, denoise_clip_audio, materialize_trimmed_clip,
            ai_runtime_status, ai_preflight_decide, ai_preflight_ready, ai_resource_check, ai_resource_requirements,
            generate_ltx_bridge, generate_minimax_h3, generate_minimax_h3_ref, extend_clip_ltx, bridge_clips_ltx, retake_ltx_bridge,
            generate_moss_sfx, generate_ltx_foley, edit_clip_kiwi, transcribe_media, track_sam2,
            cache_mask_track, preview_mask_effect, render_mask_effect,
            generate_zit_image, generate_zimage_images, inspect_zimage_lora, list_zimage_loras, edit_qwen_image, remove_image_background, flatten_image_canvas,
            caption_zimage_lora_images, rewrite_qwen_edit_prompt, start_zimage_lora_training, zimage_lora_training_status,
            pause_zimage_lora_training, resume_zimage_lora_training, cancel_zimage_lora_training,
            export_image, upscale_image, write_edited_image, read_image_base64,
            photo_depth_map, photo_inpaint, photo_select,
            analyze_clip_pauses, save_project, load_project, read_subtitle_file,
            queue_ai_job, list_jobs, export_sequence, export_timeline, timeline_preview_path,
            element_list, element_import, element_generate, element_update, element_build_proxy, media_read_bytes,
            wan_animate_status, character_generate, clean_plate_build, spritesheet_folder, spritesheet_write,
            motion_runtime_status, track_points, video_depth,
            render_video_start, render_video_frame, render_video_finish, render_video_cancel,
            programme_export_start, programme_export_finish, programme_export_cancel, text_outlines,
            spikes::spike_prepare_media, spikes::spike_read_file, spikes::spike_reference_frame,
            spikes::spike_encode_start, spikes::spike_encode_frame, spikes::spike_encode_finish,
            spikes::spike_app_usage, spikes::spike_save_results, spikes::spike_wan_alpha, spikes::spike_http_start
        ])
        .run(tauri::generate_context!())
        .expect("failed to run Digital Forge Video Studio");
}

#[cfg(test)]
mod tests {
    use super::domain::Rational;

    #[test]
    fn rejects_invalid_timebase() {
        assert!(Rational { numerator: 24, denominator: 0 }.validate().is_err());
    }
}
