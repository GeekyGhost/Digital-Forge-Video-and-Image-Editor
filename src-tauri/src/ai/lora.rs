//! Z-Image dataset captioning and isolated LoRA training jobs.

use super::*;
use parking_lot::Mutex as ParkingMutex;
use serde_json::Value;
use std::{collections::HashMap, sync::Arc};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptionRequest {
    pub image_paths: Vec<String>,
    pub profile: String,
    pub trigger: String,
    pub class_name: String,
    #[serde(default = "default_caption_backend")]
    pub backend: String,
}

fn default_caption_backend() -> String { "qwen".into() }

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptionedImage {
    pub image_path: String,
    pub caption: String,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub quality_flags: Vec<String>,
    pub confidence: f64,
    pub width: u32,
    pub height: u32,
    #[serde(default)]
    pub duplicate_of: Option<String>,
    #[serde(default)]
    pub raw_caption: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CaptionResult { pub items: Vec<CaptionedImage>, pub log: String }

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingRequest {
    pub project_id: String,
    pub name: String,
    pub profile: String,
    pub trigger: String,
    pub class_name: String,
    pub images: Vec<CaptionedImage>,
    pub resolution: u32,
    pub steps: u32,
    pub rank: u32,
    pub alpha: u32,
    pub learning_rate: f64,
    pub seed: i64,
    #[serde(default = "default_checkpoint_every")]
    pub checkpoint_every: u32,
    #[serde(default = "default_checkpoint_limit")]
    pub checkpoint_limit: u32,
    #[serde(default)]
    pub initial_lora_path: Option<String>,
    /// "base" (undistilled Z-Image) or "turbo" (Z-Image-Turbo with the frozen training adapter).
    #[serde(default = "default_training_model")]
    pub training_model: String,
}

fn default_training_model() -> String { "base".into() }
fn default_checkpoint_every() -> u32 { 100 }
fn default_checkpoint_limit() -> u32 { 5 }

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingCheckpoint {
    pub step: u32,
    pub path: String,
    pub weights: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TrainingJob {
    pub id: String,
    pub state: String,
    pub progress: f64,
    pub step: u32,
    pub total: u32,
    pub message: String,
    pub output_path: Option<String>,
    pub checkpoint_path: Option<String>,
    pub checkpoints: Vec<TrainingCheckpoint>,
    pub error: Option<String>,
    pub process_id: Option<u32>,
    pub run_directory: String,
    pub config_path: String,
}

#[derive(Clone, Default)]
pub struct LoraJobManager(Arc<ParkingMutex<HashMap<String, TrainingJob>>>);

impl LoraJobManager {
    pub fn get(&self, id: &str) -> Option<TrainingJob> { self.0.lock().get(id).cloned() }
    fn put(&self, job: TrainingJob) { self.0.lock().insert(job.id.clone(), job); }
    fn update(&self, id: &str, apply: impl FnOnce(&mut TrainingJob)) {
        if let Some(job) = self.0.lock().get_mut(id) { apply(job); }
    }
}

fn trainer_python(root: &Path) -> PathBuf {
    root.join("vendor/Z-Image-Trainer/.venv/Scripts/python.exe")
}

fn trainer_script(root: &Path) -> PathBuf {
    root.join("vendor/Z-Image-Trainer/diffusers/examples/dreambooth/train_dreambooth_lora_z_image.py")
}

/// ostris' Z-Image-Turbo training adapter, downloaded by Setup-ZImage-Trainer.bat. It is held active
/// and frozen during Turbo training only and is never loaded for generation.
pub fn turbo_training_adapter(root: &Path) -> PathBuf {
    root.join("models/Z-Image-Turbo-Training-Adapter/zimage_turbo_training_adapter_v2.safetensors")
}

fn caption_worker(root: &Path) -> PathBuf {
    root.join("ai-runtime/aiv_runtime/zimage_caption_worker.py")
}

fn train_worker(root: &Path) -> PathBuf {
    root.join("ai-runtime/aiv_runtime/zimage_train_worker.py")
}

pub(crate) fn qwen_base(root: &Path) -> Option<PathBuf> {
    let installed = root.join("models/Qwen-Image-Edit-2511");
    if installed.join("text_encoder/config.json").is_file() && installed.join("processor/tokenizer.json").is_file() {
        return Some(installed);
    }
    let snapshots = root.join("cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots");
    fs::read_dir(snapshots).ok()?.filter_map(Result::ok).map(|entry| entry.path()).find(|path| {
        path.join("text_encoder/config.json").is_file() && path.join("processor/tokenizer.json").is_file()
    })
}

pub fn trainer_ready(root: &Path) -> bool {
    root.join("vendor/Z-Image-Trainer/.aiv-ready.json").is_file()
        && trainer_python(root).is_file()
        && trainer_script(root).is_file()
        && train_worker(root).is_file()
}

pub fn captioner_ready(root: &Path) -> bool {
    trainer_ready(root) && caption_worker(root).is_file() && qwen_base(root).is_some()
}

fn validate_profile(profile: &str) -> Result<(), String> {
    if matches!(profile, "subject" | "character" | "style" | "product" | "concept") { Ok(()) }
    else { Err("LoRA profile must be subject, character, style, product, or concept".into()) }
}

fn validate_token(value: &str, label: &str) -> Result<(), String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed.len() > 80 || trimmed.chars().any(char::is_control) {
        Err(format!("{label} must contain 1 to 80 printable characters"))
    } else { Ok(()) }
}

fn safe_name(value: &str) -> String {
    value.to_ascii_lowercase().chars().map(|character| {
        if character.is_ascii_alphanumeric() { character } else { '-' }
    }).collect::<String>().split('-').filter(|part| !part.is_empty()).take(8).collect::<Vec<_>>().join("-")
}

pub(crate) fn offline_command(root: &Path, script: &Path) -> Command {
    let mut command = Command::new(trainer_python(root));
    command.arg(script)
        .current_dir(root)
        .env("PYTHONPATH", root.join("ai-runtime"))
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .env("HF_HOME", root.join("cache/huggingface"))
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .stdin(Stdio::null());
    command
}

pub async fn caption(request: CaptionRequest) -> Result<CaptionResult, String> {
    let root = project_root();
    if !captioner_ready(&root) {
        return Err("the Z-Image trainer/captioner runtime is incomplete; run scripts\\setup-zimage-trainer.ps1".into());
    }
    validate_profile(&request.profile)?;
    validate_token(&request.trigger, "trigger")?;
    validate_token(&request.class_name, "class name")?;
    if !matches!(request.backend.as_str(), "qwen" | "gemma") { return Err("caption backend must be qwen or gemma".into()); }
    if request.image_paths.is_empty() || request.image_paths.len() > 100 {
        return Err("choose between 1 and 100 training images".into());
    }
    for path in &request.image_paths {
        if !Path::new(path).is_file() { return Err(format!("training image not found: {path}")); }
    }
    let (model, processor) = if request.backend == "gemma" {
        let gemma = root.join("models/gemma-3-12b");
        if !gemma.join("config.json").is_file() { return Err("Gemma 3 is not installed".into()); }
        (gemma.clone(), gemma)
    } else {
        let qwen = qwen_base(&root).ok_or("Qwen 2511 support files are missing")?;
        (qwen.join("text_encoder"), qwen.join("processor"))
    };
    let mut command = offline_command(&root, &caption_worker(&root));
    command.args(["--backend", &request.backend])
        .args(["--model-path", model.to_string_lossy().as_ref()])
        .args(["--processor-path", processor.to_string_lossy().as_ref()])
        .args(["--profile", &request.profile, "--trigger", &request.trigger, "--class-name", &request.class_name]);
    for path in &request.image_paths { command.args(["--image-path", path]); }
    let output = command.output().await.map_err(|error| format!("failed to start caption worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() { return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code())); }
    let payload: Value = stdout.lines().rev()
        .find_map(|line| serde_json::from_str::<Value>(line).ok().filter(|value| value["event"] == "complete"))
        .ok_or("caption worker returned no completion event")?;
    let items: Vec<CaptionedImage> = serde_json::from_value(payload["items"].clone())
        .map_err(|error| format!("invalid caption worker response: {error}"))?;
    Ok(CaptionResult { items, log: if stderr.is_empty() { stdout } else { format!("{stdout}\n{stderr}") } })
}

async fn launch_training(
    root: &Path,
    config_path: &Path,
    mut job: TrainingJob,
    manager: LoraJobManager,
    permit: AiPermit,
) -> Result<TrainingJob, String> {
    let mut command = offline_command(root, &train_worker(root));
    command.args(["--config", config_path.to_string_lossy().as_ref()])
        .args(["--trainer-script", trainer_script(root).to_string_lossy().as_ref()])
        .stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true);
    let mut child = command.spawn().map_err(|error| format!("failed to start trainer: {error}"))?;
    job.process_id = child.id();
    let stdout = child.stdout.take().ok_or("trainer stdout is unavailable")?;
    let stderr = child.stderr.take().ok_or("trainer stderr is unavailable")?;
    manager.put(job.clone());
    let task_manager = manager.clone();
    let task_id = job.id.clone();
    tokio::spawn(async move {
        let stdout_manager = task_manager.clone();
        let stdout_id = task_id.clone();
        let stdout_task = tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let Ok(value) = serde_json::from_str::<Value>(&line) else { continue };
                match value["event"].as_str().unwrap_or("") {
                    "prepared" => stdout_manager.update(&stdout_id, |item| item.message = "Dataset prepared; loading the Z-Image model".into()),
                    "training" => stdout_manager.update(&stdout_id, |item| item.message = if item.step > 0 { "Restoring checkpoint and resuming LoRA training".into() } else { "Training LoRA".into() }),
                    "progress" => stdout_manager.update(&stdout_id, |item| {
                        item.step = value["step"].as_u64().unwrap_or(item.step as u64) as u32;
                        item.total = value["total"].as_u64().unwrap_or(item.total as u64) as u32;
                        item.progress = if item.total == 0 { 0.0 } else { item.step as f64 / item.total as f64 };
                        item.message = value["message"].as_str().unwrap_or("Training LoRA").to_owned();
                    }),
                    "checkpoint" => stdout_manager.update(&stdout_id, |item| {
                        if let Ok(checkpoints) = serde_json::from_value::<Vec<TrainingCheckpoint>>(value["checkpoints"].clone()) {
                            item.checkpoints = checkpoints;
                        }
                        item.message = format!("Saved checkpoint at step {}", item.checkpoints.last().map(|point| point.step).unwrap_or(item.step));
                    }),
                    "paused" => stdout_manager.update(&stdout_id, |item| {
                        item.state = "paused".into();
                        item.step = value["step"].as_u64().unwrap_or(item.step as u64) as u32;
                        item.progress = if item.total == 0 { 0.0 } else { item.step as f64 / item.total as f64 };
                        item.checkpoint_path = value["checkpointPath"].as_str().map(str::to_owned);
                        if let Ok(checkpoints) = serde_json::from_value::<Vec<TrainingCheckpoint>>(value["checkpoints"].clone()) {
                            item.checkpoints = checkpoints;
                        }
                        item.message = format!("Paused safely at checkpoint {}. GPU memory has been released.", item.step);
                    }),
                    "complete" => stdout_manager.update(&stdout_id, |item| {
                        item.output_path = value["outputPath"].as_str().map(str::to_owned);
                    }),
                    "failed" => stdout_manager.update(&stdout_id, |item| item.error = value["error"].as_str().map(str::to_owned)),
                    _ => {}
                }
            }
        });
        let stderr_task = tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            let mut tail = Vec::new();
            while let Ok(Some(line)) = lines.next_line().await {
                tail.push(line);
                tail = tail.into_iter().rev().take(30).collect::<Vec<_>>().into_iter().rev().collect();
            }
            tail.join("\n")
        });
        let status = child.wait().await;
        let _ = stdout_task.await;
        let stderr_tail = stderr_task.await.unwrap_or_default();
        task_manager.update(&task_id, |item| {
            item.process_id = None;
            match status {
                Ok(exit) if exit.success() && item.state == "paused" => {}
                Ok(exit) if exit.success() && item.output_path.is_some() => {
                    item.state = "complete".into(); item.progress = 1.0; item.step = item.total; item.message = "LoRA training complete".into();
                }
                Ok(_exit) if item.state == "canceling" => { item.state = "canceled".into(); item.message = "Training canceled".into(); }
                Ok(exit) => { item.state = "failed".into(); item.error.get_or_insert_with(|| format!("trainer exited with {exit}: {stderr_tail}")); }
                Err(error) => { item.state = "failed".into(); item.error = Some(format!("trainer process failed: {error}")); }
            }
        });
        drop(permit);
    });
    Ok(job)
}

pub async fn start(
    request: TrainingRequest,
    manager: LoraJobManager,
    permit: AiPermit,
) -> Result<TrainingJob, String> {
    let root = project_root();
    if !trainer_ready(&root) { return Err("the Z-Image trainer runtime is incomplete".into()); }
    let turbo = match request.training_model.as_str() {
        "base" => false,
        "turbo" => true,
        _ => return Err("training model must be base or turbo".into()),
    };
    let model_path = root.join(if turbo { "models/Z-Image-Turbo" } else { "models/Z-Image" });
    if !model_path.join("model_index.json").is_file() {
        return Err(if turbo { "Z-Image-Turbo is not installed".into() } else { "Z-Image Base is not installed".into() });
    }
    let assistant_lora = if turbo {
        let adapter = turbo_training_adapter(&root);
        if !adapter.is_file() {
            return Err("the Z-Image-Turbo training adapter is missing; run Setup-ZImage-Trainer.bat to download it".into());
        }
        Some(adapter)
    } else { None };
    validate_profile(&request.profile)?;
    validate_token(&request.trigger, "trigger")?;
    validate_token(&request.class_name, "class name")?;
    if !(3..=100).contains(&request.images.len()) { return Err("training needs between 3 and 100 approved images".into()); }
    for image in &request.images {
        if !Path::new(&image.image_path).is_file() { return Err(format!("training image not found: {}", image.image_path)); }
        if image.caption.trim().is_empty() || !image.caption.contains(request.trigger.trim()) {
            return Err(format!("every training caption must contain the trigger token: {}", image.image_path));
        }
        if image.duplicate_of.is_some() { return Err(format!("remove duplicate training image before starting: {}", image.image_path)); }
    }
    if !matches!(request.resolution, 512 | 768 | 1024) { return Err("training resolution must be 512, 768, or 1024".into()); }
    if !(100..=10_000).contains(&request.steps) { return Err("training steps must be between 100 and 10000".into()); }
    if !(4..=128).contains(&request.rank) || !(1..=256).contains(&request.alpha) { return Err("invalid LoRA rank or alpha".into()); }
    if !(0.000001..=0.01).contains(&request.learning_rate) { return Err("learning rate is outside the supported range".into()); }
    if !(25..=2000).contains(&request.checkpoint_every) { return Err("checkpoint interval must be between 25 and 2000 steps".into()); }
    if !(1..=50).contains(&request.checkpoint_limit) { return Err("checkpoint limit must be between 1 and 50".into()); }
    let initial_lora = request.initial_lora_path.as_deref().map(str::trim).filter(|value| !value.is_empty()).map(|path| {
        let info = super::image::inspect_zimage_lora(path)?;
        if info.base_model.as_deref().is_some_and(|model| !model.to_ascii_lowercase().contains("z-image")) {
            return Err("the starting LoRA was not trained for Z-Image".into());
        }
        if info.rank.is_some_and(|rank| rank != request.rank) || info.alpha.is_some_and(|alpha| alpha != request.alpha) {
            return Err(format!("starting LoRA requires rank {} and alpha {}; match those settings before continuing",
                info.rank.unwrap_or(request.rank), info.alpha.unwrap_or(request.alpha)));
        }
        Ok(info.weights)
    }).transpose()?;
    let project = Uuid::parse_str(&request.project_id).map_err(|_| "invalid project id".to_owned())?;
    let name = safe_name(&request.name);
    if name.is_empty() { return Err("LoRA name must contain a letter or number".into()); }
    let id = Uuid::new_v4().to_string();
    let run_directory = root.join("projects").join(project.to_string()).join("lora-training").join(&id);
    fs::create_dir_all(&run_directory).map_err(|error| format!("could not create training directory: {error}"))?;
    let output_directory = root.join("models/loras/z-image").join(&name);
    if output_directory.exists() && fs::read_dir(&output_directory)
        .map_err(|error| format!("could not inspect existing LoRA folder: {error}"))?.next().is_some()
    {
        return Err(format!("a LoRA named {name} already exists"));
    }
    let config_path = run_directory.join("training.json");
    let pause_file = run_directory.join("pause.request");
    let config = serde_json::json!({
        "projectId": request.project_id, "name": name, "displayName": request.name.trim(), "profile": request.profile,
        "trigger": request.trigger, "className": request.class_name, "images": request.images,
        "resolution": request.resolution, "steps": request.steps, "rank": request.rank,
        "alpha": request.alpha, "learningRate": request.learning_rate, "seed": request.seed,
        "checkpointEvery": request.checkpoint_every, "checkpointLimit": request.checkpoint_limit,
        "initialLoraPath": initial_lora,
        "trainingModel": request.training_model, "assistantLoraPath": assistant_lora,
        "modelPath": model_path, "runDirectory": run_directory,
        "outputDirectory": output_directory, "pauseFile": pause_file,
    });
    fs::write(&config_path, serde_json::to_vec_pretty(&config).map_err(|error| error.to_string())?)
        .map_err(|error| format!("could not write training configuration: {error}"))?;
    let job = TrainingJob {
        id, state: "running".into(), progress: 0.0, step: 0, total: request.steps,
        message: if initial_lora.is_some() { "Preparing dataset and starting from existing LoRA weights".into() } else { "Preparing the captioned dataset".into() },
        output_path: None, checkpoint_path: None, checkpoints: Vec::new(), error: None, process_id: None,
        run_directory: run_directory.to_string_lossy().into_owned(), config_path: config_path.to_string_lossy().into_owned(),
    };
    launch_training(&root, &config_path, job, manager, permit).await
}

pub fn pause(id: &str, manager: &LoraJobManager) -> Result<TrainingJob, String> {
    let job = manager.get(id).ok_or("training job not found")?;
    if job.state == "pausing" { return Ok(job); }
    if job.state != "running" { return Err("only a running training job can be paused".into()); }
    let pause_file = Path::new(&job.run_directory).join("pause.request");
    fs::write(&pause_file, b"pause after the next optimizer step")
        .map_err(|error| format!("could not request a safe training pause: {error}"))?;
    manager.update(id, |item| {
        item.state = "pausing".into();
        item.message = "Saving a resumable checkpoint after the current optimizer step".into();
    });
    manager.get(id).ok_or("training job disappeared".into())
}

pub async fn resume(
    id: &str,
    target_steps: u32,
    manager: LoraJobManager,
    permit: AiPermit,
) -> Result<TrainingJob, String> {
    let root = project_root();
    let mut job = manager.get(id).ok_or("training job not found")?;
    if job.state != "paused" { return Err("only a paused training job can be resumed".into()); }
    let checkpoint = job.checkpoint_path.clone().ok_or("paused job has no resumable checkpoint")?;
    if !Path::new(&checkpoint).join("optimizer.bin").is_file()
        || !Path::new(&checkpoint).join("pytorch_lora_weights.safetensors").is_file()
    {
        return Err("the selected checkpoint is incomplete and cannot restore optimizer state".into());
    }
    if target_steps <= job.step || target_steps > 10_000 {
        return Err(format!("target total steps must be between {} and 10000", job.step + 1));
    }
    let config_path = PathBuf::from(&job.config_path);
    let mut config: Value = serde_json::from_slice(&fs::read(&config_path)
        .map_err(|error| format!("could not read training configuration: {error}"))?)
        .map_err(|error| format!("invalid training configuration: {error}"))?;
    config["steps"] = Value::from(target_steps);
    config["resumeFromCheckpoint"] = Value::from(checkpoint);
    fs::write(&config_path, serde_json::to_vec_pretty(&config).map_err(|error| error.to_string())?)
        .map_err(|error| format!("could not update training configuration: {error}"))?;
    let pause_file = Path::new(&job.run_directory).join("pause.request");
    if pause_file.exists() { fs::remove_file(&pause_file).map_err(|error| format!("could not clear pause request: {error}"))?; }
    job.state = "running".into();
    job.total = target_steps;
    job.progress = job.step as f64 / target_steps as f64;
    job.message = format!("Restoring full training state from checkpoint {}", job.step);
    job.error = None;
    job.output_path = None;
    launch_training(&root, &config_path, job, manager, permit).await
}

pub async fn cancel(id: &str, manager: &LoraJobManager) -> Result<TrainingJob, String> {
    let job = manager.get(id).ok_or("training job not found")?;
    if job.state == "paused" {
        manager.update(id, |item| {
            item.state = "canceled".into();
            item.message = "Paused training session closed; saved checkpoints were retained".into();
        });
        return manager.get(id).ok_or("training job disappeared".into());
    }
    if !matches!(job.state.as_str(), "running" | "pausing" | "canceling") { return Ok(job); }
    let pid = job.process_id.ok_or("training process is no longer running")?;
    manager.update(id, |item| { item.state = "canceling".into(); item.message = "Stopping trainer".into(); });
    #[cfg(target_os = "windows")]
    {
        let output = Command::new("taskkill").args(["/PID", &pid.to_string(), "/T", "/F"])
            .output().await.map_err(|error| format!("could not stop trainer: {error}"))?;
        if !output.status.success() {
            let detail = String::from_utf8_lossy(&output.stderr);
            return Err(format!("could not stop trainer process {pid}: {detail}"));
        }
    }
    #[cfg(not(target_os = "windows"))]
    {
        let output = Command::new("kill").args(["-TERM", &pid.to_string()]).output().await
            .map_err(|error| format!("could not stop trainer: {error}"))?;
        if !output.status.success() { return Err(format!("could not stop trainer process {pid}")); }
    }
    manager.get(id).ok_or("training job disappeared".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_lora_names_stay_inside_the_output_folder() {
        assert_eq!(safe_name(" My Hero / Version 2 "), "my-hero-version-2");
        assert_eq!(safe_name("../../"), "");
    }

    #[test]
    fn profiles_are_explicit() {
        assert!(validate_profile("subject").is_ok());
        assert!(validate_profile("character").is_ok());
        assert!(validate_profile("anything").is_err());
    }
}
