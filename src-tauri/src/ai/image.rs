//! Still-image generation, editing, and matting in an isolated Python environment.

use super::*;
use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::Value;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ZitRequest {
    pub project_id: String,
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub seed: i64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ZImageRequest {
    pub project_id: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub guidance_scale: f64,
    pub seed: i64,
    #[serde(default = "default_candidate_count")]
    pub candidates: u32,
    #[serde(default)]
    pub lora_path: Option<String>,
    #[serde(default = "default_lora_strength")]
    pub lora_strength: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ZImageLoraInfo {
    pub folder: String,
    pub weights: String,
    pub name: String,
    pub trigger_phrase: Option<String>,
    pub trigger: Option<String>,
    pub class_name: Option<String>,
    pub base_model: Option<String>,
    pub profile: Option<String>,
    pub rank: Option<u32>,
    pub alpha: Option<u32>,
    pub step: Option<u32>,
    pub kind: Option<String>,
}

pub fn list_zimage_loras() -> Vec<ZImageLoraInfo> {
    let directory = project_root().join("models/loras/z-image");
    let Ok(entries) = fs::read_dir(directory) else { return Vec::new() };
    let mut loras = entries
        .filter_map(Result::ok)
        .filter_map(|entry| inspect_zimage_lora(entry.path().to_string_lossy().as_ref()).ok())
        .collect::<Vec<_>>();
    loras.sort_by(|left, right| left.name.to_lowercase().cmp(&right.name.to_lowercase()));
    loras
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QwenEditRequest {
    pub project_id: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: String,
    pub image_paths: Vec<String>,
    pub steps: u32,
    pub true_cfg_scale: f64,
    pub seed: i64,
    #[serde(default = "default_pixel_budget")]
    pub pixel_budget: u32,
    /// Exact output size ("Output size" override). 0 x 0 follows Image 1 at about a megapixel.
    #[serde(default)]
    pub output_width: u32,
    #[serde(default)]
    pub output_height: u32,
}

/// The edit's output-size arguments: none when following Image 1, else the size (the worker snaps
/// it to 16). Both sides or neither, 256..2048 a side (so never more than 2048 x 2048).
pub fn edit_output_size(width: u32, height: u32) -> Result<Option<(u32, u32)>, String> {
    match (width, height) {
        (0, 0) => Ok(None),
        (0, _) | (_, 0) => Err("set both the output width and height, or neither".into()),
        (w, h) if !(256..=2048).contains(&w) || !(256..=2048).contains(&h) => Err("the output size must be 256 to 2048 pixels a side".into()),
        (w, h) => Ok(Some((w, h))),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackgroundRemovalRequest {
    pub project_id: String,
    pub image_path: String,
    #[serde(default)]
    pub blur: u32,
    #[serde(default)]
    pub dilate: u32,
    #[serde(default)]
    pub erode: u32,
    #[serde(default = "default_threshold")]
    pub threshold: i32,
    #[serde(default)]
    pub invert: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CanvasLayerRequest {
    pub image_path: String,
    pub center_x: f64,
    pub center_y: f64,
    pub width: u32,
    pub height: u32,
    pub rotation: f64,
    pub opacity: f64,
    pub visible: bool,
    #[serde(default)]
    pub crop_left: f64,
    #[serde(default)]
    pub crop_top: f64,
    #[serde(default)]
    pub crop_right: f64,
    #[serde(default)]
    pub crop_bottom: f64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CanvasFlattenRequest {
    pub project_id: String,
    pub layers: Vec<CanvasLayerRequest>,
    pub crop: String,
    pub canvas_width: u32,
    pub canvas_height: u32,
    #[serde(default)]
    pub padding: u32,
    #[serde(default = "default_canvas_background")]
    pub background: String,
}

fn default_pixel_budget() -> u32 { 4_000_000 }
fn default_candidate_count() -> u32 { 1 }
fn default_lora_strength() -> f64 { 1.0 }
fn default_threshold() -> i32 { -1 }
fn default_canvas_background() -> String { "transparent".into() }

pub fn inspect_zimage_lora(path: &str) -> Result<ZImageLoraInfo, String> {
    let selected = Path::new(path);
    let (folder, selected_weights) = if selected.is_file() {
        if selected.extension().and_then(|value| value.to_str()) != Some("safetensors") {
            return Err("choose a Z-Image LoRA folder or .safetensors file".into());
        }
        (selected.parent().ok_or("the selected LoRA file has no parent folder")?, Some(selected.to_path_buf()))
    } else if selected.is_dir() {
        (selected, None)
    } else {
        return Err("choose an existing Z-Image LoRA folder or .safetensors file".into());
    };
    let manifest_path = folder.join("aiv-lora.json");
    let manifest = if manifest_path.is_file() {
        let bytes = fs::read(&manifest_path).map_err(|error| format!("could not read LoRA manifest: {error}"))?;
        serde_json::from_slice::<Value>(&bytes).map_err(|error| format!("invalid LoRA manifest: {error}"))?
    } else { Value::Null };
    let weights = selected_weights.or_else(|| ["weightsFile", "weights"].into_iter().find_map(|key| {
        let value = manifest[key].as_str()?;
        let candidate = PathBuf::from(value);
        let candidate = if candidate.is_absolute() { candidate } else { folder.join(candidate) };
        candidate.is_file().then_some(candidate)
    })).or_else(|| {
        let named = fs::read_dir(folder).ok()?.filter_map(Result::ok).map(|entry| entry.path())
            .find(|candidate| candidate.file_name().and_then(|name| name.to_str())
                .is_some_and(|name| name.ends_with("-Z-LoRA.safetensors")));
        named.or_else(|| {
            let legacy = folder.join("pytorch_lora_weights.safetensors");
            legacy.is_file().then_some(legacy)
        })
    }).ok_or("the selected folder contains no Z-Image LoRA weights")?;
    let name = manifest["name"].as_str().map(str::to_owned).unwrap_or_else(|| {
        folder.file_name().unwrap_or_default().to_string_lossy().into_owned()
    });
    let trigger_phrase = manifest["triggerPhrase"].as_str().map(str::to_owned).or_else(|| {
        Some(format!("{} {}", manifest["trigger"].as_str()?, manifest["className"].as_str()?))
    });
    Ok(ZImageLoraInfo {
        folder: folder.to_string_lossy().into_owned(),
        weights: weights.to_string_lossy().into_owned(),
        name,
        trigger_phrase,
        trigger: manifest["trigger"].as_str().map(str::to_owned),
        class_name: manifest["className"].as_str().map(str::to_owned),
        base_model: manifest["baseModel"].as_str().map(str::to_owned),
        profile: manifest["profile"].as_str().map(str::to_owned),
        rank: manifest["rank"].as_u64().map(|value| value as u32),
        alpha: manifest["alpha"].as_u64().map(|value| value as u32),
        step: manifest["step"].as_u64().map(|value| value as u32),
        kind: manifest["kind"].as_str().map(str::to_owned),
    })
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageResult {
    pub output_path: String,
    pub mask_path: Option<String>,
    pub seed: Option<i64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_active: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lora_path: Option<String>,
    pub log: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageBatchResult {
    pub images: Vec<ImageResult>,
}

pub fn python(root: &Path) -> PathBuf {
    root.join("vendor/Image-Studio/.venv/Scripts/python.exe")
}

fn worker(root: &Path) -> PathBuf {
    root.join("ai-runtime/aiv_runtime/image_studio/worker.py")
}

fn zit_model(root: &Path) -> PathBuf { root.join("models/Z-Image-Turbo") }
fn zimage_base_model(root: &Path) -> PathBuf { root.join("models/Z-Image") }
fn qwen_model(root: &Path) -> PathBuf { root.join("models/qwen_image_edit_2511_fp8_e4m3fn_scaled.safetensors") }
fn qwen_lora(root: &Path) -> PathBuf { root.join("models/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors") }
fn birefnet_model(root: &Path) -> PathBuf { root.join("models/BiRefNet") }

fn qwen_base(root: &Path) -> Option<PathBuf> {
    let installed = root.join("models/Qwen-Image-Edit-2511");
    if installed.join("model_index.json").is_file()
        && installed.join("transformer/config.json").is_file()
        && installed.join("vae/config.json").is_file()
    {
        return Some(installed);
    }
    let snapshots = root.join("cache/huggingface/hub/models--Qwen--Qwen-Image-Edit-2511/snapshots");
    fs::read_dir(snapshots).ok()?.filter_map(Result::ok).map(|entry| entry.path()).find(|path| {
        path.join("model_index.json").is_file()
            && path.join("transformer/config.json").is_file()
            && path.join("vae/config.json").is_file()
    })
}

pub fn state_label(root: &Path) -> &'static str {
    if !python(root).is_file() { "venv-missing" }
    else if !worker(root).is_file() { "adapter-missing" }
    else if !zit_model(root).join("model_index.json").is_file() { "zit-missing" }
    else if !qwen_model(root).is_file() || !qwen_lora(root).is_file() || qwen_base(root).is_none() { "qwen-missing" }
    else if !birefnet_model(root).join("config.json").is_file() { "birefnet-missing" }
    else { "ready" }
}

pub fn zit_ready(root: &Path) -> bool {
    python(root).is_file() && worker(root).is_file() && zit_model(root).join("model_index.json").is_file()
}

pub fn zimage_base_ready(root: &Path) -> bool {
    python(root).is_file() && worker(root).is_file()
        && zimage_base_model(root).join("model_index.json").is_file()
}

pub fn qwen_ready(root: &Path) -> bool {
    python(root).is_file() && worker(root).is_file() && qwen_model(root).is_file()
        && qwen_lora(root).is_file() && qwen_base(root).is_some()
}

pub fn birefnet_ready(root: &Path) -> bool {
    python(root).is_file() && worker(root).is_file() && birefnet_model(root).join("config.json").is_file()
}

fn output_dir(root: &Path, project_id: &str) -> Result<PathBuf, String> {
    let id = Uuid::parse_str(project_id).map_err(|_| "invalid project id".to_owned())?;
    let path = root.join("projects").join(id.to_string()).join("media/images");
    fs::create_dir_all(&path).map_err(|error| format!("could not create project image directory: {error}"))?;
    Ok(path)
}

/// Words that cost filename characters without helping anyone recognise the picture. Kept in step
/// with the same list in `src/domain/naming.ts`, which names the video files the save dialog writes.
const NAME_NOISE: &[&str] = &[
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "and", "or", "by",
    "is", "are", "be", "this", "that", "it", "its", "as", "from", "into", "over", "under",
    "very", "highly", "extremely", "detailed", "quality", "masterpiece",
];

/// Lowercase, hyphenated, ASCII-only fragment of `hint`: at most five content words and 44 chars.
///
/// Deliberately lossy. Five words is enough to tell two generations apart in a media grid, and a
/// longer name simply gets truncated by the tile that shows it, which puts the useful part off the
/// end. A hint made entirely of noise still gets a name -- the filter falls back to the raw words.
fn slug(hint: &str) -> String {
    const MAX_WORDS: usize = 5;
    const MAX_LEN: usize = 44;
    let flattened: String = hint.to_lowercase().chars()
        .map(|value| if value.is_ascii_alphanumeric() { value } else { ' ' })
        .collect();
    let words: Vec<&str> = flattened.split_whitespace().collect();
    if words.is_empty() {
        return String::new();
    }
    let meaningful: Vec<&str> = words.iter().copied().filter(|word| !NAME_NOISE.contains(word)).collect();
    let source = if meaningful.is_empty() { words } else { meaningful };
    let mut kept: Vec<&str> = Vec::new();
    for word in source {
        if kept.len() >= MAX_WORDS {
            break;
        }
        if kept.is_empty() {
            // Every slug is allowed one word even if that word is enormous; truncating it beats
            // returning nothing, because nothing means the name falls back to the bare prefix.
            kept.push(if word.len() > MAX_LEN { &word[..MAX_LEN] } else { word });
            continue;
        }
        let joined = kept.iter().map(|value| value.len() + 1).sum::<usize>() + word.len();
        if joined > MAX_LEN {
            break;
        }
        kept.push(word);
    }
    kept.join("-")
}

/// The readable part of an existing filename, so a cutout or an upscale inherits whatever made the
/// picture it came from recognisable instead of starting over from nothing.
fn stem_hint(path: &str) -> String {
    Path::new(path).file_stem().map(|value| value.to_string_lossy().into_owned()).unwrap_or_default()
}

/// `zit-rusted-lighthouse-dusk.png`, falling back to `zit.png` when there is nothing to name it for.
///
/// The old scheme was `{prefix}-{timestamp}-{uuid}.png`. That is unique and completely opaque: a
/// Project Media grid of those tells you nothing, and the part a truncated tile label actually
/// shows is the part that carries no information. Uniqueness now comes from a counter checked
/// against the directory, which keeps the meaning at the front where it stays visible.
fn output_path(root: &Path, project_id: &str, prefix: &str, hint: &str) -> Result<PathBuf, String> {
    let directory = output_dir(root, project_id)?;
    let fragment = slug(hint);
    let stem = if fragment.is_empty() { prefix.to_owned() } else { format!("{prefix}-{fragment}") };
    let first = directory.join(format!("{stem}.png"));
    if !first.exists() {
        return Ok(first);
    }
    for index in 2..=9999u32 {
        let candidate = directory.join(format!("{stem}-{index:02}.png"));
        if !candidate.exists() {
            return Ok(candidate);
        }
    }
    // Ten thousand runs of one prompt. Give up on readability rather than on writing the file.
    Ok(directory.join(format!("{stem}-{}.png", Uuid::new_v4().simple())))
}

fn command(root: &Path, mode: &str, model: &Path, output: &Path) -> Command {
    let mut command = Command::new(python(root));
    command
        .args(["-m", "aiv_runtime.image_studio.worker", mode])
        .args(["--model-path", model.to_string_lossy().as_ref()])
        .args(["--output-path", output.to_string_lossy().as_ref()])
        .current_dir(root)
        .env("PYTHONPATH", root.join("ai-runtime"))
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .env("HF_HOME", root.join("cache/huggingface"))
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("QWEN_OFFLOAD_MODE", "df_stream")
        .env("QWEN_STREAM_RESERVE_GB", "4.0")
        .env("QWEN_STREAM_PIN", "1")
        .stdin(Stdio::null());
    command
}

fn result(output: std::process::Output, expected: &Path) -> Result<ImageResult, String> {
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code()));
    }
    if !expected.is_file() { return Err("the image worker completed without creating output".into()); }
    let payload: Value = stdout.lines().rev()
        .find_map(|line| serde_json::from_str::<Value>(line).ok().filter(|value| value["event"] == "complete"))
        .ok_or("the image worker returned no completion event")?;
    Ok(ImageResult {
        output_path: expected.to_string_lossy().into_owned(),
        mask_path: payload["maskPath"].as_str().map(str::to_owned),
        seed: payload["seed"].as_i64(),
        width: payload["width"].as_u64().map(|value| value as u32),
        height: payload["height"].as_u64().map(|value| value as u32),
        lora_active: payload["loraActive"].as_bool(),
        lora_path: payload["loraPath"].as_str().map(str::to_owned),
        log: if stderr.is_empty() { stdout } else { format!("{stdout}\n{stderr}") },
    })
}

fn batch_result(output: std::process::Output, expected: &[PathBuf]) -> Result<ImageBatchResult, String> {
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code()));
    }
    if expected.iter().any(|path| !path.is_file()) {
        return Err("the Z-Image worker completed without creating every candidate".into());
    }
    let payload: Value = stdout.lines().rev()
        .find_map(|line| serde_json::from_str::<Value>(line).ok().filter(|value| value["event"] == "complete"))
        .ok_or("the Z-Image worker returned no completion event")?;
    let rows = payload["outputs"].as_array().ok_or("the Z-Image worker returned no candidate list")?;
    if rows.len() != expected.len() {
        return Err("the Z-Image worker returned an incomplete candidate list".into());
    }
    let log = if stderr.is_empty() { stdout } else { format!("{stdout}\n{stderr}") };
    let images = rows.iter().zip(expected).map(|(row, path)| ImageResult {
        output_path: path.to_string_lossy().into_owned(),
        mask_path: None,
        seed: row["seed"].as_i64(),
        width: row["width"].as_u64().map(|value| value as u32),
        height: row["height"].as_u64().map(|value| value as u32),
        lora_active: row["loraActive"].as_bool(),
        lora_path: row["loraPath"].as_str().map(str::to_owned),
        log: log.clone(),
    }).collect();
    Ok(ImageBatchResult { images })
}

pub async fn generate(request: ZitRequest) -> Result<ImageResult, String> {
    let root = project_root();
    if !zit_ready(&root) { return Err("the ZiT Image Studio runtime is incomplete".into()); }
    if request.prompt.trim().is_empty() { return Err("an image prompt is required".into()); }
    if request.width < 256 || request.height < 256 || request.width % 16 != 0 || request.height % 16 != 0 {
        return Err("ZiT width and height must be multiples of 16 and at least 256".into());
    }
    if request.width as u64 * request.height as u64 > 4_194_304 { return Err("ZiT output is limited to 4 MP".into()); }
    let output_path = output_path(&root, &request.project_id, "zit", &request.prompt)?;
    let output = command(&root, "generate", &zit_model(&root), &output_path)
        .args(["--prompt", &request.prompt, "--width", &request.width.to_string(), "--height", &request.height.to_string()])
        .args(["--steps", &request.steps.clamp(1, 50).to_string(), "--seed", &request.seed.to_string()])
        .output().await.map_err(|error| format!("failed to start ZiT worker: {error}"))?;
    result(output, &output_path)
}

pub async fn generate_zimage(request: ZImageRequest) -> Result<ImageBatchResult, String> {
    let root = project_root();
    if !matches!(request.model.as_str(), "turbo" | "base") {
        return Err("Z-Image model must be turbo or base".into());
    }
    let model = if request.model == "base" { zimage_base_model(&root) } else { zit_model(&root) };
    let ready = if request.model == "base" { zimage_base_ready(&root) } else { zit_ready(&root) };
    if !ready { return Err(format!("Z-Image {} is not installed", request.model)); }
    if request.prompt.trim().is_empty() { return Err("an image prompt is required".into()); }
    if request.width < 256 || request.height < 256 || request.width % 16 != 0 || request.height % 16 != 0 {
        return Err("Z-Image width and height must be multiples of 16 and at least 256".into());
    }
    if request.width as u64 * request.height as u64 > 4_194_304 {
        return Err("Z-Image output is limited to 4 MP".into());
    }
    let candidates = request.candidates.clamp(1, 4);
    let mut paths = Vec::with_capacity(candidates as usize);
    for index in 0..candidates {
        let prefix = if request.model == "base" { "zimage" } else { "zit" };
        let hint = if candidates == 1 { request.prompt.clone() } else { format!("{} candidate {}", request.prompt, index + 1) };
        paths.push(output_path(&root, &request.project_id, prefix, &hint)?);
    }
    let serialized_paths = serde_json::to_string(&paths.iter()
        .map(|path| path.to_string_lossy().into_owned()).collect::<Vec<_>>())
        .map_err(|error| format!("could not serialize candidate paths: {error}"))?;
    if let Some(path) = request.lora_path.as_deref() {
        if !Path::new(path).exists() { return Err(format!("Z-Image LoRA not found: {path}")); }
    }
    let mut worker = command(&root, "generate", &model, &paths[0]);
    worker.args(["--model-variant", &request.model, "--prompt", &request.prompt])
        .args(["--negative-prompt", &request.negative_prompt])
        .args(["--width", &request.width.to_string(), "--height", &request.height.to_string()])
        .args(["--steps", &request.steps.clamp(1, 50).to_string()])
        .args(["--cfg", &request.guidance_scale.clamp(0.0, 20.0).to_string()])
        .args(["--seed", &request.seed.to_string(), "--output-paths-json", &serialized_paths])
        .args(["--lora-scale", &request.lora_strength.clamp(0.0, 3.0).to_string()]);
    if let Some(path) = request.lora_path.as_deref() { worker.args(["--lora-path", path]); }
    let output = worker.output().await.map_err(|error| format!("failed to start Z-Image worker: {error}"))?;
    batch_result(output, &paths)
}

pub async fn edit(request: QwenEditRequest) -> Result<ImageResult, String> {
    let root = project_root();
    if !qwen_ready(&root) { return Err("the Qwen Image Edit runtime is incomplete".into()); }
    if request.prompt.trim().is_empty() { return Err("an edit instruction is required".into()); }
    if !(1..=3).contains(&request.image_paths.len()) { return Err("Qwen requires one to three images".into()); }
    for path in &request.image_paths { if !Path::new(path).is_file() { return Err(format!("Qwen input not found: {path}")); } }
    let size = edit_output_size(request.output_width, request.output_height)?;
    let output_path = output_path(&root, &request.project_id, "qwen-edit", &request.prompt)?;
    let base = qwen_base(&root).ok_or("the local Qwen support snapshot is missing")?;
    let mut worker = command(&root, "edit", &qwen_model(&root), &output_path);
    worker.args(["--base-model-path", base.to_string_lossy().as_ref()])
        .args(["--lora-path", qwen_lora(&root).to_string_lossy().as_ref()])
        .args(["--prompt", &request.prompt, "--negative-prompt", &request.negative_prompt])
        .args(["--steps", &request.steps.clamp(1, 40).to_string(), "--cfg", &request.true_cfg_scale.clamp(0.0, 20.0).to_string()])
        .args(["--seed", &request.seed.to_string(), "--pixel-budget", &request.pixel_budget.clamp(262_144, 8_000_000).to_string()]);
    for path in &request.image_paths { worker.args(["--image-path", path]); }
    if let Some((width, height)) = size { worker.args(["--out-width", &width.to_string(), "--out-height", &height.to_string()]); }
    let output = worker.output().await.map_err(|error| format!("failed to start Qwen worker: {error}"))?;
    result(output, &output_path)
}

pub async fn remove_background(request: BackgroundRemovalRequest) -> Result<ImageResult, String> {
    let root = project_root();
    if !birefnet_ready(&root) { return Err("the BiRefNet runtime is incomplete".into()); }
    if !Path::new(&request.image_path).is_file() { return Err("the BiRefNet input image is missing".into()); }
    let output_path = output_path(&root, &request.project_id, "cutout", &stem_hint(&request.image_path))?;
    let mask_path = output_path.with_file_name(format!("{}-mask.png", output_path.file_stem().unwrap_or_default().to_string_lossy()));
    let mut worker = command(&root, "remove-background", &birefnet_model(&root), &output_path);
    worker.args(["--image-path", &request.image_path, "--mask-output-path", mask_path.to_string_lossy().as_ref()])
        .args(["--blur", &request.blur.clamp(0, 5).to_string(), "--dilate", &request.dilate.clamp(0, 10).to_string()])
        .args(["--erode", &request.erode.clamp(0, 10).to_string(), "--threshold", &request.threshold.clamp(-1, 255).to_string()]);
    if request.invert { worker.arg("--invert"); }
    let output = worker.output().await.map_err(|error| format!("failed to start BiRefNet worker: {error}"))?;
    let mut parsed = result(output, &output_path)?;
    parsed.mask_path = Some(mask_path.to_string_lossy().into_owned());
    Ok(parsed)
}

pub async fn flatten_canvas(request: CanvasFlattenRequest) -> Result<ImageResult, String> {
    let root = project_root();
    if !python(&root).is_file() || !worker(&root).is_file() { return Err("the Image Studio compositor runtime is incomplete".into()); }
    if request.layers.is_empty() || !request.layers.iter().any(|layer| layer.visible) { return Err("the canvas has no visible layers".into()); }
    if !matches!(request.crop.as_str(), "tight" | "frame") { return Err("canvas crop must be tight or frame".into()); }
    if !matches!(request.background.as_str(), "transparent" | "black" | "white") { return Err("canvas background must be transparent, black, or white".into()); }
    if request.canvas_width == 0 || request.canvas_height == 0 || request.canvas_width > 8192 || request.canvas_height > 8192 {
        return Err("canvas dimensions must be between 1 and 8192 pixels".into());
    }
    for layer in request.layers.iter().filter(|layer| layer.visible) {
        if !Path::new(&layer.image_path).is_file() { return Err(format!("canvas layer input not found: {}", layer.image_path)); }
        if layer.width == 0 || layer.height == 0 { return Err("visible canvas layer dimensions must be positive".into()); }
        let crops = [layer.crop_left, layer.crop_top, layer.crop_right, layer.crop_bottom];
        if crops.iter().any(|value| !value.is_finite() || !(0.0..100.0).contains(value))
            || layer.crop_left + layer.crop_right >= 100.0
            || layer.crop_top + layer.crop_bottom >= 100.0
        {
            return Err("canvas layer crop bounds must leave visible width and height".into());
        }
    }
    let output_path = output_path(&root, &request.project_id, "canvas", "")?;
    let layers_json = serde_json::to_string(&request.layers).map_err(|error| format!("could not serialize canvas layers: {error}"))?;
    let output = command(&root, "composite", Path::new(""), &output_path)
        .args(["--layers-json", &layers_json, "--crop", &request.crop])
        .args(["--width", &request.canvas_width.to_string(), "--height", &request.canvas_height.to_string()])
        .args(["--padding", &request.padding.min(4096).to_string(), "--background", &request.background])
        .output().await.map_err(|error| format!("failed to start the canvas compositor: {error}"))?;
    result(output, &output_path)
}

// ---------------------------------------------------------------- export and upscale

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageBytesRequest {
    pub path: String,
}

/// Read a picture out of the project as base64.
///
/// Photo Edit has to call `getImageData` on the picture it is grading, and a canvas is only
/// readable if everything drawn into it came from somewhere the page is allowed to read. The asset
/// protocol serves from its own origin, so whether that holds depends on headers this app does not
/// control. A data URL is same-origin by definition and can never taint the canvas, which makes
/// this the fallback: the workspace tries the cheap path first and only comes here if it is refused.
pub async fn read_image_base64(request: ImageBytesRequest) -> Result<String, String> {
    const MAX_BYTES: u64 = 96 * 1024 * 1024;
    let path = Path::new(&request.path);
    let meta = fs::metadata(path)
        .map_err(|error| format!("could not open {}: {error}", path.display()))?;
    if !meta.is_file() {
        return Err(format!("not a file: {}", path.display()));
    }
    if meta.len() > MAX_BYTES {
        return Err(format!("{} is too large to open for editing", path.display()));
    }
    let bytes = fs::read(path)
        .map_err(|error| format!("could not read {}: {error}", path.display()))?;
    Ok(STANDARD.encode(bytes))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EditedImageRequest {
    pub project_id: String,
    /// A complete PNG file in base64, without the `data:` prefix a canvas puts in front of it.
    pub png_base64: String,
    /// The picture this was graded from. Only used to name the output after it.
    #[serde(default)]
    pub source_path: String,
    pub width: u32,
    pub height: u32,
}

/// Store a picture the webview rendered.
///
/// Photo Edit does its grade in a canvas, so the pixels the user is looking at are the pixels that
/// get saved -- there is no second implementation in Python that could disagree with the preview.
/// The cost of that is having to carry the finished PNG back across the bridge as base64, which is
/// why there is a ceiling here: refusing a transfer that is obviously too big beats an opaque
/// out-of-memory somewhere further in. 64 MB of base64 is around a 48 MP PNG, far past anything the
/// canvas will hand us.
pub async fn write_edited_image(request: EditedImageRequest) -> Result<ImageResult, String> {
    const MAX_BASE64: usize = 64 * 1024 * 1024;
    if request.png_base64.len() > MAX_BASE64 {
        return Err("the edited image is too large to save; reduce the working resolution".into());
    }
    let bytes = STANDARD.decode(request.png_base64.as_bytes())
        .map_err(|error| format!("the edited image was not valid base64: {error}"))?;
    // A truncated transfer is far easier to diagnose here than as a corrupt file three steps later.
    if !bytes.starts_with(&[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]) {
        return Err("the edited image did not arrive as a PNG".into());
    }
    let root = project_root();
    let output = output_path(&root, &request.project_id, "edit", &stem_hint(&request.source_path))?;
    fs::write(&output, &bytes)
        .map_err(|error| format!("could not write the edited image: {error}"))?;
    Ok(ImageResult {
        output_path: output.to_string_lossy().into_owned(),
        mask_path: None,
        seed: None,
        width: Some(request.width),
        height: Some(request.height),
        lora_active: None,
        lora_path: None,
        log: format!("wrote {} bytes to {}", bytes.len(), output.display()),
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageExportRequest {
    pub source_path: String,
    /// Where the user chose to put it, through a save dialog. Not project storage.
    pub output_path: String,
    /// `png`, `jpeg`, `webp` or `tiff`. Blank infers from the output suffix.
    #[serde(default)]
    pub format: String,
    #[serde(default = "default_quality")]
    pub quality: u32,
    #[serde(default)]
    pub lossless: bool,
    /// `#rrggbb` to composite transparency onto. Required to flatten alpha into JPEG.
    #[serde(default)]
    pub matte: Option<String>,
    /// Optional resize on the way out. Both must be set, or neither.
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
}

fn default_quality() -> u32 { 92 }

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageUpscaleRequest {
    pub source_path: String,
    pub project_id: String,
    /// `esrgan` or `nvvfx`. They are installed independently and neither implies the other.
    #[serde(default = "default_backend")]
    pub backend: String,
    /// A weight filename from `models/real-esrgan/`. Unused by the nvvfx backend.
    #[serde(default)]
    pub model_file: Option<String>,
    /// A `VideoSuperRes.QualityLevel` name. Unused by the esrgan backend.
    #[serde(default = "default_ai_quality")]
    pub quality: String,
    #[serde(default = "default_tile")]
    pub tile: u32,
    /// Optional exact target. The model can only produce its own integer factor, so a target that
    /// differs from the model's output is reached by resampling afterwards -- down, normally.
    #[serde(default)]
    pub target_width: Option<u32>,
    #[serde(default)]
    pub target_height: Option<u32>,
}

fn default_tile() -> u32 { 512 }
fn default_backend() -> String { "esrgan".to_owned() }
fn default_ai_quality() -> String { "HIGH".to_owned() }

fn export_worker(root: &Path) -> PathBuf {
    root.join("ai-runtime/aiv_runtime/export_image.py")
}

/// Write an image out to a path, format and quality the user chose.
///
/// The alpha rule is enforced here as well as in the worker, because reaching the worker at all
/// means having already written a partial file in some cases -- and "your cutout came back with a
/// black edge" is precisely the failure this whole command exists to prevent.
pub async fn export_image(request: ImageExportRequest) -> Result<ImageResult, String> {
    let root = project_root();
    if !python(&root).is_file() || !export_worker(&root).is_file() {
        return Err("the Image Studio runtime is incomplete".into());
    }
    let source = Path::new(&request.source_path);
    if !source.is_file() {
        return Err(format!("image not found: {}", request.source_path));
    }
    let destination = PathBuf::from(request.output_path.trim());
    if destination == source {
        return Err("an export cannot overwrite its own source".into());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("could not create the export folder: {error}"))?;
    }
    let format = request.format.trim().to_ascii_lowercase();
    if !format.is_empty() && !matches!(format.as_str(), "png" | "jpeg" | "webp" | "tiff") {
        return Err("format must be png, jpeg, webp or tiff".into());
    }
    if let Some(matte) = request.matte.as_deref() {
        let hex = matte.trim().trim_start_matches('#');
        if hex.len() != 6 || !hex.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err("matte must be a #rrggbb colour".into());
        }
    }
    if request.width.is_some() != request.height.is_some() {
        return Err("an export resize needs both a width and a height".into());
    }

    let mut command = Command::new(python(&root));
    command
        .arg(export_worker(&root))
        .args(["--input", request.source_path.as_str()])
        .args(["--output", destination.to_string_lossy().as_ref()])
        .args(["--quality", &request.quality.clamp(1, 100).to_string()])
        .current_dir(&root)
        .env("PYTHONPATH", root.join("ai-runtime"))
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::null());
    if !format.is_empty() {
        command.args(["--format", &format]);
    }
    if request.lossless {
        command.arg("--lossless");
    }
    if let Some(matte) = request.matte.as_deref() {
        command.args(["--matte", matte]);
    }
    if let (Some(width), Some(height)) = (request.width, request.height) {
        command.args(["--width", &width.to_string(), "--height", &height.to_string()]);
    }
    let output = command.output().await
        .map_err(|error| format!("failed to start the image exporter: {error}"))?;
    result(output, &destination)
}

/// Run Real-ESRGAN over one still, optionally resampling to an exact size afterwards.
///
/// The output lands in project storage rather than at a path of the user's choosing -- an upscale
/// produces a new asset for the bin, and getting it onto disk somewhere else is what `export_image`
/// is for. Keeping the two separate means an upscale never has to ask where to save.
pub async fn upscale_image(request: ImageUpscaleRequest) -> Result<ImageResult, String> {
    use crate::ai::upscale;

    let root = project_root();
    if !Path::new(&request.source_path).is_file() {
        return Err(format!("image not found: {}", request.source_path));
    }
    if request.target_width.is_some() != request.target_height.is_some() {
        return Err("an upscale target needs both a width and a height".into());
    }
    let nvvfx = request.backend == "nvvfx";

    // Backend-specific arguments, validated before a worker is spawned.
    let mut backend_args: Vec<String> = vec!["--backend".into(), request.backend.clone()];
    if nvvfx {
        if !upscale::nvvfx_ready(&root) {
            return Err(format!(
                "NVIDIA VFX is not installed ({}). Run: Install-RTX-Upscale.bat -AcceptLicense",
                upscale::nvvfx_state(&root)
            ));
        }
        if !upscale::NVVFX_QUALITY_LEVELS.contains(&request.quality.as_str()) {
            return Err(format!(
                "unknown VideoSuperRes quality level: {}. Expected one of {}",
                request.quality, upscale::NVVFX_QUALITY_LEVELS.join(", ")
            ));
        }
        // VideoSuperRes is built around a fixed output size, so unlike the spandrel path the
        // target is required rather than optional.
        let (Some(width), Some(height)) = (request.target_width, request.target_height) else {
            return Err("the NVIDIA VFX backend needs an explicit target width and height".into());
        };
        backend_args.extend([
            "--quality".into(), request.quality.clone(),
            "--target-width".into(), width.to_string(),
            "--target-height".into(), height.to_string(),
        ]);
    } else {
        if !upscale::ready(&root) {
            return Err(format!(
                "the Real-ESRGAN pack is not ready ({}). Run: .venv\\Scripts\\python.exe scripts\\models.py install real-esrgan",
                upscale::state_label(&root)
            ));
        }
        let model_file = request.model_file.as_deref()
            .ok_or("the Real-ESRGAN backend needs a model file")?;
        if !upscale::AI_MODEL_FILES.contains(&model_file) {
            return Err(format!("unknown Real-ESRGAN model: {model_file}"));
        }
        let weights = root.join("models/real-esrgan").join(model_file);
        if !weights.is_file() {
            return Err(format!("Real-ESRGAN weights not installed: {}", weights.display()));
        }
        backend_args.extend([
            "--weights".into(), weights.to_string_lossy().into_owned(),
            "--tile".into(), request.tile.min(2048).to_string(),
        ]);
    }
    let output_path = output_path(&root, &request.project_id, "upscale", &stem_hint(&request.source_path))?;

    let output = Command::new(python(&root))
        .arg(root.join("ai-runtime/aiv_runtime/upscale.py"))
        .args(["--input", request.source_path.as_str()])
        .args(["--output", output_path.to_string_lossy().as_ref()])
        .args(&backend_args)
        .current_dir(&root)
        .env("PYTHONPATH", root.join("ai-runtime"))
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("failed to start the upscale worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code()));
    }
    if !output_path.is_file() {
        return Err("the upscale worker finished without writing an image".into());
    }

    // The resample stage, when the model's own factor did not land on the requested size. Pillow
    // rather than scale_cuda: this is one still, already in a Python process's reach, and routing
    // it through an NVENC-only filter chain would be slower and would need a codec that stills
    // do not otherwise involve.
    if let (false, Some(width), Some(height)) = (nvvfx, request.target_width, request.target_height) {
        let scaled = output_path.with_extension("scaled.png");
        // The AI stage has already written into the project's media folder. If the resample fails
        // -- a missing export_image.py is enough -- returning here would leave the un-resampled
        // upscale behind as an asset the user never asked for, one per attempt.
        let published = async {
            let resized = export_image(ImageExportRequest {
                source_path: output_path.to_string_lossy().into_owned(),
                output_path: scaled.to_string_lossy().into_owned(),
                format: "png".into(), quality: 100, lossless: true, matte: None,
                width: Some(width), height: Some(height),
            }).await?;
            fs::rename(&resized.output_path, &output_path)
                .map_err(|error| format!("could not publish the resampled upscale: {error}"))
        }.await;
        if let Err(error) = published {
            let _ = fs::remove_file(&scaled);
            let _ = fs::remove_file(&output_path);
            return Err(error);
        }
    }

    let payload: Value = stdout.lines().rev()
        .find_map(|line| serde_json::from_str::<Value>(line).ok().filter(|value| value["event"] == "done"))
        .unwrap_or(Value::Null);
    Ok(ImageResult {
        output_path: output_path.to_string_lossy().into_owned(),
        mask_path: None,
        seed: None,
        width: request.target_width.or_else(|| payload["width"].as_u64().map(|v| v as u32)),
        height: request.target_height.or_else(|| payload["height"].as_u64().map(|v| v as u32)),
        lora_active: None,
        lora_path: None,
        log: if stderr.is_empty() { stdout } else { format!("{stdout}\n{stderr}") },
    })
}

#[cfg(test)]
mod edit_size_tests {
    use super::edit_output_size;

    #[test]
    fn the_output_size_is_both_sides_or_neither_and_in_range() {
        assert_eq!(edit_output_size(0, 0), Ok(None));
        assert_eq!(edit_output_size(1344, 768), Ok(Some((1344, 768))));
        assert_eq!(edit_output_size(2048, 2048), Ok(Some((2048, 2048))));
        assert!(edit_output_size(1344, 0).is_err());
        assert!(edit_output_size(0, 768).is_err());
        assert!(edit_output_size(128, 768).is_err());
        assert!(edit_output_size(4096, 768).is_err());
    }
}
