//! The three model-backed steps Photo Edit can ask for: depth, generative fill and click-selection.
//!
//! Depth and fill use the utility venv. SAM 2 selection can also use the isolated Image Studio
//! environment, which avoids coupling its native dependencies to unrelated workers. Every feature
//! is gated on both a healthy Python environment and its weights being present.

use super::*;
use serde_json::Value;
use sha2::{Digest, Sha256};

pub fn depth_model(root: &Path) -> PathBuf {
    root.join("models/depth-anything-v2")
}

pub fn inpaint_model(root: &Path) -> PathBuf {
    root.join("models/lama/big-lama.pt")
}

fn adapter(root: &Path, name: &str) -> PathBuf {
    root.join("ai-runtime/aiv_runtime").join(name)
}

/// A transformers directory is only usable with its config; the weights alone will not load.
pub fn depth_ready(root: &Path) -> bool {
    super::venv_python_usable(&super::utility_python(root))
        && depth_model(root).join("config.json").is_file()
        && adapter(root, "photo_depth.py").is_file()
}

pub fn inpaint_ready(root: &Path) -> bool {
    super::venv_python_usable(&super::utility_python(root))
        && inpaint_model(root).is_file()
        && adapter(root, "photo_inpaint.py").is_file()
}

/// Shares the video tracker's pack, so installing SAM 2 once covers both.
pub fn select_ready(root: &Path) -> bool {
    super::sam2_python(root).is_some()
        && super::sam2_model_installed(root)
        && adapter(root, "photo_select.py").is_file()
}

/// Everything Photo Edit derives lives here, beside the project it belongs to, so deleting a
/// project takes its caches with it rather than leaving them in a shared scratch directory.
fn photo_dir(root: &Path, project_id: &str, leaf: &str) -> Result<PathBuf, String> {
    let id = Uuid::parse_str(project_id).map_err(|_| "invalid project id".to_owned())?;
    let path = root.join("projects").join(id.to_string()).join("photoedit").join(leaf);
    fs::create_dir_all(&path)
        .map_err(|error| format!("could not create the Photo Edit cache directory: {error}"))?;
    Ok(path)
}

/// A short, filesystem-safe name for a cache key. Truncated because 16 bytes of SHA-256 is already
/// far past any chance of collision among the handful of edits one project will hold.
fn digest(key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    hasher.finalize().iter().take(16).map(|byte| format!("{byte:02x}")).collect()
}

fn python_command(root: &Path, script: &str) -> Command {
    python_command_with(root, script, super::utility_python(root))
}

fn python_command_with(root: &Path, script: &str, python: PathBuf) -> Command {
    let mut command = Command::new(python);
    command
        .arg(adapter(root, script))
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

fn finish(output: std::process::Output, expected: &Path) -> Result<image::ImageResult, String> {
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() {
        return Err(super::concise_worker_error_with_status(&stdout, &stderr, output.status.code()));
    }
    if !expected.is_file() {
        return Err("the worker finished without writing its output".into());
    }
    let payload: Value = stdout.lines().rev()
        .find_map(|line| serde_json::from_str::<Value>(line).ok().filter(|value| value["event"] == "complete"))
        .unwrap_or(Value::Null);
    Ok(image::ImageResult {
        output_path: expected.to_string_lossy().into_owned(),
        mask_path: None,
        seed: None,
        width: payload["width"].as_u64().map(|value| value as u32),
        height: payload["height"].as_u64().map(|value| value as u32),
        lora_active: None,
        lora_path: None,
        log: if stderr.is_empty() { stdout } else { format!("{stdout}\n{stderr}") },
    })
}

/* ------------------------------------------------------------------- depth */

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhotoDepthRequest {
    pub project_id: String,
    pub image_path: String,
}

/// Depth is keyed on the source path alone: the same picture always produces the same map, and the
/// map does not depend on anything the user can change in the workspace. So a second relight node,
/// or reopening the document tomorrow, is a cache hit rather than another model run.
pub async fn depth(request: PhotoDepthRequest) -> Result<image::ImageResult, String> {
    let root = project_root();
    if !depth_ready(&root) {
        return Err("the depth model is not installed".into());
    }
    if !Path::new(&request.image_path).is_file() {
        return Err("the picture to measure is missing".into());
    }
    let output = photo_dir(&root, &request.project_id, "depth")?
        .join(format!("{}.png", digest(&request.image_path)));
    if output.is_file() {
        return Ok(image::ImageResult {
            output_path: output.to_string_lossy().into_owned(),
            mask_path: None, seed: None, width: None, height: None,
            lora_active: None, lora_path: None,
            log: "depth map already computed".into(),
        });
    }
    let result = python_command(&root, "photo_depth.py")
        .args(["--model-path", depth_model(&root).to_string_lossy().as_ref()])
        .args(["--image-path", &request.image_path])
        .args(["--output-path", output.to_string_lossy().as_ref()])
        .output().await
        .map_err(|error| format!("failed to start the depth worker: {error}"))?;
    finish(result, &output)
}

/* ------------------------------------------------------------------ fill */

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhotoInpaintRequest {
    pub project_id: String,
    pub image_path: String,
    /// The rasterised mask as a base64 PNG, white where the fill should happen.
    pub mask_png_base64: String,
    /// Everything about the document that shaped this fill. Changing any of it misses the cache.
    pub cache_key: String,
}

pub async fn inpaint(request: PhotoInpaintRequest) -> Result<image::ImageResult, String> {
    use base64::{engine::general_purpose::STANDARD, Engine};
    let root = project_root();
    if !inpaint_ready(&root) {
        return Err("the LaMa inpainting model is not installed".into());
    }
    if !Path::new(&request.image_path).is_file() {
        return Err("the picture to edit is missing".into());
    }
    let name = digest(&request.cache_key);
    let cache = photo_dir(&root, &request.project_id, "fill")?;
    let output = cache.join(format!("{name}.png"));
    if output.is_file() {
        return Ok(image::ImageResult {
            output_path: output.to_string_lossy().into_owned(),
            mask_path: None, seed: None, width: None, height: None,
            lora_active: None, lora_path: None,
            log: "fill already computed".into(),
        });
    }

    let mask_bytes = STANDARD.decode(request.mask_png_base64.as_bytes())
        .map_err(|error| format!("the mask was not valid base64: {error}"))?;
    if !mask_bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        return Err("the mask did not arrive as a PNG".into());
    }
    // The mask is written beside the result and kept: it is what the fill was made from, and having
    // it on disk is the difference between a diagnosable bad fill and a mysterious one.
    let mask_path = cache.join(format!("{name}-mask.png"));
    fs::write(&mask_path, &mask_bytes)
        .map_err(|error| format!("could not write the fill mask: {error}"))?;

    let result = python_command(&root, "photo_inpaint.py")
        .args(["--model-path", inpaint_model(&root).to_string_lossy().as_ref()])
        .args(["--image-path", &request.image_path])
        .args(["--mask-path", mask_path.to_string_lossy().as_ref()])
        .args(["--output-path", output.to_string_lossy().as_ref()])
        .output().await
        .map_err(|error| format!("failed to start the inpainting worker: {error}"))?;
    finish(result, &output)
}

/// LaMa over one picture and a white-on-black mask file, into `output` (VFX Phase 5: the holes a
/// clean plate could not see behind). The caller holds the resource permit.
pub async fn lama_fill(image: &Path, mask: &Path, output: &Path) -> Result<(), String> {
    let root = project_root();
    if !inpaint_ready(&root) {
        return Err("LaMa is not installed".into());
    }
    let result = python_command(&root, "photo_inpaint.py")
        .args(["--model-path", inpaint_model(&root).to_string_lossy().as_ref()])
        .args(["--image-path", image.to_string_lossy().as_ref()])
        .args(["--mask-path", mask.to_string_lossy().as_ref()])
        .args(["--output-path", output.to_string_lossy().as_ref()])
        .output().await
        .map_err(|error| format!("failed to start the inpainting worker: {error}"))?;
    finish(result, output).map(|_| ())
}

/* ----------------------------------------------------------------- select */

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhotoPoint {
    pub x: f32,
    pub y: f32,
    pub include: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PhotoSelectRequest {
    pub project_id: String,
    pub image_path: String,
    pub points: Vec<PhotoPoint>,
}

pub async fn select(request: PhotoSelectRequest) -> Result<image::ImageResult, String> {
    let root = project_root();
    if !select_ready(&root) {
        return Err("the SAM 2 pack is not installed".into());
    }
    if !Path::new(&request.image_path).is_file() {
        return Err("the picture to select in is missing".into());
    }
    if request.points.is_empty() {
        return Err("click the subject first".into());
    }
    if !request.points.iter().any(|point| point.include) {
        return Err("at least one point must be an include point".into());
    }
    // Keyed on the clicks as well as the picture, so refining a selection writes a new file rather
    // than overwriting the one an existing mask is still pointing at.
    let key = request.points.iter()
        .map(|point| format!("{:.4},{:.4},{}", point.x, point.y, u8::from(point.include)))
        .collect::<Vec<_>>().join(";");
    let output = photo_dir(&root, &request.project_id, "select")?
        .join(format!("{}.png", digest(&format!("{} {key}", request.image_path))));
    if output.is_file() {
        return Ok(image::ImageResult {
            output_path: output.to_string_lossy().into_owned(),
            mask_path: None, seed: None, width: None, height: None,
            lora_active: None, lora_path: None,
            log: "selection already computed".into(),
        });
    }

    let python = super::sam2_python(&root)
        .ok_or_else(|| format!("SAM 2 is unavailable ({})", super::sam2_state(&root)))?;
    let mut command = python_command_with(&root, "photo_select.py", python);
    command
        .args(["--model-root", super::sam2_root(&root).to_string_lossy().as_ref()])
        .args(["--image-path", &request.image_path])
        .args(["--output-path", output.to_string_lossy().as_ref()]);
    for point in &request.points {
        command.args(["--point", &format!("{},{},{}", point.x, point.y, u8::from(point.include))]);
    }
    let result = command.output().await
        .map_err(|error| format!("failed to start the selection worker: {error}"))?;
    finish(result, &output)
}
