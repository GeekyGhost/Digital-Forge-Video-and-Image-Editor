//! Background removal for timeline clips: a matte sequence (`.aiv/cache/masks/<id>/masks/*.png`,
//! the SAM 2 matte-cache layout) that the timeline applies as the clip's alpha.
//!
//! Three methods (`ai-runtime/aiv_runtime/clip_matte.py` does the work):
//! - `birefnet`: automatic, soft-edged, per frame (Image Studio environment, `BackgroundRemoval` job);
//! - `sam2`: the subject picked with points, on one or several frames (track assist gives the page
//!   CoTracker3 positions to prompt with), propagated both ways (`Sam2Track` job);
//! - `sam2-birefnet`: SAM 2 picks the subject, BiRefNet gives it soft edges (the character matte's
//!   combine): the SAM 2 step, then the BiRefNet step guided by it.
//!
//! Each step is cached on the clip's identity (path, size, modification time) and the request, so
//! changing only the refine step reuses the SAM 2 track.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, path::{Path, PathBuf}, process::Stdio};
use tokio::process::Command;

use super::MaskTrackResult;

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipMattePrompt {
    /// Frame in the range (0 = the range's first frame).
    pub frame: u32,
    /// Normalised to the source picture.
    pub x: f64,
    pub y: f64,
    pub include: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipMatteRequest {
    pub video_path: String,
    /// Source seconds; `end_time <= start_time` is a still image.
    #[serde(default)]
    pub start_time: f64,
    #[serde(default)]
    pub end_time: f64,
    /// `birefnet`, `sam2` or `sam2-birefnet`.
    pub method: String,
    #[serde(default)]
    pub prompts: Vec<ClipMattePrompt>,
    /// BiRefNet's causal smoothing, 0..0.6.
    #[serde(default)]
    pub smooth: f64,
}

pub const METHODS: &[&str] = &["birefnet", "sam2", "sam2-birefnet"];
/// Prompts SAM 2 is given at most (track assist adds a few per frame it re-prompts on).
pub const MAX_PROMPTS: usize = 600;

impl ClipMatteRequest {
    pub fn uses_sam2(&self) -> bool { self.method.starts_with("sam2") }
    pub fn uses_birefnet(&self) -> bool { self.method.ends_with("birefnet") }
}

pub fn validate(request: &ClipMatteRequest) -> Result<(), String> {
    if !METHODS.contains(&request.method.as_str()) { return Err(format!("unknown background removal method {}", request.method)); }
    if !Path::new(&request.video_path).is_file() { return Err("the clip's media file is missing".into()); }
    if request.start_time < 0.0 || !request.start_time.is_finite() || !request.end_time.is_finite() {
        return Err("the clip's range is invalid".into());
    }
    if request.uses_sam2() {
        if !request.prompts.iter().any(|prompt| prompt.include) {
            return Err("pick the subject first: at least one include point".into());
        }
        if request.prompts.len() > MAX_PROMPTS { return Err(format!("too many points ({}); at most {MAX_PROMPTS}", request.prompts.len())); }
        if request.prompts.iter().any(|p| !(0.0..=1.0).contains(&p.x) || !(0.0..=1.0).contains(&p.y)) {
            return Err("a point is outside the clip".into());
        }
    }
    if !(0.0..=0.6).contains(&request.smooth) { return Err("edge smoothing must be between 0 and 0.6".into()); }
    Ok(())
}

fn clip_identity(hasher: &mut Sha256, request: &ClipMatteRequest) -> Result<(), String> {
    let metadata = fs::metadata(&request.video_path).map_err(|error| format!("cannot inspect the clip: {error}"))?;
    hasher.update(request.video_path.as_bytes());
    hasher.update(metadata.len().to_le_bytes());
    if let Ok(Ok(since)) = metadata.modified().map(|time| time.duration_since(std::time::UNIX_EPOCH)) {
        hasher.update(since.as_nanos().to_le_bytes());
    }
    hasher.update(request.start_time.to_le_bytes());
    hasher.update(request.end_time.to_le_bytes());
    Ok(())
}

/// (the SAM 2 step's cache id when there is one, the final cache id).
pub fn cache_ids(request: &ClipMatteRequest) -> Result<(Option<String>, String), String> {
    let sam2 = if request.uses_sam2() {
        let mut hasher = Sha256::new();
        hasher.update(b"clip-matte-sam2-v1");
        clip_identity(&mut hasher, request)?;
        hasher.update(serde_json::to_vec(&request.prompts).map_err(|error| error.to_string())?);
        Some(format!("{:x}", hasher.finalize()))
    } else { None };
    let last = match (request.method.as_str(), &sam2) {
        ("sam2", Some(id)) => id.clone(),
        ("sam2-birefnet", Some(id)) => {
            let mut hasher = Sha256::new();
            hasher.update(b"clip-matte-refine-v1");
            hasher.update(id.as_bytes());
            hasher.update(request.smooth.to_le_bytes());
            format!("{:x}", hasher.finalize())
        }
        _ => {
            let mut hasher = Sha256::new();
            hasher.update(b"clip-matte-birefnet-v1");
            clip_identity(&mut hasher, request)?;
            hasher.update(request.smooth.to_le_bytes());
            format!("{:x}", hasher.finalize())
        }
    };
    Ok((sam2, last))
}

pub fn cache_dir(root: &Path, id: &str) -> PathBuf { root.join(".aiv/cache/masks").join(id) }

fn read_manifest(dir: &Path) -> Option<MaskTrackResult> {
    let text = fs::read_to_string(dir.join("manifest.json")).ok()?;
    let mut result: MaskTrackResult = serde_json::from_str(&text).ok()?;
    result.reused = true;
    result.log = "Reused the cached matte.".into();
    Some(result)
}

/// The finished matte, if this exact request was made before (no job needs to start).
pub fn cached(root: &Path, request: &ClipMatteRequest) -> Result<Option<MaskTrackResult>, String> {
    let (_, last) = cache_ids(request)?;
    Ok(read_manifest(&cache_dir(root, &last)))
}

/// The worker's arguments for one step (after the script path).
pub fn worker_args(mode: &str, request: &ClipMatteRequest, cache: &Path, id: &str, model_root: &Path, guide: Option<&Path>) -> Vec<String> {
    let mut args = vec![
        "--mode".into(), mode.into(), "--video-path".into(), request.video_path.clone(),
        "--start-time".into(), format!("{:.6}", request.start_time), "--end-time".into(), format!("{:.6}", request.end_time),
        "--cache-dir".into(), cache.to_string_lossy().into_owned(), "--cache-id".into(), id.into(),
        "--model-root".into(), model_root.to_string_lossy().into_owned(),
    ];
    if mode == "sam2" {
        for prompt in &request.prompts {
            args.push("--prompt".into());
            args.push(format!("{},{:.6},{:.6},{}", prompt.frame, prompt.x, prompt.y, if prompt.include { 1 } else { 0 }));
        }
    } else {
        args.extend(["--smooth".into(), format!("{:.3}", request.smooth)]);
        if let Some(guide) = guide { args.extend(["--guide-dir".into(), guide.to_string_lossy().into_owned()]); }
    }
    args
}

async fn run(root: &Path, python: &Path, args: Vec<String>, log_name: &str) -> Result<MaskTrackResult, String> {
    let mut command = Command::new(python);
    #[cfg(windows)]
    command.creation_flags(0x0800_0000);
    let output = command.arg(root.join("ai-runtime/aiv_runtime/clip_matte.py")).args(&args).current_dir(root)
        .env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8").env("PYTHONUTF8", "1")
        .env("HF_HUB_OFFLINE", "1").env("TRANSFORMERS_OFFLINE", "1")
        .stdin(Stdio::null()).output().await
        .map_err(|error| format!("could not start the background removal worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() { return Err(super::concise_worker_error_with_status(&stdout, &stderr, output.status.code())); }
    let line = stdout.lines().rev().find(|line| line.trim_start().starts_with('{'))
        .ok_or_else(|| "the background removal worker returned no manifest".to_owned())?;
    let mut result: MaskTrackResult = serde_json::from_str(line).map_err(|error| format!("the matte manifest was unreadable: {error}"))?;
    result.log = super::worker_diagnostics(&stdout, &stderr);
    super::record_worker_log(root, log_name, &result.log);
    Ok(result)
}

/// The SAM 2 step (callers hold the `Sam2Track` gate).
pub async fn sam2_step(root: &Path, request: &ClipMatteRequest) -> Result<MaskTrackResult, String> {
    let (Some(id), _) = cache_ids(request)? else { return Err("this method does not use SAM 2".into()) };
    let dir = cache_dir(root, &id);
    if let Some(done) = read_manifest(&dir) { return Ok(done); }
    let python = super::sam2_python(root).ok_or_else(|| format!("SAM 2 is unavailable ({})", super::sam2_state(root)))?;
    fs::create_dir_all(&dir).map_err(|error| format!("cannot create the matte cache: {error}"))?;
    run(root, &python, worker_args("sam2", request, &dir, &id, &super::sam2_root(root), None), "clip-matte-sam2").await
}

/// The BiRefNet step, refining `guide` when there is one (callers hold the `BackgroundRemoval` gate).
pub async fn birefnet_step(root: &Path, request: &ClipMatteRequest, guide: Option<&MaskTrackResult>) -> Result<MaskTrackResult, String> {
    let (_, id) = cache_ids(request)?;
    let dir = cache_dir(root, &id);
    if let Some(done) = read_manifest(&dir) { return Ok(done); }
    if !super::image::birefnet_ready(root) { return Err("BiRefNet is not installed (models\\BiRefNet and the Image Studio environment); run Setup-Image-Studio.bat".into()); }
    let python = super::image::python(root);
    fs::create_dir_all(&dir).map_err(|error| format!("cannot create the matte cache: {error}"))?;
    let guide_dir = guide.map(|matte| PathBuf::from(&matte.cache_path));
    run(root, &python, worker_args("birefnet", request, &dir, &id, &root.join("models/BiRefNet"), guide_dir.as_deref()), "clip-matte-birefnet").await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(method: &str, path: &Path) -> ClipMatteRequest {
        ClipMatteRequest {
            video_path: path.to_string_lossy().into_owned(), start_time: 1.0, end_time: 4.0, method: method.into(),
            prompts: vec![ClipMattePrompt { frame: 12, x: 0.25, y: 0.5, include: true }, ClipMattePrompt { frame: 12, x: 0.9, y: 0.1, include: false }],
            smooth: 0.2,
        }
    }

    fn clip() -> PathBuf {
        let path = std::env::temp_dir().join(format!("clip-matte-{}.mp4", uuid::Uuid::new_v4()));
        fs::write(&path, b"not really a video").unwrap();
        path
    }

    #[test]
    fn requests_are_checked() {
        let path = clip();
        assert!(validate(&request("sam2", &path)).is_ok());
        assert!(validate(&request("chroma", &path)).is_err());
        let mut no_subject = request("sam2-birefnet", &path);
        no_subject.prompts.retain(|p| !p.include);
        assert!(validate(&no_subject).unwrap_err().contains("include point"));
        let mut auto = request("birefnet", &path);
        auto.prompts.clear();
        assert!(validate(&auto).is_ok(), "BiRefNet needs no points");
        let mut outside = request("sam2", &path);
        outside.prompts[0].x = 1.2;
        assert!(validate(&outside).is_err());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn the_refine_step_reuses_the_sam2_track_and_ids_follow_the_request() {
        let path = clip();
        let (sam2, last) = cache_ids(&request("sam2", &path)).unwrap();
        assert_eq!(sam2.as_deref(), Some(last.as_str()), "SAM 2 alone is its own final matte");
        let (refine_sam2, refined) = cache_ids(&request("sam2-birefnet", &path)).unwrap();
        assert_eq!(refine_sam2, sam2, "the same picks give the same SAM 2 track");
        assert_ne!(refined, last);
        let mut smoother = request("sam2-birefnet", &path);
        smoother.smooth = 0.4;
        let (again, other) = cache_ids(&smoother).unwrap();
        assert_eq!(again, sam2, "changing only the refine step keeps the track");
        assert_ne!(other, refined);
        let (none, auto) = cache_ids(&request("birefnet", &path)).unwrap();
        assert!(none.is_none() && auto != last);
        let _ = fs::remove_file(path);
    }

    #[test]
    fn worker_arguments_carry_the_prompts_or_the_guide() {
        let path = clip();
        let sam2 = worker_args("sam2", &request("sam2", &path), Path::new("C:/c/x"), "x", Path::new("C:/m/sam2"), None).join(" ");
        assert!(sam2.contains("--mode sam2") && sam2.contains("--start-time 1.000000 --end-time 4.000000"), "{sam2}");
        assert!(sam2.contains("--prompt 12,0.250000,0.500000,1 --prompt 12,0.900000,0.100000,0"), "{sam2}");
        let refine = worker_args("birefnet", &request("sam2-birefnet", &path), Path::new("C:/c/y"), "y", Path::new("C:/m/BiRefNet"), Some(Path::new("C:/c/x"))).join(" ");
        assert!(refine.contains("--smooth 0.200 --guide-dir C:/c/x") && !refine.contains("--prompt"), "{refine}");
        let _ = fs::remove_file(path);
    }
}
