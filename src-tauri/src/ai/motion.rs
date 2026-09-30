//! VFX Phase 4: the motion runtime -- CoTracker3 point tracks and Video Depth Anything Small depth
//! sequences, each a Python worker in `runtimes/motion/.venv` (built by `Setup-Motion.bat`).
//!
//! Both results are cached under `.aiv/cache/` keyed on the clip (path, size, modification time)
//! and the request, so asking again is free and never takes the GPU gate:
//! - tracks: `.aiv/cache/tracks/<sha256>.json` (the worker's JSON; the app stores the samples inline
//!   in the Photo Edit document, and this file is the provenance);
//! - depth: `.aiv/cache/depth/<sha256>/masks/00000.png ...` plus `manifest.json`, the same layout as
//!   a SAM 2 matte cache, so the compositor reads it with the sequence loader it already has.
//!
//! The callers in `lib.rs` hold `JobKind::PointTrack` / `JobKind::VideoDepth` permits.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{fs, path::{Path, PathBuf}, process::Stdio, time::Instant};
use tokio::process::Command;

pub const COTRACKER_LICENCE: &str = "CoTracker3: CC-BY-NC 4.0 (Meta Platforms), non-commercial use only";
pub const DEPTH_LICENCE: &str = "Video Depth Anything Small: Apache-2.0 (ByteDance)";

pub fn python(project: &Path) -> PathBuf {
    project.join("runtimes/motion/.venv/Scripts/python.exe")
}
pub fn cotracker_checkpoint(project: &Path) -> PathBuf {
    project.join("models/motion/scaled_offline.pth")
}
pub fn depth_checkpoint(project: &Path) -> PathBuf {
    project.join("models/motion/video_depth_anything_vits.pth")
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MotionStatus {
    /// The environment exists (`Setup-Motion.bat` has run).
    pub ready: bool,
    pub cotracker: bool,
    pub depth: bool,
    pub cotracker_licence: &'static str,
    pub depth_licence: &'static str,
    /// What is missing, in words, when something is.
    pub detail: String,
}

pub fn status(project: &Path) -> MotionStatus {
    let ready = python(project).is_file();
    let cotracker = ready && cotracker_checkpoint(project).is_file() && project.join("vendor/co-tracker/cotracker").is_dir();
    let depth = ready && depth_checkpoint(project).is_file() && project.join("vendor/Video-Depth-Anything/video_depth_anything").is_dir();
    let detail = if !ready {
        "The motion runtime is not set up. Run Setup-Motion.bat.".to_owned()
    } else if !cotracker || !depth {
        let mut missing = Vec::new();
        if !cotracker { missing.push("CoTracker3"); }
        if !depth { missing.push("Video Depth Anything"); }
        format!("{} missing; rerun Setup-Motion.bat.", missing.join(" and "))
    } else {
        String::new()
    };
    MotionStatus { ready, cotracker, depth, cotracker_licence: COTRACKER_LICENCE, depth_licence: DEPTH_LICENCE, detail }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TrackPointsRequest {
    pub video_path: String,
    /// The source frame the points were picked on.
    pub query_frame: u32,
    /// Continuous source pixels (a pixel's centre is i + 0.5).
    pub points: Vec<[f64; 2]>,
    #[serde(default)]
    pub start_frame: u32,
    /// 0 is to the end of the clip (the worker caps it).
    #[serde(default)]
    pub frame_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VideoDepthRequest {
    pub video_path: String,
}

pub fn validate_track(request: &TrackPointsRequest) -> Result<(), String> {
    if request.points.is_empty() { return Err("pick at least one point to track".into()); }
    if request.points.len() > 64 { return Err("track at most 64 points at a time".into()); }
    if request.points.iter().any(|[x, y]| !x.is_finite() || !y.is_finite() || *x < 0.0 || *y < 0.0) {
        return Err("a point is outside the clip".into());
    }
    if request.query_frame < request.start_frame {
        return Err("the frame the point was picked on is before the tracked range".into());
    }
    if request.frame_count > 0 && request.query_frame >= request.start_frame + request.frame_count {
        return Err("the frame the point was picked on is after the tracked range".into());
    }
    Ok(())
}

/// The clip's identity: its path, size and modification time. A re-encoded file is a new clip.
fn clip_hasher(video: &Path) -> Result<Sha256, String> {
    let metadata = fs::metadata(video).map_err(|error| format!("cannot inspect {}: {error}", video.display()))?;
    if !metadata.is_file() { return Err(format!("not a file: {}", video.display())); }
    let mut hasher = Sha256::new();
    hasher.update(video.to_string_lossy().as_bytes());
    hasher.update(metadata.len().to_le_bytes());
    if let Ok(since) = metadata.modified().map(|time| time.duration_since(std::time::UNIX_EPOCH)) {
        if let Ok(since) = since { hasher.update(since.as_nanos().to_le_bytes()); }
    }
    Ok(hasher)
}

pub fn track_cache_path(project: &Path, request: &TrackPointsRequest) -> Result<PathBuf, String> {
    let mut hasher = clip_hasher(Path::new(&request.video_path))?;
    hasher.update(b"cotracker3-v1");
    hasher.update(serde_json::to_vec(request).map_err(|error| error.to_string())?);
    Ok(project.join(".aiv/cache/tracks").join(format!("{:x}.json", hasher.finalize())))
}

pub fn depth_cache_dir(project: &Path, request: &VideoDepthRequest) -> Result<PathBuf, String> {
    let mut hasher = clip_hasher(Path::new(&request.video_path))?;
    hasher.update(b"video-depth-anything-small-v1");
    Ok(project.join(".aiv/cache/depth").join(format!("{:x}", hasher.finalize())))
}

/// A finished track for this request, if one is cached.
pub fn cached_track(project: &Path, request: &TrackPointsRequest) -> Result<Option<Value>, String> {
    let path = track_cache_path(project, request)?;
    let Ok(text) = fs::read_to_string(&path) else { return Ok(None) };
    let mut value: Value = serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    value["reused"] = Value::Bool(true);
    value["cachePath"] = Value::String(path.to_string_lossy().into_owned());
    Ok(Some(value))
}

/// A finished depth sequence for this clip, if one is cached.
pub fn cached_depth(project: &Path, request: &VideoDepthRequest) -> Result<Option<Value>, String> {
    let dir = depth_cache_dir(project, request)?;
    let Ok(text) = fs::read_to_string(dir.join("manifest.json")) else { return Ok(None) };
    let mut value: Value = serde_json::from_str(&text).map_err(|error| format!("{}: {error}", dir.display()))?;
    value["reused"] = Value::Bool(true);
    value["cachePath"] = Value::String(dir.to_string_lossy().into_owned());
    Ok(Some(value))
}

/// The last lines of a worker log, for an error message.
fn log_tail(path: &Path) -> String {
    let text = fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty()).collect();
    lines[lines.len().saturating_sub(6)..].join("\n")
}

async fn run_worker(project: &Path, module: &str, args: Vec<String>, log: &Path) -> Result<(), String> {
    let python = python(project);
    if !python.is_file() { return Err("The motion runtime is not set up. Run Setup-Motion.bat.".into()); }
    if let Some(parent) = log.parent() { fs::create_dir_all(parent).map_err(|error| error.to_string())?; }
    let log_file = fs::File::create(log).map_err(|error| error.to_string())?;
    let mut command = Command::new(&python);
    #[cfg(windows)]
    command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    let status = command
        .args(["-m", module])
        .args(&args)
        .current_dir(project.join("ai-runtime"))
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("HF_HUB_OFFLINE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log_file))
        .status()
        .await
        .map_err(|error| format!("could not start the {module} worker: {error}"))?;
    if !status.success() {
        return Err(format!("{module} failed ({status}):\n{}\nSee {}", log_tail(log), log.display()));
    }
    Ok(())
}

/// Track the request's points with CoTracker3 (uncached; see `cached_track`).
pub async fn track_points(project: &Path, request: &TrackPointsRequest) -> Result<Value, String> {
    validate_track(request)?;
    let state = status(project);
    if !state.cotracker { return Err(state.detail); }
    let out = track_cache_path(project, request)?;
    let started = Instant::now();
    let mut args = vec![
        "--video".to_owned(), request.video_path.clone(),
        "--out".to_owned(), out.to_string_lossy().into_owned(),
        "--checkpoint".to_owned(), cotracker_checkpoint(project).to_string_lossy().into_owned(),
        "--repo".to_owned(), project.join("vendor/co-tracker").to_string_lossy().into_owned(),
        "--query-frame".to_owned(), request.query_frame.to_string(),
        "--start-frame".to_owned(), request.start_frame.to_string(),
        "--frame-count".to_owned(), request.frame_count.to_string(),
    ];
    for [x, y] in &request.points { args.push("--point".into()); args.push(format!("{x},{y}")); }
    run_worker(project, "aiv_runtime.motion_track", args, &out.with_extension("log")).await?;
    let mut value: Value = serde_json::from_str(&fs::read_to_string(&out).map_err(|error| format!("the tracker wrote nothing: {error}"))?)
        .map_err(|error| format!("the tracker's JSON is invalid: {error}"))?;
    value["reused"] = Value::Bool(false);
    value["cachePath"] = Value::String(out.to_string_lossy().into_owned());
    value["wallSeconds"] = serde_json::json!(started.elapsed().as_secs_f64());
    Ok(value)
}

/// A depth sequence for the whole clip (uncached; see `cached_depth`).
pub async fn video_depth(project: &Path, request: &VideoDepthRequest) -> Result<Value, String> {
    let state = status(project);
    if !state.depth { return Err(state.detail); }
    let dir = depth_cache_dir(project, request)?;
    let started = Instant::now();
    let args = vec![
        "--video".to_owned(), request.video_path.clone(),
        "--cache-dir".to_owned(), dir.to_string_lossy().into_owned(),
        "--checkpoint".to_owned(), depth_checkpoint(project).to_string_lossy().into_owned(),
        "--repo".to_owned(), project.join("vendor/Video-Depth-Anything").to_string_lossy().into_owned(),
    ];
    run_worker(project, "aiv_runtime.video_depth", args, &dir.join("worker.log")).await?;
    let mut value = cached_depth(project, request)?.ok_or("the depth worker wrote no manifest")?;
    value["reused"] = Value::Bool(false);
    value["wallSeconds"] = serde_json::json!(started.elapsed().as_secs_f64());
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aiv-motion-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn request(video: &Path, x: f64) -> TrackPointsRequest {
        TrackPointsRequest { video_path: video.to_string_lossy().into_owned(), query_frame: 4, points: vec![[x, 20.5]], start_frame: 0, frame_count: 0 }
    }

    #[test]
    fn status_names_what_is_missing() {
        let project = scratch("status");
        let state = status(&project);
        assert!(!state.ready && !state.cotracker && !state.depth);
        assert!(state.detail.contains("Setup-Motion.bat"));
        assert!(state.cotracker_licence.contains("CC-BY-NC"));
        fs::create_dir_all(project.join("runtimes/motion/.venv/Scripts")).unwrap();
        fs::write(python(&project), b"").unwrap();
        let state = status(&project);
        assert!(state.ready && !state.cotracker);
        assert!(state.detail.contains("CoTracker3 and Video Depth Anything missing"), "{}", state.detail);
    }

    #[test]
    fn cache_keys_follow_the_clip_and_the_request() {
        let project = scratch("keys");
        let video = project.join("clip.mp4");
        fs::write(&video, b"one").unwrap();
        let a = track_cache_path(&project, &request(&video, 10.5)).unwrap();
        assert_eq!(a, track_cache_path(&project, &request(&video, 10.5)).unwrap());
        assert_ne!(a, track_cache_path(&project, &request(&video, 11.5)).unwrap());
        assert!(a.starts_with(project.join(".aiv/cache/tracks")));
        let depth = depth_cache_dir(&project, &VideoDepthRequest { video_path: video.to_string_lossy().into_owned() }).unwrap();
        fs::write(&video, b"a different, longer clip").unwrap();
        assert_ne!(a, track_cache_path(&project, &request(&video, 10.5)).unwrap());
        assert_ne!(depth, depth_cache_dir(&project, &VideoDepthRequest { video_path: video.to_string_lossy().into_owned() }).unwrap());
        assert!(track_cache_path(&project, &request(&project.join("missing.mp4"), 1.0)).is_err());
    }

    #[test]
    fn cached_results_are_marked_reused() {
        let project = scratch("cached");
        let video = project.join("clip.mp4");
        fs::write(&video, b"clip").unwrap();
        let req = request(&video, 3.5);
        assert!(cached_track(&project, &req).unwrap().is_none());
        let path = track_cache_path(&project, &req).unwrap();
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, br#"{"ok":true,"points":[]}"#).unwrap();
        let value = cached_track(&project, &req).unwrap().unwrap();
        assert_eq!(value["reused"], Value::Bool(true));
        let depth = VideoDepthRequest { video_path: video.to_string_lossy().into_owned() };
        let dir = depth_cache_dir(&project, &depth).unwrap();
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("manifest.json"), br#"{"ok":true,"frames":3}"#).unwrap();
        assert_eq!(cached_depth(&project, &depth).unwrap().unwrap()["frames"], 3);
    }

    #[test]
    fn requests_are_validated() {
        let video = PathBuf::from("clip.mp4");
        assert!(validate_track(&request(&video, 1.0)).is_ok());
        assert!(validate_track(&TrackPointsRequest { points: vec![], ..request(&video, 1.0) }).is_err());
        assert!(validate_track(&TrackPointsRequest { points: vec![[-1.0, 2.0]], ..request(&video, 1.0) }).is_err());
        assert!(validate_track(&TrackPointsRequest { start_frame: 5, ..request(&video, 1.0) }).is_err());
        assert!(validate_track(&TrackPointsRequest { frame_count: 4, ..request(&video, 1.0) }).is_err());
    }
}
