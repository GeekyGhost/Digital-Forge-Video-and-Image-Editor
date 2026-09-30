//! VFX Phase 5: Wan Animate 2 character insertion. One character image and a driving clip (the plate's
//! own range, or any other clip) become a video of that character performing the clip's motion, framed
//! like the plate; SAM 2 and BiRefNet matte it into an RGBA element that goes into the element library.
//!
//! - `generate` runs `aiv_runtime.wan_animate generate` in `vendor/Wan-Animate/.venv` (bake if needed,
//!   then encode and animate, each its own process). The caller holds `JobKind::WanAnimate`. The worker
//!   prints one JSON event per line; each is handed to `on_event` for the status line.
//! - `matte` runs `aiv_runtime.character_matte` in the Image Studio environment (BiRefNet lives there)
//!   over the colour frames and an optional SAM 2 track of them (`ai::cache_mask_track`, the caller
//!   holds `JobKind::Sam2Track` for that). The caller holds `JobKind::BackgroundRemoval` for the matte.
//! - `register` writes the element manifest: fps is the driving clip's, and `extra.animate` keeps the
//!   plate crop the output covers, which is how the editor lays the element exactly over the plate.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs, path::{Path, PathBuf}, process::Stdio, time::Instant};
use tokio::{io::{AsyncBufReadExt, BufReader}, process::Command};

pub const LICENCE: &str = "Wan2.2-Animate-2-14B-Distilled: Apache-2.0 (Wan-AI / Alibaba)";
pub const SOURCE: &str = "wan-animate";

pub fn python(project: &Path) -> PathBuf {
    project.join("vendor/Wan-Animate/.venv/Scripts/python.exe")
}
fn models(project: &Path) -> PathBuf {
    project.join("models/Wan-Animate-2")
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnimateStatus {
    /// The environment exists and the weights needed at run time are present.
    pub ready: bool,
    /// The FP8 bake exists (otherwise the first run bakes it, which needs the BF16 shards).
    pub baked: bool,
    pub licence: &'static str,
    pub detail: String,
}

pub fn status(project: &Path) -> AnimateStatus {
    let env = python(project).is_file();
    let distilled = models(project).join("distilled");
    let weights = ["transformer/config.json", "text_encoder/config.json", "image_encoder/config.json", "vae/config.json",
                   "scheduler/scheduler_config.json", "model_index.json"]
        .iter().all(|name| distilled.join(name).is_file());
    let baked = models(project).join("baked/wan-animate-bake.json").is_file();
    let shards = distilled.join("transformer/diffusion_pytorch_model.safetensors.index.json").is_file();
    let detail = if !env {
        "Wan Animate 2 is not set up. Run Setup-Wan-Animate.bat.".to_owned()
    } else if !weights {
        "Wan Animate 2's weights are incomplete; rerun Setup-Wan-Animate.bat (downloads resume).".to_owned()
    } else if !baked && !shards {
        "The FP8 bake is missing and so are the transformer shards; rerun Setup-Wan-Animate.bat.".to_owned()
    } else {
        String::new()
    };
    AnimateStatus { ready: detail.is_empty(), baked, licence: LICENCE, detail }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CharacterRequest {
    pub name: String,
    /// The character image (any size; it is padded, never stretched).
    pub character_path: String,
    pub prompt: String,
    /// The driving clip and the source frames [first_frame, first_frame + frame_count) of it.
    pub driving_video: String,
    pub first_frame: u32,
    pub frame_count: u32,
    /// The driving clip's size and rate: the output keeps its aspect (cropped a few pixels to 16 px
    /// steps) and its frames one-to-one.
    pub plate_width: u32,
    pub plate_height: u32,
    pub fps: f64,
    #[serde(default = "default_steps")]
    pub steps: u32,
    #[serde(default)]
    pub seed: u64,
    /// Output pixels; 512 x 512 is the 16 GB preset.
    #[serde(default = "default_area")]
    pub area: u32,
    /// SAM 2 prompts on the first driving frame, normalised 0..1 to the DRIVING CLIP (the editor takes
    /// them from the plate's actor matte). Mapped into the output with the worker's crop
    /// (`output_points`) once it is known. Empty: BiRefNet alone.
    #[serde(default)]
    pub actor_points: Vec<super::Sam2Point>,
    /// The actor's (centroid x, feet line, standing height) normalised to the driving clip. The worker
    /// scales and places the character picture to match: Animate 2 keeps the REFERENCE's framing, so an
    /// unframed full-height picture comes out taller than the actor (first 4090 run, 2026-09-29).
    #[serde(default)]
    pub actor_frame: Option<[f64; 3]>,
    /// The actor's union box over the range (x0, y0, x1, y1) normalised to the driving clip. The worker
    /// drives with that neighbourhood (grown by a margin, at the output aspect) instead of the whole
    /// frame, so the character gets most of the output's pixels.
    #[serde(default)]
    pub actor_region: Option<[f64; 4]>,
}

fn default_steps() -> u32 { 10 }
fn default_area() -> u32 { 512 * 512 }

/// What SAM 2 tracks at most (`sam2_track.MAX_TRACK_FRAMES`); a longer range is refused up front
/// rather than after an Animate run of many minutes.
pub const MAX_FRAMES: u32 = 900;

pub fn validate(request: &CharacterRequest) -> Result<(), String> {
    if request.name.trim().is_empty() { return Err("name the character".into()); }
    if !Path::new(&request.character_path).is_file() { return Err("pick a character image".into()); }
    if !Path::new(&request.driving_video).is_file() { return Err("the driving clip is missing".into()); }
    if request.frame_count < 5 { return Err("the driving range needs at least 5 frames".into()); }
    if request.frame_count > MAX_FRAMES {
        return Err(format!("the driving range is {} frames; at most {MAX_FRAMES} can be matted", request.frame_count));
    }
    if request.plate_width < 64 || request.plate_height < 64 { return Err("the driving clip is too small".into()); }
    if !(request.fps > 0.0 && request.fps <= 240.0) { return Err("the driving clip has no frame rate".into()); }
    if !(1..=50).contains(&request.steps) { return Err("steps must be 1 to 50 (10 is the distilled preset)".into()); }
    if !(256 * 256..=1024 * 1024).contains(&request.area) { return Err("the output area must be 256^2 to 1024^2 pixels".into()); }
    if request.actor_points.iter().any(|p| !(0.0..=1.0).contains(&p.x) || !(0.0..=1.0).contains(&p.y)) {
        return Err("SAM 2 points are normalised 0..1".into());
    }
    if let Some([cx, feet, standing]) = request.actor_frame {
        if !(-0.5..=1.5).contains(&cx) || !(0.0..=1.5).contains(&feet) || !(0.01..=1.5).contains(&standing) {
            return Err("the actor framing is outside the clip".into());
        }
    }
    if let Some([x0, y0, x1, y1]) = request.actor_region {
        if !(0.0..=1.0).contains(&x0) || !(0.0..=1.0).contains(&y0) || !(x0 < x1 && x1 <= 1.0) || !(y0 < y1 && y1 <= 1.0) {
            return Err("the actor region is outside the clip".into());
        }
    }
    if !request.actor_points.is_empty() && !request.actor_points.iter().any(|p| p.include) {
        return Err("SAM 2 needs at least one include point".into());
    }
    Ok(())
}

/// Driving-clip-normalised points as output-normalised ones: the output covers `crop` (x, y, w, h in
/// the clip's pixels). Points outside the crop are dropped (SAM 2 refuses them).
pub fn output_points(points: &[super::Sam2Point], crop: [f64; 4], plate_width: u32, plate_height: u32) -> Vec<super::Sam2Point> {
    let [x, y, w, h] = crop;
    points.iter().filter_map(|point| {
        let nx = (point.x * plate_width as f64 - x) / w;
        let ny = (point.y * plate_height as f64 - y) / h;
        ((0.0..=1.0).contains(&nx) && (0.0..=1.0).contains(&ny))
            .then(|| super::Sam2Point { x: (nx * 1e4).round() / 1e4, y: (ny * 1e4).round() / 1e4, include: point.include })
    }).collect()
}

/// The crop the worker reported in its stats.
pub fn reported_crop(stats: &Value) -> Option<[f64; 4]> {
    let values: Vec<f64> = stats.get("crop")?.as_array()?.iter().filter_map(|v| v.as_f64()).collect();
    (values.len() == 4 && values[2] > 0.0 && values[3] > 0.0).then(|| [values[0], values[1], values[2], values[3]])
}

pub fn worker_args(request: &CharacterRequest, out_dir: &Path) -> Vec<String> {
    let mut args: Vec<String> = ["-m", "aiv_runtime.wan_animate", "generate"].iter().map(|v| v.to_string()).collect();
    for (flag, value) in [
        ("--character", request.character_path.clone()),
        ("--prompt", request.prompt.clone()),
        ("--driving-video", request.driving_video.clone()),
        ("--first-frame", request.first_frame.to_string()),
        ("--frame-count", request.frame_count.to_string()),
        ("--plate-width", request.plate_width.to_string()),
        ("--plate-height", request.plate_height.to_string()),
        ("--fps", format!("{}", request.fps)),
        ("--steps", request.steps.to_string()),
        ("--seed", request.seed.to_string()),
        ("--area", request.area.to_string()),
        ("--out-dir", out_dir.to_string_lossy().into_owned()),
        ("--actor-frame", request.actor_frame.map(|[cx, feet, standing]| format!("{cx},{feet},{standing}")).unwrap_or_default()),
        ("--actor-region", request.actor_region.map(|[x0, y0, x1, y1]| format!("{x0},{y0},{x1},{y1}")).unwrap_or_default()),
    ] {
        args.push(flag.into());
        args.push(value);
    }
    args
}

/// Runs one Animate 2 generation into `out_dir` (`frames/00000.png ...`, `animate.mp4`, `stats.json`,
/// `worker.log`). Returns the worker's stats plus exit status and wall time.
pub async fn generate(project: &Path, request: &CharacterRequest, out_dir: &Path, on_event: impl Fn(Value)) -> Result<Value, String> {
    let python = python(project);
    if !python.is_file() { return Err("Wan Animate 2 is not set up. Run Setup-Wan-Animate.bat.".into()); }
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let log = fs::File::create(out_dir.join("worker.log")).map_err(|error| error.to_string())?;
    let started = Instant::now();
    let mut command = Command::new(&python);
    #[cfg(windows)]
    command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    let mut child = command.args(worker_args(request, out_dir))
        .current_dir(project.join("ai-runtime"))
        .env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8")
        .env("HF_HUB_OFFLINE", "1").env("TRANSFORMERS_OFFLINE", "1")
        .stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::from(log))
        .spawn().map_err(|error| format!("could not start the Wan Animate worker: {error}"))?;
    if let Some(stdout) = child.stdout.take() {
        let mut lines = BufReader::new(stdout).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Ok(event) = serde_json::from_str::<Value>(line.trim()) {
                if event.get("event").is_some() { on_event(event); }
            }
        }
    }
    let status = child.wait().await.map_err(|error| error.to_string())?;
    let mut stats: Value = fs::read_to_string(out_dir.join("stats.json")).ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_else(|| serde_json::json!({ "ok": false }));
    stats["exitStatus"] = Value::String(status.to_string());
    stats["wallSeconds"] = serde_json::json!(started.elapsed().as_secs_f64());
    Ok(stats)
}

/// RGBA `frame_%04d.png` into `element_dir` from the colour frames and an optional SAM 2 cache.
pub async fn matte(project: &Path, frames_dir: &Path, sam2_cache: Option<&Path>, element_dir: &Path) -> Result<Value, String> {
    let python = super::image::python(project);
    if !python.is_file() { return Err("the Image Studio environment (BiRefNet) is missing; run Setup-Image-Studio.bat".into()); }
    let birefnet = if super::image::birefnet_ready(project) { project.join("models/BiRefNet") } else { PathBuf::new() };
    if birefnet.as_os_str().is_empty() && sam2_cache.is_none() {
        return Err("neither BiRefNet nor a SAM 2 track is available to cut the character out".into());
    }
    let mut command = Command::new(python);
    #[cfg(windows)]
    command.creation_flags(0x0800_0000);
    command.arg(project.join("ai-runtime/aiv_runtime/character_matte.py"))
        .arg("--frames-dir").arg(frames_dir)
        .arg("--out-dir").arg(element_dir)
        .arg("--birefnet-root").arg(&birefnet);
    if let Some(cache) = sam2_cache { command.arg("--sam2-dir").arg(cache); }
    let output = command.current_dir(project)
        .env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8").env("PYTHONUTF8", "1")
        .env("HF_HUB_OFFLINE", "1").env("TRANSFORMERS_OFFLINE", "1")
        .stdin(Stdio::null()).output().await
        .map_err(|error| format!("could not start the matte worker: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr.chars().rev().take(800).collect::<Vec<_>>().into_iter().rev().collect();
        return Err(format!("the matte failed: {}", tail.trim()));
    }
    stdout.lines().rev().find_map(|line| serde_json::from_str::<Value>(line.trim()).ok())
        .ok_or_else(|| "the matte worker returned no stats".into())
}

/// Register the matted frames in `dir` as an element. `extra.animate` carries the plate crop the output
/// covers (`crop` = x, y, w, h in the driving clip's pixels) and the driving range.
pub fn register(dir: &Path, id: &str, request: &CharacterRequest, stats: &Value, matte: &Value) -> Result<super::elements::ElementEntry, String> {
    use super::elements::{png_sequence, png_size, write_manifest, ElementManifest, PATTERN};
    // Only the matte's `frame_%04d.png`: anything else in the folder (a stray picture) is not a frame.
    let files: Vec<PathBuf> = png_sequence(dir).map_err(|error| format!("the matte wrote no frames: {error}"))?
        .into_iter().filter(|path| path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with("frame_"))).collect();
    if files.is_empty() { return Err(format!("the matte wrote no frame_*.png into {}", dir.display())); }
    let (width, height) = png_size(&fs::read(&files[0]).map_err(|error| error.to_string())?)?;
    let mut extra = serde_json::Map::new();
    extra.insert("animate".into(), serde_json::json!({
        "crop": stats.get("crop"), "outputWidth": stats.get("width"), "outputHeight": stats.get("height"),
        "plateWidth": request.plate_width, "plateHeight": request.plate_height,
        "drivingVideo": request.driving_video, "firstFrame": request.first_frame, "frameCount": request.frame_count,
        "characterPath": request.character_path, "steps": request.steps, "licence": LICENCE, "actorFrame": request.actor_frame, "actorRegion": request.actor_region,
    }));
    extra.insert("generation".into(), stats.clone());
    extra.insert("matte".into(), matte.clone());
    let manifest = ElementManifest {
        version: 1, id: id.into(), name: request.name.trim().to_string(), frame_count: files.len() as u32,
        fps: request.fps, width, height, pattern: PATTERN.into(), proxy: None,
        loop_in: 0, loop_out: None, crossfade: 0, source: SOURCE.into(),
        prompt: Some(request.prompt.clone()), seed: Some(request.seed), created_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        extra,
    };
    write_manifest(dir, &manifest)?;
    Ok(super::elements::ElementEntry { dir: dir.to_string_lossy().into_owned(), manifest })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(dir: &Path) -> CharacterRequest {
        let image = dir.join("hero.png");
        let clip = dir.join("plate.mp4");
        fs::write(&image, b"x").unwrap();
        fs::write(&clip, b"x").unwrap();
        CharacterRequest {
            name: "Hero".into(), character_path: image.to_string_lossy().into(), prompt: "a knight".into(),
            driving_video: clip.to_string_lossy().into(), first_frame: 30, frame_count: 124, plate_width: 960, plate_height: 544,
            fps: 24.0, steps: 10, seed: 7, area: 512 * 512, actor_points: vec![super::super::Sam2Point { x: 0.5, y: 0.4, include: true }],
            actor_frame: Some([0.49, 0.755, 0.555]),
            actor_region: Some([0.415, 0.2, 0.733, 0.757]),
        }
    }

    fn temp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aiv-animate-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn validation_names_the_problem() {
        let dir = temp("validate");
        let ok = request(&dir);
        assert!(validate(&ok).is_ok());
        for (change, needle) in [
            (CharacterRequest { frame_count: 4, ..ok.clone() }, "at least 5"),
            (CharacterRequest { frame_count: 901, ..ok.clone() }, "at most 900"),
            (CharacterRequest { steps: 0, ..ok.clone() }, "steps"),
            (CharacterRequest { actor_frame: Some([0.5, 0.8, 0.0]), ..ok.clone() }, "framing"),
            (CharacterRequest { actor_region: Some([0.6, 0.2, 0.5, 0.8]), ..ok.clone() }, "region"),
            (CharacterRequest { area: 100, ..ok.clone() }, "area"),
            (CharacterRequest { character_path: "missing.png".into(), ..ok.clone() }, "character image"),
            (CharacterRequest { actor_points: vec![super::super::Sam2Point { x: 0.5, y: 0.5, include: false }], ..ok.clone() }, "include point"),
        ] {
            let error = validate(&change).unwrap_err();
            assert!(error.contains(needle), "{error}");
        }
    }

    #[test]
    fn worker_arguments_carry_the_range_and_the_plate() {
        let dir = temp("args");
        let args = worker_args(&request(&dir), Path::new("out"));
        let joined = args.join(" ");
        assert!(joined.starts_with("-m aiv_runtime.wan_animate generate"));
        for pair in ["--first-frame 30", "--frame-count 124", "--plate-width 960", "--plate-height 544", "--fps 24", "--steps 10", "--area 262144", "--actor-frame 0.49,0.755,0.555", "--actor-region 0.415,0.2,0.733,0.757"] {
            assert!(joined.contains(pair), "{pair} in {joined}");
        }
    }

    #[test]
    fn actor_points_map_through_the_crop_into_the_output() {
        use super::super::Sam2Point;
        let points = [Sam2Point { x: 0.5, y: 0.5, include: true }, Sam2Point { x: 0.001, y: 0.5, include: true }, Sam2Point { x: 0.25, y: 0.2, include: false }];
        let mapped = output_points(&points, [4.0, 0.0, 952.0, 544.0], 960, 544);
        assert_eq!(mapped.len(), 2, "the point left of the crop is dropped");
        assert!((mapped[0].x - 0.5).abs() < 1e-4 && (mapped[0].y - 0.5).abs() < 1e-4);
        assert!((mapped[1].x - (240.0 - 4.0) / 952.0).abs() < 1e-4 && !mapped[1].include);
        assert_eq!(reported_crop(&serde_json::json!({ "crop": [4, 0, 952, 544] })), Some([4.0, 0.0, 952.0, 544.0]));
        assert_eq!(reported_crop(&serde_json::json!({ "crop": [4, 0] })), None);
    }

    #[test]
    fn status_reports_what_is_missing() {
        let dir = temp("status");
        let missing = status(&dir);
        assert!(!missing.ready && missing.detail.contains("Setup-Wan-Animate.bat"));
    }

    #[test]
    fn register_keeps_the_plate_crop_for_placement() {
        let dir = temp("register");
        let png = {
            let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, b'I', b'H', b'D', b'R'];
            bytes.extend(672u32.to_be_bytes());
            bytes.extend(384u32.to_be_bytes());
            bytes.extend([8, 6, 0, 0, 0]);
            bytes
        };
        for index in 0..3 { fs::write(dir.join(format!("frame_{index:04}.png")), &png).unwrap(); }
        let stats = serde_json::json!({ "crop": [4, 0, 952, 544], "width": 672, "height": 384, "ok": true });
        let entry = register(&dir, "hero-1", &request(&dir), &stats, &serde_json::json!({ "frames": 3 })).unwrap();
        assert_eq!((entry.manifest.width, entry.manifest.height, entry.manifest.frame_count), (672, 384, 3));
        assert_eq!(entry.manifest.source, "wan-animate");
        assert_eq!(entry.manifest.fps, 24.0);
        assert_eq!(entry.manifest.extra["animate"]["crop"], serde_json::json!([4, 0, 952, 544]));
        assert_eq!(entry.manifest.extra["animate"]["firstFrame"], 30);
        let reread = super::super::elements::read_manifest(&dir).unwrap();
        assert_eq!(reread.extra["animate"]["plateWidth"], 960);
    }
}
