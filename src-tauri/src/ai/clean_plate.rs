//! VFX Phase 5: a clean plate for a locked-off shot, so the actor a generated character replaces
//! does not show where the character does not cover him (hair above a helmet, a sleeve past an arm).
//!
//! `ai-runtime/aiv_runtime/clean_plate.py` (Image Studio environment: numpy, Pillow, OpenCV) builds
//! the background behind the actor from the range and its neighbouring frames and writes a patch
//! element: per plate frame, that background with alpha = the actor matte grown and feathered,
//! cropped to the actor's neighbourhood. Pixels no frame shows are LaMa-filled when LaMa is
//! installed (Telea otherwise). The element records its placement exactly as a character does
//! (`extra.animate`: the plate crop it covers, 1:1, and the range), so the editor lays it over the
//! plate with the same code, between the plate and the character.

use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use serde::Deserialize;
use serde_json::Value;
use tokio::process::Command;

pub const SOURCE: &str = "clean-plate";
/// SAM 2's cap, the same as a character's range.
pub const MAX_FRAMES: u32 = 900;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CleanPlateRequest {
    pub name: String,
    /// The plate's source video.
    pub video: String,
    /// The plate's range in source frames (decode index), as it drives a character.
    pub first_frame: u32,
    pub frame_count: u32,
    /// Frames in the whole source clip: context frames are looked for either side of the range.
    #[serde(default)]
    pub total_frames: u32,
    pub fps: f64,
    pub plate_width: u32,
    pub plate_height: u32,
    /// The actor's SAM 2 track (`mask_cache.py` layout: `<cache>/masks/00000.png ...`).
    pub matte_cache: String,
    /// The source frame `masks/00000.png` belongs to.
    #[serde(default)]
    pub matte_first: u32,
}

pub fn validate(request: &CleanPlateRequest) -> Result<(), String> {
    if request.name.trim().is_empty() { return Err("the clean plate needs a name".into()); }
    if !Path::new(&request.video).is_file() { return Err(format!("the plate's video is missing: {}", request.video)); }
    if request.frame_count == 0 || request.frame_count > MAX_FRAMES {
        return Err(format!("a clean plate covers 1 to {MAX_FRAMES} frames; this range has {}", request.frame_count));
    }
    if request.plate_width == 0 || request.plate_height == 0 { return Err("the plate has no size yet".into()); }
    if !(request.fps > 0.0) { return Err("the plate has no frame rate".into()); }
    if !masks_dir(request).is_dir() {
        return Err(format!("the actor matte is missing (no masks folder in {}); run SAM 2 on the plate first", request.matte_cache));
    }
    Ok(())
}

pub fn masks_dir(request: &CleanPlateRequest) -> PathBuf {
    Path::new(&request.matte_cache).join("masks")
}

pub fn worker_args(project: &Path, request: &CleanPlateRequest, out_dir: &Path, still: Option<&Path>) -> Vec<OsString> {
    let mut args: Vec<OsString> = vec![project.join("ai-runtime/aiv_runtime/clean_plate.py").into()];
    let mut push = |flag: &str, value: String| { args.push(flag.into()); args.push(value.into()); };
    push("--video", request.video.clone());
    push("--first", request.first_frame.to_string());
    push("--count", request.frame_count.to_string());
    push("--total-frames", request.total_frames.to_string());
    push("--fps", format!("{}", request.fps));
    push("--plate-width", request.plate_width.to_string());
    push("--plate-height", request.plate_height.to_string());
    push("--matte-dir", masks_dir(request).to_string_lossy().into_owned());
    push("--matte-first", request.matte_first.to_string());
    push("--out-dir", out_dir.to_string_lossy().into_owned());
    if let Some(still) = still { push("--still", still.to_string_lossy().into_owned()); }
    args
}

/// The worker's last JSON line; an error when it reports `ok: false` or printed none.
pub fn parse_report(stdout: &str, stderr: &str) -> Result<Value, String> {
    let report = stdout.lines().rev().find_map(|line| serde_json::from_str::<Value>(line.trim()).ok());
    match report {
        Some(value) if value.get("ok").and_then(Value::as_bool) == Some(true) => Ok(value),
        Some(value) => Err(format!("the clean plate failed: {}", value.get("error").and_then(Value::as_str).unwrap_or("no reason given"))),
        None => {
            let tail: String = stderr.chars().rev().take(800).collect::<Vec<_>>().into_iter().rev().collect();
            Err(format!("the clean plate worker printed no report: {}", tail.trim()))
        }
    }
}

/// One worker run into `out_dir`: build the still and the patch, or (with `still`) the patch only.
pub async fn run(project: &Path, request: &CleanPlateRequest, out_dir: &Path, still: Option<&Path>) -> Result<Value, String> {
    let python = super::image::python(project);
    if !python.is_file() { return Err("the Image Studio environment is missing; run Setup-Image-Studio.bat".into()); }
    fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let mut command = Command::new(python);
    #[cfg(windows)]
    command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    let output = command.args(worker_args(project, request, out_dir, still))
        .current_dir(project)
        .env("PYTHONUNBUFFERED", "1").env("PYTHONIOENCODING", "utf-8").env("PYTHONUTF8", "1")
        .stdin(Stdio::null()).output().await
        .map_err(|error| format!("could not start the clean plate worker: {error}"))?;
    parse_report(&String::from_utf8_lossy(&output.stdout), &String::from_utf8_lossy(&output.stderr))
}

/// Register the patch frames in `dir` as an element placed like a character over the same plate.
pub fn register(dir: &Path, id: &str, request: &CleanPlateRequest, stats: &Value) -> Result<super::elements::ElementEntry, String> {
    use super::elements::{png_sequence, png_size, write_manifest, ElementManifest, PATTERN};
    let files: Vec<PathBuf> = png_sequence(dir).map_err(|error| format!("the clean plate wrote no frames: {error}"))?
        .into_iter().filter(|path| path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with("frame_"))).collect();
    if files.is_empty() { return Err(format!("the clean plate wrote no frame_*.png into {}", dir.display())); }
    let (width, height) = png_size(&fs::read(&files[0]).map_err(|error| error.to_string())?)?;
    let mut extra = serde_json::Map::new();
    // The same placement record as a character (character.ts `animateRecord`): the patch covers
    // `crop` of each plate frame 1:1, frame i on source frame firstFrame + i.
    extra.insert("animate".into(), serde_json::json!({
        "crop": stats.get("crop"), "outputWidth": width, "outputHeight": height,
        "plateWidth": request.plate_width, "plateHeight": request.plate_height,
        "drivingVideo": request.video, "firstFrame": request.first_frame, "frameCount": request.frame_count,
    }));
    extra.insert("cleanPlate".into(), stats.clone());
    let manifest = ElementManifest {
        version: 1, id: id.into(), name: request.name.trim().to_string(), frame_count: files.len() as u32,
        fps: request.fps, width, height, pattern: PATTERN.into(), proxy: None,
        loop_in: 0, loop_out: None, crossfade: 0, source: SOURCE.into(),
        prompt: None, seed: None, created_at: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true),
        extra,
    };
    write_manifest(dir, &manifest)?;
    Ok(super::elements::ElementEntry { dir: dir.to_string_lossy().into_owned(), manifest })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aiv-clean-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn request(dir: &Path) -> CleanPlateRequest {
        let video = dir.join("plate.mp4");
        fs::write(&video, b"x").unwrap();
        fs::create_dir_all(dir.join("track/masks")).unwrap();
        CleanPlateRequest {
            name: "Dancer clean plate".into(), video: video.to_string_lossy().into(), first_frame: 30, frame_count: 75,
            total_frames: 3733, fps: 25.0, plate_width: 1920, plate_height: 1080,
            matte_cache: dir.join("track").to_string_lossy().into(), matte_first: 30,
        }
    }

    #[test]
    fn validation_names_the_problem() {
        let dir = temp("validate");
        let good = request(&dir);
        assert!(validate(&good).is_ok());
        assert!(validate(&CleanPlateRequest { frame_count: 901, ..good.clone() }).unwrap_err().contains("900"));
        assert!(validate(&CleanPlateRequest { matte_cache: dir.join("none").to_string_lossy().into(), ..good.clone() }).unwrap_err().contains("SAM 2"));
        assert!(validate(&CleanPlateRequest { video: "missing.mp4".into(), ..good.clone() }).unwrap_err().contains("missing"));
        assert!(validate(&CleanPlateRequest { plate_width: 0, ..good }).unwrap_err().contains("size"));
    }

    #[test]
    fn worker_args_carry_the_range_the_track_and_an_optional_still() {
        let dir = temp("args");
        let request = request(&dir);
        let text = |args: Vec<OsString>| args.into_iter().map(|arg| arg.to_string_lossy().into_owned()).collect::<Vec<_>>();
        let args = text(worker_args(Path::new("C:/proj"), &request, Path::new("C:/proj/elements/x"), None));
        assert!(args[0].ends_with("clean_plate.py"));
        let value = |flag: &str| args[args.iter().position(|arg| arg == flag).unwrap() + 1].clone();
        assert_eq!(value("--first"), "30");
        assert_eq!(value("--count"), "75");
        assert_eq!(value("--total-frames"), "3733");
        assert_eq!(value("--matte-first"), "30");
        assert!(value("--matte-dir").ends_with("masks"));
        assert!(!args.contains(&"--still".to_string()));
        let again = text(worker_args(Path::new("C:/proj"), &request, Path::new("C:/x"), Some(Path::new("C:/x/clean-filled.png"))));
        assert_eq!(again.last().unwrap(), "C:/x/clean-filled.png");
    }

    #[test]
    fn the_report_is_the_last_json_line() {
        let ok = parse_report("noise\n{\"ok\": true, \"holes\": 0}\n", "").unwrap();
        assert_eq!(ok["holes"], 0);
        let refused = parse_report("{\"ok\": false, \"error\": \"the shot is not locked off\"}", "").unwrap_err();
        assert!(refused.contains("not locked off"));
        assert!(parse_report("", "Traceback: boom").unwrap_err().contains("boom"));
    }

    #[test]
    fn register_places_it_like_a_character() {
        let dir = temp("register");
        let request = request(&dir);
        let out = dir.join("element");
        fs::create_dir_all(&out).unwrap();
        // A PNG header is enough for the size probe; clean.png is not a frame.
        let png = |path: &Path| {
            let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13, b'I', b'H', b'D', b'R'];
            bytes.extend(742u32.to_be_bytes());
            bytes.extend(730u32.to_be_bytes());
            bytes.extend([8, 6, 0, 0, 0]);
            fs::write(path, bytes).unwrap();
        };
        png(&out.join("frame_0000.png"));
        png(&out.join("frame_0001.png"));
        png(&out.join("clean.png"));
        let stats = serde_json::json!({ "ok": true, "crop": [733, 152, 742, 730], "holes": 0 });
        let entry = register(&out, "dancer-clean", &request, &stats).unwrap();
        assert_eq!(entry.manifest.frame_count, 2);
        assert_eq!(entry.manifest.source, "clean-plate");
        let animate = &entry.manifest.extra["animate"];
        assert_eq!(animate["crop"], serde_json::json!([733, 152, 742, 730]));
        assert_eq!(animate["outputWidth"], 742);
        assert_eq!(animate["firstFrame"], 30);
        assert_eq!(animate["drivingVideo"], serde_json::Value::String(request.video.clone()));
    }
}
