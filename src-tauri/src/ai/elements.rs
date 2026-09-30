//! The element library (VFX Phase 3): RGBA elements with real alpha -- Wan Alpha generations and
//! imported PNG sequences -- kept inside the project at `elements/<id>/`:
//!
//! - `frame_0000.png ...`: the master, straight RGBA, lossless. Final renders read this.
//! - `proxy.mp4`: a stacked H.264 preview (colour left, alpha as luma right; BT.709, limited range,
//!   tagged), decoded by WebCodecs for scrubbing. Built on import/generation; optional.
//! - `manifest.json`: size, rate, frame count, loop defaults, provenance. Unknown keys survive a
//!   rewrite (`extra`), so an older build does not strip what a newer one wrote.
//!
//! Nothing here touches another project's environment or any shared cache.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{fs, path::{Path, PathBuf}, process::Stdio};
use tokio::process::Command;

pub const ELEMENTS_DIR: &str = "elements";
pub const PATTERN: &str = "frame_%04d.png";
pub const PROXY: &str = "proxy.mp4";
const MANIFEST: &str = "manifest.json";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ElementManifest {
    pub version: u32,
    pub id: String,
    pub name: String,
    pub frame_count: u32,
    pub fps: f64,
    pub width: u32,
    pub height: u32,
    pub pattern: String,
    /// File name of the stacked proxy inside the folder, when one was built.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy: Option<String>,
    /// Loop defaults copied onto a layer when it is placed: [loopIn, loopOut) and a crossfade.
    #[serde(default)]
    pub loop_in: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loop_out: Option<u32>,
    #[serde(default)]
    pub crossfade: u32,
    /// "wan-alpha" or "import".
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    pub created_at: String,
    #[serde(flatten)]
    pub extra: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ElementEntry {
    /// The element's folder: a frames layer's `media.path`.
    pub dir: String,
    pub manifest: ElementManifest,
}

pub fn root(project: &Path) -> PathBuf {
    project.join(ELEMENTS_DIR)
}

/// A folder name that is readable and unique: `smoke-wisps-3f9a1c2e`.
pub fn new_id(name: &str) -> String {
    let slug: String = name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect::<String>()
        .split('-').filter(|part| !part.is_empty()).take(5).collect::<Vec<_>>().join("-");
    let short = uuid::Uuid::new_v4().simple().to_string()[..8].to_string();
    if slug.is_empty() { format!("element-{short}") } else { format!("{slug}-{short}") }
}

pub fn read_manifest(dir: &Path) -> Result<ElementManifest, String> {
    let text = fs::read_to_string(dir.join(MANIFEST)).map_err(|error| format!("{}: {error}", dir.display()))?;
    serde_json::from_str(&text).map_err(|error| format!("{}: {error}", dir.join(MANIFEST).display()))
}

/// Written to a temporary file and renamed, so a crash never leaves half a manifest.
pub fn write_manifest(dir: &Path, manifest: &ElementManifest) -> Result<(), String> {
    let text = serde_json::to_string_pretty(manifest).map_err(|error| error.to_string())?;
    let pending = dir.join("manifest.json.pending");
    fs::write(&pending, text).map_err(|error| error.to_string())?;
    fs::rename(&pending, dir.join(MANIFEST)).map_err(|error| error.to_string())
}

fn entry(dir: &Path, manifest: ElementManifest) -> ElementEntry {
    ElementEntry { dir: dir.to_string_lossy().into_owned(), manifest }
}

/// Every element with a readable manifest, newest first. Folders without one are skipped.
pub fn list(project: &Path) -> Vec<ElementEntry> {
    let Ok(read) = fs::read_dir(root(project)) else { return Vec::new() };
    let mut out: Vec<ElementEntry> = read.filter_map(Result::ok)
        .filter(|item| item.path().is_dir())
        .filter_map(|item| read_manifest(&item.path()).ok().map(|manifest| entry(&item.path(), manifest)))
        .collect();
    out.sort_by(|a, b| b.manifest.created_at.cmp(&a.manifest.created_at));
    out
}

pub fn find(project: &Path, id: &str) -> Result<(PathBuf, ElementManifest), String> {
    if id.is_empty() || id.contains(['/', '\\', '.']) {
        return Err(format!("not an element id: {id}"));
    }
    let dir = root(project).join(id);
    let manifest = read_manifest(&dir)?;
    Ok((dir, manifest))
}

/// Width and height from a PNG's IHDR chunk, without decoding it.
pub fn png_size(bytes: &[u8]) -> Result<(u32, u32), String> {
    const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    if bytes.len() < 24 || bytes[..8] != SIGNATURE || &bytes[12..16] != b"IHDR" {
        return Err("not a PNG".into());
    }
    let width = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
    let height = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
    if width == 0 || height == 0 { return Err("PNG has no size".into()); }
    Ok((width, height))
}

/// Sort key that orders `f2.png` before `f10.png`.
pub fn natural_key(name: &str) -> Vec<(u8, u64, String)> {
    let mut out = Vec::new();
    let mut digits = String::new();
    let mut text = String::new();
    for c in name.chars() {
        if c.is_ascii_digit() {
            if !text.is_empty() { out.push((1, 0, std::mem::take(&mut text).to_lowercase())); }
            digits.push(c);
        } else {
            if !digits.is_empty() { out.push((0, digits.parse().unwrap_or(u64::MAX), std::mem::take(&mut digits))); }
            text.push(c);
        }
    }
    if !digits.is_empty() { out.push((0, digits.parse().unwrap_or(u64::MAX), digits)); }
    if !text.is_empty() { out.push((1, 0, text.to_lowercase())); }
    out
}

/// The PNG files of a folder in natural order.
pub fn png_sequence(folder: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = fs::read_dir(folder).map_err(|error| format!("{}: {error}", folder.display()))?
        .filter_map(Result::ok)
        .map(|item| item.path())
        .filter(|path| path.is_file() && path.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("png")))
        .collect();
    files.sort_by_key(|path| natural_key(&path.file_name().unwrap_or_default().to_string_lossy()));
    if files.is_empty() { return Err(format!("no PNG files in {}", folder.display())); }
    Ok(files)
}

fn now() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Copy a folder of RGBA PNGs into a new element as `frame_%04d.png`. Every frame must be one size.
pub fn import_folder(project: &Path, folder: &Path, name: &str, fps: f64) -> Result<ElementEntry, String> {
    let files = png_sequence(folder)?;
    let (width, height) = png_size(&fs::read(&files[0]).map_err(|error| error.to_string())?)?;
    let id = new_id(name);
    let dir = root(project).join(&id);
    fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let copy = || -> Result<(), String> {
        for (index, file) in files.iter().enumerate() {
            let bytes = fs::read(file).map_err(|error| format!("{}: {error}", file.display()))?;
            let size = png_size(&bytes).map_err(|error| format!("{}: {error}", file.display()))?;
            if size != (width, height) {
                return Err(format!("{} is {}x{}, the first frame is {width}x{height}", file.display(), size.0, size.1));
            }
            fs::write(dir.join(format!("frame_{index:04}.png")), bytes).map_err(|error| error.to_string())?;
        }
        Ok(())
    };
    if let Err(error) = copy() {
        // Nothing was registered yet; leave the partial folder out of the listing by not writing a
        // manifest. It is not deleted: this app asks before removing anything.
        return Err(error);
    }
    let manifest = ElementManifest {
        version: 1, id: id.clone(), name: name.trim().to_string(), frame_count: files.len() as u32,
        fps: if fps > 0.0 { fps } else { 24.0 }, width, height, pattern: PATTERN.into(),
        proxy: None, loop_in: 0, loop_out: None, crossfade: 0, source: "import".into(),
        prompt: None, seed: None, created_at: now(), extra: Default::default(),
    };
    write_manifest(&dir, &manifest)?;
    Ok(entry(&dir, manifest))
}

/// Register a finished Wan Alpha run in `dir` (which already holds its frames) as an element.
pub fn register_generation(dir: &Path, id: &str, name: &str, prompt: &str, seed: u64, stats: &Value) -> Result<ElementEntry, String> {
    let files = png_sequence(dir).map_err(|error| format!("the generation wrote no frames: {error}"))?;
    let (width, height) = png_size(&fs::read(&files[0]).map_err(|error| error.to_string())?)?;
    let mut extra = serde_json::Map::new();
    extra.insert("generation".into(), stats.clone());
    let manifest = ElementManifest {
        version: 1, id: id.into(), name: name.trim().to_string(), frame_count: files.len() as u32,
        fps: super::wan_alpha::FPS, width, height, pattern: PATTERN.into(), proxy: None,
        loop_in: 0, loop_out: None, crossfade: 0, source: "wan-alpha".into(),
        prompt: Some(prompt.into()), seed: Some(seed), created_at: now(), extra,
    };
    write_manifest(dir, &manifest)?;
    Ok(entry(dir, manifest))
}

/// ffmpeg arguments for the stacked proxy (the Phase 0 0.5 recipe, which measured alpha error
/// 0.12-0.17/255 and 981 fps decode).
pub fn proxy_args(dir: &Path, fps: f64) -> Vec<String> {
    let input = dir.join(PATTERN).to_string_lossy().into_owned();
    let output = dir.join(PROXY).to_string_lossy().into_owned();
    [
        "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-framerate", &format!("{fps}"), "-start_number", "0", "-i", &input,
        "-filter_complex",
        "[0:v]split[c][a];[c]scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[co];[a]alphaextract,scale=out_range=tv,format=yuv420p[al];[co][al]hstack=inputs=2[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "12", "-g", "16", "-bf", "2",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
        "-movflags", "+faststart", &output,
    ].iter().map(|arg| arg.to_string()).collect()
}

/// Build (or rebuild) the stacked proxy and record it in the manifest. A missing ffmpeg is an error
/// the caller reports; the element itself stays usable from its PNG master.
pub async fn build_proxy(dir: &Path) -> Result<ElementManifest, String> {
    let mut manifest = read_manifest(dir)?;
    let mut command = Command::new("ffmpeg");
    #[cfg(windows)]
    command.creation_flags(0x0800_0000);
    let output = command.args(proxy_args(dir, manifest.fps)).stdin(Stdio::null()).output().await
        .map_err(|error| format!("could not start ffmpeg (is it on PATH?): {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr.chars().rev().take(600).collect::<Vec<_>>().into_iter().rev().collect();
        return Err(format!("ffmpeg could not build the proxy: {tail}"));
    }
    manifest.proxy = Some(PROXY.into());
    write_manifest(dir, &manifest)?;
    Ok(manifest)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ElementUpdate {
    pub id: String,
    pub name: Option<String>,
    pub loop_in: Option<u32>,
    pub loop_out: Option<u32>,
    pub crossfade: Option<u32>,
}

/// Rename an element or change its loop defaults. The frames are never touched.
pub fn update(project: &Path, change: ElementUpdate) -> Result<ElementEntry, String> {
    let (dir, mut manifest) = find(project, &change.id)?;
    if let Some(name) = change.name { if !name.trim().is_empty() { manifest.name = name.trim().into(); } }
    if let Some(value) = change.loop_in { manifest.loop_in = value.min(manifest.frame_count.saturating_sub(1)); }
    if let Some(value) = change.loop_out { manifest.loop_out = Some(value.clamp(manifest.loop_in + 1, manifest.frame_count)); }
    if let Some(value) = change.crossfade { manifest.crossfade = value; }
    write_manifest(&dir, &manifest)?;
    Ok(entry(&dir, manifest))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn png(width: u32, height: u32) -> Vec<u8> {
        let mut bytes = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 13];
        bytes.extend_from_slice(b"IHDR");
        bytes.extend_from_slice(&width.to_be_bytes());
        bytes.extend_from_slice(&height.to_be_bytes());
        bytes.extend_from_slice(&[8, 6, 0, 0, 0, 0, 0, 0, 0]);
        bytes
    }

    fn temp_project(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aiv-elements-{tag}-{}", uuid::Uuid::new_v4().simple()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reads_png_size_and_rejects_other_files() {
        assert_eq!(png_size(&png(832, 480)).unwrap(), (832, 480));
        assert!(png_size(b"GIF89a....................").is_err());
        assert!(png_size(&png(0, 5)).is_err());
    }

    #[test]
    fn natural_order_puts_2_before_10() {
        let mut names = vec!["f10.png", "f2.png", "f1.png", "F3.png"];
        names.sort_by_key(|name| natural_key(name));
        assert_eq!(names, ["f1.png", "f2.png", "F3.png", "f10.png"]);
    }

    #[test]
    fn ids_are_readable_and_safe() {
        let id = new_id("Smoke / wisps: take 2!");
        assert!(id.starts_with("smoke-wisps-take-2-"), "{id}");
        assert!(!id.contains(['/', '\\', '.', ' ']));
        assert!(new_id("!!!").starts_with("element-"));
        assert!(find(Path::new("."), "../x").is_err());
    }

    #[test]
    fn imports_a_folder_in_natural_order_and_lists_it() {
        let project = temp_project("import");
        let source = project.join("source");
        fs::create_dir_all(&source).unwrap();
        for (name, marker) in [("smoke10.png", 10u8), ("smoke2.png", 2), ("smoke1.png", 1), ("notes.txt", 0)] {
            let mut bytes = png(4, 3);
            bytes.push(marker);
            fs::write(source.join(name), bytes).unwrap();
        }
        let made = import_folder(&project, &source, "Smoke wisps", 16.0).unwrap();
        assert_eq!(made.manifest.frame_count, 3);
        assert_eq!((made.manifest.width, made.manifest.height), (4, 3));
        let dir = PathBuf::from(&made.dir);
        let last = fs::read(dir.join("frame_0002.png")).unwrap();
        assert_eq!(*last.last().unwrap(), 10, "smoke10 must be the third frame");
        let listed = list(&project);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].manifest, made.manifest);
        fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn refuses_frames_of_different_sizes() {
        let project = temp_project("sizes");
        let source = project.join("source");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("a1.png"), png(4, 3)).unwrap();
        fs::write(source.join("a2.png"), png(5, 3)).unwrap();
        assert!(import_folder(&project, &source, "bad", 16.0).unwrap_err().contains("first frame is 4x3"));
        assert!(list(&project).is_empty(), "a failed import must not be listed");
        fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn manifest_keeps_unknown_keys_and_updates_loops() {
        let project = temp_project("manifest");
        let source = project.join("source");
        fs::create_dir_all(&source).unwrap();
        for index in 0..10 { fs::write(source.join(format!("f{index}.png")), png(2, 2)).unwrap(); }
        let made = import_folder(&project, &source, "Loop me", 24.0).unwrap();
        let dir = PathBuf::from(&made.dir);
        let mut raw: Value = serde_json::from_str(&fs::read_to_string(dir.join(MANIFEST)).unwrap()).unwrap();
        raw["futureField"] = serde_json::json!({ "kept": true });
        fs::write(dir.join(MANIFEST), raw.to_string()).unwrap();
        let updated = update(&project, ElementUpdate { id: made.manifest.id.clone(), name: None, loop_in: Some(2), loop_out: Some(50), crossfade: Some(3) }).unwrap();
        assert_eq!((updated.manifest.loop_in, updated.manifest.loop_out, updated.manifest.crossfade), (2, Some(10), 3));
        let reread: Value = serde_json::from_str(&fs::read_to_string(dir.join(MANIFEST)).unwrap()).unwrap();
        assert_eq!(reread["futureField"]["kept"], true);
        fs::remove_dir_all(&project).ok();
    }

    #[test]
    fn proxy_arguments_are_tagged_bt709_limited() {
        let args = proxy_args(Path::new("/e/x"), 16.0);
        let joined = args.join(" ");
        assert!(joined.contains("-color_range tv") && joined.contains("-colorspace bt709"));
        assert!(joined.contains("alphaextract") && joined.contains("hstack=inputs=2"));
        assert!(args.last().unwrap().ends_with("proxy.mp4"));
    }
}
