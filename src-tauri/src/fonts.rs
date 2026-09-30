//! VFX Phase 6: glyph outlines for "Convert text to outlines".
//!
//! The page lays text out with the canvas (so the outlines land exactly where the text layer drew
//! its letters) and asks here for each character's outline in font units. Fonts are the ones the
//! machine has installed -- `C:\Windows\Fonts` and the per-user `%LOCALAPPDATA%\Microsoft\Windows\Fonts`
//! on Windows, the usual font folders elsewhere -- read locally with ttf-parser (MIT OR Apache-2.0).
//! Nothing is bundled or copied.
//!
//! Finding a family: every font file's `name` and `OS/2` tables are read once per launch (only those
//! tables, by seeking, so a gigabyte of CJK fonts costs a few kilobytes each), then the family is
//! matched case-insensitively on its typographic or legacy family name, and the face nearest the
//! wanted weight and style wins. An absent family falls back to what Chromium uses for the generic
//! the page names (`sans-serif` -> Arial, `serif` -> Times New Roman, `monospace` -> Consolas), and the
//! answer says which file was used, so the page can tell the user when it is not the one asked for.

use serde::Serialize;
use std::{
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    sync::OnceLock,
};

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct FaceInfo {
    pub path: String,
    pub index: u32,
    pub family: String,
    /// Legacy (name ID 1) family, when it differs from the typographic one ("Arial Black" vs "Arial").
    pub legacy_family: String,
    pub weight: u16,
    pub italic: bool,
}

/// One outline command in font units, y up (as the font stores it): `M x y`, `L x y`, `Q x1 y1 x y`,
/// `C x1 y1 x2 y2 x y`, `Z`.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Command(pub char, pub Vec<f32>);

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Glyph {
    pub ch: String,
    pub advance: f32,
    pub commands: Vec<Command>,
    /// False when the font has no glyph for this character (the canvas then used a fallback font).
    pub found: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FontOutlines {
    pub face: FaceInfo,
    pub units_per_em: u16,
    pub ascender: i16,
    pub descender: i16,
    pub glyphs: Vec<Glyph>,
    /// True when the face is not the family asked for (a generic fallback).
    pub substituted: bool,
    /// True when the weight or italic asked for is not the face's own (the canvas synthesises it).
    pub synthetic: bool,
}

fn font_dirs() -> Vec<PathBuf> {
    if let Ok(list) = std::env::var("AIV_FONT_DIRS") {
        return std::env::split_paths(&list).collect();
    }
    let mut dirs = Vec::new();
    if cfg!(windows) {
        let windir = std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into());
        dirs.push(Path::new(&windir).join("Fonts"));
        if let Ok(local) = std::env::var("LOCALAPPDATA") {
            dirs.push(Path::new(&local).join("Microsoft").join("Windows").join("Fonts"));
        }
    } else {
        dirs.push("/usr/share/fonts".into());
        dirs.push("/usr/local/share/fonts".into());
        if let Ok(home) = std::env::var("HOME") { dirs.push(Path::new(&home).join(".fonts")); }
        dirs.push("/Library/Fonts".into());
        dirs.push("/System/Library/Fonts".into());
    }
    dirs
}

fn font_files(dir: &Path, out: &mut Vec<PathBuf>, depth: u32) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if depth < 4 { font_files(&path, out, depth + 1); }
            continue;
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
        if matches!(ext.as_str(), "ttf" | "otf" | "ttc" | "otc") { out.push(path); }
    }
}

fn read_at(file: &mut std::fs::File, offset: u64, length: usize) -> Option<Vec<u8>> {
    if length > 64 << 20 { return None; }
    file.seek(SeekFrom::Start(offset)).ok()?;
    let mut buffer = vec![0u8; length];
    file.read_exact(&mut buffer).ok()?;
    Some(buffer)
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> { b.get(at..at + 4).map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]])) }
fn u16_at(b: &[u8], at: usize) -> Option<u16> { b.get(at..at + 2).map(|s| u16::from_be_bytes([s[0], s[1]])) }

/// The faces in one file, from their `name` and `OS/2` tables only.
fn faces_in(path: &Path) -> Vec<FaceInfo> {
    let Ok(mut file) = std::fs::File::open(path) else { return Vec::new() };
    let Some(head) = read_at(&mut file, 0, 12) else { return Vec::new() };
    let offsets: Vec<u32> = if &head[0..4] == b"ttcf" {
        let count = u32_at(&head, 8).unwrap_or(0).min(256) as usize;
        match read_at(&mut file, 12, count * 4) {
            Some(list) => (0..count).filter_map(|i| u32_at(&list, i * 4)).collect(),
            None => return Vec::new(),
        }
    } else {
        vec![0]
    };
    let mut out = Vec::new();
    for (index, offset) in offsets.into_iter().enumerate() {
        let Some(directory) = read_at(&mut file, offset as u64, 12) else { continue };
        let tables = u16_at(&directory, 4).unwrap_or(0) as usize;
        let Some(records) = read_at(&mut file, offset as u64 + 12, tables * 16) else { continue };
        let (mut name, mut os2) = (None, None);
        for t in 0..tables {
            let r = &records[t * 16..t * 16 + 16];
            let (at, len) = (u32_at(r, 8).unwrap_or(0) as u64, u32_at(r, 12).unwrap_or(0) as usize);
            match &r[0..4] {
                b"name" => name = read_at(&mut file, at, len),
                b"OS/2" => os2 = read_at(&mut file, at, len),
                _ => {}
            }
        }
        let Some(name) = name else { continue };
        let Some(table) = ttf_parser::name::Table::parse(&name) else { continue };
        let pick = |id: u16| -> Option<String> {
            let mut english = None;
            let mut any = None;
            for entry in table.names {
                if entry.name_id != id || !entry.is_unicode() { continue; }
                let Some(text) = entry.to_string() else { continue };
                if entry.language_id == 0x0409 || entry.language_id == 0 { english.get_or_insert(text); } else { any.get_or_insert(text); }
            }
            english.or(any)
        };
        let legacy = pick(1).unwrap_or_default();
        let family = pick(16).unwrap_or_else(|| legacy.clone());
        if family.is_empty() { continue; }
        let (weight, italic) = os2.as_deref().and_then(ttf_parser::os2::Table::parse)
            .map(|t| (t.weight().to_number(), t.style() != ttf_parser::Style::Normal))
            .unwrap_or((400, false));
        out.push(FaceInfo { path: path.to_string_lossy().into_owned(), index: index as u32, family, legacy_family: legacy, weight, italic });
    }
    out
}

static INDEX: OnceLock<Vec<FaceInfo>> = OnceLock::new();

/// Every installed face, read once per launch.
pub fn installed_faces() -> &'static [FaceInfo] {
    INDEX.get_or_init(|| {
        let mut files = Vec::new();
        for dir in font_dirs() { font_files(&dir, &mut files, 0); }
        files.sort();
        files.dedup();
        files.iter().flat_map(|path| faces_in(path)).collect()
    })
}

/// What Chromium on Windows resolves the CSS generics to (and a Linux stand-in for tests).
fn generic_fallback(generic: &str) -> &'static [&'static str] {
    match generic {
        "serif" => &["Times New Roman", "DejaVu Serif", "Liberation Serif"],
        "monospace" => &["Consolas", "Courier New", "DejaVu Sans Mono", "Liberation Mono"],
        "cursive" => &["Comic Sans MS", "DejaVu Sans"],
        _ => &["Arial", "Liberation Sans", "DejaVu Sans"],
    }
}

/// The face for `family` nearest `weight` and `italic`; `None` when no family by that name exists.
pub fn find_face<'a>(faces: &'a [FaceInfo], family: &str, weight: u16, italic: bool) -> Option<&'a FaceInfo> {
    let wanted = family.trim().trim_matches(['"', '\'']).to_lowercase();
    faces.iter()
        .filter(|face| face.family.to_lowercase() == wanted || face.legacy_family.to_lowercase() == wanted)
        .min_by_key(|face| {
            // Style first, then weight distance, then prefer the typographic-family match.
            let style = if face.italic == italic { 0 } else { 10_000 };
            let exact_family = if face.family.to_lowercase() == wanted { 0 } else { 1 };
            style + (face.weight as i32 - weight as i32).unsigned_abs() as i32 * 2 + exact_family
        })
}

struct Builder(Vec<Command>);

impl ttf_parser::OutlineBuilder for Builder {
    fn move_to(&mut self, x: f32, y: f32) { self.0.push(Command('M', vec![x, y])); }
    fn line_to(&mut self, x: f32, y: f32) { self.0.push(Command('L', vec![x, y])); }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) { self.0.push(Command('Q', vec![x1, y1, x, y])); }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) { self.0.push(Command('C', vec![x1, y1, x2, y2, x, y])); }
    fn close(&mut self) { self.0.push(Command('Z', vec![])); }
}

/// Outlines of every distinct character of `text` in `family` (or its generic fallback).
pub fn outlines(family: &str, generic: &str, weight: u16, italic: bool, text: &str) -> Result<FontOutlines, String> {
    let faces = installed_faces();
    let (face_info, substituted) = match find_face(faces, family, weight, italic) {
        Some(face) => (face.clone(), false),
        None => generic_fallback(generic).iter().find_map(|name| find_face(faces, name, weight, italic)).cloned()
            .map(|face| (face, true))
            .ok_or_else(|| format!("no installed font named \"{family}\", and no {generic} fallback was found"))?,
    };
    let data = std::fs::read(&face_info.path).map_err(|error| format!("could not read {}: {error}", face_info.path))?;
    let face = ttf_parser::Face::parse(&data, face_info.index).map_err(|error| format!("could not parse {}: {error}", face_info.path))?;
    let mut seen = std::collections::BTreeSet::new();
    let mut glyphs = Vec::new();
    for ch in text.chars() {
        if ch.is_control() || !seen.insert(ch) { continue; }
        let id = face.glyph_index(ch);
        let mut builder = Builder(Vec::new());
        if let Some(id) = id { face.outline_glyph(id, &mut builder); }
        let advance = id.and_then(|id| face.glyph_hor_advance(id)).unwrap_or(0) as f32;
        glyphs.push(Glyph { ch: ch.to_string(), advance, commands: builder.0, found: id.is_some() });
    }
    let synthetic = (face_info.weight as i32 - weight as i32).abs() >= 200 || face_info.italic != italic;
    Ok(FontOutlines {
        units_per_em: face.units_per_em(), ascender: face.ascender(), descender: face.descender(),
        face: face_info, glyphs, substituted, synthetic,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_the_nearest_weight_and_style() {
        let face = |family: &str, legacy: &str, weight: u16, italic: bool| FaceInfo {
            path: format!("{family}-{weight}-{italic}"), index: 0, family: family.into(), legacy_family: legacy.into(), weight, italic,
        };
        let faces = vec![
            face("Arial", "Arial", 400, false), face("Arial", "Arial", 700, false), face("Arial", "Arial", 400, true),
            face("Arial", "Arial Black", 900, false), face("Segoe UI", "Segoe UI", 400, false),
        ];
        assert_eq!(find_face(&faces, "arial", 400, false).unwrap().weight, 400);
        assert_eq!(find_face(&faces, "Arial", 650, false).unwrap().weight, 700);
        assert!(find_face(&faces, "Arial", 400, true).unwrap().italic);
        assert_eq!(find_face(&faces, "\"Arial Black\"", 400, false).unwrap().weight, 900);
        assert!(find_face(&faces, "Inter", 400, false).is_none());
    }

    /// Runs where the machine has fonts (a Windows install; a Linux CI container has DejaVu): a real file end to end.
    #[test]
    fn reads_real_outlines_when_fonts_are_installed() {
        let faces = installed_faces();
        let Some(any) = faces.iter().find(|face| ["Arial", "DejaVu Sans", "Liberation Sans"].contains(&face.family.as_str())) else { return };
        let got = outlines("No Such Family 123", "sans-serif", 400, false, "Ab A\u{1}").unwrap();
        assert!(got.substituted);
        assert_eq!(got.glyphs.iter().map(|g| g.ch.as_str()).collect::<Vec<_>>(), vec!["A", "b", " "], "distinct, no controls");
        let a = &got.glyphs[0];
        assert!(a.found && a.advance > 0.0 && a.commands.first().map(|c| c.0) == Some('M') && a.commands.iter().any(|c| c.0 == 'Z'));
        assert!(got.glyphs[2].commands.is_empty(), "a space has no outline");
        assert!(got.units_per_em >= 16);
        let direct = outlines(&any.family, "sans-serif", any.weight, any.italic, "x").unwrap();
        assert!(!direct.substituted && !direct.synthetic);
    }
}

/// Manual check (run with `--ignored`): writes the outlines of `$AIV_OUTLINE_TEXT` in
/// `$AIV_OUTLINE_FAMILY` to `$AIV_OUTLINE_FILE`, for comparing against the canvas in a browser.
#[cfg(test)]
mod dump {
    #[test]
    #[ignore]
    fn dump_outlines() {
        let family = std::env::var("AIV_OUTLINE_FAMILY").unwrap_or_else(|_| "DejaVu Sans".into());
        let text = std::env::var("AIV_OUTLINE_TEXT").unwrap_or_else(|_| "Hag".into());
        let got = super::outlines(&family, "sans-serif", 400, false, &text).unwrap();
        std::fs::write(std::env::var("AIV_OUTLINE_FILE").unwrap(), serde_json::to_string(&got).unwrap()).unwrap();
    }
}
