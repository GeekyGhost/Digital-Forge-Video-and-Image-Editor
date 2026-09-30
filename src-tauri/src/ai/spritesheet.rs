//! VFX Phase 5: spritesheet export files. The page renders the frames with the compositor, packs the
//! atlas and writes the Godot `.tres` and JSON text (src/domain/spritesheet.ts); this saves them under
//! `exports/spritesheets/<name>[-NN]/`. File bodies arrive raw (an atlas can be tens of MB), named by
//! headers, and names are checked: one plain file name, never a path.

use std::{fs, path::{Path, PathBuf}};

pub const DIR: &str = "exports/spritesheets";

/// `[a-z0-9-]`, at most 40 characters, never empty: the same rule as the page's `sheetSlug`.
pub fn slug(name: &str) -> String {
    let lowered: String = name.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    let joined = lowered.split('-').filter(|part| !part.is_empty()).collect::<Vec<_>>().join("-");
    let cut: String = joined.chars().take(40).collect();
    let cut = cut.trim_matches('-').to_owned();
    if cut.is_empty() { "spritesheet".into() } else { cut }
}

/// A new, empty export folder: `<slug>`, then `<slug>-02` ... Returns its name (not the path).
pub fn new_folder(root: &Path, name: &str) -> Result<String, String> {
    let base = root.join(DIR);
    fs::create_dir_all(&base).map_err(|error| format!("could not create {}: {error}", base.display()))?;
    let stem = slug(name);
    for counter in 1..1000 {
        let folder = if counter == 1 { stem.clone() } else { format!("{stem}-{counter:02}") };
        let path = base.join(&folder);
        if !path.exists() {
            fs::create_dir(&path).map_err(|error| format!("could not create {}: {error}", path.display()))?;
            return Ok(folder);
        }
    }
    Err("too many spritesheet exports with this name".into())
}

fn plain_name(name: &str) -> bool {
    !name.is_empty() && name.len() <= 80 && !name.starts_with('.')
        && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

/// Write one file of an export. Only `.png`, `.tres` and `.json` into an existing export folder.
pub fn write(root: &Path, folder: &str, file: &str, bytes: &[u8]) -> Result<PathBuf, String> {
    if !plain_name(folder) || folder.contains('.') { return Err(format!("not an export folder name: {folder}")); }
    if !plain_name(file) { return Err(format!("not a plain file name: {file}")); }
    let extension = Path::new(file).extension().and_then(|value| value.to_str()).unwrap_or("").to_ascii_lowercase();
    if !["png", "tres", "json"].contains(&extension.as_str()) { return Err(format!("{file}: only .png, .tres and .json are written")); }
    let dir = root.join(DIR).join(folder);
    if !dir.is_dir() { return Err(format!("the export folder {folder} does not exist")); }
    let path = dir.join(file);
    fs::write(&path, bytes).map_err(|error| format!("could not write {}: {error}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aiv-sheet-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn slugs_match_the_page() {
        assert_eq!(slug("  Hero Knight / walk!! "), "hero-knight-walk");
        assert_eq!(slug("***"), "spritesheet");
        assert_eq!(slug(&"a".repeat(60)).len(), 40);
    }

    #[test]
    fn folders_are_new_and_numbered() {
        let root = temp("folders");
        assert_eq!(new_folder(&root, "Hero").unwrap(), "hero");
        assert_eq!(new_folder(&root, "Hero").unwrap(), "hero-02");
        assert!(root.join("exports/spritesheets/hero-02").is_dir());
    }

    #[test]
    fn writes_only_plain_names_of_known_kinds_into_an_existing_folder() {
        let root = temp("write");
        let folder = new_folder(&root, "Hero").unwrap();
        let path = write(&root, &folder, "hero.tres", b"[gd_resource]").unwrap();
        assert_eq!(fs::read(&path).unwrap(), b"[gd_resource]");
        for (folder, file) in [("hero", "../x.png"), ("hero", "a/b.png"), ("..", "x.png"), ("hero", "x.exe"), ("hero", ".png"), ("missing", "x.png")] {
            assert!(write(&root, folder, file, b"x").is_err(), "{folder}/{file}");
        }
    }
}
