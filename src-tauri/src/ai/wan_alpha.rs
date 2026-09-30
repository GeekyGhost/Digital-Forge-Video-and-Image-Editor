//! Wan Alpha generation (VFX Phase 3), moved out of the Phase 0 spike (`spikes/rust/spikes.rs`,
//! which now calls this). One generation is bake-if-needed -> encode -> denoise -> decode, each its
//! own Python process, run by `aiv_runtime.wan_alpha generate`. The caller holds the
//! `JobKind::WanAlpha` permit, so the resource guard checks it first and the job monitor logs its
//! real peak to `.aiv/resource-log.jsonl`.

use serde_json::Value;
use std::{path::{Path, PathBuf}, process::Stdio, time::Instant};
use tokio::process::Command;

pub fn python(project: &Path) -> PathBuf {
    project.join("vendor/Wan-Alpha/.venv/Scripts/python.exe")
}

/// Wan2.1's VAE packs 4 frames per latent: 4n+1 frames, 17 to 81 (the Gaussian mask is sized for 81).
pub fn validate_frames(frames: u32) -> Result<(), String> {
    if !(17..=81).contains(&frames) || (frames - 1) % 4 != 0 {
        return Err("frames must be 4n+1 between 17 and 81 (33 and 81 are the tested presets)".into());
    }
    Ok(())
}

/// The frame rate Wan2.1 generates at. Elements are tagged with it.
pub const FPS: f64 = 16.0;

/// Runs one generation into `out_dir` (RGBA `frame_%04d.png`, `stats.json`, `checkerboard.mp4`,
/// and a `work/` folder of intermediates). Returns the worker's stats plus exit status and wall time.
pub async fn generate(project: &Path, prompt: &str, frames: u32, seed: u64, out_dir: &Path) -> Result<Value, String> {
    validate_frames(frames)?;
    let python = python(project);
    if !python.is_file() {
        return Err("Wan-Alpha is not set up. Run Setup-Wan-Alpha.bat.".into());
    }
    std::fs::create_dir_all(out_dir).map_err(|error| error.to_string())?;
    let log = std::fs::File::create(out_dir.join("worker.log")).map_err(|error| error.to_string())?;
    let started = Instant::now();
    let mut command = Command::new(&python);
    #[cfg(windows)]
    command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    let status = command
        .args(["-m", "aiv_runtime.wan_alpha", "generate"])
        .args(["--prompt", prompt])
        .args(["--frames", &frames.to_string()])
        .args(["--seed", &seed.to_string()])
        .arg("--out-dir").arg(out_dir)
        .current_dir(project.join("ai-runtime"))
        .env("PYTHONUNBUFFERED", "1")
        .env("PYTHONIOENCODING", "utf-8")
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::from(log))
        .status()
        .await
        .map_err(|error| format!("could not start the Wan-Alpha worker: {error}"))?;
    let mut stats: Value = std::fs::read_to_string(out_dir.join("stats.json")).ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_else(|| serde_json::json!({ "ok": false }));
    stats["exitStatus"] = Value::String(status.to_string());
    stats["wallSeconds"] = serde_json::json!(started.elapsed().as_secs_f64());
    stats["outDir"] = Value::String(out_dir.to_string_lossy().into_owned());
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_must_be_four_n_plus_one_in_range() {
        for ok in [17, 33, 81] { assert!(validate_frames(ok).is_ok()); }
        for bad in [0, 16, 32, 85, 82] { assert!(validate_frames(bad).is_err()); }
    }
}
