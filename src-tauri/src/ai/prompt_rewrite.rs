//! Rewrite a Qwen-Image-Edit instruction with Qwen's own prompt-enhancer rules
//! (`ai-runtime/aiv_runtime/qwen_edit_rewrite.py`), on the Qwen2.5-VL-7B that Qwen-Image-Edit-2511
//! ships as its text encoder, at 4-bit. It shares the LoRA captioner's environment (the Z-Image
//! trainer venv, which carries bitsandbytes) and its model files, so it is ready exactly when the
//! captioner is.

use super::lora::{offline_command, qwen_base, trainer_ready};
use super::{concise_worker_error_with_status, project_root};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RewriteRequest {
    pub prompt: String,
    /// Picture 1 (the image being edited), then the references, as the edit will send them.
    #[serde(default)]
    pub image_paths: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RewriteResult {
    pub prompt: String,
    pub raw: String,
    pub log: String,
}

pub const MAX_PICTURES: usize = 3;
pub const MAX_PROMPT_CHARS: usize = 4000;

fn worker(root: &Path) -> PathBuf { root.join("ai-runtime/aiv_runtime/qwen_edit_rewrite.py") }

pub fn ready(root: &Path) -> bool {
    trainer_ready(root) && worker(root).is_file() && qwen_base(root).is_some()
}

pub fn validate(request: &RewriteRequest) -> Result<(), String> {
    let prompt = request.prompt.trim();
    if prompt.is_empty() { return Err("write an edit instruction first".into()); }
    if prompt.chars().count() > MAX_PROMPT_CHARS { return Err(format!("the instruction is longer than {MAX_PROMPT_CHARS} characters")); }
    if request.image_paths.len() > MAX_PICTURES { return Err(format!("at most {MAX_PICTURES} images")); }
    for path in &request.image_paths {
        if !Path::new(path).is_file() { return Err(format!("image not found: {path}")); }
    }
    Ok(())
}

/// The completion event's prompt and raw answer from the worker's stdout (JSON lines).
fn completion(stdout: &str) -> Option<(String, String)> {
    stdout.lines().rev().find_map(|line| {
        let value: Value = serde_json::from_str(line).ok()?;
        if value["event"] != "complete" { return None; }
        let prompt = value["prompt"].as_str()?.trim().to_owned();
        (!prompt.is_empty()).then(|| (prompt, value["raw"].as_str().unwrap_or_default().to_owned()))
    })
}

pub async fn rewrite(request: RewriteRequest) -> Result<RewriteResult, String> {
    validate(&request)?;
    let root = project_root();
    if !ready(&root) {
        return Err("the prompt rewriter needs the Z-Image trainer runtime and Qwen-Image-Edit-2511's text encoder; run Setup-ZImage-Trainer.bat".into());
    }
    let qwen = qwen_base(&root).ok_or("Qwen-Image-Edit-2511's text encoder is missing")?;
    let mut command = offline_command(&root, &worker(&root));
    command.args(["--model-path", qwen.join("text_encoder").to_string_lossy().as_ref()])
        .args(["--processor-path", qwen.join("processor").to_string_lossy().as_ref()])
        .args(["--prompt", request.prompt.trim()]);
    for path in &request.image_paths { command.args(["--image-path", path]); }
    let output = command.output().await.map_err(|error| format!("failed to start the prompt rewriter: {error}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    if !output.status.success() { return Err(concise_worker_error_with_status(&stdout, &stderr, output.status.code())); }
    let (prompt, raw) = completion(&stdout).ok_or("the prompt rewriter returned no rewrite")?;
    Ok(RewriteResult { prompt, raw, log: if stderr.is_empty() { stdout } else { format!("{stdout}\n{stderr}") } })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requests_are_checked() {
        let dir = std::env::temp_dir().join(format!("aiv-rewrite-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let image = dir.join("a.png");
        std::fs::write(&image, b"x").unwrap();
        let path = image.to_string_lossy().into_owned();
        let ok = RewriteRequest { prompt: " fix the chair ".into(), image_paths: vec![path.clone()] };
        assert!(validate(&ok).is_ok());
        assert!(validate(&RewriteRequest { prompt: "  ".into(), image_paths: vec![] }).is_err());
        assert!(validate(&RewriteRequest { prompt: "x".into(), image_paths: vec![path.clone(); 4] }).is_err());
        assert!(validate(&RewriteRequest { prompt: "x".into(), image_paths: vec![dir.join("missing.png").to_string_lossy().into_owned()] }).is_err());
        assert!(validate(&RewriteRequest { prompt: "x".repeat(MAX_PROMPT_CHARS + 1), image_paths: vec![] }).is_err());
        assert!(validate(&RewriteRequest { prompt: "x".into(), image_paths: vec![] }).is_ok(), "text-only is allowed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_last_completion_event_is_the_rewrite() {
        let stdout = "{\"event\":\"loading\",\"pictures\":2}\nnoise\n{\"event\":\"complete\",\"prompt\":\" Remove the arms \",\"raw\":\"{...}\"}";
        assert_eq!(completion(stdout), Some(("Remove the arms".to_owned(), "{...}".to_owned())));
        assert_eq!(completion("{\"event\":\"complete\",\"prompt\":\"  \"}"), None);
        assert_eq!(completion("{\"event\":\"loading\"}"), None);
    }
}
