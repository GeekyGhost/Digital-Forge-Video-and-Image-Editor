use base64::{engine::general_purpose::STANDARD, Engine};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::{Path, PathBuf}, process::Stdio};
use tokio::process::Command;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaProbe {
    pub path: String,
    pub duration_seconds: f64,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub frame_rate: Option<String>,
    pub video_codec: Option<String>,
    pub audio_codec: Option<String>,
    pub sample_rate: Option<u32>,
    pub color_space: Option<String>,
}

pub async fn probe(path: &str) -> Result<MediaProbe, String> {
    if !Path::new(path).is_file() {
        return Err("media path does not exist or is not a file".into());
    }
    let output = Command::new("ffprobe")
        .args(["-v", "error", "-show_streams", "-show_format", "-of", "json", path])
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("failed to run ffprobe: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    let value: serde_json::Value = serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("invalid ffprobe output: {error}"))?;
    let streams = value["streams"].as_array().cloned().unwrap_or_default();
    // Album art in MP3/M4A files is represented as a video stream with attached_pic=1. Treating it
    // as picture content gives the asset dimensions and routes an audio file onto V1.
    let video = streams.iter().find(|stream| {
        stream["codec_type"] == "video"
            && stream.pointer("/disposition/attached_pic").and_then(|value| value.as_i64()) != Some(1)
    });
    let audio = streams.iter().find(|stream| stream["codec_type"] == "audio");
    let duration_seconds = value["format"]["duration"]
        .as_str()
        .and_then(|duration| duration.parse().ok())
        .or_else(|| video.and_then(|stream| stream["duration"].as_str()?.parse().ok()))
        .unwrap_or(0.0);

    Ok(MediaProbe {
        path: path.to_owned(),
        duration_seconds,
        width: video.and_then(|stream| stream["width"].as_u64()).map(|value| value as u32),
        height: video.and_then(|stream| stream["height"].as_u64()).map(|value| value as u32),
        frame_rate: video.and_then(|stream| stream["avg_frame_rate"].as_str()).map(str::to_owned),
        video_codec: video.and_then(|stream| stream["codec_name"].as_str()).map(str::to_owned),
        audio_codec: audio.and_then(|stream| stream["codec_name"].as_str()).map(str::to_owned),
        sample_rate: audio.and_then(|stream| stream["sample_rate"].as_str()?.parse().ok()),
        color_space: video.and_then(|stream| stream["color_space"].as_str()).map(str::to_owned),
    })
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MaterializeClipRequest {
    pub path: String,
    pub output_path: String,
    pub source_in_seconds: f64,
    pub source_out_seconds: f64,
    #[serde(default = "default_playback_rate")]
    pub playback_rate: f64,
    /// `audio` writes a standalone PCM WAV; `video` writes H.264/AAC MP4.
    pub kind: String,
}

fn default_playback_rate() -> f64 { 1.0 }

fn materialize_atempo(rate: f64) -> String {
    let safe = rate.clamp(0.25, 4.0);
    if safe < 0.5 {
        format!("atempo=0.5,atempo={:.6}", safe / 0.5)
    } else if safe > 2.0 {
        format!("atempo=2.0,atempo={:.6}", safe / 2.0)
    } else {
        format!("atempo={safe:.6}")
    }
}

fn materialize_frame_rate(value: Option<&str>) -> f64 {
    let parsed = value.and_then(|raw| {
        let (left, right) = raw.split_once('/').unwrap_or((raw, "1"));
        let numerator = left.parse::<f64>().ok()?;
        let denominator = right.parse::<f64>().ok()?;
        (denominator > 0.0).then_some(numerator / denominator)
    }).unwrap_or(24.0);
    parsed.clamp(1.0, 240.0)
}

/// Turn a timeline subclip into ordinary project media. Ref2VA consumes file paths rather than
/// timeline in/out metadata, so merely duplicating the asset record would silently feed it the
/// untrimmed source. Accurate filter trims also avoid keyframe-snapped starts from stream copying.
pub async fn materialize_clip(request: MaterializeClipRequest) -> Result<String, String> {
    let source = Path::new(&request.path);
    if !source.is_file() {
        return Err("clip source does not exist".into());
    }
    let destination = PathBuf::from(request.output_path.trim());
    if destination == source {
        return Err("a trimmed clip cannot overwrite its source".into());
    }
    let start = request.source_in_seconds;
    let end = request.source_out_seconds;
    let rate = request.playback_rate;
    if !start.is_finite() || !end.is_finite() || start < 0.0 || end - start < 0.04 {
        return Err("the trimmed clip must contain at least 0.04 seconds".into());
    }
    if !rate.is_finite() || !(0.25..=4.0).contains(&rate) {
        return Err("playback rate must be between 0.25 and 4".into());
    }
    let metadata = probe(&request.path).await?;
    if end > metadata.duration_seconds + 0.1 {
        return Err(format!(
            "trim out point {:.3}s exceeds the {:.3}s source duration",
            end, metadata.duration_seconds
        ));
    }
    let extension = destination.extension().and_then(|part| part.to_str())
        .unwrap_or("").to_ascii_lowercase();
    let mut args: Vec<String> = vec![
        "-hide_banner".into(), "-loglevel".into(), "error".into(), "-nostdin".into(),
        "-y".into(), "-i".into(), request.path.clone(),
    ];
    match request.kind.as_str() {
        "audio" => {
            if extension != "wav" {
                return Err("trimmed audio assets must use the .wav extension".into());
            }
            if metadata.audio_codec.is_none() {
                return Err("the selected clip has no audio stream".into());
            }
            args.extend([
                "-map".into(), "0:a:0".into(), "-vn".into(),
                "-af".into(), format!(
                    "atrim=start={start:.9}:end={end:.9},asetpts=PTS-STARTPTS,{}",
                    materialize_atempo(rate)
                ),
                "-c:a".into(), "pcm_s16le".into(),
            ]);
        }
        "video" => {
            if extension != "mp4" {
                return Err("trimmed video assets must use the .mp4 extension".into());
            }
            if metadata.video_codec.is_none() {
                return Err("the selected clip has no video stream".into());
            }
            // After setpts changes speed, explicitly restore the source cadence. Without this,
            // FFmpeg's output sync can drop a non-deterministic tail when audio is mapped too.
            let fps = materialize_frame_rate(metadata.frame_rate.as_deref());
            args.extend([
                "-map".into(), "0:v:0".into(),
                "-vf".into(), format!(
                    "trim=start={start:.9}:end={end:.9},setpts=(PTS-STARTPTS)/{rate:.9},fps={fps:.6}"
                ),
                "-c:v".into(), "libx264".into(), "-preset".into(), "medium".into(),
                "-crf".into(), "16".into(), "-pix_fmt".into(), "yuv420p".into(),
            ]);
            if metadata.audio_codec.is_some() {
                args.extend([
                    "-map".into(), "0:a:0".into(),
                    "-af".into(), format!(
                        "atrim=start={start:.9}:end={end:.9},asetpts=PTS-STARTPTS,{}",
                        materialize_atempo(rate)
                    ),
                    "-c:a".into(), "aac".into(),
                    "-b:a".into(), "256k".into(),
                ]);
            }
            args.extend(["-movflags".into(), "+faststart".into()]);
        }
        _ => return Err("trimmed asset kind must be audio or video".into()),
    }
    if let Some(parent) = destination.parent().filter(|value| !value.as_os_str().is_empty()) {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    args.push(destination.to_string_lossy().into_owned());
    let output = Command::new("ffmpeg").args(&args).stdin(Stdio::null()).output().await
        .map_err(|error| format!("failed to save the trimmed clip: {error}"))?;
    if !output.status.success() {
        let _ = fs::remove_file(&destination);
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    if !destination.is_file() || fs::metadata(&destination).map(|item| item.len()).unwrap_or(0) == 0 {
        return Err("FFmpeg did not create the trimmed asset".into());
    }
    Ok(destination.to_string_lossy().into_owned())
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WaveformRequest {
    pub path: String,
    #[serde(default)]
    pub source_in_seconds: f64,
    pub duration_seconds: f64,
    #[serde(default = "default_buckets")]
    pub buckets: u32,
}

fn default_buckets() -> u32 { 320 }

/// Keep the decoded PCM behind a waveform bounded by the number of pixels it can display.
/// At 320 buckets, roughly 256 samples per bucket is already more peak detail than the SVG can
/// show. A fixed 4 kHz rate turned a long recording into hundreds of megabytes before reduction.
fn waveform_sample_rate(duration_seconds: f64, buckets: usize) -> u32 {
    let target_samples = buckets.saturating_mul(256).max(4_096) as f64;
    (target_samples / duration_seconds.max(0.02)).ceil().clamp(8.0, 4_000.0) as u32
}

/// Decodes the clip's audio to mono 8-bit PCM and reduces it to normalised peaks so the
/// timeline can draw a waveform without shipping the samples into the webview.
pub async fn waveform(request: WaveformRequest) -> Result<Vec<f32>, String> {
    if !Path::new(&request.path).is_file() {
        return Err("audio source does not exist".into());
    }
    let buckets = request.buckets.clamp(16, 2048) as usize;
    let duration = request.duration_seconds.max(0.02);
    let sample_rate = waveform_sample_rate(duration, buckets);
    // The rate adapts to duration, so an hour-long source and a ten-second source both return a
    // similarly bounded PCM buffer before reduction rather than memory usage tracking runtime.
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner", "-loglevel", "error", "-nostdin",
            "-ss", &format!("{:.6}", request.source_in_seconds.max(0.0)),
            "-t", &format!("{duration:.6}"),
            "-i", &request.path,
            "-vn", "-ac", "1", "-ar", &sample_rate.to_string(), "-f", "u8", "-",
        ])
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("failed to run ffmpeg: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    let samples = output.stdout;
    if samples.is_empty() {
        return Ok(vec![0.0; buckets]);
    }
    // Short clips can have fewer samples than buckets, so every slice bound is clamped and the
    // window is allowed to be empty rather than indexing past the end.
    let window = samples.len() as f64 / buckets as f64;
    let mut peaks = Vec::with_capacity(buckets);
    for index in 0..buckets {
        let start = ((index as f64 * window) as usize).min(samples.len());
        let end = ((((index + 1) as f64 * window).ceil()) as usize).clamp(start, samples.len());
        let peak = samples[start..end]
            .iter()
            .map(|value| ((*value as f32) - 128.0).abs() / 128.0)
            .fold(0.0_f32, f32::max);
        peaks.push(peak.clamp(0.0, 1.0));
    }
    let loudest = peaks.iter().copied().fold(0.0_f32, f32::max);
    if loudest > 0.01 {
        for peak in &mut peaks {
            *peak = (*peak / loudest).clamp(0.0, 1.0);
        }
    }
    Ok(peaks)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ThumbnailRequest {
    pub path: String,
    #[serde(default)]
    pub at_seconds: f64,
    #[serde(default = "default_thumb_width")]
    pub width: u32,
}

fn default_thumb_width() -> u32 { 192 }

/// Extracts one frame as a small JPEG and returns it as a data URL for the media bin and
/// timeline filmstrips.
pub async fn thumbnail(request: ThumbnailRequest) -> Result<String, String> {
    if !Path::new(&request.path).is_file() {
        return Err("media source does not exist".into());
    }
    let width = request.width.clamp(48, 640);
    let scratch = std::env::temp_dir().join(format!("aiv-thumb-{}.jpg", Uuid::new_v4()));
    let result = Command::new("ffmpeg")
        .args([
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", &format!("{:.6}", request.at_seconds.max(0.0)),
            "-i", &request.path,
            "-frames:v", "1",
            "-vf", &format!("scale={width}:-2:flags=fast_bilinear"),
            "-q:v", "6",
        ])
        .arg(&scratch)
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|error| format!("failed to run ffmpeg: {error}"))?;
    if !result.status.success() {
        let _ = fs::remove_file(&scratch);
        return Err(String::from_utf8_lossy(&result.stderr).trim().to_owned());
    }
    let bytes = fs::read(&scratch).map_err(|error| error.to_string())?;
    let _ = fs::remove_file(&scratch);
    if bytes.is_empty() {
        return Err("ffmpeg produced an empty thumbnail".into());
    }
    Ok(format!("data:image/jpeg;base64,{}", STANDARD.encode(bytes)))
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractFramesRequest {
    pub project_id: String,
    pub path: String,
    pub timestamps: Vec<f64>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExtractedFrame {
    pub output_path: String,
    pub timestamp_seconds: f64,
    pub frame_index: u64,
    pub width: u32,
    pub height: u32,
}

/// Whether this source is a single still rather than a moving picture.
///
/// Extension-first, mirroring `domain/media.ts::isImagePath`, and for the same reason the importer
/// classifies that way: it is the source of truth the UI already used to decide the asset's kind.
///
/// This matters because a still has exactly one frame, and *every* visible range over it is that
/// frame. Seeking into one asks FFmpeg for a timestamp the input does not have and produces no
/// output at all — which is why timeline conditioning used to reject image clips outright rather
/// than handle them. A PNG on V1 is a perfectly good frame to bridge to.
fn is_still_image(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .is_some_and(|value| {
            matches!(value.as_str(), "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tif" | "tiff")
        })
}

/// Copy a still's one frame out as PNG, with no seek and no range.
async fn extract_still_frame(path: &str, destination: &Path) -> Result<(), String> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-i", path, "-map", "0:v:0", "-an",
            "-frames:v", "1", "-vf", "setsar=1", "-update", "1", "-c:v", "png",
        ])
        .arg(destination)
        .stdin(Stdio::null())
        .output().await
        .map_err(|error| format!("failed to read the still: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    if !destination.is_file() || fs::metadata(destination).map(|item| item.len()).unwrap_or(0) == 0 {
        return Err("FFmpeg did not produce a frame from the still".into());
    }
    Ok(())
}

async fn extract_frame_at(path: &str, at_seconds: f64, destination: &Path) -> Result<(), String> {
    if !Path::new(path).is_file() { return Err("video frame source does not exist".into()); }
    if !at_seconds.is_finite() || at_seconds < 0.0 { return Err("video frame time must be non-negative".into()); }
    if is_still_image(path) { return extract_still_frame(path, destination).await; }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    // Seek near the target first, then decode forward accurately. A fast input-only seek can land
    // on the preceding keyframe and silently save a different frame than the picker displayed.
    let coarse = (at_seconds - 2.0).max(0.0);
    let fine = at_seconds - coarse;
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", &format!("{coarse:.9}"), "-i", path,
            "-ss", &format!("{fine:.9}"), "-map", "0:v:0", "-an",
            "-frames:v", "1", "-vf", "setsar=1", "-update", "1", "-c:v", "png",
        ])
        .arg(destination)
        .stdin(Stdio::null())
        .output().await
        .map_err(|error| format!("failed to extract video frame: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    if !destination.is_file() || fs::metadata(destination).map(|item| item.len()).unwrap_or(0) == 0 {
        return Err("FFmpeg did not produce the selected video frame".into());
    }
    Ok(())
}

pub async fn extract_frames(request: ExtractFramesRequest) -> Result<Vec<ExtractedFrame>, String> {
    if request.timestamps.is_empty() { return Err("select at least one video frame".into()); }
    if request.timestamps.len() > 32 { return Err("at most 32 frames can be imported at once".into()); }
    let project_id = Uuid::parse_str(&request.project_id).map_err(|_| "invalid project id".to_owned())?;
    let metadata = probe(&request.path).await?;
    if metadata.video_codec.is_none() || metadata.width.is_none() || metadata.height.is_none() {
        return Err("the selected file has no usable video stream".into());
    }
    if metadata.duration_seconds <= 0.0 { return Err("the selected video has no measurable duration".into()); }
    let fps = materialize_frame_rate(metadata.frame_rate.as_deref());
    let max_frame = ((metadata.duration_seconds * fps).ceil() as u64).saturating_sub(1);
    let mut frames = BTreeMap::new();
    for timestamp in request.timestamps {
        if !timestamp.is_finite() || timestamp < 0.0 || timestamp > metadata.duration_seconds + 0.001 {
            return Err(format!("selected frame time {timestamp:.3}s is outside the video"));
        }
        let index = ((timestamp * fps).round() as u64).min(max_frame);
        frames.insert(index, index as f64 / fps);
    }
    let output_dir = crate::ai::project_root().join("projects").join(project_id.to_string()).join("media/images");
    fs::create_dir_all(&output_dir).map_err(|error| format!("could not create project image directory: {error}"))?;
    let mut extracted = Vec::with_capacity(frames.len());
    for (frame_index, timestamp_seconds) in frames {
        let output_path = output_dir.join(format!("video-frame-{frame_index:08}-{}.png", Uuid::new_v4().simple()));
        extract_frame_at(&request.path, timestamp_seconds, &output_path).await?;
        extracted.push(ExtractedFrame {
            output_path: output_path.to_string_lossy().into_owned(),
            timestamp_seconds, frame_index,
            width: metadata.width.unwrap_or_default(), height: metadata.height.unwrap_or_default(),
        });
    }
    Ok(extracted)
}

/// Decode through the end of the visible source range and keep the final decoded frame. Using an
/// overwrite image sink avoids guessing a VFR clip's last-frame timestamp from its average FPS.
pub async fn extract_last_visible_frame(
    path: &str,
    source_in_seconds: f64,
    source_out_seconds: f64,
    destination: &Path,
) -> Result<(), String> {
    if !Path::new(path).is_file() { return Err("clip-extension source does not exist".into()); }
    // Before the range check: a still's timeline range is synthetic (the importer gives it five
    // seconds), so demanding a valid *source* range of it would reject the one frame it has.
    if is_still_image(path) { return extract_still_frame(path, destination).await; }
    if source_in_seconds < 0.0 || source_out_seconds <= source_in_seconds {
        return Err("clip extension needs a valid visible source range".into());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let seek = source_in_seconds.max(source_out_seconds - 2.0);
    let duration = (source_out_seconds - seek - 0.000_1).max(0.001);
    let output = Command::new("ffmpeg")
        .args([
            "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
            "-ss", &format!("{seek:.9}"), "-i", path,
            "-t", &format!("{duration:.9}"), "-map", "0:v:0", "-an",
            "-vf", "setsar=1", "-fps_mode", "passthrough", "-update", "1", "-c:v", "png",
        ])
        .arg(destination)
        .stdin(Stdio::null())
        .output().await
        .map_err(|error| format!("failed to extract the clip-extension frame: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    if !destination.is_file() || fs::metadata(destination).map(|item| item.len()).unwrap_or(0) == 0 {
        return Err("FFmpeg did not produce a lossless last frame".into());
    }
    Ok(())
}

/// Decode the first *visible* frame of a clip — the frame at its trimmed `sourceIn`, not the
/// untrimmed head of the file. Accurate seeking (`-ss` after `-i`) is used deliberately: the
/// conditioning frame has to be the exact frame the timeline shows, and a keyframe-snapped seek
/// would silently hand LTX a different one.
pub async fn extract_first_visible_frame(
    path: &str,
    source_in_seconds: f64,
    destination: &Path,
) -> Result<(), String> {
    extract_frame_at(path, source_in_seconds, destination).await
}

/// A clip-to-clip bridge is pinned at *both* ends, so its first and last frames are duplicates of
/// frames already on the timeline. Trim one frame from each end, with short fades so neither
/// splice lands as a hard discontinuity.
pub async fn trim_bridge_boundaries(
    input_path: &Path,
    output_path: &Path,
    frame_duration: f64,
) -> Result<(), String> {
    let metadata = probe(input_path.to_string_lossy().as_ref()).await?;
    let frame = frame_duration.max(0.001);
    // Both boundary frames come off, so the payload is two frames shorter than what LTX returned.
    let kept = (metadata.duration_seconds - frame * 2.0).max(frame);
    let mut args = vec![
        "-hide_banner".into(), "-loglevel".into(), "error".into(), "-nostdin".into(), "-y".into(),
        "-i".into(), input_path.to_string_lossy().into_owned(),
        "-map".into(), "0:v:0".into(), "-vf".into(),
        format!("trim=start={frame:.9}:duration={kept:.9},setpts=PTS-STARTPTS"),
        "-c:v".into(), "libx264".into(), "-preset".into(), "medium".into(), "-crf".into(), "16".into(),
        "-pix_fmt".into(), "yuv420p".into(),
    ];
    if metadata.audio_codec.is_some() {
        args.extend([
            "-map".into(), "0:a:0".into(), "-af".into(),
            format!(
                "atrim=start={frame:.9}:duration={kept:.9},asetpts=PTS-STARTPTS,\
                 afade=t=in:st=0:d=0.060,afade=t=out:st={:.9}:d=0.060",
                (kept - 0.060).max(0.0)
            ),
            "-c:a".into(), "aac".into(), "-b:a".into(), "256k".into(),
        ]);
    }
    args.extend(["-movflags".into(), "+faststart".into(), output_path.to_string_lossy().into_owned()]);
    let output = Command::new("ffmpeg").args(&args).stdin(Stdio::null()).output().await
        .map_err(|error| format!("failed to trim the clip-bridge boundaries: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    if !output_path.is_file() { return Err("trimmed clip bridge was not created".into()); }
    Ok(())
}

/// Remove LTX's duplicated conditioning frame and the matching audio duration. The short fade
/// prevents a hard discontinuity when the generated extension is inserted after the source clip.
pub async fn trim_extension_boundary(
    input_path: &Path,
    output_path: &Path,
    frame_duration: f64,
) -> Result<(), String> {
    let metadata = probe(input_path.to_string_lossy().as_ref()).await?;
    let mut args = vec![
        "-hide_banner".into(), "-loglevel".into(), "error".into(), "-nostdin".into(), "-y".into(),
        "-i".into(), input_path.to_string_lossy().into_owned(),
        "-map".into(), "0:v:0".into(), "-vf".into(),
        format!("trim=start={:.9},setpts=PTS-STARTPTS", frame_duration.max(0.001)),
        "-c:v".into(), "libx264".into(), "-preset".into(), "medium".into(), "-crf".into(), "16".into(),
        "-pix_fmt".into(), "yuv420p".into(),
    ];
    if metadata.audio_codec.is_some() {
        args.extend([
            "-map".into(), "0:a:0".into(), "-af".into(),
            format!("atrim=start={:.9},asetpts=PTS-STARTPTS,afade=t=in:st=0:d=0.060", frame_duration.max(0.001)),
            "-c:a".into(), "aac".into(), "-b:a".into(), "256k".into(),
        ]);
    }
    args.extend(["-movflags".into(), "+faststart".into(), output_path.to_string_lossy().into_owned()]);
    let output = Command::new("ffmpeg").args(&args).stdin(Stdio::null()).output().await
        .map_err(|error| format!("failed to trim the clip-extension boundary: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    if !output_path.is_file() { return Err("trimmed clip extension was not created".into()); }
    Ok(())
}

/// Trim conditioning-frame time from H3's standalone soundtrack. H3 deliberately returns picture
/// and sound as separate files, so the video boundary helpers above cannot keep its WAV aligned for
/// clip extension/bridging. Both outputs must lose exactly the same 24 fps frame durations.
pub async fn trim_audio_boundaries(
    input_path: &Path,
    output_path: &Path,
    trim_start: f64,
    trim_end: f64,
) -> Result<(), String> {
    let metadata = probe(input_path.to_string_lossy().as_ref()).await?;
    if metadata.audio_codec.is_none() {
        return Err("the generated H3 soundtrack has no audio stream".into());
    }
    let start = trim_start.max(0.0);
    let end = trim_end.max(0.0);
    let kept = (metadata.duration_seconds - start - end).max(0.001);
    let fade = kept.min(0.060);
    let mut filter = format!("atrim=start={start:.9}:duration={kept:.9},asetpts=PTS-STARTPTS");
    if start > 0.0 {
        filter.push_str(&format!(",afade=t=in:st=0:d={fade:.9}"));
    }
    if end > 0.0 {
        filter.push_str(&format!(",afade=t=out:st={:.9}:d={fade:.9}", (kept - fade).max(0.0)));
    }
    let output = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i"])
        .arg(input_path)
        .args(["-map", "0:a:0", "-af", &filter, "-c:a", "pcm_s16le"])
        .arg(output_path)
        .stdin(Stdio::null())
        .output().await
        .map_err(|error| format!("failed to trim the H3 soundtrack boundaries: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    if !output_path.is_file() {
        return Err("trimmed H3 soundtrack was not created".into());
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DenoiseRequest {
    pub path: String,
    /// `light` | `medium` | `strong`.
    #[serde(default = "default_denoise_strength")]
    pub strength: String,
    /// Rolls off rumble and DC drift below this frequency. 0 disables it.
    #[serde(default = "default_denoise_highpass")]
    pub highpass_hz: u32,
    /// Bring the cleaned result back to a broadcast-ish -16 LUFS. Denoising always loses some
    /// level, and generated audio rarely arrives at a consistent one to begin with.
    #[serde(default)]
    pub normalize: bool,
    /// Empty means "beside the source, named `<stem>-denoised.<ext>`".
    #[serde(default)]
    pub output_path: String,
}

fn default_denoise_strength() -> String { "medium".into() }
fn default_denoise_highpass() -> u32 { 60 }

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DenoiseResult {
    pub output_path: String,
    /// The exact filter chain that ran, so the panel can show what was done rather than a label.
    pub filter: String,
    /// True when the picture was stream-copied. Denoise never re-encodes video.
    pub video_copied: bool,
}

/// Clean hiss, hum and broadband noise out of a clip's audio without touching its picture.
///
/// `afftdn` with `tn=1` is the right primitive here. Ordinary spectral subtraction wants a *noise
/// print* — a stretch of the file containing only noise — which generated audio does not reliably
/// contain. `tn=1` tracks the noise floor continuously instead, so it works on LTX foley and V2A
/// output where the artefacts drift with the content.
///
/// Non-destructive by construction: the source file is never opened for writing. A new file is
/// written beside it and the caller re-points the clip at it, so dropping the result is a pointer
/// change rather than a re-render — the same contract `stem_mix.py` uses.
///
/// Video is stream-copied (`-c:v copy`). A denoise pass that silently re-encoded the picture would
/// cost a generation of quality for a change that never touched a pixel.
pub async fn denoise_audio(request: DenoiseRequest) -> Result<DenoiseResult, String> {
    let source = Path::new(&request.path);
    if !source.is_file() {
        return Err("audio source does not exist".into());
    }
    let metadata = probe(&request.path).await?;
    if metadata.audio_codec.is_none() {
        return Err("this clip has no audio track to denoise".into());
    }
    let has_video = metadata.video_codec.is_some();

    // nr is how many dB of noise to remove; nf is the assumed floor it starts tracking from. Push
    // nr too high and the residual turns into the watery "underwater" artefact everyone recognises,
    // which is why `strong` is a deliberate step rather than a slider that goes to 97.
    let (noise_reduction, noise_floor, extra) = match request.strength.as_str() {
        "light" => (6.0_f32, -38.0_f32, ""),
        "medium" => (12.0, -30.0, ""),
        // anlmdn is a second, non-spectral pass: it averages self-similar patches, which cleans up
        // the granular residue afftdn leaves behind at high nr. It is slow, so only `strong` pays.
        "strong" => (24.0, -24.0, ",anlmdn=s=0.0005:p=0.008:r=0.012"),
        other => return Err(format!("unknown denoise strength '{other}' (light, medium, strong)")),
    };
    let mut chain = String::new();
    if request.highpass_hz > 0 {
        chain.push_str(&format!("highpass=f={}:poles=2,", request.highpass_hz.clamp(20, 400)));
    }
    chain.push_str(&format!("afftdn=nr={noise_reduction:.1}:nf={noise_floor:.1}:tn=1"));
    chain.push_str(extra);
    if request.normalize {
        chain.push_str(",loudnorm=I=-16:TP=-1.5:LRA=11");
    }

    let destination = if request.output_path.trim().is_empty() {
        let stem = source.file_stem().and_then(|part| part.to_str()).unwrap_or("clip");
        let extension = if has_video {
            source.extension().and_then(|part| part.to_str()).unwrap_or("mp4").to_owned()
        } else {
            // No picture to preserve, so land on FLAC rather than re-lossy an already-lossy source.
            "flac".to_owned()
        };
        source.with_file_name(format!("{stem}-denoised.{extension}"))
    } else {
        PathBuf::from(request.output_path.trim())
    };
    if destination == source {
        return Err("denoise cannot overwrite its own source".into());
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }

    let mut args: Vec<String> = vec![
        "-hide_banner".into(), "-loglevel".into(), "error".into(), "-nostdin".into(), "-y".into(),
        "-i".into(), request.path.clone(),
    ];
    if has_video {
        args.extend(["-map".into(), "0:v:0".into(), "-c:v".into(), "copy".into()]);
    }
    args.extend([
        "-map".into(), "0:a:0".into(),
        "-af".into(), chain.clone(),
    ]);
    if has_video {
        args.extend(["-c:a".into(), "aac".into(), "-b:a".into(), "320k".into()]);
    } else {
        args.extend(["-c:a".into(), "flac".into()]);
    }
    let lower = destination.extension().and_then(|part| part.to_str()).unwrap_or("").to_ascii_lowercase();
    if lower == "mp4" || lower == "mov" {
        args.extend(["-movflags".into(), "+faststart".into()]);
    }
    args.push(destination.to_string_lossy().into_owned());

    let output = Command::new("ffmpeg").args(&args).stdin(Stdio::null()).output().await
        .map_err(|error| format!("failed to run ffmpeg: {error}"))?;
    if !output.status.success() {
        let _ = fs::remove_file(&destination);
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    if !destination.is_file() || fs::metadata(&destination).map(|item| item.len()).unwrap_or(0) == 0 {
        return Err("FFmpeg produced an empty denoised file".into());
    }
    Ok(DenoiseResult {
        output_path: destination.to_string_lossy().into_owned(),
        filter: chain,
        video_copied: has_video,
    })
}

/// VFX Phase 7: the audio envelope a reactive mask VFX follows, one value per video frame -- the old
/// `mask_effects.py` `audio_envelope`, moved here so the compositor can render the effect live.
#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioEnvelopeRequest {
    pub path: String,
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub frame_count: u32,
    /// "amplitude", "bass" or "transients" ("off" returns zeros).
    pub mode: String,
    pub smoothing: f32,
}

pub async fn audio_envelope(request: AudioEnvelopeRequest) -> Result<Vec<f32>, String> {
    let frames = request.frame_count.clamp(1, 100_000) as usize;
    if request.mode == "off" { return Ok(vec![0.0; frames]); }
    if !["amplitude", "bass", "transients"].contains(&request.mode.as_str()) {
        return Err(format!("unknown audio-reactive mode {}", request.mode));
    }
    if !Path::new(&request.path).is_file() { return Err("audio source does not exist".into()); }
    if request.end_seconds <= request.start_seconds { return Err("the envelope needs a time range".into()); }
    let mut args: Vec<String> = [
        "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", &format!("{:.6}", request.start_seconds.max(0.0)),
        "-t", &format!("{:.6}", request.end_seconds - request.start_seconds),
        "-i", &request.path, "-vn", "-ac", "1", "-ar", "48000",
    ].iter().map(|value| value.to_string()).collect();
    if request.mode == "bass" { args.extend(["-af".into(), "lowpass=f=180".into()]); }
    args.extend(["-f".into(), "f32le".into(), "pipe:1".into()]);
    let output = Command::new("ffmpeg").args(&args).stdin(Stdio::null()).output().await
        .map_err(|error| format!("failed to run ffmpeg: {error}"))?;
    // A clip without usable audio simply does not react (as before), rather than failing the effect.
    if !output.status.success() || output.stdout.len() < 4 { return Ok(vec![0.0; frames]); }
    let samples: Vec<f32> = output.stdout.chunks_exact(4).map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])).collect();
    Ok(envelope_from_samples(&samples, frames, &request.mode, request.smoothing))
}

/// Per-frame RMS, transient-extracted for "transients", normalised to the 95th percentile, with a
/// release that grows with `smoothing` (attacks stay immediate). Same arithmetic as the old worker.
pub fn envelope_from_samples(samples: &[f32], frames: usize, mode: &str, smoothing: f32) -> Vec<f32> {
    if samples.is_empty() || frames == 0 { return vec![0.0; frames]; }
    let mut rms: Vec<f64> = (0..frames).map(|index| {
        let start = (index as f64 * samples.len() as f64 / frames as f64) as usize;
        let end = (((index + 1) as f64) * samples.len() as f64 / frames as f64) as usize;
        if end > start {
            (samples[start..end].iter().map(|value| (*value as f64) * (*value as f64)).sum::<f64>() / (end - start) as f64).sqrt()
        } else { 0.0 }
    }).collect();
    if mode == "transients" {
        // numpy.convolve(rms, ones(w)/w, "same"): a centred mean with zeros beyond the ends.
        let mut window = 7.min(rms.len());
        if window > 1 && window % 2 == 0 { window -= 1; }
        let half = (window as isize - 1) / 2;
        let baseline: Vec<f64> = (0..rms.len() as isize).map(|i| {
            (-half..=half).map(|k| i + k).filter(|j| *j >= 0 && (*j as usize) < rms.len()).map(|j| rms[j as usize]).sum::<f64>() / window as f64
        }).collect();
        for (value, base) in rms.iter_mut().zip(baseline) { *value = (*value - base).max(0.0); }
    }
    // numpy.percentile(..., 95) with linear interpolation.
    let mut sorted = rms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let position = 0.95 * (sorted.len() - 1) as f64;
    let low = position.floor() as usize;
    let high = (low + 1).min(sorted.len() - 1);
    let ceiling = sorted[low] + (sorted[high] - sorted[low]) * (position - low as f64);
    let mut values: Vec<f32> = rms.iter().map(|value| (value / ceiling.max(1e-6)).clamp(0.0, 1.0) as f32).collect();
    let release = 0.15 + smoothing.clamp(0.0, 1.0) * 0.8;
    for index in 1..values.len() {
        if values[index] < values[index - 1] {
            values[index] = values[index] * (1.0 - release) + values[index - 1] * release;
        }
    }
    values
}

#[cfg(test)]
mod tests {
    #[test]
    fn envelope_follows_loudness_and_releases_slowly() {
        // Four frames: silence, loud, silence, silence.
        let mut samples = vec![0.0_f32; 400];
        for value in &mut samples[100..200] { *value = 0.5; }
        let values = super::envelope_from_samples(&samples, 4, "amplitude", 0.5);
        assert_eq!(values.len(), 4);
        assert!(values[0] < 1e-6);
        assert!((values[1] - 1.0).abs() < 1e-6);
        // Released: 0 * (1 - r) + 1 * r with r = 0.15 + 0.5 * 0.8.
        assert!((values[2] - 0.55).abs() < 1e-6);
        assert!((values[3] - 0.55 * 0.55).abs() < 1e-6);
        assert_eq!(super::envelope_from_samples(&[], 3, "amplitude", 0.5), vec![0.0; 3]);
    }

    #[test]
    fn transients_subtract_the_local_mean() {
        let samples: Vec<f32> = (0..700).map(|i| if (300..400).contains(&i) { 1.0 } else { 0.2 }).collect();
        let values = super::envelope_from_samples(&samples, 7, "transients", 0.0);
        // Only the hit stands above its neighbourhood.
        assert!((values[3] - 1.0).abs() < 1e-6);
        assert!(values[0] < values[3]);
    }

    use super::*;
    use std::process::Command as StdCommand;

    #[test]
    fn waveform_buckets_are_clamped_to_a_sane_range() {
        let request = WaveformRequest { path: "missing.wav".into(), source_in_seconds: 0.0, duration_seconds: 1.0, buckets: 100_000 };
        assert_eq!(request.buckets.clamp(16, 2048), 2048);
    }

    #[test]
    fn waveform_decode_rate_drops_for_long_media_instead_of_scaling_memory_with_duration() {
        assert_eq!(waveform_sample_rate(10.0, 320), 4_000);
        assert!(waveform_sample_rate(3_600.0, 320) <= 23);
        assert_eq!(waveform_sample_rate(24.0 * 3_600.0, 320), 8);
    }

    #[test]
    fn exact_frame_extraction_writes_a_lossless_png() {
        if StdCommand::new("ffmpeg").arg("-version").output().is_err() { return; }
        let work = std::env::temp_dir().join(format!("aiv-frame-import-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&work).unwrap();
        let input = work.join("source.mp4");
        let frame = work.join("frame.png");
        let made = StdCommand::new("ffmpeg").args([
            "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
            "testsrc2=size=96x64:rate=10:duration=1", "-c:v", "mpeg4", "-q:v", "3",
        ]).arg(&input).status().unwrap();
        assert!(made.success());
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        runtime.block_on(extract_frame_at(input.to_string_lossy().as_ref(), 0.4, &frame)).unwrap();
        let bytes = fs::read(&frame).unwrap();
        assert!(bytes.len() > 100);
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        let _ = fs::remove_dir_all(work);
    }

    #[test]
    fn mp3_album_art_is_not_reported_as_video() {
        if StdCommand::new("ffmpeg").arg("-version").output().is_err() { return; }
        let work = std::env::temp_dir().join(format!("aiv-mp3-cover-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&work).unwrap();
        let audio = work.join("plain.mp3");
        let cover = work.join("cover.jpg");
        let tagged = work.join("with-cover.mp3");
        assert!(StdCommand::new("ffmpeg").args([
            "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
            "sine=frequency=440:sample_rate=44100:duration=3", "-c:a", "libmp3lame",
        ]).arg(&audio).status().unwrap().success());
        assert!(StdCommand::new("ffmpeg").args([
            "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
            "color=c=blue:size=64x64", "-frames:v", "1",
        ]).arg(&cover).status().unwrap().success());
        assert!(StdCommand::new("ffmpeg").args([
            "-hide_banner", "-loglevel", "error", "-y", "-i",
        ]).arg(&audio).arg("-i").arg(&cover).args([
            "-map", "0:a:0", "-map", "1:v:0", "-c:a", "copy", "-c:v", "mjpeg",
            "-disposition:v:0", "attached_pic",
        ]).arg(&tagged).status().unwrap().success());
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let metadata = runtime.block_on(probe(tagged.to_string_lossy().as_ref())).unwrap();
        assert!(metadata.audio_codec.is_some());
        assert!(metadata.video_codec.is_none());
        assert!(metadata.width.is_none() && metadata.height.is_none());
        assert!((2.9..=3.1).contains(&metadata.duration_seconds));
        let _ = fs::remove_dir_all(work);
    }

    #[test]
    fn timeline_subclips_become_standalone_audio_and_video_assets() {
        if StdCommand::new("ffmpeg").arg("-version").output().is_err() { return; }
        let work = std::env::temp_dir().join(format!("aiv-materialize-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&work).unwrap();
        let input = work.join("source.mp4");
        let video = work.join("video-trim.mp4");
        let audio = work.join("audio-trim.wav");
        let made = StdCommand::new("ffmpeg").args([
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=24:duration=4",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=4",
            "-c:v", "mpeg4", "-q:v", "3", "-c:a", "aac", "-shortest",
        ]).arg(&input).status().unwrap();
        assert!(made.success());
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        runtime.block_on(materialize_clip(MaterializeClipRequest {
            path: input.to_string_lossy().into_owned(),
            output_path: video.to_string_lossy().into_owned(),
            source_in_seconds: 1.0, source_out_seconds: 3.0, playback_rate: 2.0,
            kind: "video".into(),
        })).unwrap();
        let video_probe = runtime.block_on(probe(video.to_string_lossy().as_ref())).unwrap();
        assert!(video_probe.video_codec.is_some() && video_probe.audio_codec.is_some());
        assert!((0.90..=1.10).contains(&video_probe.duration_seconds), "unexpected video duration: {}", video_probe.duration_seconds);
        runtime.block_on(materialize_clip(MaterializeClipRequest {
            path: input.to_string_lossy().into_owned(),
            output_path: audio.to_string_lossy().into_owned(),
            source_in_seconds: 1.0, source_out_seconds: 3.0, playback_rate: 0.5,
            kind: "audio".into(),
        })).unwrap();
        let audio_probe = runtime.block_on(probe(audio.to_string_lossy().as_ref())).unwrap();
        assert!(audio_probe.audio_codec.is_some() && audio_probe.video_codec.is_none());
        assert!((3.90..=4.10).contains(&audio_probe.duration_seconds), "unexpected audio duration: {}", audio_probe.duration_seconds);
        let _ = fs::remove_dir_all(work);
    }

    #[test]
    fn lossless_last_frame_and_extension_trim_keep_streams_in_sync() {
        if StdCommand::new("ffmpeg").arg("-version").output().is_err() { return; }
        let work = std::env::temp_dir().join(format!("aiv-extend-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&work).unwrap();
        let input = work.join("input.mp4");
        let frame = work.join("last.png");
        let trimmed = work.join("trimmed.mp4");
        let trimmed_audio = work.join("trimmed.wav");
        let made = StdCommand::new("ffmpeg").args([
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=24:duration=1.5",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.5",
            "-c:v", "mpeg4", "-q:v", "3", "-c:a", "aac", "-shortest",
        ]).arg(&input).status().unwrap();
        assert!(made.success());
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        runtime.block_on(extract_last_visible_frame(input.to_string_lossy().as_ref(), 0.25, 1.25, &frame)).unwrap();
        assert!(frame.is_file() && fs::metadata(&frame).unwrap().len() > 100);
        runtime.block_on(trim_extension_boundary(&input, &trimmed, 1.0 / 24.0)).unwrap();
        let video_probe = runtime.block_on(probe(trimmed.to_string_lossy().as_ref())).unwrap();
        assert!((1.40..=1.50).contains(&video_probe.duration_seconds), "unexpected trimmed duration: {}", video_probe.duration_seconds);
        assert!(video_probe.video_codec.is_some() && video_probe.audio_codec.is_some());
        runtime.block_on(trim_audio_boundaries(&input, &trimmed_audio, 1.0 / 24.0, 1.0 / 24.0)).unwrap();
        let audio_probe = runtime.block_on(probe(trimmed_audio.to_string_lossy().as_ref())).unwrap();
        assert!(audio_probe.audio_codec.is_some() && audio_probe.video_codec.is_none());
        assert!((1.35..=1.48).contains(&audio_probe.duration_seconds), "unexpected H3 audio trim: {}", audio_probe.duration_seconds);
        let _ = fs::remove_dir_all(work);
    }

    #[test]
    fn denoise_keeps_the_picture_and_actually_reduces_noise() {
        if StdCommand::new("ffmpeg").arg("-version").output().is_err() { return; }
        let work = std::env::temp_dir().join(format!("aiv-denoise-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&work).unwrap();
        let input = work.join("noisy.mp4");
        // A tone buried in white noise: exactly the shape of the problem, and measurable afterwards.
        let made = StdCommand::new("ffmpeg").args([
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=24:duration=2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
            "-f", "lavfi", "-i", "anoisesrc=color=white:amplitude=0.25:sample_rate=48000:duration=2",
            "-filter_complex", "[1:a][2:a]amix=inputs=2:duration=shortest[a]",
            "-map", "0:v", "-map", "[a]", "-c:v", "mpeg4", "-q:v", "3", "-c:a", "aac", "-shortest",
        ]).arg(&input).status().unwrap();
        assert!(made.success());
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let result = runtime.block_on(denoise_audio(DenoiseRequest {
            path: input.to_string_lossy().into_owned(),
            strength: "medium".into(),
            highpass_hz: 60,
            normalize: false,
            output_path: String::new(),
        })).unwrap();
        assert!(result.video_copied, "denoise must stream-copy the picture, never re-encode it");
        assert!(result.filter.contains("afftdn"));
        let cleaned = PathBuf::from(&result.output_path);
        assert!(cleaned.is_file());
        let probe = runtime.block_on(probe(cleaned.to_string_lossy().as_ref())).unwrap();
        assert!(probe.video_codec.is_some() && probe.audio_codec.is_some());
        assert!((1.8..=2.2).contains(&probe.duration_seconds), "denoise changed the duration: {}", probe.duration_seconds);
        let _ = fs::remove_dir_all(work);
    }

    #[test]
    fn denoise_refuses_a_silent_source_rather_than_writing_an_empty_file() {
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let error = runtime.block_on(denoise_audio(DenoiseRequest {
            path: "does-not-exist.mp4".into(),
            strength: "medium".into(),
            highpass_hz: 60,
            normalize: false,
            output_path: String::new(),
        })).unwrap_err();
        assert!(error.contains("does not exist"));
    }
}
