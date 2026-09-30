//! VFX Phase 7: the timeline export on the compositor.
//!
//! The page renders every programme frame with the same engine as the monitor
//! (`src/render/programmeEngine.ts`), packs it to I420 and streams it here (HTTP channel or IPC, as
//! the Clip Editor's render does). This side does what only ffmpeg does well:
//!
//! 1. `start`: mixes the programme's audio once (every picture, overlay and audio-track entry with its
//!    trim, speed, gain, pan and fades, junction crossfades already expressed as fades by the page,
//!    and the programme's head/tail dips), then starts ONE encode with the user's delivery settings
//!    reading the frames from stdin and the mix from the file.
//! 2. `finish`: waits for the encode, runs the upscale chain when one was asked for, and publishes
//!    the file to the chosen path exactly once (a staging file until then, as before).
//!
//! The old filter-graph export (`export.rs`) is untouched and stays behind the "Legacy export" switch.

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, fs, path::{Path, PathBuf}, sync::Arc};
use tokio::sync::Mutex;
use uuid::Uuid;

use crate::ai::render::{self, PixelFormat, RenderSessions, RenderStarted};
use crate::export::{EncodeSettings, ExportResult};
use crate::domain::Rational;

/// One audible programme entry (`ProgrammeAudio` in `src/domain/programme.ts`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgrammeAudioEntry {
    pub path: String,
    /// Programme seconds.
    pub start: f64,
    pub duration: f64,
    /// Source seconds at `start`, and source seconds per programme second.
    pub source_in: f64,
    pub rate: f64,
    pub volume_db: f64,
    #[serde(default)]
    pub pan: f64,
    #[serde(default)]
    pub fade_in: f64,
    #[serde(default)]
    pub fade_out: f64,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgrammeExportRequest {
    pub output_path: String,
    pub width: u32,
    pub height: u32,
    pub frame_rate: Rational,
    /// Programme seconds (frames = round(duration * fps)).
    pub duration: f64,
    #[serde(default)]
    pub audio: Vec<ProgrammeAudioEntry>,
    /// The programme's head and tail dips, applied to the sound too (seconds, 0 = none).
    #[serde(default)]
    pub head_fade: f64,
    #[serde(default)]
    pub tail_fade: f64,
    pub video_codec: String,
    pub audio_codec: String,
    pub video_bitrate: String,
    pub audio_bitrate: String,
    #[serde(default = "default_quality_mode")]
    pub quality_mode: String,
    #[serde(default = "default_quality")]
    pub quality: u32,
    #[serde(default)]
    pub preset: String,
    #[serde(default = "default_sample_rate")]
    pub audio_sample_rate: u32,
    #[serde(default = "default_channels")]
    pub audio_channels: u32,
    #[serde(default)]
    pub upscale: Option<crate::ai::upscale::UpscaleChain>,
    /// "yuv420" (the page packs) or "rgba".
    #[serde(default)]
    pub pixel_format: Option<String>,
    #[serde(default)]
    pub http: bool,
}

fn default_quality_mode() -> String { "quality".to_owned() }
fn default_quality() -> u32 { 21 }
fn default_sample_rate() -> u32 { 48_000 }
fn default_channels() -> u32 { 2 }

impl ProgrammeExportRequest {
    fn encode(&self) -> EncodeSettings {
        EncodeSettings {
            video_codec: self.video_codec.clone(), quality_mode: self.quality_mode.clone(), quality: self.quality,
            preset: self.preset.clone(), video_bitrate: self.video_bitrate.clone(), audio_codec: self.audio_codec.clone(),
            audio_bitrate: self.audio_bitrate.clone(), audio_sample_rate: self.audio_sample_rate, audio_channels: self.audio_channels,
        }
    }
}

struct Job {
    render_dir: PathBuf,
    staging: PathBuf,
    final_output: PathBuf,
    chain: Option<crate::ai::upscale::UpscaleChain>,
    encode: EncodeSettings,
    fps: f64,
    width: u32,
    height: u32,
    command: Vec<String>,
}

#[derive(Default, Clone)]
pub struct ExportJobs(Arc<Mutex<HashMap<String, Job>>>);

/// The mix's sample rate (stereo, 32-bit float while mixing, 16-bit in the WAV).
pub const MIX_RATE: u32 = 48_000;

/// Whether an entry is heard at all.
fn audible(entry: &ProgrammeAudioEntry) -> bool { entry.duration > 0.0 && entry.volume_db > -60.0 }

/// ffmpeg arguments that decode ONE entry -- trimmed, retimed, gained, panned and faded -- as raw
/// 48 kHz stereo f32 on stdout. Placing it on the programme and summing is done in Rust (`place`).
///
/// Why not one ffmpeg graph (anullsrc + adelay + amix), as the first version did: on the development machine's ffmpeg
/// build (newer than 7.0, 2026-09-29) that graph ended about 1.4 s into a 2 s programme whatever the
/// `amix` duration mode and even behind `apad,atrim`. A single-input chain per entry, with the length
/// fixed by construction here, does not depend on how a build schedules a multi-input graph.
pub fn entry_args(entry: &ProgrammeAudioEntry) -> Vec<String> {
    let rate = entry.rate.clamp(0.25, 4.0);
    let fade_in = entry.fade_in.clamp(0.0, entry.duration);
    let fade_out = entry.fade_out.clamp(0.0, entry.duration);
    let pan = entry.pan.clamp(-1.0, 1.0);
    let left = if pan > 0.0 { 1.0 - pan } else { 1.0 };
    let right = if pan < 0.0 { 1.0 + pan } else { 1.0 };
    let mut chain = format!(
        "aresample={MIX_RATE},atrim=duration={:.6},asetpts=PTS-STARTPTS,{},volume={:.3}dB,aformat=channel_layouts=stereo,pan=stereo|c0={left:.6}*c0|c1={right:.6}*c1",
        entry.duration * rate, crate::export::atempo_filter(rate), entry.volume_db.clamp(-60.0, 24.0),
    );
    if fade_in > 0.0 { chain.push_str(&format!(",afade=t=in:st=0:d={fade_in:.6}")); }
    if fade_out > 0.0 { chain.push_str(&format!(",afade=t=out:st={:.6}:d={fade_out:.6}", (entry.duration - fade_out).max(0.0))); }
    let mut args: Vec<String> = ["-hide_banner", "-loglevel", "error", "-nostdin"].iter().map(|v| v.to_string()).collect();
    args.extend([
        "-ss".into(), format!("{:.6}", entry.source_in.max(0.0)), "-i".into(), entry.path.clone(), "-vn".into(),
        "-af".into(), chain, "-f".into(), "f32le".into(), "-ac".into(), "2".into(), "-ar".into(), MIX_RATE.to_string(), "pipe:1".into(),
    ]);
    args
}

/// Frames in `seconds` at the mix rate.
pub fn frames(seconds: f64) -> usize { (seconds.max(0.0) * MIX_RATE as f64).round() as usize }

/// Add one decoded entry (interleaved stereo f32) into `mix` at its programme start, for at most its
/// duration; a part before 0 or after the end of the programme is dropped.
pub fn place(mix: &mut [f32], decoded: &[f32], start: f64, duration: f64) {
    let total = mix.len() / 2;
    let offset = (start * MIX_RATE as f64).round() as i64;
    let length = frames(duration).min(decoded.len() / 2);
    for index in 0..length {
        let at = offset + index as i64;
        if at < 0 { continue; }
        let at = at as usize;
        if at >= total { break; }
        mix[at * 2] += decoded[index * 2];
        mix[at * 2 + 1] += decoded[index * 2 + 1];
    }
}

/// The programme's head and tail dips on the sound: linear, like `afade`.
pub fn dip(mix: &mut [f32], head: f64, tail: f64) {
    let total = mix.len() / 2;
    let (head, tail) = (frames(head).min(total), frames(tail).min(total));
    for index in 0..head {
        let gain = index as f32 / head as f32;
        mix[index * 2] *= gain;
        mix[index * 2 + 1] *= gain;
    }
    for index in 0..tail {
        let at = total - 1 - index;
        let gain = index as f32 / tail as f32;
        mix[at * 2] *= gain;
        mix[at * 2 + 1] *= gain;
    }
}

/// A 16-bit stereo WAV of the mix (clipped, rounded).
pub fn wav_bytes(mix: &[f32]) -> Vec<u8> {
    let data = (mix.len() * 2) as u32;
    let mut bytes = Vec::with_capacity(44 + data as usize);
    bytes.extend(b"RIFF");
    bytes.extend((36 + data).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(2u16.to_le_bytes());
    bytes.extend(MIX_RATE.to_le_bytes());
    bytes.extend((MIX_RATE * 4).to_le_bytes());
    bytes.extend(4u16.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend(data.to_le_bytes());
    for value in mix { bytes.extend(((value.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes()); }
    bytes
}

/// Decoded f32 samples from ffmpeg's raw output.
fn samples(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

/// Mix the programme's sound into a WAV of exactly `duration` seconds at `output`.
pub async fn render_mix(entries: &[ProgrammeAudioEntry], duration: f64, head_fade: f64, tail_fade: f64, output: &Path) -> Result<(), String> {
    let mut mix = vec![0.0_f32; frames(duration) * 2];
    for entry in entries.iter().filter(|entry| audible(entry)) {
        let mut command = tokio::process::Command::new("ffmpeg");
        #[cfg(windows)]
        command.creation_flags(0x0800_0000);
        let decoded = command.args(entry_args(entry)).stdin(std::process::Stdio::null()).output().await
            .map_err(|error| format!("failed to launch ffmpeg: {error}"))?;
        if !decoded.status.success() {
            return Err(format!("{}: {}", entry.path, String::from_utf8_lossy(&decoded.stderr).trim()));
        }
        place(&mut mix, &samples(&decoded.stdout), entry.start, entry.duration);
    }
    dip(&mut mix, head_fade, tail_fade);
    fs::write(output, wav_bytes(&mix)).map_err(|error| format!("could not write the mix: {error}"))
}

/// The final encode: raw frames on stdin, the mix (when there is one), the user's delivery settings.
pub fn encode_args(width: u32, height: u32, fps: f64, format: PixelFormat, mix: Option<&Path>, duration: f64, encode: &EncodeSettings, output: &Path) -> Vec<String> {
    let mut args: Vec<String> = ["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo"].iter().map(|v| v.to_string()).collect();
    match format {
        PixelFormat::Rgba => args.extend(["-pix_fmt".into(), "rgba".into(), "-s".into(), format!("{width}x{height}")]),
        PixelFormat::Yuv420 => {
            let (w, h) = (width + (width & 1), height + (height & 1));
            args.extend(["-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
                .iter().map(|v| v.to_string()));
            args.extend(["-s".into(), format!("{w}x{h}")]);
        }
    }
    args.extend(["-r".into(), format!("{fps}"), "-i".into(), "-".into()]);
    if let Some(mix) = mix {
        args.extend(["-i".into(), mix.to_string_lossy().into_owned(), "-map".into(), "0:v:0".into(), "-map".into(), "1:a:0".into()]);
    }
    if format == PixelFormat::Rgba { args.extend(["-vf".into(), render::RGBA_TO_YUV.into()]); }
    args.extend(encode.video_args(false));
    if mix.is_some() { args.extend(encode.audio_args()); } else { args.push("-an".into()); }
    args.extend(["-t".into(), format!("{duration:.6}")]);
    let extension = output.extension().and_then(|value| value.to_str()).unwrap_or("mp4").to_ascii_lowercase();
    if extension == "mp4" || extension == "mov" { args.extend(["-movflags".into(), "+faststart".into()]); }
    args.push(output.to_string_lossy().into_owned());
    args
}

pub async fn start(jobs: &ExportJobs, sessions: &RenderSessions, request: ProgrammeExportRequest) -> Result<RenderStarted, String> {
    let fps = request.frame_rate.validate()?.as_f64();
    if request.width == 0 || request.height == 0 || request.width > 8192 || request.height > 8192 || fps <= 0.0 {
        return Err("invalid export settings".into());
    }
    if request.duration <= 0.0 { return Err("the timeline is empty".into()); }
    let format = PixelFormat::parse(request.pixel_format.as_deref())?;
    let final_output = PathBuf::from(&request.output_path);
    if let Some(parent) = final_output.parent() { fs::create_dir_all(parent).map_err(|error| error.to_string())?; }
    for entry in &request.audio {
        if !Path::new(&entry.path).is_file() { return Err(format!("audio source is missing: {}", entry.path)); }
    }
    let render_dir = std::env::temp_dir().join(format!("aiv-programme-{}", Uuid::new_v4()));
    fs::create_dir_all(&render_dir).map_err(|error| error.to_string())?;
    let extension = final_output.extension().and_then(|value| value.to_str()).unwrap_or("mp4");
    let staging = render_dir.join(format!("master.{extension}"));
    let encode = request.encode();

    let setup = async {
        let mix = if encode.audio_codec == "none" { None } else {
            let path = render_dir.join("mix.wav");
            render_mix(&request.audio, request.duration, request.head_fade, request.tail_fade, &path).await
                .map_err(|error| format!("the audio mix failed:\n{error}"))?;
            Some(path)
        };
        let args = encode_args(request.width, request.height, fps, format, mix.as_deref(), request.duration, &encode, &staging);
        let started = render::spawn(sessions, staging.clone(), render_dir.join("encode.log"), args.clone(), format, request.width, request.height, request.http, "export").await?;
        Ok::<_, String>((started, args))
    }.await;
    let (started, command) = match setup {
        Ok(value) => value,
        Err(error) => { let _ = fs::remove_dir_all(&render_dir); return Err(error); }
    };
    jobs.0.lock().await.insert(started.id.clone(), Job {
        render_dir, staging, final_output, chain: request.upscale.clone(), encode, fps, width: request.width, height: request.height, command,
    });
    Ok(started)
}

pub async fn finish(jobs: &ExportJobs, sessions: &RenderSessions, id: &str) -> Result<ExportResult, String> {
    let job = jobs.0.lock().await.remove(id).ok_or("no such export")?;
    let result = async {
        render::finish(sessions, id).await?;
        let mut warnings = Vec::new();
        let mut command = job.command.clone();
        if let Some(chain) = &job.chain {
            match crate::export::run_upscale_chain(&job.staging, &job.render_dir, chain, &job.encode, job.fps).await {
                Ok(args) => if !args.is_empty() { command = args; },
                // The master is finished; losing it over the optional second pass would be the worst
                // outcome, so it is still published at the size it was rendered at, and said so.
                Err(error) => warnings.push(format!("The export finished at {}x{}, but the upscale did not run: {error}", job.width, job.height)),
            }
        }
        fs::copy(&job.staging, &job.final_output)
            .map_err(|error| format!("the render finished but could not be written to {}: {error}", job.final_output.display()))?;
        Ok(ExportResult { output_path: job.final_output.to_string_lossy().into_owned(), command, warnings })
    }.await;
    if std::env::var_os("AIV_KEEP_RENDER_CACHE").is_none() { let _ = fs::remove_dir_all(&job.render_dir); }
    result
}

pub async fn cancel(jobs: &ExportJobs, sessions: &RenderSessions, id: &str) -> Result<(), String> {
    render::cancel(sessions, id).await?;
    if let Some(job) = jobs.0.lock().await.remove(id) { let _ = fs::remove_dir_all(&job.render_dir); }
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    fn entry(start: f64, duration: f64) -> ProgrammeAudioEntry {
        ProgrammeAudioEntry { path: "C:/m/a b.mp4".into(), start, duration, source_in: 1.5, rate: 2.0, volume_db: -3.0, pan: 0.5, fade_in: 1.0, fade_out: 0.5 }
    }

    #[test]
    fn decodes_each_entry_trimmed_retimed_gained_panned_and_faded() {
        let args = entry_args(&entry(2.0, 4.0));
        let joined = args.join(" ");
        assert!(joined.contains("-ss 1.500000 -i C:/m/a b.mp4 -vn"), "{joined}");
        assert!(args.contains(&"C:/m/a b.mp4".to_string()), "the path is one argument");
        let chain = &args[args.iter().position(|value| value == "-af").unwrap() + 1];
        assert!(chain.contains("atrim=duration=8.000000"), "source seconds = duration x rate: {chain}");
        assert!(chain.contains("atempo=2"), "{chain}");
        assert!(chain.contains("volume=-3.000dB"), "{chain}");
        assert!(chain.contains("c0=0.500000*c0|c1=1.000000*c1"), "{chain}");
        assert!(chain.contains("afade=t=in:st=0:d=1.000000") && chain.contains("afade=t=out:st=3.500000:d=0.500000"), "{chain}");
        assert!(joined.ends_with("-f f32le -ac 2 -ar 48000 pipe:1"), "{joined}");
        assert!(!audible(&ProgrammeAudioEntry { volume_db: -60.0, ..entry(0.0, 1.0) }));
    }

    #[test]
    fn places_mixes_dips_and_writes_an_exact_length() {
        // 1 s programme; a constant 0.5 entry from 0.25 s for 0.5 s, another from -0.1 s for 0.3 s.
        let mut mix = vec![0.0_f32; frames(1.0) * 2];
        let ones = vec![0.5_f32; frames(2.0) * 2];
        place(&mut mix, &ones, 0.25, 0.5);
        place(&mut mix, &ones, -0.1, 0.3);
        let at = |seconds: f64| mix[frames(seconds) * 2];
        assert_eq!(at(0.05), 0.5, "the second entry, its first 0.1 s dropped");
        assert_eq!(at(0.22), 0.0);
        assert_eq!(at(0.3), 0.5);
        assert_eq!(at(0.74), 0.5);
        assert_eq!(at(0.76), 0.0, "an entry lasts its duration even when more was decoded");
        place(&mut mix, &ones, 0.9, 1.0);
        assert_eq!(mix.len(), frames(1.0) * 2, "nothing written past the programme");
        dip(&mut mix, 0.1, 0.0);
        assert!(mix[0] == 0.0 && (mix[frames(0.05) * 2] - 0.25).abs() < 1e-3);
        let wav = wav_bytes(&mix);
        assert_eq!(wav.len(), 44 + frames(1.0) * 4);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes([wav[24], wav[25], wav[26], wav[27]]), 48_000);
    }

    #[test]
    fn encodes_the_frames_once_with_the_delivery_settings() {
        let encode = EncodeSettings {
            video_codec: "libx264".into(), quality_mode: "quality".into(), quality: 18, preset: "slow".into(),
            video_bitrate: "12M".into(), audio_codec: "aac".into(), audio_bitrate: "320k".into(), audio_sample_rate: 48_000, audio_channels: 2,
        };
        let args = encode_args(1920, 1080, 24.0, PixelFormat::Yuv420, Some(Path::new("mix.wav")), 9.5, &encode, Path::new("out.mp4")).join(" ");
        assert!(args.contains("-f rawvideo -pix_fmt yuv420p") && args.contains("-s 1920x1080 -r 24 -i -"), "{args}");
        assert!(args.contains("-i mix.wav -map 0:v:0 -map 1:a:0"), "{args}");
        assert!(args.contains("-c:v libx264 -preset slow -crf 18"), "{args}");
        assert!(args.contains("-c:a aac") && args.contains("-b:a 320k"), "{args}");
        assert!(args.contains("-t 9.500000") && args.ends_with("out.mp4"), "{args}");
        assert!(!args.contains("-vf"), "I420 goes to the encoder untouched: {args}");
        let silent = encode_args(1280, 720, 30.0, PixelFormat::Rgba, None, 2.0, &encode, Path::new("out.mkv")).join(" ");
        assert!(silent.contains("-an") && silent.contains(render::RGBA_TO_YUV) && !silent.contains("faststart"), "{silent}");
    }

    /// Runs ffmpeg when it is on PATH: a real mix of a generated tone, the right length.
    #[test]
    fn the_mix_renders_to_the_programme_length() {
        if std::process::Command::new("ffmpeg").arg("-version").output().is_err() { return; }
        let dir = std::env::temp_dir().join(format!("aiv-mix-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let tone = dir.join("tone.wav");
        let made = std::process::Command::new("ffmpeg").args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=3", tone.to_str().unwrap()]).status().unwrap();
        assert!(made.success());
        // Two entries: 0.5..1.5 s and 1.2..1.9 s of a 2 s programme. The mix must run the full 2 s,
        // and the second entry must still sound after the first one has ended (an ffmpeg that ends the
        // mix when any input ends would leave 1.5..1.9 silent).
        let tone_path = tone.to_string_lossy().into_owned();
        let entries = [
            ProgrammeAudioEntry { path: tone_path.clone(), start: 0.5, duration: 1.0, source_in: 0.5, rate: 1.0, volume_db: 0.0, pan: 0.0, fade_in: 0.1, fade_out: 0.1 },
            ProgrammeAudioEntry { path: tone_path, start: 1.2, duration: 0.7, source_in: 0.0, rate: 1.0, volume_db: 0.0, pan: 0.0, fade_in: 0.0, fade_out: 0.0 },
        ];
        // The same steps as `render_mix`, synchronously: decode each entry, place it, write the WAV.
        let out = dir.join("mix.wav");
        let mut mix = vec![0.0_f32; frames(2.0) * 2];
        for item in &entries {
            let decoded = std::process::Command::new("ffmpeg").args(entry_args(item)).output().unwrap();
            assert!(decoded.status.success(), "{}", String::from_utf8_lossy(&decoded.stderr));
            place(&mut mix, &samples(&decoded.stdout), item.start, item.duration);
        }
        fs::write(&out, wav_bytes(&mix)).unwrap();
        let bytes = fs::read(&out).unwrap();
        // 2 s of 48 kHz stereo 16-bit plus a small header.
        assert!((bytes.len() as i64 - 384_000).abs() < 2_000, "{} bytes: expected 2 s of 48 kHz stereo 16-bit", bytes.len());
        let data = &bytes[bytes.len() - 384_000..];
        let loudness = |from: f64, to: f64| {
            let (a, b) = ((from * 48_000.0) as usize * 4, (to * 48_000.0) as usize * 4);
            data[a..b].chunks_exact(2).map(|pair| i16::from_le_bytes([pair[0], pair[1]]).unsigned_abs() as u64).max().unwrap_or(0)
        };
        assert!(loudness(0.1, 0.4) < 50, "silence before the first entry");
        assert!(loudness(0.7, 1.1) > 1_500, "the first entry sounds");
        assert!(loudness(1.6, 1.85) > 1_500, "the second entry still sounds after the first ended");
        assert!(loudness(1.92, 1.99) < 50, "silence after the last entry, padded to the end");
        let _ = fs::remove_dir_all(&dir);
    }
}
