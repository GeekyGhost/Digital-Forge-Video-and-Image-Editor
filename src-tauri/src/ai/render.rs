//! Clip Editor VFX render (VFX Phase 4b): the VFX mode composes every frame of the clip's range with
//! the same compositor its preview uses, and streams each one here as raw RGBA; ffmpeg encodes H.264.
//! The finished file comes back to the timeline as a new asset (replace the range, or insert after).
//!
//! The RGB -> YUV conversion is done by ffmpeg with an explicit BT.709 limited-range matrix, and the
//! stream is TAGGED to match. Tagging alone does not convert (ffmpeg's default RGB->YUV matrix is
//! BT.601): that mismatch is what caused the upscale "coloured lines" (see the colour-conversion
//! note in the project), so both halves are here on purpose.
//!
//! Output goes to `projects/<id>/media/videos/`.
//!
//! VFX Phase 6: the page may instead send planar I420 it packed itself (`src/render/yuv420.ts`: the
//! same BT.709 limited matrix, 2x2 average chroma = ffmpeg's `flags=area+accurate_rnd`, which the RGBA
//! path now asks for too, so both paths give the same bytes to 1/255), and send it over a per-render
//! localhost HTTP channel (`frame_channel.rs`) instead of Tauri IPC. The IPC command stays as the
//! fallback and accepts either format.

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, path::{Path, PathBuf}, process::Stdio, sync::{atomic::{AtomicU8, Ordering}, Arc}, time::Instant};
use tokio::{io::AsyncWriteExt, process::{Child, ChildStdin, Command}, sync::Mutex};

/// Shared with each render's HTTP channel thread, hence the `Arc`.
#[derive(Default, Clone)]
pub struct RenderSessions(Arc<Mutex<HashMap<String, Session>>>);

/// What a render's frames are: straight RGBA (ffmpeg converts), or planar I420 BT.709 limited at the
/// even size (the page converted).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PixelFormat {
    Rgba,
    Yuv420,
}

impl PixelFormat {
    pub fn parse(value: Option<&str>) -> Result<Self, String> {
        match value.unwrap_or("rgba") {
            "rgba" => Ok(Self::Rgba),
            "yuv420" => Ok(Self::Yuv420),
            other => Err(format!("unknown pixel format {other}")),
        }
    }

    /// Bytes per frame: RGBA at the given size, I420 at the size rounded up to even.
    pub fn frame_bytes(self, width: u32, height: u32) -> usize {
        match self {
            Self::Rgba => width as usize * height as usize * 4,
            Self::Yuv420 => {
                let (w, h) = (width as usize + (width as usize & 1), height as usize + (height as usize & 1));
                w * h * 3 / 2
            }
        }
    }
}

struct Session {
    child: Child,
    stdin: Option<ChildStdin>,
    frames: u32,
    frame_bytes: usize,
    output: PathBuf,
    log: PathBuf,
    /// Time spent inside `write_all` into ffmpeg: the encoder's back-pressure, apart from IPC.
    write_seconds: f64,
    encoder: &'static str,
    /// The localhost channel, closed when the session ends (finish, cancel, or drop).
    channel: Option<super::frame_channel::FrameChannel>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderStarted {
    pub id: String,
    pub output_path: String,
    pub pixel_format: PixelFormat,
    pub frame_bytes: usize,
    /// Where to POST frames instead of the IPC command; `None` when it was not asked for or could not
    /// listen (then IPC it is).
    pub http: Option<super::frame_channel::Endpoint>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderFinished {
    pub output_path: String,
    pub frames: u32,
    /// Seconds spent writing frames into ffmpeg (encoder back-pressure), for the page's timing line.
    pub write_seconds: f64,
    pub encoder: String,
}

/// NVENC refuses tiny frames ("Frame Dimension less than the minimum supported value": 17x9 failed on
/// the 4090, 2026-09-28; the H.264 minimum is about 145x49 and varies by GPU). Anything under this
/// safe floor goes to libx264, which takes any even size.
pub fn nvenc_fits(width: u32, height: u32) -> bool {
    width >= 256 && height >= 144
}

/// 0 unknown, 1 works, 2 does not. NVENC is probed once per launch with a one-frame encode.
static NVENC: AtomicU8 = AtomicU8::new(0);

async fn nvenc_works() -> bool {
    match NVENC.load(Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    let mut command = Command::new("ffmpeg");
    #[cfg(windows)]
    command.creation_flags(0x0800_0000);
    let works = command.args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=s=256x256", "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"])
        .stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null())
        .status().await.map(|status| status.success()).unwrap_or(false);
    NVENC.store(if works { 1 } else { 2 }, Ordering::Relaxed);
    works
}

/// The plate's own sound for the rendered span, muxed in so a VFX render never drops the audio.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioSpan {
    pub path: String,
    pub start: f64,
    pub duration: f64,
}

/// The RGBA path's conversion: pad to even, BT.709 limited, 2x2-average chroma. `yuv420.ts` does the
/// same on the page (a test holds the two to 1/255); `area+accurate_rnd` is what makes ffmpeg's chroma
/// the plain block average (its default bicubic chroma differs by up to 10 levels on grain).
pub const RGBA_TO_YUV: &str = "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_color_matrix=bt709:out_range=tv:flags=area+accurate_rnd,format=yuv420p";

/// ffmpeg arguments: raw frames on stdin -> H.264 yuv420p, BT.709 limited, tagged.
/// - RGBA: ffmpeg converts (`RGBA_TO_YUV`); odd sizes are padded by one pixel (4:2:0 needs even).
/// - I420: already converted and padded by the page (`width`/`height` are the frame's, the stream is
///   the even size); it goes to the encoder untouched, tagged as what it is.
/// With `audio`, that span of the source's first audio stream is added as AAC (`?`: a source without
/// audio still renders).
#[allow(clippy::too_many_arguments)]
pub fn ffmpeg_args(width: u32, height: u32, fps: f64, crf: u32, nvenc: bool, format: PixelFormat, audio: Option<&AudioSpan>, output: &Path) -> Vec<String> {
    let mut args: Vec<String> = ["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo"].iter().map(|value| value.to_string()).collect();
    match format {
        PixelFormat::Rgba => {
            args.extend(["-pix_fmt".into(), "rgba".into(), "-s".into(), format!("{width}x{height}")]);
        }
        PixelFormat::Yuv420 => {
            let (w, h) = (width + (width & 1), height + (height & 1));
            args.extend(["-pix_fmt", "yuv420p", "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
                .iter().map(|value| value.to_string()));
            args.extend(["-s".into(), format!("{w}x{h}")]);
        }
    }
    args.extend(["-r".into(), format!("{fps}"), "-i".into(), "-".into()]);
    if let Some(span) = audio {
        args.extend([
            "-ss".into(), format!("{:.6}", span.start.max(0.0)), "-t".into(), format!("{:.6}", span.duration.max(0.0)),
            "-i".into(), span.path.clone(), "-map".into(), "0:v:0".into(), "-map".into(), "1:a:0?".into(),
            "-c:a".into(), "aac".into(), "-b:a".into(), "192k".into(),
        ]);
    }
    if format == PixelFormat::Rgba {
        args.extend(["-vf".into(), RGBA_TO_YUV.into()]);
    }
    // The RGB -> YUV conversion above is ffmpeg's either way (NVENC's own RGBA conversion measured
    // 3.2/255 off in Phase 0); the encoder only ever sees yuv420p. First 4090 runs (2026-09-28): each
    // 960x544 frame took about half a second to send with libx264 "medium".
    if nvenc {
        args.extend(["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "19", "-b:v", "0"].iter().map(|value| value.to_string()));
    } else {
        args.extend(["-c:v".into(), "libx264".into(), "-preset".into(), "fast".into(), "-crf".into(), crf.to_string()]);
    }
    args.extend(["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-movflags", "+faststart"]
        .iter().map(|value| value.to_string()));
    args.push(output.to_string_lossy().into_owned());
    args
}

/// `<slug>-render.mp4`, then `-02`, `-03` ... in `projects/<id>/media/videos/`.
pub fn output_path(root: &Path, project_id: &str, hint: &str) -> Result<PathBuf, String> {
    let id = uuid::Uuid::parse_str(project_id).map_err(|_| "invalid project id".to_owned())?;
    let dir = root.join("projects").join(id.to_string()).join("media/videos");
    std::fs::create_dir_all(&dir).map_err(|error| format!("could not create {}: {error}", dir.display()))?;
    let stem: String = hint.to_lowercase().chars()
        .map(|value| if value.is_ascii_alphanumeric() { value } else { '-' })
        .collect::<String>().split('-').filter(|part| !part.is_empty()).take(5).collect::<Vec<_>>().join("-");
    let stem = if stem.is_empty() { "clip-vfx".to_owned() } else { stem.chars().take(44).collect() };
    for counter in 1..10_000 {
        let name = if counter == 1 { format!("{stem}-render.mp4") } else { format!("{stem}-render-{counter:02}.mp4") };
        let path = dir.join(name);
        if !path.exists() { return Ok(path); }
    }
    Err("too many renders with this name".into())
}

#[allow(clippy::too_many_arguments)]
pub async fn start(
    sessions: &RenderSessions, root: &Path, project_id: &str, name: &str, width: u32, height: u32, fps: f64,
    audio: Option<AudioSpan>, format: PixelFormat, http: bool,
) -> Result<RenderStarted, String> {
    if width == 0 || height == 0 || width > 8192 || height > 8192 { return Err(format!("cannot render {width}x{height}")); }
    if !(fps > 0.0 && fps <= 240.0) { return Err(format!("cannot render at {fps} fps")); }
    let output = output_path(root, project_id, name)?;
    let log = output.with_extension("ffmpeg.log");
    let nvenc = nvenc_fits(width, height) && nvenc_works().await;
    let args = ffmpeg_args(width, height, fps, 16, nvenc, format, audio.as_ref(), &output);
    spawn(sessions, output, log, args, format, width, height, http, if nvenc { "h264_nvenc" } else { "libx264" }).await
}

/// VFX Phase 7: a render session with the caller's own ffmpeg arguments (raw frames on stdin in
/// `format` at `width` x `height`), e.g. the timeline export's final encode. Frames arrive exactly as
/// for `start`: the IPC command or the localhost channel.
#[allow(clippy::too_many_arguments)]
pub async fn spawn(
    sessions: &RenderSessions, output: PathBuf, log: PathBuf, args: Vec<String>, format: PixelFormat,
    width: u32, height: u32, http: bool, encoder: &'static str,
) -> Result<RenderStarted, String> {
    let log_file = std::fs::File::create(&log).map_err(|error| error.to_string())?;
    let mut command = Command::new("ffmpeg");
    #[cfg(windows)]
    command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    let mut child = command.args(&args)
        .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::from(log_file))
        .spawn().map_err(|error| format!("could not start ffmpeg (is it on PATH?): {error}"))?;
    let stdin = child.stdin.take();
    let id = uuid::Uuid::new_v4().simple().to_string();
    let frame_bytes = format.frame_bytes(width, height);
    // The channel's thread writes through the same `frame` as the IPC command, on this runtime.
    let channel = if http {
        let (runtime, shared, session) = (tokio::runtime::Handle::current(), sessions.clone(), id.clone());
        super::frame_channel::FrameChannel::start(frame_bytes, Arc::new(move |bytes: &[u8]| {
            runtime.block_on(frame(&shared, &session, bytes))
        })).ok()
    } else {
        None
    };
    let endpoint = channel.as_ref().map(|channel| channel.endpoint.clone());
    sessions.0.lock().await.insert(id.clone(), Session {
        child, stdin, frames: 0, frame_bytes, output: output.clone(), log,
        write_seconds: 0.0, encoder, channel,
    });
    Ok(RenderStarted { id, output_path: output.to_string_lossy().into_owned(), pixel_format: format, frame_bytes, http: endpoint })
}

pub async fn frame(sessions: &RenderSessions, id: &str, bytes: &[u8]) -> Result<u32, String> {
    let mut map = sessions.0.lock().await;
    let session = map.get_mut(id).ok_or("no such render")?;
    if bytes.len() != session.frame_bytes {
        return Err(format!("a frame must be {} bytes, got {}", session.frame_bytes, bytes.len()));
    }
    let stdin = session.stdin.as_mut().ok_or("the render is already finished")?;
    let started = Instant::now();
    let written = stdin.write_all(bytes).await;
    session.write_seconds += started.elapsed().as_secs_f64();
    if let Err(error) = written {
        let tail = std::fs::read_to_string(&session.log).unwrap_or_default();
        return Err(format!("ffmpeg stopped accepting frames ({error}): {}", tail.trim()));
    }
    session.frames += 1;
    Ok(session.frames)
}

pub async fn finish(sessions: &RenderSessions, id: &str) -> Result<RenderFinished, String> {
    let mut session = sessions.0.lock().await.remove(id).ok_or("no such render")?;
    drop(session.channel.take());
    drop(session.stdin.take());
    let status = session.child.wait().await.map_err(|error| error.to_string())?;
    let tail = std::fs::read_to_string(&session.log).unwrap_or_default();
    if !status.success() || session.frames == 0 {
        return Err(format!("ffmpeg failed ({status}, {} frames): {}", session.frames, tail.trim()));
    }
    let _ = std::fs::remove_file(&session.log);
    Ok(RenderFinished {
        output_path: session.output.to_string_lossy().into_owned(), frames: session.frames,
        write_seconds: session.write_seconds, encoder: session.encoder.to_owned(),
    })
}

/// Stop a render and remove its partial file.
pub async fn cancel(sessions: &RenderSessions, id: &str) -> Result<(), String> {
    let Some(mut session) = sessions.0.lock().await.remove(id) else { return Ok(()) };
    drop(session.channel.take());
    drop(session.stdin.take());
    let _ = session.child.kill().await;
    let _ = std::fs::remove_file(&session.output);
    let _ = std::fs::remove_file(&session.log);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_with_bt709_and_tags_it() {
        let args = ffmpeg_args(961, 544, 23.976, 16, false, PixelFormat::Rgba, None, Path::new("out.mp4")).join(" ");
        assert!(args.contains("-pix_fmt rgba -s 961x544 -r 23.976 -i -"), "{args}");
        assert!(args.contains("scale=out_color_matrix=bt709:out_range=tv:flags=area+accurate_rnd,format=yuv420p"), "{args}");
        assert!(args.contains("pad=ceil(iw/2)*2:ceil(ih/2)*2"), "{args}");
        assert!(args.contains("-colorspace bt709") && args.contains("-color_range tv"), "{args}");
        assert!(args.ends_with("out.mp4"));
        assert!(!args.contains("-c:a"));
        let span = AudioSpan { path: "C:/m/a b.mp4".into(), start: 2.5, duration: 1.25 };
        assert!(args.contains("-c:v libx264"));
        assert!(!nvenc_fits(17, 9) && !nvenc_fits(255, 1080) && !nvenc_fits(1920, 143));
        assert!(nvenc_fits(256, 144) && nvenc_fits(960, 544) && nvenc_fits(3840, 2160));
        let nv = ffmpeg_args(960, 544, 24.0, 16, true, PixelFormat::Rgba, None, Path::new("out.mp4")).join(" ");
        assert!(nv.contains("-c:v h264_nvenc") && nv.contains("format=yuv420p") && !nv.contains("libx264"), "{nv}");
        let with_audio = ffmpeg_args(960, 544, 24.0, 16, false, PixelFormat::Rgba, Some(&span), Path::new("out.mp4"));
        let joined = with_audio.join(" ");
        assert!(joined.contains("-ss 2.500000 -t 1.250000 -i C:/m/a b.mp4 -map 0:v:0 -map 1:a:0? -c:a aac"), "{joined}");
        assert!(with_audio.contains(&"C:/m/a b.mp4".to_string()), "the path is one argument");

        // I420 from the page: tagged on the way in, padded size, no conversion filter at all.
        let yuv = ffmpeg_args(961, 543, 25.0, 16, true, PixelFormat::Yuv420, None, Path::new("out.mp4")).join(" ");
        assert!(yuv.contains("-f rawvideo -pix_fmt yuv420p -color_range tv -colorspace bt709 -color_primaries bt709 -color_trc bt709 -s 962x544 -r 25 -i -"), "{yuv}");
        assert!(!yuv.contains("-vf") && !yuv.contains("scale="), "{yuv}");
        assert!(yuv.contains("-c:v h264_nvenc") && yuv.ends_with("-color_trc bt709 -movflags +faststart out.mp4"), "{yuv}");
        assert_eq!(PixelFormat::Yuv420.frame_bytes(961, 543), 962 * 544 * 3 / 2);
        assert_eq!(PixelFormat::Rgba.frame_bytes(961, 543), 961 * 543 * 4);
        assert_eq!(PixelFormat::parse(None), Ok(PixelFormat::Rgba));
        assert_eq!(PixelFormat::parse(Some("yuv420")), Ok(PixelFormat::Yuv420));
        assert!(PixelFormat::parse(Some("nv12")).is_err());
    }

    #[test]
    fn names_count_up_inside_the_project() {
        let root = std::env::temp_dir().join(format!("aiv-render-{}", std::process::id()));
        let project = "3f1c2a8e-1b2c-4d5e-8f90-123456789abc";
        let first = output_path(&root, project, "Captain: smoke shot!").unwrap();
        assert!(first.ends_with("media/videos/captain-smoke-shot-render.mp4"), "{}", first.display());
        std::fs::write(&first, b"x").unwrap();
        let second = output_path(&root, project, "Captain: smoke shot!").unwrap();
        assert!(second.to_string_lossy().ends_with("captain-smoke-shot-render-02.mp4"));
        assert!(output_path(&root, "not-a-uuid", "x").is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A plain test driving its own runtime: the crate's tokio has no `macros` feature for `#[tokio::test]`.
    #[test]
    fn encodes_frames_when_ffmpeg_is_present() {
        if std::process::Command::new("ffmpeg").arg("-version").output().is_err() { return; }
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(encode_three_frames());
    }

    async fn encode_three_frames() {
        let root = std::env::temp_dir().join(format!("aiv-render-e2e-{}", std::process::id()));
        let sessions = RenderSessions::default();
        let project = "3f1c2a8e-1b2c-4d5e-8f90-123456789abc";
        let started = start(&sessions, &root, project, "e2e", 17, 9, 24.0, None, PixelFormat::Rgba, false).await.unwrap();
        assert!(frame(&sessions, &started.id, &[0u8; 10]).await.is_err());
        for value in [0u8, 128, 255] {
            frame(&sessions, &started.id, &vec![value; 17 * 9 * 4]).await.unwrap();
        }
        let done = finish(&sessions, &started.id).await.unwrap();
        assert_eq!(done.frames, 3);
        assert!(std::fs::metadata(&done.output_path).unwrap().len() > 0);

        // With the plate's audio: a one-second tone, of which half a second is muxed in.
        let tone = root.join("tone.m4a");
        let made = std::process::Command::new("ffmpeg").args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1"])
            .arg(&tone).status().unwrap();
        assert!(made.success());
        let span = AudioSpan { path: tone.to_string_lossy().into_owned(), start: 0.25, duration: 0.5 };
        let started = start(&sessions, &root, project, "e2e audio", 16, 8, 24.0, Some(span), PixelFormat::Rgba, false).await.unwrap();
        for _ in 0..12 { frame(&sessions, &started.id, &vec![90u8; 16 * 8 * 4]).await.unwrap(); }
        let done = finish(&sessions, &started.id).await.unwrap();
        let probe = std::process::Command::new("ffprobe").args(["-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0"])
            .arg(&done.output_path).output().unwrap();
        let kinds = String::from_utf8_lossy(&probe.stdout);
        assert!(kinds.contains("video") && kinds.contains("audio"), "{kinds}");

        // On a machine with NVENC (an RTX 4090, say; not a GPU-less CI container), the smallest size it is used for.
        if nvenc_works().await {
            let started = start(&sessions, &root, project, "e2e nvenc", 256, 144, 24.0, None, PixelFormat::Rgba, false).await.unwrap();
            for value in [30u8, 120, 210, 90, 60, 180] { frame(&sessions, &started.id, &vec![value; 256 * 144 * 4]).await.unwrap(); }
            let done = finish(&sessions, &started.id).await.unwrap();
            assert_eq!((done.frames, done.encoder.as_str()), (6, "h264_nvenc"));
            assert!(std::fs::metadata(&done.output_path).unwrap().len() > 0);
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// I420 frames over the localhost channel (the Phase 6 path), end to end: the page's bytes reach
    /// the encoder untouched, the file is tagged BT.709 limited, and the channel closes with the render.
    #[test]
    fn yuv_frames_over_http_when_ffmpeg_is_present() {
        if std::process::Command::new("ffmpeg").arg("-version").output().is_err() { return; }
        tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap().block_on(yuv_over_http());
    }

    async fn yuv_over_http() {
        use std::io::{BufRead as _, Read as _, Write as _};
        let root = std::env::temp_dir().join(format!("aiv-render-yuv-{}", std::process::id()));
        let sessions = RenderSessions::default();
        let project = "3f1c2a8e-1b2c-4d5e-8f90-123456789abc";
        // 33x17 -> the stream is 34x18.
        let started = start(&sessions, &root, project, "e2e yuv", 33, 17, 24.0, None, PixelFormat::Yuv420, true).await.unwrap();
        assert_eq!((started.pixel_format, started.frame_bytes), (PixelFormat::Yuv420, 34 * 18 * 3 / 2));
        let endpoint = started.http.clone().expect("the channel listens");
        let planes = |y: u8, u: u8, v: u8| { let mut f = vec![y; 34 * 18]; f.extend(vec![u; 17 * 9]); f.extend(vec![v; 17 * 9]); f };
        let frames = vec![planes(60, 90, 170), planes(200, 128, 128), planes(120, 160, 100)];
        let sent = frames.clone();
        let (url, token) = (endpoint.url.clone(), endpoint.token.clone());
        // The client blocks, so it runs off the runtime thread; this thread keeps driving ffmpeg's pipe.
        let statuses = tokio::task::spawn_blocking(move || {
            let address = url.trim_start_matches("http://").trim_end_matches("/frame").to_owned();
            let mut client = std::net::TcpStream::connect(address).unwrap();
            let mut reader = std::io::BufReader::new(client.try_clone().unwrap());
            let mut statuses = Vec::new();
            for frame in &sent {
                let mut request = format!("POST /frame HTTP/1.1\r\nHost: 127.0.0.1\r\nx-render-token: {token}\r\nContent-Length: {}\r\n\r\n", frame.len()).into_bytes();
                request.extend_from_slice(frame);
                client.write_all(&request).unwrap();
                let mut status = String::new();
                reader.read_line(&mut status).unwrap();
                let mut length = 0usize;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line.trim_end().is_empty() { break; }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") { length = value.trim().parse().unwrap(); }
                }
                let mut body = vec![0u8; length];
                reader.read_exact(&mut body).unwrap();
                statuses.push(status.trim_end().to_owned());
            }
            statuses
        }).await.unwrap();
        assert_eq!(statuses, vec!["HTTP/1.1 204 No Content"; 3]);
        // The IPC path still takes frames for the same render (the page's fallback).
        assert_eq!(frame(&sessions, &started.id, &planes(90, 110, 140)).await.unwrap(), 4);
        let done = finish(&sessions, &started.id).await.unwrap();
        assert_eq!(done.frames, 4);
        let address = endpoint.url.trim_start_matches("http://").trim_end_matches("/frame").to_owned();
        assert!(std::net::TcpStream::connect(address).is_err(), "the channel closed with the render");

        let probe = std::process::Command::new("ffprobe").args(["-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,pix_fmt,color_range,color_space", "-of", "csv=p=0"]).arg(&done.output_path).output().unwrap();
        assert_eq!(String::from_utf8_lossy(&probe.stdout).trim(), "34,18,yuv420p,tv,bt709");
        let decoded = std::process::Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(&done.output_path)
            .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"]).output().unwrap().stdout;
        let size = 34 * 18 * 3 / 2;
        assert_eq!(decoded.len(), size * 4);
        let expected = [(60u8, 90u8, 170u8), (200, 128, 128), (120, 160, 100), (90, 110, 140)];
        for (index, (y, u, v)) in expected.iter().enumerate() {
            let frame = &decoded[index * size..(index + 1) * size];
            let mean = |range: std::ops::Range<usize>| frame[range.clone()].iter().map(|&b| b as f64).sum::<f64>() / range.len() as f64;
            let (my, mu, mv) = (mean(0..34 * 18), mean(34 * 18..34 * 18 + 153), mean(34 * 18 + 153..size));
            assert!((my - *y as f64).abs() < 1.0 && (mu - *u as f64).abs() < 1.0 && (mv - *v as f64).abs() < 1.0,
                "frame {index}: {my:.1} {mu:.1} {mv:.1}, sent {y} {u} {v}");
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
