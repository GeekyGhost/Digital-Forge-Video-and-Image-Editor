//! VFX Phase 0 spikes (tasks 0.2-0.5), Tauri side. Throwaway by design.
//!
//! Included from `src-tauri/src/lib.rs` with `#[path = "../../spikes/rust/spikes.rs"] mod spikes;`.
//! To remove the spikes: delete `spikes/`, that `mod` line, the `spikes::` entries in
//! `generate_handler!`, the `.manage(spikes::Sessions::default())` and `.setup(...)` lines, the
//! `spikes` input in `vite.config.ts`, and "spikes" in `capabilities/default.json`.
//!
//! The spike window only opens when the app starts with `AIV_SPIKES=1` (Run-VFX-Spikes.bat).
//! Everything it writes goes to `scratch/vfx-spikes/` inside the project.

use serde::Serialize;
use serde_json::Value;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    process::Stdio,
    sync::atomic::{AtomicU8, Ordering},
    time::Instant,
};
use tauri::{ipc::{InvokeBody, Request, Response}, State};
use tokio::{
    io::AsyncWriteExt,
    process::{Child, ChildStdin, Command},
    sync::Mutex,
};

const RECIPE: &str = include_str!("../media-recipe.json");

fn root() -> PathBuf {
    crate::ai::project_root().join("scratch").join("vfx-spikes")
}

fn media_dir() -> PathBuf {
    root().join("media")
}

/// Opens the spike window when the app was started with AIV_SPIKES=1.
pub fn open_if_requested(app: &tauri::App) {
    if std::env::var("AIV_SPIKES").ok().as_deref() != Some("1") {
        return;
    }
    // WebView2 on a hybrid laptop renders on the integrated GPU by default (the first run measured
    // "Intel(R) UHD Graphics" for both WebGL2 and WebGPU). Force the discrete GPU for this window.
    // Browser args are fixed per WebView2 environment, so the spike window gets its own data
    // directory (inside the project); Tauri's default args are repeated because this replaces them.
    let built = tauri::WebviewWindowBuilder::new(app, "spikes", tauri::WebviewUrl::App("spikes/index.html".into()))
        .title("VFX Phase 0 spikes")
        .inner_size(1280.0, 900.0)
        .data_directory(root().join("webview2-data"))
        .additional_browser_args("--disable-features=msWebOOUI,msPdfOOUI,msSmartScreenProtection --force_high_performance_gpu")
        .build();
    if let Err(error) = built {
        eprintln!("could not open the spike window: {error}");
    }
}

fn command(program: &str) -> Command {
    let mut command = Command::new(program);
    #[cfg(windows)]
    command.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    command.stdin(Stdio::null());
    command
}

async fn ffmpeg(args: &[String]) -> Result<Vec<u8>, String> {
    let output = command("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args(args)
        .output()
        .await
        .map_err(|error| format!("could not start ffmpeg (is it on PATH?): {error}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let tail: String = stderr.chars().rev().take(800).collect::<Vec<_>>().into_iter().rev().collect();
        return Err(format!("ffmpeg failed ({}): {tail}", output.status));
    }
    Ok(output.stdout)
}

fn recipe() -> Value {
    serde_json::from_str(RECIPE).expect("spikes/media-recipe.json is valid JSON")
}

fn strings(value: &Value) -> Vec<String> {
    value.as_array().map(|items| items.iter().filter_map(|item| item.as_str().map(str::to_owned)).collect()).unwrap_or_default()
}

fn fill(args: &[String], values: &[(&str, String)]) -> Vec<String> {
    args.iter().map(|arg| {
        let mut out = arg.clone();
        for (key, value) in values {
            out = out.replace(&format!("{{{key}}}"), value);
        }
        out
    }).collect()
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ClipInfo {
    id: String,
    path: String,
    codec: String,
    width: u64,
    height: u64,
    frames: u64,
    fps: u64,
    index_strip: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AlphaMedia {
    width: u64,
    height: u64,
    frames: u64,
    fps: u64,
    png_paths: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stacked_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vp9_alpha_path: Option<String>,
    errors: Vec<String>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MediaSet {
    directory: String,
    clips: Vec<ClipInfo>,
    alpha: AlphaMedia,
    ffmpeg_version: String,
}

/// Generates the test media once (later calls reuse the files).
#[tauri::command]
pub async fn spike_prepare_media() -> Result<MediaSet, String> {
    let recipe = recipe();
    let dir = media_dir();
    std::fs::create_dir_all(&dir).map_err(|error| format!("could not create {}: {error}", dir.display()))?;
    let mut clips = Vec::new();
    for clip in recipe["videoClips"].as_array().cloned().unwrap_or_default() {
        let get = |key: &str| clip[key].as_u64().unwrap_or(0);
        let codec = clip["codec"].as_str().unwrap_or("h264").to_owned();
        let out = dir.join(clip["file"].as_str().unwrap_or("clip.mp4"));
        let mut info = ClipInfo {
            id: clip["id"].as_str().unwrap_or("?").to_owned(),
            path: path_string(&out),
            codec: codec.clone(),
            width: get("width"),
            height: get("height"),
            frames: get("frames"),
            fps: get("fps"),
            index_strip: true,
            error: None,
        };
        if !out.is_file() {
            let values = [("W", info.width.to_string()), ("H", info.height.to_string()), ("N", info.frames.to_string()), ("FPS", info.fps.to_string())];
            let mut args = fill(&strings(&recipe["videoInputArgs"]), &values);
            args.extend(strings(&recipe["encoderArgs"][codec.as_str()]));
            if clip["tagged"].as_bool().unwrap_or(true) {
                args.extend(strings(&recipe["tagArgs"]));
            }
            args.extend(strings(&recipe["containerArgs"]));
            args.push(path_string(&out));
            if let Err(error) = ffmpeg(&args).await {
                info.error = Some(error);
            }
        }
        clips.push(info);
    }

    let alpha = &recipe["alpha"];
    let get = |key: &str| alpha[key].as_u64().unwrap_or(0);
    let png_dir = dir.join("alpha-png");
    std::fs::create_dir_all(&png_dir).map_err(|error| error.to_string())?;
    let (width, height, frames, fps) = (get("width"), get("height"), get("frames"), get("fps"));
    // ffmpeg on Windows accepts forward slashes, which keeps the %04d pattern unambiguous.
    let dir_arg = path_string(&png_dir).replace('\\', "/");
    let values = [("W", width.to_string()), ("H", height.to_string()), ("N", frames.to_string()), ("FPS", fps.to_string()), ("DIR", dir_arg)];
    let png_paths: Vec<PathBuf> = (0..frames).map(|i| png_dir.join(format!("frame_{i:04}.png"))).collect();
    let mut errors = Vec::new();
    if !png_paths.last().map(|path| path.is_file()).unwrap_or(false) {
        if let Err(error) = ffmpeg(&fill(&strings(&alpha["pngArgs"]), &values)).await {
            errors.push(format!("png: {error}"));
        }
    }
    let stacked = dir.join("alpha-stacked.mp4");
    let vp9 = dir.join("alpha-vp9.webm");
    for (label, key, out) in [("stacked", "stackedArgs", &stacked), ("vp9", "vp9Args", &vp9)] {
        if out.is_file() {
            continue;
        }
        let mut with_out = values.to_vec();
        with_out.push(("OUT", path_string(out)));
        if let Err(error) = ffmpeg(&fill(&strings(&alpha[key]), &with_out)).await {
            errors.push(format!("{label}: {error}"));
        }
    }
    let version = command("ffmpeg").arg("-version").output().await
        .map(|output| String::from_utf8_lossy(&output.stdout).lines().next().unwrap_or("").to_owned())
        .unwrap_or_else(|error| format!("ffmpeg not found: {error}"));
    Ok(MediaSet {
        directory: path_string(&dir),
        clips,
        alpha: AlphaMedia {
            width, height, frames, fps,
            png_paths: png_paths.iter().map(|path| path_string(path)).collect(),
            stacked_path: stacked.is_file().then(|| path_string(&stacked)),
            vp9_alpha_path: vp9.is_file().then(|| path_string(&vp9)),
            errors,
        },
        ffmpeg_version: version,
    })
}

/// Only files under scratch/vfx-spikes can be read: the page never needs anything else.
fn checked(path: &str) -> Result<PathBuf, String> {
    let wanted = std::fs::canonicalize(path).map_err(|error| format!("{path}: {error}"))?;
    let allowed = std::fs::canonicalize(root()).map_err(|error| error.to_string())?;
    if !wanted.starts_with(&allowed) {
        return Err(format!("{path} is outside {}", allowed.display()));
    }
    Ok(wanted)
}

#[tauri::command]
pub async fn spike_read_file(path: String) -> Result<Response, String> {
    let path = checked(&path)?;
    let bytes = tokio::fs::read(&path).await.map_err(|error| error.to_string())?;
    Ok(Response::new(bytes))
}

/// Frame `index` as RGBA via ffmpeg with an explicit BT.709 matrix (images: plain RGBA).
#[tauri::command]
pub async fn spike_reference_frame(path: String, index: u32) -> Result<Response, String> {
    let path = checked(&path)?;
    let recipe = recipe();
    let image = matches!(path.extension().and_then(|value| value.to_str()).map(str::to_ascii_lowercase).as_deref(), Some("png" | "jpg" | "jpeg"));
    let key = if image { "referenceImageArgs" } else { "referenceVideoArgs" };
    let args = fill(&strings(&recipe[key]), &[("IN", path_string(&path)), ("INDEX", index.to_string())]);
    Ok(Response::new(ffmpeg(&args).await?))
}

// ---- readback -> ffmpeg

pub struct Session {
    stdin: Option<ChildStdin>,
    child: Option<Child>,
    frames: u64,
    bytes: u64,
    write_seconds: f64,
    output_path: PathBuf,
    stderr_path: PathBuf,
}

#[derive(Default)]
pub struct Sessions(Mutex<HashMap<String, Session>>);

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EncodeSession {
    id: String,
    encoder: String,
    output_path: String,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EncodeStats {
    frames: u64,
    bytes: u64,
    write_seconds: f64,
    drain_seconds: f64,
    output_bytes: u64,
    output_frames: Option<u64>,
    output_pix_fmt: Option<String>,
    stderr_tail: String,
}

/// 0 unknown, 1 works, 2 does not.
static NVENC: AtomicU8 = AtomicU8::new(0);

async fn nvenc_works() -> bool {
    match NVENC.load(Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    let args: Vec<String> = ["-f", "lavfi", "-i", "color=s=256x256", "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"]
        .iter().map(|arg| arg.to_string()).collect();
    let works = ffmpeg(&args).await.is_ok();
    NVENC.store(if works { 1 } else { 2 }, Ordering::Relaxed);
    works
}

#[tauri::command]
pub async fn spike_encode_start(width: u32, height: u32, fps: u32, encoder: String, pix_fmt: String, sessions: State<'_, Sessions>) -> Result<EncodeSession, String> {
    if pix_fmt != "rgba" && pix_fmt != "yuv420p" {
        return Err(format!("unsupported pixel format {pix_fmt}"));
    }
    let chosen = match encoder.as_str() {
        "null" => "null".to_owned(),
        "h264_nvenc" => if nvenc_works().await { "h264_nvenc".to_owned() } else { "libx264".to_owned() },
        "libx264" => "libx264".to_owned(),
        other => return Err(format!("unknown encoder {other}")),
    };
    let dir = root();
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let id = uuid::Uuid::new_v4().simple().to_string();
    let output_path = dir.join(format!("readback-{width}x{height}-{}.mp4", &id[..8]));
    let stderr_path = dir.join(format!("readback-{}.ffmpeg.log", &id[..8]));
    let mut session = Session { stdin: None, child: None, frames: 0, bytes: 0, write_seconds: 0.0, output_path: output_path.clone(), stderr_path: stderr_path.clone() };
    if chosen != "null" {
        let log = std::fs::File::create(&stderr_path).map_err(|error| error.to_string())?;
        let mut args: Vec<String> = ["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt"]
            .iter().map(|arg| arg.to_string()).collect();
        args.push(pix_fmt.clone());
        if pix_fmt == "yuv420p" {
            // The page already converted with BT.709 limited range; say so, so nothing converts again
            // and the output is tagged to match.
            args.extend(["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"].iter().map(|arg| arg.to_string()));
        }
        args.extend(["-s".into(), format!("{width}x{height}"), "-r".into(), fps.to_string(), "-i".into(), "-".into()]);
        if pix_fmt == "yuv420p" {
            args.extend(["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"].iter().map(|arg| arg.to_string()));
        }
        if chosen == "h264_nvenc" {
            // RGBA goes straight to NVENC, which converts on the GPU (no CPU swscale).
            args.extend(["-c:v", "h264_nvenc", "-preset", "p4", "-g", "48", "-bf", "0"].iter().map(|arg| arg.to_string()));
        } else {
            args.extend(["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "48", "-bf", "0"].iter().map(|arg| arg.to_string()));
        }
        args.push(path_string(&output_path));
        let mut child = command("ffmpeg")
            .args(&args)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::from(log))
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| format!("could not start ffmpeg: {error}"))?;
        session.stdin = child.stdin.take();
        session.child = Some(child);
    }
    sessions.0.lock().await.insert(id.clone(), session);
    Ok(EncodeSession { id, encoder: chosen, output_path: path_string(&output_path) })
}

/// The frame arrives as the raw request body (a Uint8Array argument on the JS side).
#[tauri::command]
pub async fn spike_encode_frame(request: Request<'_>, sessions: State<'_, Sessions>) -> Result<(), String> {
    let InvokeBody::Raw(bytes) = request.body() else {
        return Err("expected a raw binary body".into());
    };
    let id = request.headers().get("x-spike-session").and_then(|value| value.to_str().ok()).ok_or("missing x-spike-session header")?.to_owned();
    write_frame(&sessions, &id, bytes).await
}

async fn write_frame(sessions: &Sessions, id: &str, bytes: &[u8]) -> Result<(), String> {
    let mut map = sessions.0.lock().await;
    let session = map.get_mut(id).ok_or("no such encode session")?;
    let started = Instant::now();
    if let Some(stdin) = session.stdin.as_mut() {
        stdin.write_all(bytes).await.map_err(|error| format!("ffmpeg stdin: {error}"))?;
    }
    session.write_seconds += started.elapsed().as_secs_f64();
    session.frames += 1;
    session.bytes += bytes.len() as u64;
    Ok(())
}

// ---- round 2: a localhost HTTP channel for frames, to compare with the Tauri IPC raw body
//
// Round 1 measured the IPC raw body at ~100 MB/s. This is a deliberately minimal HTTP/1.1
// listener on 127.0.0.1 (random port, keep-alive, CORS + Private Network Access preflight). It
// only accepts `POST /frame` carrying the per-launch token and an existing session id, and all it
// can do is write into that session's ffmpeg stdin. It exists only while the spike window runs.

#[derive(Clone, Serialize)]
pub struct HttpEndpoint {
    url: String,
    token: String,
}

static HTTP_ENDPOINT: std::sync::OnceLock<HttpEndpoint> = std::sync::OnceLock::new();

#[tauri::command]
pub fn spike_http_start(app: tauri::AppHandle) -> Result<HttpEndpoint, String> {
    if let Some(endpoint) = HTTP_ENDPOINT.get() {
        return Ok(endpoint.clone());
    }
    let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|error| format!("could not listen on localhost: {error}"))?;
    let port = listener.local_addr().map_err(|error| error.to_string())?.port();
    let token = uuid::Uuid::new_v4().simple().to_string();
    let endpoint = HttpEndpoint { url: format!("http://127.0.0.1:{port}/frame"), token: token.clone() };
    std::thread::Builder::new().name("spike-http".into()).spawn(move || {
        for stream in listener.incoming().flatten() {
            let app = app.clone();
            let token = token.clone();
            let _ = std::thread::Builder::new().name("spike-http-conn".into()).spawn(move || {
                use tauri::Manager as _;
                serve_connection(stream, &token, |session, bytes| {
                    let sessions = app.state::<Sessions>();
                    tauri::async_runtime::block_on(write_frame(&sessions, session, bytes))
                })
            });
        }
    }).map_err(|error| error.to_string())?;
    Ok(HTTP_ENDPOINT.get_or_init(|| endpoint).clone())
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Private-Network: true\r\n";

fn respond(stream: &mut std::net::TcpStream, status: &str, extra: &str, body: &str) -> std::io::Result<()> {
    use std::io::Write as _;
    write!(stream, "HTTP/1.1 {status}\r\n{CORS}{extra}Content-Length: {}\r\nConnection: keep-alive\r\n\r\n{body}", body.len())
}

/// One keep-alive connection. `write` receives (session id, frame bytes) for each accepted frame.
fn serve_connection(stream: std::net::TcpStream, token: &str, write: impl Fn(&str, &[u8]) -> Result<(), String>) {
    use std::io::{BufRead as _, Read as _};
    let _ = stream.set_nodelay(true);
    let Ok(mut writer) = stream.try_clone() else { return };
    let mut reader = std::io::BufReader::with_capacity(1 << 20, stream);
    let mut body = Vec::new();
    loop {
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let mut parts = line.split_whitespace();
        let method = parts.next().unwrap_or("").to_owned();
        let path = parts.next().unwrap_or("").to_owned();
        let (mut length, mut session, mut given_token) = (0usize, String::new(), String::new());
        loop {
            let mut header = String::new();
            if reader.read_line(&mut header).unwrap_or(0) == 0 {
                return;
            }
            let header = header.trim_end();
            if header.is_empty() {
                break;
            }
            if let Some((name, value)) = header.split_once(':') {
                let value = value.trim();
                match name.trim().to_ascii_lowercase().as_str() {
                    "content-length" => length = value.parse().unwrap_or(0),
                    "x-spike-session" => session = value.to_owned(),
                    "x-spike-token" => given_token = value.to_owned(),
                    _ => {}
                }
            }
        }
        if method == "OPTIONS" {
            let extra = "Access-Control-Allow-Methods: POST\r\nAccess-Control-Allow-Headers: content-type, x-spike-session, x-spike-token\r\nAccess-Control-Max-Age: 600\r\n";
            if respond(&mut writer, "204 No Content", extra, "").is_err() { return; }
            continue;
        }
        body.resize(length, 0);
        if reader.read_exact(&mut body).is_err() {
            return;
        }
        let result = if method != "POST" || path != "/frame" {
            Err("404 Not Found".to_owned())
        } else if given_token != token {
            Err("403 Forbidden".to_owned())
        } else {
            write(&session, &body).map_err(|error| format!("500 {error}"))
        };
        let sent = match result {
            Ok(()) => respond(&mut writer, "204 No Content", "", ""),
            Err(error) => {
                let (status, message) = error.split_at(3);
                respond(&mut writer, &format!("{status} Error"), "Content-Type: text/plain\r\n", message.trim())
            }
        };
        if sent.is_err() {
            return;
        }
    }
}

#[tauri::command]
pub async fn spike_encode_finish(id: String, sessions: State<'_, Sessions>) -> Result<EncodeStats, String> {
    let mut session = sessions.0.lock().await.remove(&id).ok_or("no such encode session")?;
    let started = Instant::now();
    drop(session.stdin.take());
    let had_child = session.child.is_some();
    if let Some(mut child) = session.child.take() {
        child.wait().await.map_err(|error| error.to_string())?;
    }
    let drain_seconds = started.elapsed().as_secs_f64();
    let output_bytes = if had_child { std::fs::metadata(&session.output_path).map(|meta| meta.len()).unwrap_or(0) } else { 0 };
    let (mut output_frames, mut output_pix_fmt) = (None, None);
    if output_bytes > 0 {
        let probe = command("ffprobe")
            .args(["-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames,pix_fmt", "-of", "csv=p=0"])
            .arg(&session.output_path)
            .output().await;
        if let Ok(probe) = probe {
            let text = String::from_utf8_lossy(&probe.stdout).trim().to_owned();
            let mut parts = text.split(',');
            output_pix_fmt = parts.next().map(str::to_owned);
            output_frames = parts.next().and_then(|value| value.trim().parse().ok());
        }
    }
    let stderr_tail = std::fs::read_to_string(&session.stderr_path).unwrap_or_default();
    Ok(EncodeStats {
        frames: session.frames,
        bytes: session.bytes,
        write_seconds: session.write_seconds,
        drain_seconds,
        output_bytes,
        output_frames,
        output_pix_fmt,
        stderr_tail: stderr_tail.chars().rev().take(1200).collect::<Vec<_>>().into_iter().rev().collect(),
    })
}

// ---- CPU and memory of this app and everything it started (WebView2, ffmpeg)

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppUsage {
    private_bytes: u64,
    cpu_seconds: f64,
    processes: usize,
}

#[tauri::command]
pub async fn spike_app_usage() -> Option<AppUsage> {
    tokio::task::spawn_blocking(|| {
        let snapshot = crate::ai::resource_guard::measure();
        if !snapshot.measured {
            return None;
        }
        let owned = crate::ai::resource_guard::descendants(&snapshot.processes, snapshot.self_pid);
        let tree: Vec<_> = snapshot.processes.iter().filter(|p| p.pid == snapshot.self_pid || owned.contains(&p.pid)).collect();
        Some(AppUsage {
            private_bytes: tree.iter().map(|p| p.private_bytes).sum(),
            cpu_seconds: tree.iter().map(|p| cpu::seconds(p.pid)).sum(),
            processes: tree.len(),
        })
    }).await.ok().flatten()
}

#[cfg(any(windows, aiv_check_windows))]
mod cpu {
    #[repr(C)]
    #[derive(Default)]
    struct FileTime { low: u32, high: u32 }

    #[link(name = "kernel32")]
    extern "system" {
        fn OpenProcess(access: u32, inherit: i32, pid: u32) -> *mut std::ffi::c_void;
        fn CloseHandle(handle: *mut std::ffi::c_void) -> i32;
        fn GetProcessTimes(handle: *mut std::ffi::c_void, creation: *mut FileTime, exit: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
    }

    /// Kernel + user CPU time of `pid` in seconds (0 if it cannot be opened).
    pub fn seconds(pid: u32) -> f64 {
        let handle = unsafe { OpenProcess(0x1000, 0, pid) };
        if handle.is_null() {
            return 0.0;
        }
        let (mut a, mut b, mut kernel, mut user) = (FileTime::default(), FileTime::default(), FileTime::default(), FileTime::default());
        let ok = unsafe { GetProcessTimes(handle, &mut a, &mut b, &mut kernel, &mut user) } != 0;
        unsafe { CloseHandle(handle) };
        if !ok {
            return 0.0;
        }
        let ticks = |t: &FileTime| ((t.high as u64) << 32 | t.low as u64) as f64 / 1e7;
        ticks(&kernel) + ticks(&user)
    }
}

#[cfg(not(any(windows, aiv_check_windows)))]
mod cpu {
    pub fn seconds(_pid: u32) -> f64 { 0.0 }
}

#[tauri::command]
pub async fn spike_save_results(json: String) -> Result<String, String> {
    let dir = root();
    std::fs::create_dir_all(&dir).map_err(|error| error.to_string())?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|value| value.as_secs()).unwrap_or(0);
    let path = dir.join(format!("results-{stamp}.json"));
    std::fs::write(&path, json).map_err(|error| error.to_string())?;
    Ok(path_string(&path))
}

// ---- 0.6 Wan Alpha, through the resource guard

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WanAlphaRequest {
    label: String,
    prompt: String,
    frames: u32,
    seed: u64,
}

/// One Wan-Alpha generation (bake if needed -> encode -> denoise -> decode, each its own process),
/// gated as `JobKind::WanAlpha` so the resource guard checks it and the job monitor logs its real
/// peak commit to `.aiv/resource-log.jsonl`. Returns the worker's stats.json plus where it wrote.
#[tauri::command]
pub async fn spike_wan_alpha(
    request: WanAlphaRequest,
    gate: State<'_, crate::ai::AiExecutionGate>,
    app: tauri::AppHandle,
) -> Result<Value, String> {
    // VFX Phase 3: the generation itself lives in src-tauri/src/ai/wan_alpha.rs; the spike only
    // chooses where it writes.
    crate::ai::wan_alpha::validate_frames(request.frames)?;
    let project = crate::ai::project_root();
    if !crate::ai::wan_alpha::python(&project).is_file() {
        return Err("Wan-Alpha is not set up. Run Setup-Wan-Alpha.bat.".into());
    }
    let _permit = gate.acquire_for(crate::ai::JobKind::WanAlpha, &app).await?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|value| value.as_secs()).unwrap_or(0);
    let safe_label: String = request.label.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    let out = project.join("scratch").join("wan-alpha").join(format!("{safe_label}-{}f-{stamp}", request.frames));
    let mut stats = crate::ai::wan_alpha::generate(&project, &request.prompt, request.frames, request.seed, &out).await?;
    stats["label"] = Value::String(request.label);
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_parses_and_every_clip_has_an_encoder() {
        let recipe = recipe();
        let clips = recipe["videoClips"].as_array().unwrap();
        assert!(clips.len() >= 5);
        for clip in clips {
            let codec = clip["codec"].as_str().unwrap();
            assert!(!strings(&recipe["encoderArgs"][codec]).is_empty(), "{codec}");
        }
        for key in ["pngArgs", "stackedArgs", "vp9Args"] {
            assert!(!strings(&recipe["alpha"][key]).is_empty(), "{key}");
        }
    }

    /// One request, then its full response (status line, headers, Content-Length body), the way a
    /// browser uses a keep-alive connection. (A first version pipelined every request and
    /// half-closed the socket; Windows answered that teardown with a connection reset.)
    fn exchange(stream: &mut std::net::TcpStream, reader: &mut std::io::BufReader<std::net::TcpStream>, request: &[u8]) -> (String, String) {
        use std::io::{BufRead, Read, Write};
        stream.write_all(request).unwrap();
        let mut status = String::new();
        reader.read_line(&mut status).unwrap();
        let mut headers = String::new();
        let mut length = 0usize;
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line == "\r\n" {
                break;
            }
            if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                length = value.trim().parse().unwrap();
            }
            headers.push_str(&line);
        }
        let mut body = vec![0u8; length];
        reader.read_exact(&mut body).unwrap();
        (status.trim_end().to_owned(), headers)
    }

    #[test]
    fn http_channel_handles_preflight_keep_alive_and_bad_tokens() {
        use std::sync::{Arc, Mutex};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let received: Arc<Mutex<Vec<(String, Vec<u8>)>>> = Arc::default();
        let sink = received.clone();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            serve_connection(stream, "secret", |session, bytes| {
                sink.lock().unwrap().push((session.to_owned(), bytes.to_vec()));
                Ok(())
            });
        });
        let mut client = std::net::TcpStream::connect(address).unwrap();
        let mut reader = std::io::BufReader::new(client.try_clone().unwrap());
        let (status, headers) = exchange(&mut client, &mut reader,
            b"OPTIONS /frame HTTP/1.1\r\nHost: x\r\nAccess-Control-Request-Private-Network: true\r\n\r\n");
        assert_eq!(status, "HTTP/1.1 204 No Content");
        assert!(headers.contains("Access-Control-Allow-Private-Network: true"));
        assert!(headers.contains("Access-Control-Allow-Headers: content-type, x-spike-session, x-spike-token"));
        let frame = vec![7u8; 100_000];
        let mut statuses = Vec::new();
        for (token, session) in [("secret", "a"), ("wrong", "b"), ("secret", "c")] {
            let mut request = format!("POST /frame HTTP/1.1\r\nHost: x\r\nX-Spike-Token: {token}\r\nx-spike-session: {session}\r\nContent-Length: {}\r\n\r\n", frame.len()).into_bytes();
            request.extend_from_slice(&frame);
            statuses.push(exchange(&mut client, &mut reader, &request).0);
        }
        assert_eq!(statuses, vec!["HTTP/1.1 204 No Content", "HTTP/1.1 403 Error", "HTTP/1.1 204 No Content"]);
        drop(reader);
        drop(client);
        server.join().unwrap();
        let got = received.lock().unwrap();
        assert_eq!(got.iter().map(|(s, b)| (s.as_str(), b.len())).collect::<Vec<_>>(), vec![("a", 100_000), ("c", 100_000)]);
        assert!(got.iter().all(|(_, b)| b.iter().all(|&v| v == 7)));
    }

    #[test]
    fn fill_replaces_every_placeholder() {
        let args = fill(&["{W}x{H}".into(), "select=eq(n\\,{INDEX})".into()], &[("W", "3".into()), ("H", "4".into()), ("INDEX", "7".into())]);
        assert_eq!(args, vec!["3x4".to_owned(), "select=eq(n\\,7)".to_owned()]);
    }
}
