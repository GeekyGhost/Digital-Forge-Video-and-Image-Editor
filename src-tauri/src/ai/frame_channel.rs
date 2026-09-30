//! VFX Phase 6: a localhost HTTP channel for render frames.
//!
//! Phase 5 measured the Tauri IPC raw body at ~12 MB/s in the main window (~169 ms per 2 MB frame),
//! which made it the render's limit. Phase 0 measured a plain HTTP/1.1 POST to 127.0.0.1 at many
//! times that, so a render now opens one of these beside its ffmpeg session:
//! - bound to 127.0.0.1 on a random port, so nothing off the machine can reach it;
//! - one token per render (a random UUID), checked on every request;
//! - `POST /frame` only, the body at most the frame size; anything else is refused;
//! - keep-alive, and the CORS + Private Network Access preflight a browser sends to localhost;
//! - it lives exactly as long as its render: `stop` (or dropping it) closes the listener and every
//!   open connection. The Phase 0 spike's listener lived until the app closed.
//!
//! Plain threads, not tokio: the crate's tokio has no `net` feature, and a render has one sender.

use std::{
    io::{BufRead as _, Read as _, Write as _},
    net::{Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{atomic::{AtomicBool, Ordering}, Arc, Mutex},
};

/// What the page needs to send frames: `url` and the `x-render-token` value.
#[derive(Debug, Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Endpoint {
    pub url: String,
    pub token: String,
}

/// Receives one frame's bytes; `Ok(n)` is the frames written so far, `Err` the reason (sent as 500).
pub type Sink = Arc<dyn Fn(&[u8]) -> Result<u32, String> + Send + Sync>;

pub struct FrameChannel {
    pub endpoint: Endpoint,
    address: SocketAddr,
    stopped: Arc<AtomicBool>,
    connections: Arc<Mutex<Vec<TcpStream>>>,
}

const CORS: &str = "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Private-Network: true\r\nAccess-Control-Expose-Headers: x-frames\r\n";

impl FrameChannel {
    /// Listens on 127.0.0.1 (random port). `frame_bytes` is the exact size every frame must have.
    pub fn start(frame_bytes: usize, sink: Sink) -> Result<Self, String> {
        let listener = TcpListener::bind("127.0.0.1:0").map_err(|error| format!("could not listen on localhost: {error}"))?;
        let address = listener.local_addr().map_err(|error| error.to_string())?;
        let token = uuid::Uuid::new_v4().simple().to_string();
        let endpoint = Endpoint { url: format!("http://127.0.0.1:{}/frame", address.port()), token: token.clone() };
        let stopped = Arc::new(AtomicBool::new(false));
        let connections: Arc<Mutex<Vec<TcpStream>>> = Arc::default();
        let (stop_flag, open) = (stopped.clone(), connections.clone());
        std::thread::Builder::new().name("render-frames".into()).spawn(move || {
            for stream in listener.incoming() {
                if stop_flag.load(Ordering::SeqCst) { break; }
                let Ok(stream) = stream else { continue };
                let Ok(handle) = stream.try_clone() else { continue };
                open.lock().unwrap().push(handle);
                let (token, sink, stop_flag) = (token.clone(), sink.clone(), stop_flag.clone());
                let _ = std::thread::Builder::new().name("render-frames-conn".into()).spawn(move || {
                    serve(stream, &token, frame_bytes, &stop_flag, &*sink);
                });
            }
        }).map_err(|error| error.to_string())?;
        Ok(Self { endpoint, address, stopped, connections })
    }

    /// Closes the listener and every connection. Idempotent.
    pub fn stop(&self) {
        if self.stopped.swap(true, Ordering::SeqCst) { return; }
        for connection in self.connections.lock().unwrap().drain(..) {
            let _ = connection.shutdown(Shutdown::Both);
        }
        // Wakes the blocked `accept` so the listener thread sees the flag and exits.
        let _ = TcpStream::connect_timeout(&self.address, std::time::Duration::from_millis(200));
    }
}

impl Drop for FrameChannel {
    fn drop(&mut self) { self.stop(); }
}

fn respond(stream: &mut TcpStream, status: &str, extra: &str, body: &str) -> std::io::Result<()> {
    write!(stream, "HTTP/1.1 {status}\r\n{CORS}{extra}Content-Length: {}\r\nConnection: keep-alive\r\n\r\n{body}", body.len())?;
    stream.flush()
}

/// One keep-alive connection: requests are answered in order until the client or `stop` closes it.
fn serve(stream: TcpStream, token: &str, frame_bytes: usize, stopped: &AtomicBool, sink: &(dyn Fn(&[u8]) -> Result<u32, String> + Send + Sync)) {
    let _ = stream.set_nodelay(true);
    let Ok(mut writer) = stream.try_clone() else { return };
    let mut reader = std::io::BufReader::with_capacity(1 << 20, stream);
    let mut body = Vec::with_capacity(frame_bytes);
    loop {
        if stopped.load(Ordering::SeqCst) { return; }
        let mut line = String::new();
        match reader.read_line(&mut line) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let mut parts = line.split_whitespace();
        let method = parts.next().unwrap_or("").to_owned();
        let path = parts.next().unwrap_or("").to_owned();
        let (mut length, mut given_token) = (0usize, String::new());
        loop {
            let mut header = String::new();
            if reader.read_line(&mut header).unwrap_or(0) == 0 { return; }
            let header = header.trim_end();
            if header.is_empty() { break; }
            if let Some((name, value)) = header.split_once(':') {
                let value = value.trim();
                match name.trim().to_ascii_lowercase().as_str() {
                    "content-length" => length = value.parse().unwrap_or(usize::MAX),
                    "x-render-token" => given_token = value.to_owned(),
                    _ => {}
                }
            }
        }
        if method == "OPTIONS" {
            let extra = "Access-Control-Allow-Methods: POST\r\nAccess-Control-Allow-Headers: content-type, x-render-token\r\nAccess-Control-Max-Age: 600\r\n";
            if respond(&mut writer, "204 No Content", extra, "").is_err() { return; }
            continue;
        }
        // Never allocate or read more than one frame: a wrong size closes the connection after the answer.
        if length != frame_bytes {
            let _ = respond(&mut writer, "400 Bad Request", "Content-Type: text/plain\r\nConnection: close\r\n",
                &format!("a frame must be {frame_bytes} bytes, got {}", if length == usize::MAX { "an unreadable length".into() } else { length.to_string() }));
            return;
        }
        body.resize(length, 0);
        if reader.read_exact(&mut body).is_err() { return; }
        let answer = if method != "POST" || path != "/frame" {
            respond(&mut writer, "404 Not Found", "Content-Type: text/plain\r\n", "not found")
        } else if given_token != token {
            respond(&mut writer, "403 Forbidden", "Content-Type: text/plain\r\n", "wrong token")
        } else {
            match sink(&body) {
                Ok(frames) => respond(&mut writer, "204 No Content", &format!("x-frames: {frames}\r\n"), ""),
                Err(error) => respond(&mut writer, "500 Internal Server Error", "Content-Type: text/plain\r\n", error.replace(['\r', '\n'], " ").as_str()),
            }
        };
        if answer.is_err() { return; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exchange(client: &mut TcpStream, reader: &mut std::io::BufReader<TcpStream>, request: &[u8]) -> (String, String, String) {
        client.write_all(request).unwrap();
        let mut status = String::new();
        reader.read_line(&mut status).unwrap();
        let (mut headers, mut length) = (String::new(), 0usize);
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            if line.trim_end().is_empty() { break; }
            if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") { length = value.trim().parse().unwrap(); }
            headers.push_str(&line);
        }
        let mut body = vec![0u8; length];
        reader.read_exact(&mut body).unwrap();
        (status.trim_end().to_owned(), headers, String::from_utf8(body).unwrap())
    }

    fn post(token: &str, body: &[u8]) -> Vec<u8> {
        let mut request = format!("POST /frame HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Render-Token: {token}\r\nContent-Length: {}\r\n\r\n", body.len()).into_bytes();
        request.extend_from_slice(body);
        request
    }

    #[test]
    fn preflight_keep_alive_tokens_sizes_and_stop() {
        let got: Arc<Mutex<Vec<Vec<u8>>>> = Arc::default();
        let sink_got = got.clone();
        let channel = FrameChannel::start(1000, Arc::new(move |bytes: &[u8]| {
            let mut all = sink_got.lock().unwrap();
            all.push(bytes.to_vec());
            if bytes[0] == 9 { Err("ffmpeg stopped\naccepting".into()) } else { Ok(all.len() as u32) }
        })).unwrap();
        assert!(channel.endpoint.url.starts_with("http://127.0.0.1:") && channel.endpoint.url.ends_with("/frame"));
        assert_eq!(channel.endpoint.token.len(), 32);
        let address: SocketAddr = channel.endpoint.url.trim_start_matches("http://").trim_end_matches("/frame").parse().unwrap();
        let mut client = TcpStream::connect(address).unwrap();
        let mut reader = std::io::BufReader::new(client.try_clone().unwrap());
        let (status, headers, _) = exchange(&mut client, &mut reader,
            b"OPTIONS /frame HTTP/1.1\r\nHost: x\r\nOrigin: http://tauri.localhost\r\nAccess-Control-Request-Private-Network: true\r\nAccess-Control-Request-Headers: x-render-token\r\n\r\n");
        assert_eq!(status, "HTTP/1.1 204 No Content");
        assert!(headers.contains("Access-Control-Allow-Private-Network: true"), "{headers}");
        assert!(headers.contains("Access-Control-Allow-Headers: content-type, x-render-token"), "{headers}");
        assert!(headers.contains("Access-Control-Allow-Origin: *"), "{headers}");
        // Three frames on one connection; the wrong token is refused and never reaches the sink.
        let token = channel.endpoint.token.clone();
        let (a, b) = (vec![1u8; 1000], vec![2u8; 1000]);
        let answers: Vec<(String, String)> = [(&token[..], &a), ("nope", &b), (&token[..], &b)].iter()
            .map(|(t, body)| { let (s, h, _) = exchange(&mut client, &mut reader, &post(t, body)); (s, h) }).collect();
        assert_eq!(answers.iter().map(|(s, _)| s.as_str()).collect::<Vec<_>>(),
            vec!["HTTP/1.1 204 No Content", "HTTP/1.1 403 Forbidden", "HTTP/1.1 204 No Content"]);
        assert!(answers[2].1.contains("x-frames: 2"), "{}", answers[2].1);
        // A sink error comes back as a 500 with its message on one line.
        let (status, _, body) = exchange(&mut client, &mut reader, &post(&token, &[9u8; 1000]));
        assert_eq!((status.as_str(), body.as_str()), ("HTTP/1.1 500 Internal Server Error", "ffmpeg stopped accepting"));
        // A wrong size is refused before anything is read, and the connection closes.
        let (status, _, body) = exchange(&mut client, &mut reader, format!("POST /frame HTTP/1.1\r\nX-Render-Token: {token}\r\nContent-Length: 5000000000\r\n\r\n").as_bytes());
        assert_eq!(status, "HTTP/1.1 400 Bad Request");
        assert!(body.contains("must be 1000 bytes"), "{body}");
        assert_eq!(got.lock().unwrap().iter().map(|frame| frame[0]).collect::<Vec<_>>(), vec![1, 2, 9]);

        // A second keep-alive connection is open when the render stops: both it and the port close.
        let mut second = TcpStream::connect(address).unwrap();
        let mut second_reader = std::io::BufReader::new(second.try_clone().unwrap());
        assert_eq!(exchange(&mut second, &mut second_reader, &post(&token, &a)).0, "HTTP/1.1 204 No Content");
        drop(channel);
        let mut rest = Vec::new();
        second.set_read_timeout(Some(std::time::Duration::from_secs(5))).unwrap();
        assert_eq!(second_reader.read_to_end(&mut rest).unwrap_or(0), 0, "the open connection is closed");
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(TcpStream::connect(address).is_err(), "the listener is gone after stop");
    }
}

/// Manual check with a real browser (run with `--ignored`): listens, writes the endpoint to
/// `$AIV_CHANNEL_FILE`, and waits until 3 frames have arrived or 90 s pass.
#[cfg(test)]
mod browser_check {
    use super::*;
    #[test]
    #[ignore]
    fn serve_for_a_browser() {
        let got = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = got.clone();
        let channel = FrameChannel::start(12, Arc::new(move |_bytes: &[u8]| Ok(count.fetch_add(1, Ordering::SeqCst) + 1))).unwrap();
        std::fs::write(std::env::var("AIV_CHANNEL_FILE").unwrap(), serde_json::to_string(&channel.endpoint).unwrap()).unwrap();
        let began = std::time::Instant::now();
        while got.load(Ordering::SeqCst) < 3 && began.elapsed().as_secs() < 90 { std::thread::sleep(std::time::Duration::from_millis(50)); }
        assert_eq!(got.load(Ordering::SeqCst), 3);
    }
}
