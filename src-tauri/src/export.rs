use crate::domain::Rational;
use serde::{Deserialize, Serialize};
use std::{fs, path::{Path, PathBuf}, process::Stdio};
use tokio::process::Command;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportRequest {
    pub concat_manifest: String,
    pub output_path: String,
    pub width: u32,
    pub height: u32,
    pub frame_rate: Rational,
    pub video_codec: String,
    pub audio_codec: String,
    pub video_bitrate: String,
    pub audio_bitrate: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderClip {
    pub path: String,
    #[serde(default)]
    pub start_seconds: f64,
    pub source_in_seconds: f64,
    pub duration_seconds: f64,
    #[serde(default)]
    pub source_duration_seconds: f64,
    #[serde(default = "default_speed_rate")]
    pub speed_rate: f64,
    pub has_audio: bool,
    pub position_x: f64,
    pub position_y: f64,
    pub scale_percent: f64,
    pub rotation_degrees: f64,
    pub opacity: f64,
    pub volume_db: f64,
    /// Transition played as this clip enters. Fuses with the previous clip's `transition_out_kind`.
    pub transition_kind: String,
    pub transition_duration: f64,
    /// Transition played as this clip leaves. On the final clip this renders as a fade to black.
    #[serde(default = "default_transition_kind")]
    pub transition_out_kind: String,
    #[serde(default)]
    pub transition_out_duration: f64,
    #[serde(default)]
    pub effects: RenderEffects,
    /// How this clip is fitted into the sequence frame. Absent = the sequence default, resolved on
    /// the UI side, so an older project still renders as plain `contain`.
    #[serde(default)]
    pub conform: RenderConform,
    /// Timeline seconds this clip may borrow from BEFORE its in-point to cover an incoming
    /// dissolve. Resolved on the UI side alongside the ripple applied to the other tracks, so the
    /// handle used here and the shift applied there cannot disagree.
    #[serde(default)]
    pub head_handle_seconds: f64,
}

fn default_transition_kind() -> String { "none".to_owned() }

fn default_speed_rate() -> f64 { 1.0 }

pub(crate) fn atempo_filter(rate: f64) -> String {
    let safe = rate.clamp(0.25, 4.0);
    if safe < 0.5 {
        format!("atempo=0.5,atempo={:.6}", safe / 0.5)
    } else if safe > 2.0 {
        format!("atempo=2.0,atempo={:.6}", safe / 2.0)
    } else {
        format!("atempo={safe:.6}")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderEffects {
    pub brightness: f64,
    pub contrast: f64,
    pub saturation: f64,
    pub gamma: f64,
    pub hue: f64,
    pub blur: f64,
    pub sharpen: f64,
    pub vignette: f64,
    pub noise: f64,
}

impl Default for RenderEffects {
    fn default() -> Self {
        Self { brightness: 0.0, contrast: 1.0, saturation: 1.0, gamma: 1.0, hue: 0.0, blur: 0.0, sharpen: 0.0, vignette: 0.0, noise: 0.0 }
    }
}

/// A normalised 0..1 rectangle of the SOURCE frame, so a crop survives a change of source
/// resolution or a swap to a proxy.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SourceRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Default for SourceRect {
    fn default() -> Self { Self { x: 0.0, y: 0.0, width: 1.0, height: 1.0 } }
}

impl SourceRect {
    /// Clamp to the source and refuse a degenerate rect, which would make ffmpeg's `crop` fail with
    /// an unhelpful message halfway through a render.
    fn sane(self) -> Self {
        let width = self.width.clamp(0.01, 1.0);
        let height = self.height.clamp(0.01, 1.0);
        Self {
            x: self.x.clamp(0.0, 1.0 - width),
            y: self.y.clamp(0.0, 1.0 - height),
            width,
            height,
        }
    }

    fn is_full(self) -> bool {
        self.x <= f64::EPSILON && self.y <= f64::EPSILON && self.width >= 1.0 && self.height >= 1.0
    }
}

/// `end` absent is a static crop. `end` present is a move -- ffmpeg's `crop` takes `t` in its
/// expressions, so a Ken Burns push is a string change here rather than a keyframe engine.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConformCrop {
    pub start: SourceRect,
    #[serde(default)]
    pub end: Option<SourceRect>,
}

/// How a clip's picture is fitted into the sequence frame. Previously this was hardcoded to
/// `contain` for every V1 clip, which is why mismatched footage always gained bars.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderConform {
    /// `contain`, `cover`, `stretch` or `none`.
    #[serde(default = "default_fit")]
    pub fit: String,
    /// `black`, `blur` or `color`.
    #[serde(default = "default_pad")]
    pub pad: String,
    #[serde(default)]
    pub pad_color: Option<String>,
    #[serde(default)]
    pub pad_blur: Option<f64>,
    #[serde(default)]
    pub crop: Option<ConformCrop>,
}

fn default_fit() -> String { "contain".to_owned() }
fn default_pad() -> String { "black".to_owned() }

impl Default for RenderConform {
    fn default() -> Self {
        Self { fit: default_fit(), pad: default_pad(), pad_color: None, pad_blur: None, crop: None }
    }
}

impl RenderConform {
    /// The `crop=` filter for the source rect, or None when the whole frame is kept.
    ///
    /// A moving crop interpolates start -> end linearly in `t`. `exact=1` stops ffmpeg snapping the
    /// offset to the chroma grid, which on a slow move reads as a stutter.
    fn crop_filter(&self, duration: f64) -> Option<String> {
        let crop = self.crop?;
        let start = crop.start.sane();
        match crop.end.map(SourceRect::sane) {
            None => {
                if start.is_full() { return None; }
                Some(format!(
                    "crop=w='iw*{:.6}':h='ih*{:.6}':x='iw*{:.6}':y='ih*{:.6}':exact=1",
                    start.width, start.height, start.x, start.y
                ))
            }
            Some(end) => {
                // Guard the divisor: a zero-length clip would make the expression NaN and ffmpeg
                // would render the first frame for the whole segment with no error.
                let span = duration.max(1.0 / 1000.0);
                let lerp = |a: f64, b: f64| format!("({a:.6}+({:.6})*min(t/{span:.6},1))", b - a);
                Some(format!(
                    "crop=w='iw*{}':h='ih*{}':x='iw*{}':y='ih*{}':exact=1",
                    lerp(start.width, end.width),
                    lerp(start.height, end.height),
                    lerp(start.x, end.x),
                    lerp(start.y, end.y),
                ))
            }
        }
    }
}

/// Emit the graph fragment that conforms `input_label` into a `width` x `height` frame at
/// `output_label`. `tag` must be unique per call so the intermediate labels never collide.
///
/// Returns a fragment ending in `;`. Every branch finishes in RGBA so that whatever the caller
/// appends -- overlay, rotate with `c=none`, a transparent pad -- still has an alpha channel to
/// work with. `eq` has no alpha-capable pixel format, so anything that grades BEFORE this point
/// will have dropped alpha; that asymmetry is what produced both the green corners and the opaque
/// letterbox bars.
fn conform_graph(
    input_label: &str,
    output_label: &str,
    conform: &RenderConform,
    width: u32,
    height: u32,
    duration: f64,
    tag: &str,
    grade: &str,
) -> String {
    // Order is load-bearing: crop the source, GRADE, then fit and pad.
    //
    // Grading has to happen before the pad because `eq` has no alpha-capable pixel format -- put it
    // after and ffmpeg silently converts away from RGBA, which is what made transparent padding
    // opaque and `rotate`'s `c=none` fill green. Cropping first also means the grade only runs on
    // the pixels that survive, and the blur-fill background is built from the cropped picture
    // rather than the whole source.
    let mut head = String::new();
    if let Some(crop) = conform.crop_filter(duration) {
        head.push_str(&crop);
        head.push(',');
    }
    if !grade.is_empty() {
        head.push_str(grade.trim_end_matches(','));
        head.push(',');
    }

    match conform.fit.as_str() {
        "stretch" => format!("[{input_label}]{head}scale={width}:{height},setsar=1,format=rgba[{output_label}];"),
        "cover" => format!(
            "[{input_label}]{head}scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1,format=rgba[{output_label}];"
        ),
        // `none` keeps the source at its own size and lets the clip transform place it. Without
        // this every clip is resampled even when nothing asked for it.
        "none" => format!("[{input_label}]{head}setsar=1,format=rgba[{output_label}];"),
        _ => match conform.pad.as_str() {
            // Fill the bars with a blurred, zoomed copy of the same frame: no bars AND nothing
            // cropped away, which is the answer most people actually want for mismatched footage.
            "blur" => {
                let sigma = conform.pad_blur.unwrap_or(40.0).clamp(1.0, 200.0);
                format!(
                    "[{input_label}]{head}split=2[{tag}bl][{tag}fg];                     [{tag}bl]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},gblur=sigma={sigma:.2}[{tag}bg];                     [{tag}fg]scale={width}:{height}:force_original_aspect_ratio=decrease[{tag}fs];                     [{tag}bg][{tag}fs]overlay=(W-w)/2:(H-h)/2,setsar=1,format=rgba[{output_label}];"
                )
            }
            other => {
                let color = if other == "color" {
                    color_with_opacity(conform.pad_color.as_deref().unwrap_or("#000000"), 1.0, "000000ff")
                } else {
                    "black".to_owned()
                };
                format!(
                    "[{input_label}]{head}scale={width}:{height}:force_original_aspect_ratio=decrease,                     pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={color},setsar=1,format=rgba[{output_label}];"
                )
            }
        },
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderGraphic {
    pub start_seconds: f64,
    pub duration_seconds: f64,
    pub opacity: f64,
    pub template: String,
    pub text: String,
    pub subtitle: String,
    pub text_color: String,
    pub background_color: String,
    pub accent_color: String,
    pub font_size: f64,
    #[serde(default = "default_text_backdrop")]
    pub backdrop: String,
    #[serde(default = "default_one")]
    pub background_opacity: f64,
    #[serde(default = "default_text_padding")]
    pub background_padding: f64,
    #[serde(default = "default_text_frame_width")]
    pub background_border_width: f64,
    #[serde(default = "default_text_font")]
    pub font_family: String,
    #[serde(default = "default_text_weight")]
    pub font_weight: u32,
    #[serde(default)]
    pub italic: bool,
    #[serde(default)]
    pub stroke_width: f64,
    #[serde(default = "default_text_stroke_color")]
    pub stroke_color: String,
    #[serde(default)]
    pub shadow_offset: f64,
    #[serde(default = "default_text_shadow_color")]
    pub shadow_color: String,
    #[serde(default = "default_text_line_spacing")]
    pub line_spacing: f64,
    #[serde(default)]
    pub motion_in: String,
    #[serde(default)]
    pub motion_out: String,
    #[serde(default = "default_text_motion_duration")]
    pub motion_duration: f64,
    #[serde(default)]
    pub motion_loop: String,
    #[serde(default = "default_text_easing")]
    pub motion_easing: String,
}

fn default_text_motion_duration() -> f64 { 0.6 }
fn default_one() -> f64 { 1.0 }
fn default_text_backdrop() -> String { "box".to_owned() }
fn default_text_font() -> String { "display".to_owned() }
fn default_text_stroke_color() -> String { "#000000".to_owned() }
fn default_text_shadow_color() -> String { "#000000".to_owned() }
fn default_text_easing() -> String { "smooth".to_owned() }
fn default_text_weight() -> u32 { 700 }
fn default_text_padding() -> f64 { 16.0 }
fn default_text_frame_width() -> f64 { 3.0 }
fn default_text_line_spacing() -> f64 { 4.0 }

impl Default for RenderGraphic {
    fn default() -> Self {
        Self {
            start_seconds: 0.0, duration_seconds: 1.0, opacity: 1.0, template: "title".into(),
            text: String::new(), subtitle: String::new(), text_color: "#ffffff".into(),
            background_color: "#10131a".into(), accent_color: "#ff6b4a".into(), font_size: 48.0,
            backdrop: default_text_backdrop(), background_opacity: 1.0,
            background_padding: default_text_padding(), background_border_width: default_text_frame_width(),
            font_family: default_text_font(), font_weight: default_text_weight(), italic: false,
            stroke_width: 0.0, stroke_color: default_text_stroke_color(), shadow_offset: 0.0,
            shadow_color: default_text_shadow_color(), line_spacing: default_text_line_spacing(),
            motion_in: "none".into(), motion_out: "none".into(),
            motion_duration: default_text_motion_duration(), motion_loop: "none".into(),
            motion_easing: default_text_easing(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderAudioClip {
    pub path: String,
    pub source_in_seconds: f64,
    pub duration_seconds: f64,
    #[serde(default)]
    pub source_duration_seconds: f64,
    #[serde(default = "default_speed_rate")]
    pub speed_rate: f64,
    pub start_seconds: f64,
    pub volume_db: f64,
    #[serde(default)]
    pub pan: f64,
    #[serde(default)]
    pub fade_in_seconds: f64,
    #[serde(default)]
    pub fade_out_seconds: f64,
}

/// A clip on a picture track that composites over the programme: picture-in-picture, split
/// screen, or a background-removed cutout sitting full frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RenderOverlayClip {
    pub path: String,
    pub start_seconds: f64,
    pub source_in_seconds: f64,
    pub duration_seconds: f64,
    #[serde(default)]
    pub source_duration_seconds: f64,
    #[serde(default = "default_speed_rate")]
    pub speed_rate: f64,
    #[serde(default)]
    pub has_audio: bool,
    #[serde(default)]
    pub volume_db: f64,
    /// Placement in normalised 0..1 coordinates of the sequence frame.
    pub box_x: f64,
    pub box_y: f64,
    pub box_width: f64,
    pub box_height: f64,
    /// `contain`, `cover` or `stretch`.
    pub fit: String,
    pub opacity: f64,
    #[serde(default)]
    pub corner_radius: f64,
    #[serde(default)]
    pub border_width: f64,
    #[serde(default)]
    pub border_color: String,
    #[serde(default)]
    pub effects: RenderEffects,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TimelineExportRequest {
    pub clips: Vec<RenderClip>,
    #[serde(default)]
    pub graphics: Vec<RenderGraphic>,
    #[serde(default)]
    pub audio_clips: Vec<RenderAudioClip>,
    #[serde(default)]
    pub overlay_clips: Vec<RenderOverlayClip>,
    pub output_path: String,
    pub width: u32,
    pub height: u32,
    pub frame_rate: Rational,
    pub video_codec: String,
    pub audio_codec: String,
    pub video_bitrate: String,
    pub audio_bitrate: String,
    /// Legacy integer factor, kept so a project saved before the chain existed still renders.
    /// A payload carrying `upscale` takes precedence over this.
    #[serde(default = "default_upscale_factor")]
    pub upscale_factor: u32,
    /// The resolved upscale chain: AI stage, resample stage, final size. See `src/domain/upscale.ts`
    /// -- the readout the panel shows and these arguments come from the same resolver.
    #[serde(default)]
    pub upscale: Option<crate::ai::upscale::UpscaleChain>,
    /// `quality` (CQ/CRF) or `bitrate`. Defaults to bitrate so a payload from before this existed
    /// still renders exactly as it used to.
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
}

fn default_upscale_factor() -> u32 { 1 }
fn default_quality_mode() -> String { "bitrate".to_owned() }
fn default_quality() -> u32 { 21 }
fn default_sample_rate() -> u32 { 48_000 }
fn default_channels() -> u32 { 2 }

/// Everything the encoders need, gathered so no call site has to reassemble it and drift.
#[derive(Debug, Clone)]
pub struct EncodeSettings {
    pub video_codec: String,
    pub quality_mode: String,
    pub quality: u32,
    pub preset: String,
    pub video_bitrate: String,
    pub audio_codec: String,
    pub audio_bitrate: String,
    pub audio_sample_rate: u32,
    pub audio_channels: u32,
}

impl EncodeSettings {
    fn from_timeline(request: &TimelineExportRequest) -> Self {
        Self {
            video_codec: request.video_codec.clone(),
            quality_mode: request.quality_mode.clone(),
            quality: request.quality,
            preset: request.preset.clone(),
            video_bitrate: request.video_bitrate.clone(),
            audio_codec: request.audio_codec.clone(),
            audio_bitrate: request.audio_bitrate.clone(),
            audio_sample_rate: request.audio_sample_rate,
            audio_channels: request.audio_channels,
        }
    }

    /// Video encoder arguments.
    ///
    /// `intermediate` is the important part. The export re-encodes up to FIVE times -- per-clip
    /// segments, concat/xfade, overlays, audio mix, graphics -- and every pass used to run at the
    /// delivery bitrate, so a clip was lossily compressed five times over before anyone saw it.
    /// Intermediates now run near-lossless and only the final pass honours the user's setting.
    pub(crate) fn video_args(&self, intermediate: bool) -> Vec<String> {
        let nvenc = self.video_codec.contains("nvenc");
        let mut args: Vec<String> = vec!["-c:v".into(), self.video_codec.clone()];

        let preset = if self.preset.is_empty() {
            if nvenc { "p5".to_owned() } else { "medium".to_owned() }
        } else {
            self.preset.clone()
        };
        // A preset from the wrong family is a hard failure, not a fallback: NVENC rejects
        // `veryslow` and x264 rejects `p5`. Substitute rather than let the render die at pass four.
        let preset = match (nvenc, preset.starts_with('p') && preset.len() == 2) {
            (true, false) => "p5".to_owned(),
            (false, true) => "medium".to_owned(),
            _ => preset,
        };
        args.extend(["-preset".into(), preset]);

        if intermediate {
            // Near-visually-lossless, and cheap enough not to dominate render time.
            if nvenc {
                args.extend(["-rc".into(), "constqp".into(), "-qp".into(), "16".into()]);
            } else {
                args.extend(["-crf".into(), "14".into()]);
            }
        } else if self.quality_mode == "quality" {
            let cq = self.quality.clamp(1, 51).to_string();
            if nvenc {
                // `-b:v 0` is load-bearing: without it NVENC treats `-cq` as a ceiling on top of a
                // default bitrate cap and quietly ignores most of the quality setting.
                args.extend(["-rc".into(), "vbr".into(), "-cq".into(), cq, "-b:v".into(), "0".into()]);
            } else {
                args.extend(["-crf".into(), cq]);
            }
        } else {
            args.extend(["-b:v".into(), self.video_bitrate.clone()]);
        }

        // Pin the output format and colour tagging at every boundary. Leaving these to negotiation
        // across a five-pass pipeline is how a "transparent" fill became green in the first place.
        args.extend([
            "-pix_fmt".into(), "yuv420p".into(),
            "-colorspace".into(), "bt709".into(),
            "-color_primaries".into(), "bt709".into(),
            "-color_trc".into(), "bt709".into(),
            "-color_range".into(), "tv".into(),
        ]);
        args
    }

    /// Audio encoder arguments. `none` drops the track entirely.
    pub(crate) fn audio_args(&self) -> Vec<String> {
        if self.audio_codec == "none" {
            return vec!["-an".into()];
        }
        let mut args: Vec<String> = vec![
            "-c:a".into(), self.audio_codec.clone(),
            "-ar".into(), self.audio_sample_rate.clamp(8_000, 192_000).to_string(),
            "-ac".into(), self.audio_channels.clamp(1, 8).to_string(),
        ];
        // FLAC and PCM are uncompressed or lossless; handing them a bitrate is meaningless and
        // ffmpeg warns about it.
        if !matches!(self.audio_codec.as_str(), "flac" | "pcm_s16le" | "pcm_s24le") {
            args.extend(["-b:a".into(), self.audio_bitrate.clone()]);
        }
        args
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ExportResult {
    pub output_path: String,
    pub command: Vec<String>,
    /// Non-fatal notes the UI should surface -- e.g. a transition that had to hold a frame because
    /// the clip had no media before its in-point. Silently doing neither is how the original
    /// restart-after-a-dissolve bug went unnoticed.
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug)]
struct PreparedSegment {
    path: PathBuf,
    duration: f64,
    transition_in_kind: String,
    transition_in_duration: f64,
    transition_out_kind: String,
    transition_out_duration: f64,
}

/// A cut carries at most one rendered transition. The incoming clip's choice wins when both
/// sides are set, and the longer of the two durations is used so neither side is truncated.
fn resolve_junction(previous: &PreparedSegment, next: &PreparedSegment) -> (String, f64) {
    let out_set = previous.transition_out_kind != "none" && !previous.transition_out_kind.is_empty();
    let in_set = next.transition_in_kind != "none" && !next.transition_in_kind.is_empty();
    match (out_set, in_set) {
        (false, false) => ("none".to_owned(), 0.0),
        (true, false) => (previous.transition_out_kind.clone(), previous.transition_out_duration),
        (false, true) => (next.transition_in_kind.clone(), next.transition_in_duration),
        (true, true) => (
            next.transition_in_kind.clone(),
            previous.transition_out_duration.max(next.transition_in_duration),
        ),
    }
}

async fn run_ffmpeg(args: &[String]) -> Result<(), String> {
    if std::env::var_os("AIV_KEEP_RENDER_CACHE").is_some() {
        eprintln!("AIV ffmpeg {}", args.join(" "));
    }
    let output = Command::new("ffmpeg").args(args).stdin(Stdio::null()).output().await
        .map_err(|error| format!("failed to launch ffmpeg: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    Ok(())
}

/// The colour conversion every video output chain has to end with.
///
/// `-colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv` in `video_args`
/// only **tags** the bitstream. It does not tell swscale what conversion to perform, and
/// swscale's default when a chain ends in a bare `format=yuv420p` after any RGB stage is
/// **BT.601**. So the pixels are converted one way and labelled another, and every player then
/// applies the wrong inverse matrix. On moving footage that reads as colour banding and coloured
/// fringing -- which is exactly what the upscaled exports were doing, and why stills were fine:
/// the image path never leaves RGB.
///
/// Measured on flat colour patches through libx264 at `-qp 0`, so nothing below is a compression
/// artefact. Mean absolute error over the frame, and what pure green comes back as:
///
/// | chain | with `format=yuv420p` | with this |
/// |---|---|---|
/// | PNG frame sequence (the AI upscale encode) | 5.94, `0,215,0` | **0.86, `0,254,0`** |
/// | clip segment, RGBA via `colorchannelmixer` | 6.20, `0,213,0` | **2.67, `0,250,0`** |
/// | a chain that never leaves YUV | 0.000 | **0.000** |
///
/// Three things this deliberately does *not* say, each of which was measured and made it worse:
/// `in_range=full` (13.61 on the segment chain -- the graph's RGB carries the source's range, so
/// declaring it full is a lie), `in_color_matrix=bt709` (6.20, i.e. back to broken -- naming the
/// input matrix makes scale treat RGB input as YUV and disables the correct auto behaviour), and
/// the `colorspace` filter instead of `scale` (14.03 on the PNG chain).
///
/// Because it is an exact no-op on a chain that stays in YUV, it is applied to every video output
/// chain rather than only the ones with a visible RGB stage -- `drawbox` with an alpha colour and
/// `overlay` with an RGBA layer both pull a graph into RGB without saying so in the filter text.
pub const RGB_TO_BT709_TV: &str = "scale=out_color_matrix=bt709,format=yuv420p";

/// Append [`RGB_TO_BT709_TV`] to a `filter_complex` chain, returning the label to map instead.
fn tag_video_range(filter: &mut String, label: &str) -> String {
    if !filter.is_empty() && !filter.ends_with(';') {
        filter.push(';');
    }
    filter.push_str(&format!("[{label}]{conversion}[vcolor]", conversion = RGB_TO_BT709_TV));
    "vcolor".to_owned()
}

async fn render_gap(
    output_path: &Path,
    duration: f64,
    width: u32,
    height: u32,
    fps: f64,
    encode: &EncodeSettings,
) -> Result<(), String> {
    let mut args: Vec<String> = vec![
        "-hide_banner".into(), "-y".into(),
        "-f".into(), "lavfi".into(), "-i".into(),
        format!("color=c=black:s={}x{}:r={}:d={:.6}", width, height, fps, duration),
        "-f".into(), "lavfi".into(), "-i".into(),
        format!("anullsrc=channel_layout=stereo:sample_rate=48000:d={:.6}", duration),
    ];
    args.extend(encode.video_args(true));
    args.extend(encode.audio_args());
    args.extend(["-shortest".into(), output_path.to_string_lossy().into_owned()]);
    run_ffmpeg(&args).await
}

async fn trim_prepared_segment(
    input: &Path,
    output: &Path,
    duration: f64,
    encode: &EncodeSettings,
) -> Result<(), String> {
    let mut args: Vec<String> = vec![
        "-hide_banner".into(), "-y".into(), "-i".into(), input.to_string_lossy().into_owned(),
        "-filter_complex".into(), format!(
            "[0:v]trim=duration={duration:.9},setpts=PTS-STARTPTS[v];[0:a]atrim=duration={duration:.9},asetpts=PTS-STARTPTS[a]"
        ),
        "-map".into(), "[v]".into(), "-map".into(), "[a]".into(),
    ];
    args.extend(encode.video_args(true));
    args.extend(encode.audio_args());
    args.push(output.to_string_lossy().into_owned());
    run_ffmpeg(&args).await
}

pub async fn export(request: ExportRequest) -> Result<ExportResult, String> {
    let frame_rate = request.frame_rate.validate()?.as_f64();
    if request.width == 0 || request.height == 0 || frame_rate <= 0.0 {
        return Err("invalid export dimensions or frame rate".into());
    }
    if !PathBuf::from(&request.concat_manifest).is_file() {
        return Err("render manifest does not exist".into());
    }
    let args = vec![
        "-hide_banner".into(), "-y".into(), "-f".into(), "concat".into(), "-safe".into(), "0".into(),
        "-i".into(), request.concat_manifest.clone(), "-vf".into(),
        format!("scale={}:{},fps={frame_rate}", request.width, request.height),
        "-c:v".into(), request.video_codec.clone(), "-b:v".into(), request.video_bitrate.clone(),
        "-c:a".into(), request.audio_codec.clone(), "-b:a".into(), request.audio_bitrate.clone(),
        request.output_path.clone(),
    ];
    run_ffmpeg(&args).await?;
    Ok(ExportResult { output_path: request.output_path, command: args, warnings: Vec::new() })
}

const XFADE_TRANSITIONS: &[&str] = &[
    "fade", "wipeleft", "wiperight", "wipeup", "wipedown",
    "slideleft", "slideright", "slideup", "slidedown",
    "circlecrop", "rectcrop", "distance", "fadeblack", "fadewhite", "radial",
    "smoothleft", "smoothright", "smoothup", "smoothdown",
    "circleopen", "circleclose", "vertopen", "vertclose", "horzopen", "horzclose",
    "dissolve", "pixelize", "diagtl", "diagtr", "diagbl", "diagbr",
    "hlslice", "hrslice", "vuslice", "vdslice", "hblur", "fadegrays",
    "wipetl", "wipetr", "wipebl", "wipebr", "squeezeh", "zoomin",
    "fadefast", "fadeslow", "hlwind", "hrwind", "vuwind", "vdwind",
    "coverleft", "coverright", "coverup", "coverdown",
    "revealleft", "revealright", "revealup", "revealdown",
];

fn transition_name(value: &str) -> &str {
    if XFADE_TRANSITIONS.contains(&value) { value } else { "fade" }
}

/// A transition at the head or tail of the edit has no neighbour, so it degrades to a fade
/// from/to the colour the transition dips through.
fn fade_color(value: &str) -> &'static str {
    match value {
        "fadewhite" => "white",
        "fadegrays" => "gray",
        _ => "black",
    }
}

fn color_with_opacity(value: &str, opacity: f64, fallback: &str) -> String {
    let raw = value.trim().trim_start_matches('#');
    let valid = (raw.len() == 6 || raw.len() == 8) && raw.chars().all(|ch| ch.is_ascii_hexdigit());
    let selected = if valid { raw } else { fallback };
    let rgb = &selected[..6];
    let embedded_alpha = if selected.len() == 8 { u8::from_str_radix(&selected[6..], 16).unwrap_or(255) as f64 / 255.0 } else { 1.0 };
    format!("0x{rgb}@{:.4}", (embedded_alpha * opacity).clamp(0.0, 1.0))
}

fn escape_drawtext(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\'', "\\'").replace(':', "\\:")
        .replace('%', "\\%").replace(',', "\\,").replace('[', "\\[").replace(']', "\\]")
        .replace('\n', " ").replace('\r', " ")
}

fn text_motion_segments(graphic: &RenderGraphic, start: f64, end: f64, motion: f64) -> Vec<(String, f64, f64)> {
    let characters = graphic.text.chars().collect::<Vec<_>>();
    if characters.is_empty() { return Vec::new(); }
    let type_in = ["typewriter", "word-reveal"].contains(&graphic.motion_in.as_str());
    let type_out = ["typewriter", "word-reveal"].contains(&graphic.motion_out.as_str());
    if !type_in && !type_out { return vec![(graphic.text.clone(), start, end)]; }
    // Word reveal advances at word boundaries; typewriter advances at character boundaries. Both
    // are emitted as timed drawtext segments so the exported master matches the live preview.
    let word_mode = graphic.motion_in == "word-reveal" || graphic.motion_out == "word-reveal";
    let boundaries = if word_mode {
        let mut ends = graphic.text.char_indices()
            .filter_map(|(index, ch)| ch.is_whitespace().then_some(index))
            .collect::<Vec<_>>();
        ends.push(graphic.text.len());
        ends.sort_unstable(); ends.dedup(); ends
    } else {
        characters.iter().enumerate().map(|(index, _)| index + 1).collect::<Vec<_>>()
    };
    // Typewriter means one character per step, including long titles. The previous 60-step cap
    // skipped letters whenever the text was longer than 60 characters, so it looked like chunks
    // rather than a real type-in/type-out. Text fields are UI-limited, making the full count safe.
    let steps = boundaries.len();
    let mut result = Vec::new();
    let body_start = if type_in { start + motion } else { start };
    let body_end = if type_out { end - motion } else { end };
    if type_in {
        for step in 1..=steps {
            let boundary = boundaries[(boundaries.len() * step).div_ceil(steps).saturating_sub(1).min(boundaries.len() - 1)];
            let segment = if word_mode { graphic.text[..boundary].trim_end().to_owned() }
                else { characters[..boundary].iter().collect() };
            let left = start + motion * (step - 1) as f64 / steps as f64;
            let right = start + motion * step as f64 / steps as f64;
            result.push((segment, left, right));
        }
    }
    if body_end > body_start + 0.000_001 {
        result.push((graphic.text.clone(), body_start, body_end));
    }
    if type_out {
        for step in 0..steps {
            let remaining = steps.saturating_sub(step + 1);
            let boundary = if remaining == 0 { 0 } else {
                boundaries[(boundaries.len() * remaining).div_ceil(steps).saturating_sub(1).min(boundaries.len() - 1)]
            };
            let segment = if word_mode { graphic.text[..boundary].trim_end().to_owned() }
                else { characters[..boundary].iter().collect() };
            let left = end - motion + motion * step as f64 / steps as f64;
            let right = end - motion + motion * (step + 1) as f64 / steps as f64;
            if boundary > 0 { result.push((segment, left, right)); }
        }
    }
    result
}

fn text_ease_expression(progress: &str, easing: &str) -> String {
    match easing {
        "linear" => progress.to_owned(),
        "ease-out" => format!("1-pow(1-({progress}),3)"),
        "back" => format!("({progress})+sin(({progress})*PI)*0.18"),
        "bounce" => format!("({progress})+abs(sin(({progress})*3*PI))*(1-({progress}))*0.22"),
        _ => format!("({progress})*({progress})*(3-2*({progress}))"),
    }
}

fn graphic_font_file(graphic: &RenderGraphic) -> &'static str {
    match (graphic.font_family.as_str(), graphic.font_weight >= 700, graphic.italic) {
        ("serif", true, true) => "C\\:/Windows/Fonts/timesbi.ttf",
        ("serif", true, false) => "C\\:/Windows/Fonts/timesbd.ttf",
        ("serif", false, true) => "C\\:/Windows/Fonts/timesi.ttf",
        ("serif", false, false) => "C\\:/Windows/Fonts/times.ttf",
        ("mono", true, _) => "C\\:/Windows/Fonts/consolab.ttf",
        ("mono", false, true) => "C\\:/Windows/Fonts/consolai.ttf",
        ("mono", false, false) => "C\\:/Windows/Fonts/consola.ttf",
        (_, true, true) => "C\\:/Windows/Fonts/arialbi.ttf",
        (_, true, false) => "C\\:/Windows/Fonts/arialbd.ttf",
        (_, false, true) => "C\\:/Windows/Fonts/ariali.ttf",
        _ => "C\\:/Windows/Fonts/arial.ttf",
    }
}

fn text_motion_options(
    graphic: &RenderGraphic,
    start: f64,
    end: f64,
    motion: f64,
    text_x: &str,
    text_y: &str,
    accent: &str,
) -> (String, String, String, String) {
    let enter_progress = text_ease_expression(
        &format!("(t-{start:.6})/{motion:.6}"), &graphic.motion_easing,
    );
    let exit_progress = text_ease_expression(
        &format!("({end:.6}-t)/{motion:.6}"), &graphic.motion_easing,
    );
    let mut alpha = "1".to_owned();
    if ["fade", "magic", "zoom-blur", "tracking", "cinematic", "word-reveal"].contains(&graphic.motion_in.as_str()) {
        alpha = format!("({alpha})*if(lt(t,{:.6}),{enter_progress},1)", start + motion);
    }
    if ["fade", "magic", "zoom-blur", "tracking", "cinematic", "word-reveal"].contains(&graphic.motion_out.as_str()) {
        alpha = format!("({alpha})*if(gt(t,{:.6}),{exit_progress},1)", end - motion);
    }
    if graphic.motion_in == "glitch" || graphic.motion_out == "glitch"
        || graphic.motion_loop == "flicker" || graphic.motion_loop == "breathe" {
        alpha = format!("({alpha})*(0.82+0.18*abs(sin((t-{start:.6})*37)))");
    }

    let mut x = format!("({text_x})");
    if graphic.motion_in == "glitch" || graphic.motion_out == "glitch" {
        x.push_str(&format!("+sin((t-{start:.6})*83)*w*0.006"));
    }
    if graphic.motion_loop == "jitter" {
        x.push_str(&format!("+sin((t-{start:.6})*53)*w*0.0012"));
    }
    if graphic.motion_in == "slide-left" {
        x.push_str(&format!("+if(lt(t,{:.6}),(1-({enter_progress}))*-w*0.12,0)", start + motion));
    }
    if graphic.motion_in == "slide-right" {
        x.push_str(&format!("+if(lt(t,{:.6}),(1-({enter_progress}))*w*0.12,0)", start + motion));
    }
    if graphic.motion_out == "slide-left" {
        x.push_str(&format!("+if(gt(t,{:.6}),(1-({exit_progress}))*-w*0.12,0)", end - motion));
    }
    if graphic.motion_out == "slide-right" {
        x.push_str(&format!("+if(gt(t,{:.6}),(1-({exit_progress}))*w*0.12,0)", end - motion));
    }
    let mut y = format!("({text_y})");
    if graphic.motion_in == "slide-up" || graphic.motion_in == "cinematic" {
        y.push_str(&format!("+if(lt(t,{:.6}),(1-({enter_progress}))*h*0.08,0)", start + motion));
    }
    if graphic.motion_in == "slide-down" {
        y.push_str(&format!("+if(lt(t,{:.6}),(1-({enter_progress}))*-h*0.08,0)", start + motion));
    }
    if graphic.motion_out == "slide-down" {
        y.push_str(&format!("+if(gt(t,{:.6}),(1-({exit_progress}))*h*0.08,0)", end - motion));
    }
    if graphic.motion_out == "slide-up" || graphic.motion_out == "cinematic" {
        y.push_str(&format!("+if(gt(t,{:.6}),(1-({exit_progress}))*-h*0.08,0)", end - motion));
    }
    if graphic.motion_loop == "float" {
        y.push_str(&format!("+sin((t-{start:.6})*PI)*h*0.006"));
    }
    if graphic.motion_loop == "jitter" {
        y.push_str(&format!("+sin((t-{start:.6})*71)*h*0.001"));
    }

    let base_size = graphic.font_size.clamp(16.0, 160.0);
    let mut size_factor = "1".to_owned();
    if graphic.motion_in == "pop" {
        size_factor = format!("({size_factor})*if(lt(t,{:.6}),0.55+0.45*({enter_progress}),1)", start + motion);
    }
    if graphic.motion_in == "bounce" {
        size_factor = format!("({size_factor})*if(lt(t,{:.6}),0.42+0.58*({enter_progress}),1)", start + motion);
    }
    if ["zoom-blur", "tracking"].contains(&graphic.motion_in.as_str()) {
        let start_scale = if graphic.motion_in == "zoom-blur" { 1.45 } else { 0.84 };
        size_factor = format!("({size_factor})*if(lt(t,{:.6}),{start_scale:.2}+(1-{start_scale:.2})*({enter_progress}),1)", start + motion);
    }
    if graphic.motion_out == "shrink" {
        size_factor = format!("({size_factor})*if(gt(t,{:.6}),0.55+0.45*({end:.6}-t)/{motion:.6},1)", end - motion);
    }
    if ["zoom-blur", "tracking"].contains(&graphic.motion_out.as_str()) {
        let end_scale = if graphic.motion_out == "zoom-blur" { 1.45 } else { 0.84 };
        size_factor = format!("({size_factor})*if(gt(t,{:.6}),{end_scale:.2}+(1-{end_scale:.2})*({exit_progress}),1)", end - motion);
    }
    if graphic.motion_loop == "pulse" {
        size_factor = format!("({size_factor})*(1+0.025*sin((t-{start:.6})*2*PI))");
    }
    if graphic.motion_loop == "breathe" {
        size_factor = format!("({size_factor})*(1+0.012*sin((t-{start:.6})*1.4*PI))");
    }
    let font_size = if size_factor == "1" { format!("{base_size:.1}") } else { format!("'{base_size:.1}*({size_factor})'") };
    let stroke = graphic.stroke_width.clamp(0.0, 12.0);
    let stroke_color = color_with_opacity(&graphic.stroke_color, 1.0, "000000ff");
    let shadow_color = color_with_opacity(&graphic.shadow_color, 0.85, "000000dd");
    let mut decoration = format!(":borderw={stroke:.1}:bordercolor={stroke_color}");
    if graphic.shadow_offset > 0.0 {
        let offset = graphic.shadow_offset.clamp(0.0, 20.0);
        decoration.push_str(&format!(":shadowcolor={shadow_color}:shadowx={offset:.1}:shadowy={offset:.1}"));
    } else if graphic.motion_in == "magic" || graphic.motion_out == "magic"
        || graphic.motion_loop == "flicker" || graphic.motion_loop == "shimmer" {
        decoration.push_str(&format!(":shadowcolor={accent}:shadowx=3:shadowy=3"));
    } else if graphic.motion_in == "glitch" || graphic.motion_out == "glitch" {
        decoration.push_str(&format!(":shadowcolor={accent}:shadowx=5:shadowy=0"));
    }
    (alpha, x, y, format!("{font_size}{decoration}"))
}

async fn render_graphics(
    input_path: &Path,
    output_path: &str,
    graphics: &[RenderGraphic],
    encode: &EncodeSettings,
    timeline_duration: f64,
) -> Result<Vec<String>, String> {
    if graphics.is_empty() {
        let args = vec![
            "-hide_banner".into(), "-y".into(), "-i".into(), input_path.to_string_lossy().into_owned(),
            "-c".into(), "copy".into(), "-t".into(), format!("{timeline_duration:.9}"),
            "-movflags".into(), "+faststart".into(), output_path.into(),
        ];
        run_ffmpeg(&args).await?;
        return Ok(args);
    }

    let mut filters = String::new();
    let mut last_label = "0:v".to_owned();
    for (index, graphic) in graphics.iter().enumerate() {
        if graphic.duration_seconds <= 0.0 { continue; }
        let start = graphic.start_seconds.max(0.0);
        let end = start + graphic.duration_seconds;
        let opacity = graphic.opacity.clamp(0.0, 1.0);
        let bg = color_with_opacity(
            &graphic.background_color, opacity * graphic.background_opacity.clamp(0.0, 1.0), "10131ae6",
        );
        let accent = color_with_opacity(&graphic.accent_color, opacity, "ff6b4aff");
        let text_color = color_with_opacity(&graphic.text_color, opacity, "ffffffff");
        let (box_x, box_y, box_w, box_h, text_x, text_y, subtitle_y) = match graphic.template.as_str() {
            "title" => ("iw*0.10", "ih*0.33", "iw*0.80", "ih*0.34", "(w-text_w)/2", "h*0.43", "h*0.54"),
            "banner" => ("0", "ih*0.76", "iw", "ih*0.24", "w*0.055", "h*0.79", "h*0.88"),
            "bug" => ("iw*0.79", "ih*0.05", "iw*0.17", "ih*0.10", "w*0.815", "h*0.075", "h*0.13"),
            "caption" => ("iw*0.10", "ih*0.80", "iw*0.80", "ih*0.13", "(w-text_w)/2", "h*0.835", "h*0.90"),
            _ => ("iw*0.05", "ih*0.72", "iw*0.62", "ih*0.19", "w*0.075", "h*0.755", "h*0.835"),
        };
        let box_label = format!("g{index}box");
        let accent_label = format!("g{index}accent");
        let text_label = format!("g{index}text");
        let final_label = format!("g{index}out");
        let enable = format!("between(t,{start:.6},{end:.6})");
        let backdrop = if graphic.backdrop.is_empty() { "box" } else { graphic.backdrop.as_str() };
        let padding = graphic.background_padding.clamp(0.0, 64.0);
        let padded_x = format!("({box_x})-{padding:.1}");
        let padded_y = format!("({box_y})-{padding:.1}");
        let padded_w = format!("({box_w})+{:.1}", padding * 2.0);
        let padded_h = format!("({box_h})+{:.1}", padding * 2.0);
        match backdrop {
            "none" => filters.push_str(&format!("[{last_label}]null[{accent_label}];")),
            "outline" => filters.push_str(&format!(
                "[{last_label}]drawbox=x={padded_x}:y={padded_y}:w={padded_w}:h={padded_h}:color={accent}:t={:.1}:enable='{enable}'[{accent_label}];",
                graphic.background_border_width.clamp(1.0, 12.0),
            )),
            "underline" => filters.push_str(&format!(
                "[{last_label}]drawbox=x={padded_x}:y=({box_y})+({box_h})+{padding:.1}:w={padded_w}:h={:.1}:color={accent}:t=fill:enable='{enable}'[{accent_label}];",
                graphic.background_border_width.clamp(2.0, 12.0),
            )),
            _ => {
                filters.push_str(&format!(
                    "[{last_label}]drawbox=x={padded_x}:y={padded_y}:w={padded_w}:h={padded_h}:color={bg}:t=fill:enable='{enable}'[{box_label}];"
                ));
                filters.push_str(&format!(
                    "[{box_label}]drawbox=x={padded_x}:y={padded_y}:w=iw*0.008:h={padded_h}:color={accent}:t=fill:enable='{enable}'[{accent_label}];"
                ));
            }
        }
        let motion = graphic.motion_duration.clamp(0.05, 3.0).min(graphic.duration_seconds / 2.0);
        let (alpha, animated_x, animated_y, font_options) = text_motion_options(
            graphic, start, end, motion, text_x, text_y, &accent,
        );
        let segments = text_motion_segments(graphic, start, end, motion);
        let font_file = graphic_font_file(graphic);
        let mut text_input = accent_label.clone();
        for (segment_index, (segment, segment_start, segment_end)) in segments.iter().enumerate() {
            let segment_label = if segment_index + 1 == segments.len() {
                text_label.clone()
            } else { format!("g{index}text{segment_index}") };
            let segment_enable = format!("between(t,{segment_start:.6},{segment_end:.6})");
            filters.push_str(&format!(
                "[{text_input}]drawtext=fontfile='{font_file}':text='{}':fontcolor={}:fontsize={font_options}:line_spacing={:.1}:x='{animated_x}':y='{animated_y}':alpha='{alpha}':enable='{segment_enable}'[{segment_label}];",
                escape_drawtext(segment), text_color,
                graphic.line_spacing.clamp(0.0, 40.0),
            ));
            text_input = segment_label;
        }
        if segments.is_empty() {
            filters.push_str(&format!("[{accent_label}]null[{text_label}];"));
        }
        if graphic.subtitle.trim().is_empty() || graphic.template == "bug" {
            filters.push_str(&format!("[{text_label}]null[{final_label}];"));
        } else {
            filters.push_str(&format!(
                "[{text_label}]drawtext=fontfile='{font_file}':text='{}':fontcolor={}:fontsize={:.1}:line_spacing={:.1}:borderw={:.1}:bordercolor={}:shadowcolor={}:shadowx={:.1}:shadowy={:.1}:x={}:y={}:alpha='{}':enable='{}'[{}];",
                escape_drawtext(&graphic.subtitle), text_color, (graphic.font_size * 0.48).clamp(12.0, 96.0),
                graphic.line_spacing.clamp(0.0, 40.0), graphic.stroke_width.clamp(0.0, 8.0) * 0.5,
                color_with_opacity(&graphic.stroke_color, 1.0, "000000ff"),
                color_with_opacity(&graphic.shadow_color, 0.85, "000000dd"),
                graphic.shadow_offset.clamp(0.0, 20.0) * 0.6, graphic.shadow_offset.clamp(0.0, 20.0) * 0.6,
                text_x, subtitle_y, alpha, enable, final_label
            ));
        }
        last_label = final_label;
    }
    // No need to trim a trailing separator: `tag_video_range` needs one and adds it if missing.
    // `drawbox` with an alpha colour takes this graph through RGB even though the filter text
    // never says so, so the conversion out is named rather than left to swscale.
    let last_label = tag_video_range(&mut filters, &last_label);
    let mut args: Vec<String> = vec![
        "-hide_banner".into(), "-y".into(), "-i".into(), input_path.to_string_lossy().into_owned(),
        "-filter_complex".into(), filters, "-map".into(), format!("[{last_label}]"), "-map".into(), "0:a:0".into(),
    ];
    args.extend(encode.video_args(false));
    args.extend([
        "-c:a".into(), "copy".into(),
        "-t".into(), format!("{timeline_duration:.9}"), "-movflags".into(), "+faststart".into(), output_path.into(),
    ]);
    run_ffmpeg(&args).await?;
    Ok(args)
}

async fn render_audio_mix(
    input_path: &Path,
    output_path: &Path,
    clips: &[RenderAudioClip],
    encode: &EncodeSettings,
    timeline_duration: f64,
) -> Result<Vec<String>, String> {
    let mut args = vec![
        "-hide_banner".into(), "-y".into(), "-i".into(), input_path.to_string_lossy().into_owned(),
    ];
    let mut usable = Vec::new();
    for clip in clips {
        if clip.duration_seconds <= 0.0 { continue; }
        if !Path::new(&clip.path).is_file() { return Err(format!("audio source is missing: {}", clip.path)); }
        args.extend([
            "-ss".into(), format!("{:.6}", clip.source_in_seconds.max(0.0)),
            "-i".into(), clip.path.clone(),
        ]);
        usable.push(clip);
    }
    if usable.is_empty() {
        args.extend(["-c".into(), "copy".into(), output_path.to_string_lossy().into_owned()]);
        run_ffmpeg(&args).await?;
        return Ok(args);
    }

    let mut filter = format!("[0:a:0]aresample=48000,apad,atrim=duration={timeline_duration:.9}[base]");
    let mut inputs = "[base]".to_owned();
    for (index, clip) in usable.iter().enumerate() {
        let input_index = index + 1;
        let delay_ms = (clip.start_seconds.max(0.0) * 1000.0).round() as u64;
        let label = format!("a{input_index}");
        let speed_rate = clip.speed_rate.clamp(0.25, 4.0);
        let source_duration = if clip.source_duration_seconds > 0.0 { clip.source_duration_seconds } else { clip.duration_seconds * speed_rate };
        let rendered_duration = (source_duration / speed_rate).max(0.0);
        let fade_in = clip.fade_in_seconds.clamp(0.0, rendered_duration);
        let fade_out = clip.fade_out_seconds.clamp(0.0, rendered_duration);
        let fade_out_start = (rendered_duration - fade_out).max(0.0);
        let pan = clip.pan.clamp(-1.0, 1.0);
        let left = if pan > 0.0 { 1.0 - pan } else { 1.0 };
        let right = if pan < 0.0 { 1.0 + pan } else { 1.0 };
        let mut chain = format!(
            "aresample=48000,atrim=duration={:.6},asetpts=PTS-STARTPTS,{},volume={:.3}dB,aformat=channel_layouts=stereo,pan=stereo|c0={left:.6}*c0|c1={right:.6}*c1",
            source_duration, atempo_filter(speed_rate), clip.volume_db.clamp(-60.0, 24.0)
        );
        if fade_in > 0.0 { chain.push_str(&format!(",afade=t=in:st=0:d={fade_in:.6}")); }
        if fade_out > 0.0 { chain.push_str(&format!(",afade=t=out:st={fade_out_start:.6}:d={fade_out:.6}")); }
        chain.push_str(&format!(",adelay={delay_ms}|{delay_ms}"));
        filter.push_str(&format!(";[{input_index}:a:0]{chain}[{label}]"));
        inputs.push_str(&format!("[{label}]"));
    }
    filter.push_str(&format!(
        ";{inputs}amix=inputs={}:duration=longest:dropout_transition=0:normalize=0[aout]",
        usable.len() + 1
    ));
    args.extend([
        "-filter_complex".into(), filter,
        "-map".into(), "0:v:0".into(), "-map".into(), "[aout]".into(),
        "-c:v".into(), "copy".into(),
    ]);
    args.extend(encode.audio_args());
    args.extend([
        "-t".into(), format!("{timeline_duration:.9}"),
        "-movflags".into(), "+faststart".into(),
        output_path.to_string_lossy().into_owned(),
    ]);
    run_ffmpeg(&args).await?;
    Ok(args)
}

async fn render_overlays(
    input_path: &Path,
    output_path: &Path,
    clips: &[RenderOverlayClip],
    request: &TimelineExportRequest,
    timeline_duration: f64,
    // Whether anything re-encodes the video after this. Only the pass that actually writes the
    // delivered picture may spend the user's quality setting; every earlier one runs near-lossless.
    // `render_audio_mix` does not count -- it is `-c:v copy` and never touches the picture.
    intermediate: bool,
) -> Result<Vec<String>, String> {
    let encode = EncodeSettings::from_timeline(request);
    let usable = clips.iter().filter(|clip| clip.duration_seconds > 0.0).collect::<Vec<_>>();
    if usable.is_empty() {
        let args = vec![
            "-hide_banner".into(), "-y".into(), "-i".into(), input_path.to_string_lossy().into_owned(),
            "-c".into(), "copy".into(), output_path.to_string_lossy().into_owned(),
        ];
        run_ffmpeg(&args).await?;
        return Ok(args);
    }

    let mut args = vec!["-hide_banner".into(), "-y".into(), "-i".into(), input_path.to_string_lossy().into_owned()];
    for clip in &usable {
        if !Path::new(&clip.path).is_file() { return Err(format!("overlay source is missing: {}", clip.path)); }
        args.extend([
            "-ss".into(), format!("{:.9}", clip.source_in_seconds.max(0.0)),
            "-i".into(), clip.path.clone(),
        ]);
    }

    let mut filter = format!(
        "[0:v]setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration={timeline_duration:.9},trim=duration={timeline_duration:.9}[vbase];\
         [0:a]aresample=48000,apad,atrim=duration={timeline_duration:.9}[abase];"
    );
    let mut video_label = "vbase".to_owned();
    let mut audio_labels = vec!["[abase]".to_owned()];
    for (index, clip) in usable.iter().enumerate() {
        let input = index + 1;
        let box_width = (clip.box_width.clamp(0.01, 1.0) * request.width as f64).round().max(2.0) as u32;
        let box_height = (clip.box_height.clamp(0.01, 1.0) * request.height as f64).round().max(2.0) as u32;
        let x = (clip.box_x.clamp(0.0, 1.0) * request.width as f64).round();
        let y = (clip.box_y.clamp(0.0, 1.0) * request.height as f64).round();
        let speed = clip.speed_rate.clamp(0.25, 4.0);
        let source_duration = if clip.source_duration_seconds > 0.0 { clip.source_duration_seconds } else { clip.duration_seconds * speed };
        // Same `eq` trap as the segment path, with a different symptom. `pad=...:color=black@0`
        // asks for TRANSPARENT bars, but `eq` cannot process an alpha format, so ffmpeg inserts a
        // conversion that drops the alpha and the bars become OPAQUE black -- a contained overlay
        // then punches a black hole in the programme behind it instead of letterboxing over it.
        // Verified: base shows through after this reorder, was solid black before. So scale and
        // grade first, and only then move to RGBA and pad.
        let (fit_scale, fit_pad) = match clip.fit.as_str() {
            "stretch" => (format!("scale={box_width}:{box_height}"), String::new()),
            "cover" => (
                format!("scale={box_width}:{box_height}:force_original_aspect_ratio=increase,crop={box_width}:{box_height}"),
                String::new(),
            ),
            _ => (
                format!("scale={box_width}:{box_height}:force_original_aspect_ratio=decrease"),
                format!(",pad={box_width}:{box_height}:(ow-iw)/2:(oh-ih)/2:color=black@0"),
            ),
        };
        let effects = &clip.effects;
        let overlay_start = clip.start_seconds.max(0.0);
        let mut chain = format!(
            "trim=duration={source_duration:.9},setpts=(PTS-STARTPTS)/{speed:.9},{fit_scale},eq=brightness={:.6}:contrast={:.6}:saturation={:.6}:gamma={:.6},hue=h={:.6}",
            effects.brightness.clamp(-1.0, 1.0), effects.contrast.clamp(0.0, 2.0),
            effects.saturation.clamp(0.0, 3.0), effects.gamma.clamp(0.1, 3.0), effects.hue.clamp(-180.0, 180.0),
        );
        if effects.blur > 0.001 { chain.push_str(&format!(",gblur=sigma={:.3}", effects.blur.clamp(0.0, 20.0))); }
        if effects.sharpen > 0.001 { chain.push_str(&format!(",unsharp=5:5:{:.3}:5:5:0", effects.sharpen.clamp(0.0, 2.0))); }
        if effects.vignette > 0.001 { chain.push_str(&format!(",vignette=PI/{:.3}", 9.0 - effects.vignette.clamp(0.0, 1.0) * 6.5)); }
        if effects.noise > 0.001 { chain.push_str(&format!(",noise=alls={:.3}:allf=t+u", effects.noise.clamp(0.0, 50.0))); }
        chain.push_str(&format!(",format=rgba{fit_pad},colorchannelmixer=aa={:.6}", clip.opacity.clamp(0.0, 1.0)));
        let border = (clip.border_width.clamp(0.0, 0.1) * request.width as f64).round() as u32;
        if border > 0 {
            chain.push_str(&format!(",drawbox=x=0:y=0:w=iw:h=ih:color={}:t={border}", color_with_opacity(&clip.border_color, 1.0, "00f0ffff")));
        }
        // Give framesync a real, transparent secondary frame for the whole programme. Relying on
        // eof_action=pass alone is racy in some Windows FFmpeg builds and can truncate the master
        // at the overlay's final frame.
        chain.push_str(&format!(
            ",tpad=start_mode=add:start_duration={overlay_start:.9}:stop_mode=add:stop_duration={timeline_duration:.9}:color=black@0,trim=duration={timeline_duration:.9},setpts=PTS-STARTPTS"
        ));
        filter.push_str(&format!("[{input}:v]{chain}[ov{index}];[{video_label}][ov{index}]overlay=x={x:.0}:y={y:.0}:eof_action=pass:repeatlast=0:shortest=0[v{}];", index + 1));
        video_label = format!("v{}", index + 1);

        if clip.has_audio {
            let delay = (clip.start_seconds.max(0.0) * 1000.0).round() as u64;
            filter.push_str(&format!(
                "[{input}:a]aresample=48000,atrim=duration={source_duration:.9},asetpts=PTS-STARTPTS,{},volume={:.3}dB,adelay={delay}|{delay},apad,atrim=duration={timeline_duration:.9}[oa{index}];",
                atempo_filter(speed), clip.volume_db.clamp(-60.0, 24.0),
            ));
            audio_labels.push(format!("[oa{index}]"));
        }
    }
    filter.push_str(&format!(
        "[{video_label}]tpad=stop_mode=clone:stop_duration={timeline_duration:.9},trim=duration={timeline_duration:.9},setpts=PTS-STARTPTS[vfinal];"
    ));
    video_label = "vfinal".to_owned();
    if audio_labels.len() > 1 {
        filter.push_str(&format!("{}amix=inputs={}:duration=first:dropout_transition=0:normalize=0[aout]", audio_labels.join(""), audio_labels.len()));
    } else {
        filter.push_str("[abase]anull[aout]");
    }
    let video_label = tag_video_range(&mut filter, &video_label);
    args.extend([
        "-filter_complex".into(), filter, "-map".into(), format!("[{video_label}]"), "-map".into(), "[aout]".into(),
    ]);
    args.extend(encode.video_args(intermediate));
    args.extend(encode.audio_args());
    args.extend([
        "-t".into(), format!("{timeline_duration:.9}"), "-movflags".into(), "+faststart".into(),
        output_path.to_string_lossy().into_owned(),
    ]);
    run_ffmpeg(&args).await?;
    Ok(args)
}

async fn render_layers(
    input_path: &Path,
    output_path: &str,
    render_dir: &Path,
    request: &TimelineExportRequest,
    timeline_duration: f64,
) -> Result<Vec<String>, String> {
    let encode = EncodeSettings::from_timeline(request);
    if std::env::var_os("AIV_KEEP_RENDER_CACHE").is_some() {
        eprintln!("AIV layer duration: {timeline_duration:.9}s");
    }
    let has_after_overlays = !request.audio_clips.is_empty() || !request.graphics.is_empty();
    let overlay_output = if has_after_overlays {
        render_dir.join("base-with-overlays.mp4")
    } else {
        PathBuf::from(output_path)
    };
    let after_overlays = if request.overlay_clips.is_empty() {
        input_path.to_path_buf()
    } else {
        // Graphics is the only later pass that re-encodes the picture.
        let command = render_overlays(
            input_path, &overlay_output, &request.overlay_clips, request, timeline_duration,
            !request.graphics.is_empty(),
        ).await?;
        if !has_after_overlays { return Ok(command); }
        overlay_output
    };

    let audio_output = if request.graphics.is_empty() {
        PathBuf::from(output_path)
    } else {
        render_dir.join("base-with-a1-audio.mp4")
    };
    let mut audio_args = None;
    let layered_input = if request.audio_clips.is_empty() {
        after_overlays.clone()
    } else {
        audio_args = Some(render_audio_mix(
            &after_overlays, &audio_output, &request.audio_clips, &encode, timeline_duration,
        ).await?);
        audio_output.clone()
    };
    if request.graphics.is_empty() {
        if request.audio_clips.is_empty() {
            return render_graphics(&after_overlays, output_path, &[], &encode, timeline_duration).await;
        }
        return Ok(audio_args.expect("audio render command"));
    }
    render_graphics(&layered_input, output_path, &request.graphics, &encode, timeline_duration).await
}

/// Run the resolved upscale chain over the finished master.
///
/// Both stages are a *second* pass over an already-encoded file, which is a generation loss and is
/// the design as it stands rather than an accident: scaling inside the primary render graph would
/// give one encode instead of two, but the graph is assembled per-segment and the upscale applies
/// to the concatenated result. Recorded here so the cost is a known trade rather than a surprise.
///
/// The ordering rule -- AI first, resample second -- is enforced by construction: the AI stage runs
/// before the resample stage in this function and there is no path that reverses them.
pub(crate) async fn run_upscale_chain(
    master_path: &Path,
    work_dir: &Path,
    chain: &crate::ai::upscale::UpscaleChain,
    encode: &EncodeSettings,
    fps: f64,
) -> Result<Vec<String>, String> {
    use crate::ai::upscale;

    // Unconditional, and a handful of lines per export rather than per frame. An upscale that does
    // nothing used to be indistinguishable from one that worked: this stage sits behind minutes of
    // rendering, so "run it again and watch" is an expensive way to find out what it decided.
    eprintln!(
        "[upscale] chain: backend={} model={:?} quality={} tile={} resample={} target={}x{} codec={}",
        chain.ai_backend, chain.ai_model_file, chain.ai_quality, chain.tile, chain.resample,
        chain.width, chain.height, encode.video_codec,
    );
    if let Err(error) = chain.validate(&encode.video_codec) {
        eprintln!("[upscale] rejected before any work: {error}");
        return Err(error);
    }
    let mut last_args: Vec<String> = Vec::new();
    let mut current = master_path.to_path_buf();

    // An AI stage runs when a weight file is named (Real-ESRGAN) or the backend carries its own
    // weights (nvvfx). Only the resample-only chain has neither.
    let has_ai = chain.ai_model_file.is_some() || chain.ai_backend == "nvvfx";
    eprintln!("[upscale] ai stage: {}", if has_ai { "yes" } else { "no -- resample only" });
    if has_ai {
        let ai_path = work_dir.join("ai-upscaled-master.mp4");
        last_args = upscale::ai_upscale_video(
            &current, &ai_path, work_dir, chain, fps, encode.video_args(false),
        ).await?;
        current = ai_path;
        eprintln!("[upscale] ai stage wrote {}", current.display());
    }

    // Probe the *encoded* geometry rather than assuming it: the AI stage produced whatever its
    // weights are trained for, and only a comparison against the requested size can say whether a
    // resample is still owed. Skipping a no-op resample skips a whole extra encode.
    let probed = crate::media::probe(&current.to_string_lossy()).await;
    if let Err(error) = &probed {
        // Not fatal, but it must not pass for agreement: falling through with (0, 0) makes the
        // size comparison below trivially true and hides the fact that nothing was measured.
        eprintln!("[upscale] could not probe {}: {error}", current.display());
    }
    let (actual_width, actual_height) = probed
        .map(|meta| (meta.width.unwrap_or(0), meta.height.unwrap_or(0)))
        .unwrap_or((0, 0));
    eprintln!(
        "[upscale] after the ai stage the master is {actual_width}x{actual_height}, target is {}x{}",
        chain.width, chain.height,
    );
    if actual_width != chain.width || actual_height != chain.height {
        let resampled = work_dir.join("resampled-master.mp4");
        last_args = upscale::resample_cuda(
            &current, &resampled, chain.width, chain.height, &chain.resample,
            encode.video_args(false), &encode.video_codec,
        ).await?;
        current = resampled;
        eprintln!("[upscale] resample wrote {}", current.display());
    }

    if current == master_path {
        // Nothing ran. Reporting Ok here is what made a failed upscale look like a successful one:
        // the export completed, the file was the size it always was, and nothing was said. The
        // only honest reading is that the chain resolved to no work, which is a fault in the
        // resolver or in the payload, not a success.
        return Err(format!(
            "the chain resolved to no work at all -- the master is already {actual_width}x{actual_height} \
             and the target is {}x{}, so neither stage had anything to do",
            chain.width, chain.height,
        ));
    }
    fs::copy(&current, master_path)
        .map_err(|error| format!("failed to publish the upscaled master: {error}"))?;

    // Verify the outcome rather than trusting the steps. Every silent failure this path has had so
    // far was a step that reported success without changing the picture, so the last word belongs
    // to a measurement of the file that is actually about to be delivered.
    match crate::media::probe(&master_path.to_string_lossy()).await {
        Ok(meta) => {
            let (width, height) = (meta.width.unwrap_or(0), meta.height.unwrap_or(0));
            eprintln!("[upscale] published master is {width}x{height}");
            if width != chain.width || height != chain.height {
                return Err(format!(
                    "the chain ran but the master came out {width}x{height} instead of {}x{}",
                    chain.width, chain.height,
                ));
            }
        }
        Err(error) => eprintln!("[upscale] could not verify the published master: {error}"),
    }
    Ok(last_args)
}

pub async fn export_timeline(request: TimelineExportRequest) -> Result<ExportResult, String> {
    let mut request = request;
    let encode = EncodeSettings::from_timeline(&request);
    let mut warnings: Vec<String> = Vec::new();
    let fps = request.frame_rate.validate()?.as_f64();
    if request.clips.is_empty() { return Err("the primary video track has no enabled clips".into()); }
    if request.width == 0 || request.height == 0 || fps <= 0.0 { return Err("invalid export settings".into()); }
    let final_output = PathBuf::from(&request.output_path);
    if let Some(parent) = final_output.parent() { fs::create_dir_all(parent).map_err(|e| e.to_string())?; }
    let render_dir = std::env::temp_dir().join(format!("aiv-render-{}", Uuid::new_v4()));
    fs::create_dir_all(&render_dir).map_err(|e| e.to_string())?;

    // A payload from before the chain existed carries only `upscaleFactor`, so translate it into
    // the same structure rather than keeping a second code path alive to service it.
    let chain = request.upscale.clone().or_else(|| {
        (request.upscale_factor > 1).then(|| {
            let factor = request.upscale_factor.clamp(1, 4);
            crate::ai::upscale::UpscaleChain {
                ai_backend: "esrgan".to_owned(),
                ai_model_file: None,
                ai_quality: "HIGH".to_owned(),
                tile: 512,
                resample: "lanczos".to_owned(),
                width: (request.width.saturating_mul(factor) / 2) * 2,
                height: (request.height.saturating_mul(factor) / 2) * 2,
            }
        })
    });

    // The chain is a SECOND pass over the finished master, and an AI stage over every frame of a
    // timeline takes minutes. Rendering straight to the user's chosen path would drop a
    // finished-looking file there long before it is finished -- which is exactly what happened:
    // the output appeared at its original size while the job was still running, and looked like a
    // broken upscale rather than an unfinished one. So when a chain is queued the render goes to a
    // staging file and the destination is written exactly once, at the end.
    let staging = chain.as_ref().map(|_| {
        let extension = final_output.extension().and_then(|value| value.to_str()).unwrap_or("mp4");
        render_dir.join(format!("master-pre-upscale.{extension}"))
    });
    if let Some(path) = &staging {
        request.output_path = path.to_string_lossy().into_owned();
    }
    let output_path = staging.clone().unwrap_or_else(|| final_output.clone());

    let result = async {
        let frame = 1.0 / fps;
        let mut prepared = Vec::<PreparedSegment>::new();
        let mut timeline_cursor = 0.0_f64;
        let mut ordered_clips = request.clips.iter().collect::<Vec<_>>();
        ordered_clips.sort_by(|left, right| left.start_seconds.total_cmp(&right.start_seconds));
        for (index, clip) in ordered_clips.into_iter().enumerate() {
            if !Path::new(&clip.path).is_file() { return Err(format!("source media is missing: {}", clip.path)); }
            if clip.duration_seconds <= 0.0 { continue; }
            let clip_start = clip.start_seconds.max(0.0);
            let mut overlap = (timeline_cursor - clip_start).max(0.0);
            if clip_start > timeline_cursor + frame / 2.0 {
                let gap_duration = clip_start - timeline_cursor;
                let gap_path = render_dir.join(format!("gap-{index:04}.mp4"));
                render_gap(&gap_path, gap_duration, request.width, request.height, fps, &encode)
                    .await.map_err(|error| format!("timeline gap failed to render:\n{error}"))?;
                prepared.push(PreparedSegment {
                    path: gap_path,
                    duration: gap_duration,
                    transition_in_kind: "none".into(),
                    transition_in_duration: 0.0,
                    transition_out_kind: "none".into(),
                    transition_out_duration: 0.0,
                });
                timeline_cursor = clip_start;
                overlap = 0.0;
            } else if overlap > frame / 2.0
                && clip.transition_kind == "none"
                && prepared.last().map(|s| s.transition_out_kind == "none").unwrap_or(true)
            {
                if let Some(previous) = prepared.last_mut() {
                    let duration = (previous.duration - overlap).max(frame);
                    let trimmed = render_dir.join(format!("hard-cut-{index:04}.mp4"));
                    trim_prepared_segment(&previous.path, &trimmed, duration, &encode).await
                        .map_err(|error| format!("hard-cut overlap failed to trim:\n{error}"))?;
                    previous.path = trimmed;
                    previous.duration = duration;
                }
                timeline_cursor = clip_start;
                overlap = 0.0;
            }
            let segment = render_dir.join(format!("segment-{index:04}.mp4"));
            let video_segment = render_dir.join(format!("segment-video-{index:04}.mp4"));
            let audio_segment = render_dir.join(format!("segment-audio-{index:04}.m4a"));
            let scale = (clip.scale_percent / 100.0).clamp(0.01, 10.0);
            let opacity = clip.opacity.clamp(0.0, 1.0);
            let rotation = clip.rotation_degrees.to_radians();
            let speed_rate = clip.speed_rate.clamp(0.25, 4.0);
            let source_duration = if clip.source_duration_seconds > 0.0 {
                clip.source_duration_seconds
            } else {
                clip.duration_seconds * speed_rate
            };
            // ---- HEAD HANDLE ------------------------------------------------------------
            //
            // `xfade` consumes `D` seconds from BOTH neighbours, but the timeline butt-joins
            // clips, so the export used to come out `sum(D)` shorter than the timeline. Two
            // consequences: the incoming clip appeared to "restart" after a dissolve (it had
            // already spent its first D seconds inside the blend), and graphics, captions and A1
            // audio -- placed at absolute timeline times -- drifted EARLY in the export while
            // looking correct in the preview.
            //
            // The fix is to give the incoming clip a head handle: render it from `sourceIn - D`
            // so the D seconds the dissolve eats are frames BEFORE its in-point. When the blend
            // ends the clip is at exactly `sourceIn`, and the arithmetic closes:
            //
            //     segment.duration = D + visible          (this clip)
            //     cumulative      += segment.duration - D = visible
            //
            // so the running total matches the butt-joined timeline. The outgoing side needs no
            // handle -- its half of the dissolve is its own visible tail, which is what "consumes
            // duration from both" has always meant. The xfade block itself is unchanged.
            //
            // Speed matters: D is timeline seconds, and the source is trimmed before `setpts`
            // divides by the rate, so D timeline seconds is D*rate seconds of source.
            let speed_rate_for_handle = clip.speed_rate.clamp(0.25, 4.0);
            // Borrow only what the UI already proved exists. Anything it could not cover was taken
            // out of visible material there, and every track was shifted by the same amount, so by
            // the time the payload arrives here the timeline is already self-consistent.
            //
            // Nothing is ever frozen. Premiere is the only editor that invents frames for a
            // dissolve, and it stripes the result with a warning because it looks wrong -- worst of
            // all on short, motion-dense generated clips.
            let head_handle = clip.head_handle_seconds.max(0.0).min(clip.duration_seconds);
            let real_handle = (clip.source_in_seconds.max(0.0) / speed_rate_for_handle).min(head_handle);
            // A junction the UI had to ripple: no handle existed, so the dissolve was paid for out
            // of visible frames and the export is shorter than the timeline by that much. The
            // alternative was freezing, which is what this rework removed -- but a silent ripple is
            // its own kind of surprise, so say it plainly.
            let requested = clip.transition_duration.max(0.0);
            if clip.transition_kind != "none"
                && !clip.transition_kind.is_empty()
                && requested > frame
                && clip.head_handle_seconds + frame < requested
            {
                warnings.push(format!(
                    "clip {} has no media before its in-point, so {:.2}s of its dissolve came out of visible frames. The export is shorter than the timeline by that much.",
                    index + 1, requested - clip.head_handle_seconds.max(0.0),
                ));
            }

            let effects = &clip.effects;
            // Grade only. Fitting the picture into the frame is `conform_graph`'s job now, and
            // the user's zoom is applied after it so a crop rect stays in source coordinates.
            let mut video_filters = format!(
                "eq=brightness={:.6}:contrast={:.6}:saturation={:.6}:gamma={:.6},hue=h={:.6}",
                effects.brightness.clamp(-1.0, 1.0), effects.contrast.clamp(0.0, 2.0),
                effects.saturation.clamp(0.0, 3.0), effects.gamma.clamp(0.1, 3.0), effects.hue.clamp(-180.0, 180.0),
            );
            if effects.blur > 0.001 { video_filters.push_str(&format!(",gblur=sigma={:.3}", effects.blur.clamp(0.0, 20.0))); }
            if effects.sharpen > 0.001 { video_filters.push_str(&format!(",unsharp=5:5:{:.3}:5:5:0", effects.sharpen.clamp(0.0, 2.0))); }
            if effects.vignette > 0.001 {
                let divisor = 9.0 - effects.vignette.clamp(0.0, 1.0) * 6.5;
                video_filters.push_str(&format!(",vignette=PI/{divisor:.3}"));
            }
            if effects.noise > 0.001 { video_filters.push_str(&format!(",noise=alls={:.3}:allf=t+u", effects.noise.clamp(0.0, 50.0))); }
            // Two separate defects here produced the green bars, and BOTH have to be fixed.
            //
            // 1. `ow=rotw(iw):oh=roth(ih)` is simply wrong. `rotw()`/`roth()` take an ANGLE in
            //    radians and return the canvas needed to contain the input rotated by it; this
            //    passed the WIDTH and HEIGHT as the angle. For a 1620x1080 picture that evaluates
            //    to `rotw(1620 rad)` = 1733 and `roth(1080 rad)` = 1874 -- a canvas half again too
            //    big, produced even at zero rotation, and pure coincidence that it was ever close.
            // 2. That oversized margin is filled by `c=none`. On a format with no alpha -- which is
            //    what `eq` inside `video_filters` leaves behind -- "transparent" writes ZEROS, and
            //    YUV (0,0,0) is GREEN, not black.
            //
            // Measured on the user's own 768x512 clip at zero rotation: green columns 92..1824,
            // exactly the ~1733-wide canvas predicted above, clean once both are fixed. Skipping
            // the filter entirely at zero rotation is also a straight saving -- it was rotating a
            // canvas 1733x1874 instead of 1620x1080 on every clip in every export.
            let rotate_filter = if rotation.abs() < 1e-6 {
                String::new()
            } else {
                format!("rotate={rotation:.8}:c=none:ow=rotw({rotation:.8}):oh=roth({rotation:.8}),")
            };
            let conform = conform_graph(
                "src", "cf", &clip.conform, request.width, request.height,
                clip.duration_seconds, &format!("c{index}"), &video_filters,
            );
            // `colorchannelmixer` forces this chain through RGBA, so the conversion back out is
            // the one that has to be spelled out rather than left to swscale's BT.601 default.
            // This is the first pass every clip takes, so getting it wrong here miscolours the
            // whole export no matter what the later passes do.
            let to_video_range = RGB_TO_BT709_TV;
            let video_filter = format!(
                "[0:v:0]trim=start={:.6}:end={:.6},setpts=(PTS-STARTPTS)/{:.6}[src];\
                 {conform}\
                 [cf]scale=iw*{:.6}:ih*{:.6},{rotate_filter}colorchannelmixer=aa={:.6},setsar=1[fg];\
                 color=c=black:s={}x{}:r={}:d={:.6}[bg];\
                 [bg][fg]overlay=x={:.3}-overlay_w/2:y={:.3}-overlay_h/2:shortest=1,fps={},\
                 tpad=stop_mode=clone:stop_duration=0.100,trim=duration={:.6},setpts=PTS-STARTPTS,{to_video_range}[vout]",
                (clip.source_in_seconds.max(0.0) - real_handle * speed_rate).max(0.0),
                clip.source_in_seconds.max(0.0) + source_duration, speed_rate,
                scale, scale, opacity,
                request.width, request.height, fps, clip.duration_seconds + real_handle,
                clip.position_x, clip.position_y, fps, clip.duration_seconds + real_handle
            );
            let mut video_args: Vec<String> = vec![
                "-hide_banner".into(), "-y".into(), "-i".into(), clip.path.clone(),
                "-filter_complex".into(), video_filter, "-map".into(), "[vout]".into(),
                "-an".into(),
            ];
            video_args.extend(encode.video_args(true));
            video_args.push(video_segment.to_string_lossy().into_owned());
            run_ffmpeg(&video_args).await.map_err(|e| format!("clip {} video failed to render:\n{}", index + 1, e))?;

            // The audio segment has to match the video's length exactly -- handle included --
            // or every junction drifts by its own transition duration. `adelay` prepends the
            // frozen part as silence, mirroring the video's cloned first frame.
            let segment_duration = clip.duration_seconds + real_handle;
            let mut audio_args: Vec<String> = if clip.has_audio {
                vec![
                    "-hide_banner".into(), "-y".into(),
                    "-ss".into(), format!("{:.6}", (clip.source_in_seconds.max(0.0) - real_handle * speed_rate).max(0.0)),
                    "-i".into(), clip.path.clone(), "-t".into(), format!("{segment_duration:.6}"),
                    "-vn".into(), "-af".into(), format!(
                        "atrim=duration={:.6},asetpts=PTS-STARTPTS,{},volume={:.3}dB,apad",
                        source_duration + real_handle * speed_rate, atempo_filter(speed_rate), clip.volume_db,
                    ),
                ]
            } else {
                vec![
                    "-hide_banner".into(), "-y".into(), "-f".into(), "lavfi".into(),
                    "-i".into(), format!("anullsrc=channel_layout=stereo:sample_rate={}", encode.audio_sample_rate),
                    "-t".into(), format!("{segment_duration:.6}"),
                ]
            };
            // A segment is muxed into an intermediate .m4a, which cannot hold FLAC or raw PCM. The
            // delivery codec is applied on the final pass; forcing it this early would fail the
            // render on any of the lossless choices.
            audio_args.extend([
                "-c:a".into(), "aac".into(), "-b:a".into(), "320k".into(),
                "-ar".into(), encode.audio_sample_rate.clamp(8_000, 192_000).to_string(),
                audio_segment.to_string_lossy().into_owned(),
            ]);
            run_ffmpeg(&audio_args).await.map_err(|e| format!("clip {} audio failed to render:\n{}", index + 1, e))?;
            let mux_args = vec![
                "-hide_banner".into(), "-y".into(), "-i".into(), video_segment.to_string_lossy().into_owned(),
                "-i".into(), audio_segment.to_string_lossy().into_owned(), "-map".into(), "0:v:0".into(),
                "-map".into(), "1:a:0".into(), "-c".into(), "copy".into(), "-t".into(),
                format!("{segment_duration:.6}"), segment.to_string_lossy().into_owned(),
            ];
            run_ffmpeg(&mux_args).await.map_err(|e| format!("clip {} streams failed to mux:\n{}", index + 1, e))?;
            // A deliberate overlap on the timeline defines the transition length directly.
            let has_junction_transition = clip.transition_kind != "none"
                || prepared.last().map(|s| s.transition_out_kind != "none").unwrap_or(false);
            let transition_in_duration = if overlap > frame / 2.0 && has_junction_transition {
                overlap
            } else {
                clip.transition_duration
            };
            if overlap > frame / 2.0 && has_junction_transition {
                if let Some(previous) = prepared.last_mut() {
                    if previous.transition_out_kind != "none" {
                        previous.transition_out_duration = overlap;
                    }
                }
            }
            prepared.push(PreparedSegment {
                // Includes the head handle: `cumulative += duration - D` then lands exactly on the
                // butt-joined timeline position, which is what stops graphics and captions drifting.
                duration: segment_duration,
                path: segment,
                transition_in_kind: clip.transition_kind.clone(),
                // The achievable handle, not the requested one -- if the source could not supply
                // it and it was not frozen, the xfade must not ask for more than exists.
                // The junction is exactly as long as the material backing it. Where the UI had to
                // ripple, the clips already overlap on the timeline and `overlap` above has
                // supplied the length instead -- the two models no longer both claim the junction.
                transition_in_duration: if real_handle > 0.0 { real_handle } else { transition_in_duration },
                transition_out_kind: clip.transition_out_kind.clone(),
                transition_out_duration: if clip.transition_out_duration > 0.0 {
                    clip.transition_out_duration
                } else {
                    clip.transition_duration
                },
            });
            timeline_cursor = timeline_cursor.max(clip_start + clip.duration_seconds);
        }
        if prepared.is_empty() { return Err("the timeline contains no renderable clips".into()); }

        // The first clip's "in" and the last clip's "out" have no neighbour to blend with, so they
        // render as fades from and to a solid colour.
        let head_fade = {
            let first = &prepared[0];
            if first.transition_in_kind != "none" && !first.transition_in_kind.is_empty() {
                Some((fade_color(&first.transition_in_kind), first.transition_in_duration.max(frame)))
            } else {
                None
            }
        };
        let tail_fade = {
            let last = prepared.last().expect("at least one segment");
            if last.transition_out_kind != "none" && !last.transition_out_kind.is_empty() {
                Some((fade_color(&last.transition_out_kind), last.transition_out_duration.max(frame)))
            } else {
                None
            }
        };

        if prepared.len() == 1 && head_fade.is_none() && tail_fade.is_none() {
            let final_args = render_layers(
                &prepared[0].path, &request.output_path, &render_dir, &request, prepared[0].duration,
            ).await?;
            return Ok(ExportResult { output_path: request.output_path.clone(), command: final_args, warnings: warnings.clone() });
        }

        let mut final_args = vec!["-hide_banner".into(), "-y".into()];
        for segment in &prepared {
            final_args.extend(["-i".into(), segment.path.to_string_lossy().into_owned()]);
        }
        let mut filter = String::new();
        let mut cumulative = prepared[0].duration;
        let mut video_label = "0:v".to_owned();
        let mut audio_label = "0:a".to_owned();
        for index in 1..prepared.len() {
            let segment = &prepared[index];
            let (junction_kind, junction_duration) = resolve_junction(&prepared[index - 1], segment);
            let previous_video = format!("[{video_label}]");
            let previous_audio = format!("[{audio_label}]");
            if junction_kind == "none" {
                filter.push_str(&format!(
                    "{}{}[{}:v][{}:a]concat=n=2:v=1:a=1[v{}][a{}];",
                    previous_video, previous_audio, index, index, index, index
                ));
                cumulative += segment.duration;
            } else {
                // xfade consumes `duration` seconds from both neighbours, so it can never be longer
                // than either of them, and the offset is where the overlap begins on the running edit.
                let transition_duration = junction_duration
                    .max(frame)
                    .min((cumulative - frame).max(frame))
                    .min((segment.duration - frame).max(frame));
                let offset = (cumulative - transition_duration).max(0.0);
                filter.push_str(&format!(
                    "{}[{}:v]xfade=transition={}:duration={:.6}:offset={:.6}[v{}];{}[{}:a]acrossfade=d={:.6}[a{}];",
                    previous_video, index, transition_name(&junction_kind), transition_duration, offset, index,
                    previous_audio, index, transition_duration, index
                ));
                cumulative += segment.duration - transition_duration;
            }
            video_label = format!("v{index}");
            audio_label = format!("a{index}");
        }

        if head_fade.is_some() || tail_fade.is_some() {
            let mut video_chain = Vec::<String>::new();
            let mut audio_chain = Vec::<String>::new();
            if let Some((color, duration)) = &head_fade {
                let duration = duration.min((cumulative - frame).max(frame));
                video_chain.push(format!("fade=t=in:st=0:d={duration:.6}:color={color}"));
                audio_chain.push(format!("afade=t=in:st=0:d={duration:.6}"));
            }
            if let Some((color, duration)) = &tail_fade {
                let duration = duration.min((cumulative - frame).max(frame));
                let start = (cumulative - duration).max(0.0);
                video_chain.push(format!("fade=t=out:st={start:.6}:d={duration:.6}:color={color}"));
                audio_chain.push(format!("afade=t=out:st={start:.6}:d={duration:.6}"));
            }
            filter.push_str(&format!("[{video_label}]{}[vedge];", video_chain.join(",")));
            filter.push_str(&format!("[{audio_label}]{}[aedge];", audio_chain.join(",")));
            video_label = "vedge".to_owned();
            audio_label = "aedge".to_owned();
        }
        while filter.ends_with(';') { let _ = filter.pop(); }
        // The last pass decides what the delivered file actually contains, so it is the one that
        // must not hand swscale an unqualified conversion.
        let video_label = tag_video_range(&mut filter, &video_label);

        let has_layers = !request.graphics.is_empty() || !request.audio_clips.is_empty() || !request.overlay_clips.is_empty();
        // NOT `has_layers`. An A1 audio track adds a layer pass, but that pass is `-c:v copy` --
        // it never re-encodes the picture, so this is still the pass that writes the delivered
        // video and it has to honour the user's bitrate or CRF rather than the CRF 14 intermediate
        // profile. Getting this wrong shipped every music-over-timeline export at ~5-10x the
        // requested size with the quality setting silently discarded.
        let video_reencoded_later =
            !request.graphics.is_empty() || !request.overlay_clips.is_empty();
        let base_output = if has_layers { render_dir.join("base-with-transitions.mp4") } else { output_path.clone() };
        final_args.extend([
            "-filter_complex".into(), filter,
            "-map".into(), format!("[{video_label}]"),
            "-map".into(), format!("[{audio_label}]"),
        ]);
        final_args.extend(encode.video_args(video_reencoded_later));
        final_args.extend(encode.audio_args());
        final_args.extend([
            "-movflags".into(), "+faststart".into(), base_output.to_string_lossy().into_owned(),
        ]);
        run_ffmpeg(&final_args).await?;
        if !has_layers {
            Ok(ExportResult { output_path: request.output_path.clone(), command: final_args, warnings: warnings.clone() })
        } else {
            let layered_args = render_layers(
                &base_output, &request.output_path, &render_dir, &request, cumulative,
            ).await?;
            Ok(ExportResult { output_path: request.output_path.clone(), command: layered_args, warnings: warnings.clone() })
        }
    }.await;

    let result = match (result, chain) {
        (Ok(mut rendered), Some(chain)) => {
            let encode = EncodeSettings::from_timeline(&request);
            match run_upscale_chain(&output_path, &render_dir, &chain, &encode, fps).await {
                // `run_upscale_chain` now refuses to return Ok without having changed the master,
                // so an empty arg list here would be a contradiction rather than a no-op.
                Ok(args) => {
                    if !args.is_empty() { rendered.command = args; }
                    Ok(rendered)
                }
                // The master itself is finished. Losing it because the optional second pass
                // failed would be the worst possible outcome of asking for a bigger version, so
                // it is still published -- just at the size it was rendered at, and said so.
                Err(error) => {
                    eprintln!("[upscale] FAILED: {error}");
                    rendered.warnings.push(format!(
                        "The export finished at {}x{}, but the upscale did not run: {error}",
                        request.width, request.height
                    ));
                    Ok(rendered)
                }
            }
        }
        (other, _) => other,
    };

    // Publish once, whether or not the chain succeeded: a failed upscale still leaves a valid
    // master, and the user asked for a file at this path.
    let result = match (result, staging) {
        (Ok(mut rendered), Some(path)) => {
            match fs::copy(&path, &final_output) {
                Ok(_) => {
                    rendered.output_path = final_output.to_string_lossy().into_owned();
                    Ok(rendered)
                }
                Err(error) => Err(format!(
                    "the render finished but could not be written to {}: {error}",
                    final_output.display()
                )),
            }
        }
        (other, _) => other,
    };

    if std::env::var_os("AIV_KEEP_RENDER_CACHE").is_none() {
        let _ = fs::remove_dir_all(&render_dir);
    } else {
        eprintln!("AIV render cache retained at {}", render_dir.display());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command as StdCommand;

    fn segment(out_kind: &str, out_duration: f64, in_kind: &str, in_duration: f64) -> PreparedSegment {
        PreparedSegment {
            path: PathBuf::from("segment.mp4"),
            duration: 4.0,
            transition_in_kind: in_kind.into(),
            transition_in_duration: in_duration,
            transition_out_kind: out_kind.into(),
            transition_out_duration: out_duration,
        }
    }

    #[test]
    fn the_rgb_conversion_is_spelled_out_not_left_to_swscale() {
        // Tagging is not converting. `video_args` writes the BT.709 tags; without these scale
        // options swscale converts with BT.601 limited-range defaults and pure green round trips
        // as 0,215,0.
        assert!(RGB_TO_BT709_TV.contains("out_color_matrix=bt709"));
        assert!(RGB_TO_BT709_TV.ends_with("format=yuv420p"));
        // Each of these was measured and made the result worse. See the constant's docs.
        assert!(!RGB_TO_BT709_TV.contains("in_range"), "declaring the input range broke it");
        assert!(!RGB_TO_BT709_TV.contains("in_color_matrix"), "naming the input matrix broke it");
        assert!(!RGB_TO_BT709_TV.contains("colorspace="), "the colorspace filter was worse");
    }

    #[test]
    fn the_conversion_is_appended_once_with_its_own_label() {
        let mut filter = String::from("[0:v]null[vfinal];[0:a]anull[aout]");
        let label = tag_video_range(&mut filter, "vfinal");
        assert_eq!(label, "vcolor");
        let expected = format!("[vfinal]{conversion}[vcolor]", conversion = RGB_TO_BT709_TV);
        assert!(filter.ends_with(&expected), "{filter}");
        assert_eq!(filter.matches("[vcolor]").count(), 1);

        // A chain that already ends in a separator must not gain an empty one.
        let mut trailing = String::from("[0:v]null[v];");
        tag_video_range(&mut trailing, "v");
        assert!(!trailing.contains(";;"), "{trailing}");
    }

    fn encode_settings(codec: &str) -> EncodeSettings {
        EncodeSettings {
            video_codec: codec.into(),
            quality_mode: "quality".into(),
            quality: 20,
            preset: String::new(),
            video_bitrate: "12M".into(),
            audio_codec: "aac".into(),
            audio_bitrate: "320k".into(),
            audio_sample_rate: 48_000,
            audio_channels: 2,
        }
    }

    #[test]
    fn the_encoder_tags_match_what_the_conversion_produces() {
        let args = encode_settings("libx264").video_args(false);
        for (flag, value) in [
            ("-colorspace", "bt709"),
            ("-color_primaries", "bt709"),
            ("-color_trc", "bt709"),
            ("-color_range", "tv"),
            ("-pix_fmt", "yuv420p"),
        ] {
            let at = args.iter().position(|a| a == flag)
                .unwrap_or_else(|| panic!("{flag} is not tagged"));
            assert_eq!(args[at + 1], value, "{flag}");
        }
        // The NVENC family has to agree, or the two encoders disagree about what the pixels mean.
        let nvenc = encode_settings("h264_nvenc").video_args(true);
        assert!(nvenc.windows(2).any(|w| w[0] == "-color_range" && w[1] == "tv"));
    }

    #[test]
    fn junctions_fuse_the_outgoing_and_incoming_transitions() {
        let hard = segment("none", 0.0, "none", 0.0);
        assert_eq!(resolve_junction(&hard, &hard), ("none".to_owned(), 0.0));

        // Only the outgoing side is set: it drives the cut.
        let out_only = segment("wipeleft", 0.75, "none", 0.0);
        assert_eq!(resolve_junction(&out_only, &hard), ("wipeleft".to_owned(), 0.75));

        // Only the incoming side is set.
        let in_only = segment("none", 0.0, "circleopen", 0.4);
        assert_eq!(resolve_junction(&hard, &in_only), ("circleopen".to_owned(), 0.4));

        // Both set: the incoming style wins and neither duration is truncated.
        assert_eq!(resolve_junction(&out_only, &in_only), ("circleopen".to_owned(), 0.75));
    }

    #[test]
    fn head_and_tail_transitions_pick_a_dip_colour() {
        assert_eq!(fade_color("fadewhite"), "white");
        assert_eq!(fade_color("fadegrays"), "gray");
        assert_eq!(fade_color("wipeleft"), "black");
    }

    #[test]
    fn colors_and_text_are_safely_normalized() {
        assert_eq!(color_with_opacity("#ff000080", 0.5, "ffffffff"), "0xff0000@0.2510");
        assert_eq!(color_with_opacity("not-a-color", 1.0, "112233ff"), "0x112233@1.0000");
        assert_eq!(escape_drawtext("Director's cut: 50%"), "Director\\'s cut\\: 50\\%");
    }

    #[test]
    fn exported_typewriter_uses_one_segment_per_character_even_past_sixty() {
        let text = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-extra";
        assert!(text.chars().count() > 60);
        let graphic = RenderGraphic {
            text: text.into(), motion_in: "typewriter".into(), motion_out: "typewriter".into(),
            motion_duration: 1.0, duration_seconds: 4.0, ..Default::default()
        };
        let segments = text_motion_segments(&graphic, 0.0, 4.0, 1.0);
        let count = text.chars().count();
        let entering = &segments[..count];
        let exiting = &segments[count + 1..];
        assert_eq!(entering.len(), count);
        // The final empty state needs no drawtext node, so there are count-1 visible exit segments.
        assert_eq!(exiting.len(), count - 1);
        assert_eq!(entering[0].0.chars().count(), 1);
        assert_eq!(entering[count - 1].0.chars().count(), count);
        assert_eq!(exiting[0].0.chars().count(), count - 1);
        assert_eq!(exiting[count - 2].0.chars().count(), 1);
    }

    #[test]
    fn ffmpeg_renders_effects_and_timed_graphics() {
        if StdCommand::new("ffmpeg").arg("-version").output().is_err() {
            eprintln!("ffmpeg is unavailable; skipping render integration test");
            return;
        }
        let work = std::env::temp_dir().join(format!("aiv-export-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&work).unwrap();
        let input = work.join("input.mp4");
        let output = work.join("output.mp4");
        let make_source = StdCommand::new("ffmpeg").args([
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=1.2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
        ]).arg(&input).status().unwrap();
        assert!(make_source.success(), "failed to create test media");

        let request = TimelineExportRequest {
            clips: vec![RenderClip {
                path: input.to_string_lossy().into_owned(), start_seconds: 0.0, source_in_seconds: 0.0, duration_seconds: 1.0,
                source_duration_seconds: 1.0, speed_rate: 1.0,
                has_audio: true, position_x: 80.0, position_y: 45.0, scale_percent: 100.0,
                rotation_degrees: 0.0, opacity: 1.0, volume_db: -1.5, transition_kind: "none".into(),
                transition_duration: 0.0, transition_out_kind: "none".into(), transition_out_duration: 0.0,
                effects: RenderEffects {
                    brightness: 0.05, contrast: 1.1, saturation: 0.85, gamma: 1.05, hue: 8.0,
                    blur: 0.2, sharpen: 0.35, vignette: 0.25, noise: 2.0,
                },
                // Exercise the split/overlay blur-fill branch, which is the only conform path that
                // emits a multi-node subgraph and so the only one that can produce a broken
                // filtergraph rather than merely a wrong-looking picture.
                conform: RenderConform { fit: "contain".into(), pad: "blur".into(), pad_blur: Some(12.0), ..Default::default() },
                head_handle_seconds: 0.0,
            }, RenderClip {
                path: input.to_string_lossy().into_owned(), start_seconds: 1.5, source_in_seconds: 0.1, duration_seconds: 0.4,
                source_duration_seconds: 0.8, speed_rate: 2.0,
                has_audio: true, position_x: 80.0, position_y: 45.0, scale_percent: 100.0,
                rotation_degrees: 0.0, opacity: 1.0, volume_db: -3.0, transition_kind: "none".into(),
                transition_duration: 0.0, transition_out_kind: "none".into(), transition_out_duration: 0.0,
                effects: RenderEffects::default(),
                // A moving crop, so `crop_filter`'s interpolated `t` expression is parsed by a real
                // ffmpeg rather than only by eye.
                conform: RenderConform {
                    fit: "cover".into(),
                    crop: Some(ConformCrop {
                        start: SourceRect { x: 0.1, y: 0.1, width: 0.7, height: 0.7 },
                        end: Some(SourceRect { x: 0.0, y: 0.0, width: 1.0, height: 1.0 }),
                    }),
                    ..Default::default()
                },
                head_handle_seconds: 0.0,
            }],
            graphics: vec![RenderGraphic {
                start_seconds: 0.1, duration_seconds: 0.75, opacity: 0.9, template: "lower-third".into(),
                text: "Digital Forge".into(), subtitle: "Effects + graphics render".into(),
                text_color: "#ffffffff".into(), background_color: "#10131ae6".into(),
                accent_color: "#ff6b4aff".into(), font_size: 24.0,
                motion_in: "typewriter".into(), motion_out: "fade".into(),
                motion_duration: 0.2, motion_loop: "none".into(),
                ..Default::default()
            }, RenderGraphic {
                start_seconds: 1.5, duration_seconds: 0.35, opacity: 1.0, template: "caption".into(),
                text: "Timed caption on C1".into(), subtitle: "".into(), text_color: "#ffffffff".into(),
                background_color: "#000000cc".into(), accent_color: "#4b8ee8ff".into(), font_size: 22.0,
                motion_in: "pop".into(), motion_out: "shrink".into(),
                motion_duration: 0.1, motion_loop: "pulse".into(),
                ..Default::default()
            }, RenderGraphic {
                start_seconds: 0.4, duration_seconds: 0.4, opacity: 0.8, template: "bug".into(),
                text: "FLOAT".into(), subtitle: "".into(), text_color: "#ffffffff".into(),
                background_color: "#000000cc".into(), accent_color: "#00f0ffff".into(), font_size: 18.0,
                motion_in: "slide-up".into(), motion_out: "slide-down".into(),
                motion_duration: 0.1, motion_loop: "float".into(),
                ..Default::default()
            }, RenderGraphic {
                start_seconds: 0.9, duration_seconds: 0.4, opacity: 0.8, template: "bug".into(),
                text: "MAGIC".into(), subtitle: "".into(), text_color: "#ffffffff".into(),
                background_color: "#000000cc".into(), accent_color: "#ff51c8ff".into(), font_size: 18.0,
                motion_in: "magic".into(), motion_out: "glitch".into(),
                motion_duration: 0.1, motion_loop: "flicker".into(),
                ..Default::default()
            }],
            audio_clips: vec![RenderAudioClip {
                path: input.to_string_lossy().into_owned(), source_in_seconds: 0.2,
                duration_seconds: 0.5, source_duration_seconds: 0.5, speed_rate: 1.0,
                start_seconds: 0.25, volume_db: -9.0, pan: 0.35,
                fade_in_seconds: 0.08, fade_out_seconds: 0.12,
            }],
            overlay_clips: vec![RenderOverlayClip {
                path: input.to_string_lossy().into_owned(), start_seconds: 0.35, source_in_seconds: 0.0,
                duration_seconds: 0.45, source_duration_seconds: 0.45, speed_rate: 1.0,
                has_audio: true, volume_db: -18.0, box_x: 0.68, box_y: 0.05,
                box_width: 0.28, box_height: 0.32, fit: "cover".into(), opacity: 0.9,
                corner_radius: 0.0, border_width: 0.01, border_color: "#00f0ff".into(),
                effects: RenderEffects::default(),
            }],
            output_path: output.to_string_lossy().into_owned(), width: 160, height: 90,
            frame_rate: Rational { numerator: 24, denominator: 1 }, video_codec: "libx264".into(),
            audio_codec: "aac".into(), video_bitrate: "1M".into(), audio_bitrate: "128k".into(),
            upscale_factor: 1,
            upscale: None,
            quality_mode: "quality".into(), quality: 28, preset: "veryfast".into(),
            audio_sample_rate: 48_000, audio_channels: 2,
        };
        let runtime = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        let result = runtime.block_on(export_timeline(request));
        assert!(result.is_ok(), "timeline render failed: {result:?}");
        assert!(output.is_file() && fs::metadata(&output).unwrap().len() > 1_000, "rendered file is empty");
        let probe = StdCommand::new("ffprobe").args([
            "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
        ]).arg(&output).output().unwrap();
        assert!(probe.status.success());
        assert_eq!(String::from_utf8_lossy(&probe.stdout).trim(), "160,90");
        let duration_probe = StdCommand::new("ffprobe").args([
            "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
        ]).arg(&output).output().unwrap();
        assert!(duration_probe.status.success());
        let duration = String::from_utf8_lossy(&duration_probe.stdout).trim().parse::<f64>().unwrap();
        assert!((1.85..=2.05).contains(&duration), "timeline gap was not preserved: {duration}");
        let _ = fs::remove_dir_all(&work);
    }
}
