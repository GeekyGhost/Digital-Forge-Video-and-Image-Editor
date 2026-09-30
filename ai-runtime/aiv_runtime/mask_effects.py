"""Mask-driven visual effects for a selected clip range.

BiRefNet supplies an automatic soft foreground matte. SAM 2 supplies an interactive, temporally
propagated subject mask. The rendering stage is intentionally model-agnostic: both mask sources
feed the same small set of compositing primitives and optional audio envelope.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from aiv_runtime import sam2_track

EFFECTS = (
    "glow", "outline", "aura", "neon-edge", "electric", "energy-shield",
    "spotlight", "background-blur", "background-pixelate", "silhouette", "hologram",
    "heat-vision", "rgb-split", "subject-echo", "fireball", "disintegrate", "comic-pop",
    "power-up", "animated-matte",
)
REACTIVE_MODES = ("off", "amplitude", "bass", "transients")
MATTE_STYLES = ("subject", "flame", "rings", "burst", "clip")
PLACEMENTS = ("behind", "inside", "front")
TRANSITIONS = (
    "none", "fade", "edges-to-center", "center-to-edges", "top-to-bottom",
    "bottom-to-top", "explosion", "power-up",
)


def build_parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Render mask-driven VFX over a clip range")
    value.add_argument("--model", choices=("sam2", "birefnet"), required=True)
    value.add_argument("--model-root", required=True)
    value.add_argument("--video-path", required=True)
    value.add_argument("--output-path", required=True)
    value.add_argument("--start-time", type=float, required=True)
    value.add_argument("--end-time", type=float, required=True)
    value.add_argument("--point", action="append", default=[])
    value.add_argument("--effect", choices=EFFECTS, default="glow")
    value.add_argument("--color", default="#00f0ff")
    value.add_argument("--secondary-color", default="#ff6b1a")
    value.add_argument("--intensity", type=float, default=1.0)
    value.add_argument("--feather", type=int, default=6)
    value.add_argument("--expand", type=int, default=4)
    value.add_argument("--temporal-smooth", type=int, default=2)
    value.add_argument("--cleanup", type=int, default=120)
    value.add_argument("--effect-speed", type=float, default=1.0)
    value.add_argument("--particle-amount", type=float, default=1.0)
    value.add_argument("--transition-in", choices=TRANSITIONS, default="none")
    value.add_argument("--transition-out", choices=TRANSITIONS, default="none")
    value.add_argument("--transition-in-duration", type=float, default=0.6)
    value.add_argument("--transition-out-duration", type=float, default=0.45)
    value.add_argument("--invert", action="store_true")
    value.add_argument("--audio-reactive", choices=REACTIVE_MODES, default="off")
    value.add_argument("--reactive-amount", type=float, default=0.75)
    value.add_argument("--reactive-smoothing", type=float, default=0.45)
    value.add_argument("--placement", choices=PLACEMENTS, default="behind")
    value.add_argument("--matte-style", choices=MATTE_STYLES, default="subject")
    value.add_argument("--effect-clip-path", default="")
    value.add_argument("--effect-clip-color", choices=("original", "gradient"), default="original")
    value.add_argument("--effect-opacity", type=float, default=0.9)
    value.add_argument("--effect-x", type=float, default=0.5)
    value.add_argument("--effect-y", type=float, default=0.5)
    value.add_argument("--effect-scale", type=float, default=100.0)
    value.add_argument("--effect-rotation", type=float, default=0.0)
    value.add_argument("--loop", action="store_true")
    value.add_argument("--pulse-amount", type=float, default=0.18)
    value.add_argument("--pulse-rate", type=float, default=1.25)
    value.add_argument("--gradient-enabled", action="store_true")
    value.add_argument("--gradient-json", default="")
    value.add_argument("--matte-cache-path", default="")
    value.add_argument("--occlusion-feather", type=float, default=1.5)
    value.add_argument("--occlusion-expand", type=int, default=0)
    value.add_argument("--transition-origin", choices=("subject", "effect"), default="subject")
    value.add_argument("--blend-mode", choices=("normal", "screen", "add", "multiply", "overlay", "soft-light"), default="screen")
    value.add_argument("--key-mode", choices=("alpha", "luma", "chroma"), default="alpha")
    value.add_argument("--key-threshold", type=float, default=.16)
    value.add_argument("--key-softness", type=float, default=.12)
    value.add_argument("--key-color", default="#00ff00")
    value.add_argument("--rear-amount", type=float, default=1.0)
    value.add_argument("--subject-amount", type=float, default=.2)
    value.add_argument("--front-amount", type=float, default=.35)
    value.add_argument("--motion-blur", type=float, default=0.0)
    value.add_argument("--light-wrap", type=float, default=.18)
    value.add_argument("--color-spill", type=float, default=.12)
    value.add_argument("--grain", type=float, default=.08)
    value.add_argument("--displacement", type=float, default=0.0)
    value.add_argument("--keyframes-json", default="[]")
    value.add_argument("--preview-width", type=int, default=0)
    value.add_argument("--dry-run", action="store_true")
    return value


def parse_color(value: str) -> tuple[int, int, int]:
    raw = value.strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(character * 2 for character in raw)
    if len(raw) != 6 or any(character not in "0123456789abcdefABCDEF" for character in raw):
        raise ValueError("effect color must be #rrggbb")
    return tuple(int(raw[index:index + 2], 16) for index in (0, 2, 4))


def validate(args: argparse.Namespace) -> list[tuple[float, float, int]]:
    if not Path(args.video_path).is_file():
        raise FileNotFoundError(args.video_path)
    if not Path(args.model_root).exists():
        raise FileNotFoundError(args.model_root)
    if args.start_time < 0 or args.end_time <= args.start_time:
        raise ValueError("mask effects require 0 <= start-time < end-time")
    if Path(args.output_path).suffix.lower() != ".mp4":
        raise ValueError("mask effects output must be an MP4 file")
    if not 0 <= args.feather <= 64 or not -32 <= args.expand <= 64:
        raise ValueError("feather or expansion is outside the supported range")
    if not 0 <= args.intensity <= 2 or not 0 <= args.reactive_amount <= 2:
        raise ValueError("effect strength is outside the supported range")
    if not 0 <= args.reactive_smoothing <= 1:
        raise ValueError("audio smoothing must be between 0 and 1")
    if not 0 <= args.temporal_smooth <= 5 or not 0 <= args.cleanup <= 2000:
        raise ValueError("mask cleanup controls are outside the supported range")
    if not 0.1 <= args.effect_speed <= 3 or not 0 <= args.particle_amount <= 2:
        raise ValueError("motion controls are outside the supported range")
    if not 0.05 <= args.transition_in_duration <= 3 or not 0.05 <= args.transition_out_duration <= 3:
        raise ValueError("transition durations must be between 0.05 and 3 seconds")
    if not 0 <= args.effect_opacity <= 1 or not 0 <= args.effect_x <= 1 or not 0 <= args.effect_y <= 1:
        raise ValueError("effect opacity and position must be between 0 and 1")
    if not 10 <= args.effect_scale <= 300 or not -180 <= args.effect_rotation <= 180:
        raise ValueError("effect transform is outside the supported range")
    if not 0 <= args.pulse_amount <= 1 or not 0.1 <= args.pulse_rate <= 8:
        raise ValueError("pulse controls are outside the supported range")
    if not 0 <= args.occlusion_feather <= 32 or not -16 <= args.occlusion_expand <= 32:
        raise ValueError("occlusion matte controls are outside the supported range")
    if not 0 <= args.key_threshold <= 1 or not 0 <= args.key_softness <= 1:
        raise ValueError("key controls must be between 0 and 1")
    if any(not 0 <= value <= 1 for value in (
        args.rear_amount, args.subject_amount, args.front_amount, args.light_wrap,
        args.color_spill, args.grain,
    )):
        raise ValueError("pass and integration controls must be between 0 and 1")
    if not 0 <= args.motion_blur <= 32 or not 0 <= args.displacement <= 64:
        raise ValueError("motion blur or displacement is outside the supported range")
    if args.preview_width and not 160 <= args.preview_width <= 1920:
        raise ValueError("preview width must be between 160 and 1920 pixels")
    if args.matte_style == "clip" and not Path(args.effect_clip_path).is_file():
        raise FileNotFoundError("the imported effect clip is missing")
    parse_gradient(args.gradient_json)
    parse_color(args.color)
    parse_color(args.secondary_color)
    parse_color(args.key_color)
    parse_keyframes(args.keyframes_json)
    points = sam2_track.parse_points(args.point)
    if args.matte_cache_path:
        cache = Path(args.matte_cache_path)
        if not (cache / "manifest.json").is_file() or not (cache / "masks").is_dir():
            raise FileNotFoundError("the cached matte is missing or incomplete")
    if args.preview_width and not args.matte_cache_path:
        raise ValueError("proxy previews require a cached tracking matte")
    if args.model == "sam2" and not args.matte_cache_path:
        if not points or not any(label == 1 for _, _, label in points):
            raise ValueError("SAM 2 needs at least one include point")
    return points


def parse_keyframes(value: str) -> list[dict]:
    frames = json.loads(value or "[]")
    if not isinstance(frames, list):
        raise ValueError("VFX keyframes must be a list")
    result = []
    for frame in frames:
        if not isinstance(frame, dict) or not 0 <= float(frame.get("at", -1)) <= 1:
            raise ValueError("each VFX keyframe needs a normalized time")
        if frame.get("easing", "linear") not in ("linear", "smooth", "ease-in", "ease-out"):
            raise ValueError("unknown VFX keyframe easing")
        values = frame.get("values", {})
        required = ("effectX", "effectY", "effectScale", "effectRotation", "effectOpacity", "intensity")
        if not isinstance(values, dict) or any(name not in values for name in required):
            raise ValueError("each VFX keyframe needs a complete transform pose")
        if not 0 <= float(values["effectX"]) <= 1 or not 0 <= float(values["effectY"]) <= 1 \
                or not 10 <= float(values["effectScale"]) <= 300 \
                or not -180 <= float(values["effectRotation"]) <= 180 \
                or not 0 <= float(values["effectOpacity"]) <= 1 \
                or not 0 <= float(values["intensity"]) <= 2:
            raise ValueError("VFX keyframe values are outside the supported range")
        result.append(frame)
    return sorted(result, key=lambda frame: float(frame["at"]))


def parse_gradient(value: str) -> dict:
    """Validate the editor's portable gradient contract and supply a safe legacy default."""
    default = {
        "stops": [
            {"at": 0.0, "color": "#ffffff", "opacity": 1.0},
            {"at": 0.28, "color": "#00f0ff", "opacity": 1.0},
            {"at": 0.72, "color": "#3166ff", "opacity": 0.82},
            {"at": 1.0, "color": "#1010a0", "opacity": 0.0},
        ],
        "kind": "radial", "angle": 270.0, "cx": 0.5, "cy": 0.5, "scale": 0.72,
        "offset": 0.0, "repeat": 1.0, "mirror": False, "space": "oklab", "ease": 0.8,
        "dither": 0.7,
    }
    gradient = json.loads(value) if value else default
    if gradient.get("kind") not in ("linear", "reflected", "radial", "angular", "diamond"):
        raise ValueError("unknown gradient shape")
    if gradient.get("space") not in ("oklab", "rgb"):
        raise ValueError("unknown gradient color space")
    stops = gradient.get("stops", [])
    if len(stops) < 2:
        raise ValueError("an effect gradient needs at least two colors")
    for stop in stops:
        parse_color(str(stop.get("color", "")))
        if not 0 <= float(stop.get("at", -1)) <= 1 or not 0 <= float(stop.get("opacity", -1)) <= 1:
            raise ValueError("gradient stop position and opacity must be between 0 and 1")
    if not 0.01 <= float(gradient.get("scale", 0)) <= 4:
        raise ValueError("gradient spread is outside the supported range")
    if not 1 <= float(gradient.get("repeat", 0)) <= 12:
        raise ValueError("gradient repeat is outside the supported range")
    if not -.5 <= float(gradient.get("cx", -2)) <= 1.5 or not -.5 <= float(gradient.get("cy", -2)) <= 1.5:
        raise ValueError("gradient center is outside the supported range")
    if not -1 <= float(gradient.get("offset", -2)) <= 1 or not .25 <= float(gradient.get("ease", 0)) <= 4:
        raise ValueError("gradient shift or easing is outside the supported range")
    if not 0 <= float(gradient.get("dither", -1)) <= 1:
        raise ValueError("gradient dither must be between 0 and 1")
    return gradient


def acquire_masks(
    args: argparse.Namespace,
    points: list[tuple[float, float, int]],
    source: dict,
    scratch: Path,
    frame_count: int,
) -> list[Image.Image]:
    if args.matte_cache_path:
        cache_root = Path(args.matte_cache_path)
        manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
        paths = sorted((cache_root / "masks").glob("*.png"))
        if len(paths) < frame_count or int(manifest.get("frameCount", 0)) < frame_count:
            raise ValueError("the cached matte is incomplete; track the subject again")
        masks = []
        last = Image.new("L", (source["width"], source["height"]), 0)
        for index in range(frame_count):
            if index < len(paths):
                with Image.open(paths[index]) as cached:
                    last = cached.convert("L").resize(
                        (source["width"], source["height"]), Image.Resampling.NEAREST,
                    )
            masks.append(last.copy())
        return stabilize_masks(masks, args.temporal_smooth, args.cleanup)
    frame_paths = sorted((scratch / "frames").glob("*.jpg"))
    if args.model == "sam2":
        tracked = sam2_track.track(args, points, source, scratch, frame_count)
        masks: list[Image.Image] = []
        last = np.zeros((source["height"], source["width"]), dtype=np.uint8)
        for index in range(frame_count):
            current = tracked.get(index)
            if current is not None:
                last = current.astype(np.uint8) * 255
            masks.append(Image.fromarray(last, mode="L").resize(
                (source["width"], source["height"]), Image.Resampling.NEAREST,
            ))
        return stabilize_masks(masks, args.temporal_smooth, args.cleanup)

    from aiv_runtime.image_studio.birefnet import BiRefNetPipeline

    print(f"[mask-effects] loading BiRefNet from {args.model_root}", file=sys.stderr, flush=True)
    pipeline = BiRefNetPipeline.create(args.model_root, "cuda")
    masks = []
    for index, frame_path in enumerate(frame_paths):
        with Image.open(frame_path) as frame:
            current = np.asarray(pipeline.remove_background(frame).mask, dtype=np.float32)
        masks.append(Image.fromarray(np.clip(current, 0, 255).astype(np.uint8), mode="L"))
        if index % 24 == 0:
            print(f"[mask-effects] BiRefNet matte {index + 1}/{frame_count}", file=sys.stderr, flush=True)
    return stabilize_masks(masks, args.temporal_smooth, args.cleanup)


def _remove_small_regions(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    """Remove isolated matte islands while retaining the original soft edge when OpenCV exists."""
    if minimum_area <= 0:
        return mask
    try:
        import cv2
    except ImportError:
        return mask
    binary = (mask >= 96).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            keep[labels == label] = 1
    # Slight dilation keeps antialiased boundary pixels around surviving components.
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)
    return mask * keep.astype(np.float32)


def stabilize_masks(masks: list[Image.Image], smoothing: int, cleanup: int) -> list[Image.Image]:
    """Causal matte stabilization shared by BiRefNet and SAM 2."""
    result: list[Image.Image] = []
    previous: np.ndarray | None = None
    history = max(0, min(5, smoothing))
    memory = history / (history + 4.0) if history else 0.0
    for mask in masks:
        current = np.asarray(mask.convert("L"), dtype=np.float32)
        current = _remove_small_regions(current, cleanup)
        if previous is not None and memory:
            current = current * (1 - memory) + previous * memory
        previous = current
        result.append(Image.fromarray(np.clip(current, 0, 255).astype(np.uint8), mode="L"))
    return result


def shape_mask(mask: Image.Image, feather: int, expand: int, invert: bool) -> Image.Image:
    value = mask.convert("L")
    if expand:
        size = min(65, abs(expand) * 2 + 1)
        if size % 2 == 0:
            size += 1
        value = value.filter(ImageFilter.MaxFilter(size) if expand > 0 else ImageFilter.MinFilter(size))
    if feather:
        value = value.filter(ImageFilter.GaussianBlur(radius=feather))
    return ImageOps.invert(value) if invert else value


def extract_effect_frames(
    args: argparse.Namespace, source: dict, scratch: Path, frame_count: int,
) -> list[Path]:
    """Decode an optional VFX plate at project fps; looping is explicit rather than assumed."""
    if args.matte_style != "clip":
        return []
    output = scratch / "effect-frames"
    output.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if args.loop:
        command.extend(["-stream_loop", "-1"])
    command.extend([
        "-i", args.effect_clip_path,
        "-vf", (
            f"setpts=PTS/{args.effect_speed:.6f},fps={source['fps']:.6f},"
            f"scale={source['width']}:{source['height']}:force_original_aspect_ratio=decrease,"
            f"pad={source['width']}:{source['height']}:(ow-iw)/2:(oh-ih)/2:color=black@0,format=rgba"
        ),
        "-frames:v", str(frame_count), str(output / "%05d.png"),
    ])
    subprocess.run(command, check=True)
    frames = sorted(output.glob("*.png"))
    if not frames:
        raise ValueError("the imported effect clip contains no decodable video frames")
    return frames


def _transform_canvas(
    image: Image.Image, center: tuple[float, float], scale: float, rotation: float,
) -> Image.Image:
    """Transform a full-frame matte/plate around a normalized anchor while retaining canvas size."""
    mode = image.mode
    width, height = image.size
    factor = max(0.01, scale / 100.0)
    resized = image.resize(
        (max(1, round(width * factor)), max(1, round(height * factor))), Image.Resampling.BICUBIC,
    )
    rotated = resized.rotate(-rotation, resample=Image.Resampling.BICUBIC, expand=True)
    canvas = Image.new(mode, image.size, 0 if mode == "L" else (0, 0, 0, 0))
    left = round(center[0] * width - rotated.width / 2)
    top = round(center[1] * height - rotated.height / 2)
    if mode == "RGBA":
        canvas.alpha_composite(rotated, (left, top))
    else:
        canvas.paste(rotated, (left, top))
    return canvas


def animated_matte(subject: Image.Image, style: str, phase: float) -> Image.Image:
    """Derive reusable anime/movie mattes from the tracked silhouette."""
    subject = subject.convert("L")
    if style in ("subject", "clip"):
        return subject
    bounds = _mask_bounds(subject)
    left, top, right, bottom = bounds
    width, height = max(4, right - left), max(4, bottom - top)
    if style == "rings":
        layer = Image.new("L", subject.size, 0)
        draw = ImageDraw.Draw(layer)
        cx, cy = (left + right) / 2, (top + bottom) / 2
        for ring in range(4):
            progress = (phase + ring / 4) % 1
            radius_x = width * (.48 + progress * .55)
            radius_y = height * (.12 + progress * .26)
            y = bottom - height * (.04 + progress * .82)
            alpha = round(255 * math.sin(progress * math.pi))
            draw.ellipse((cx-radius_x, y-radius_y, cx+radius_x, y+radius_y),
                         outline=alpha, width=max(2, round(min(width, height) * .018)))
        return layer.filter(ImageFilter.GaussianBlur(max(1, min(width, height) * .008)))
    if style == "burst":
        layer = Image.new("L", subject.size, 0)
        draw = ImageDraw.Draw(layer)
        cx, cy = (left + right) / 2, (top + bottom) / 2
        maximum = math.hypot(subject.width, subject.height) * (.18 + .38 * phase)
        for ray in range(24):
            angle = ray / 24 * math.tau + math.sin(phase * math.tau + ray) * .035
            start = maximum * (.15 + (ray % 3) * .04)
            end = maximum * (.7 + (ray % 5) * .08)
            alpha = 95 + (ray % 4) * 40
            draw.line((cx+math.cos(angle)*start, cy+math.sin(angle)*start,
                       cx+math.cos(angle)*end, cy+math.sin(angle)*end),
                      fill=alpha, width=max(1, round(min(width, height) * (.008 + ray % 2 * .004))))
        return layer.filter(ImageFilter.GaussianBlur(1.2))
    # Flame silhouette: an outside rim with several vertically shifted, oscillating tongues.
    radius = max(5, min(63, (round(min(width, height) * .055) | 1)))
    outer = subject.filter(ImageFilter.MaxFilter(radius))
    rim = ImageChops.subtract(outer, subject.filter(ImageFilter.MinFilter(max(3, radius // 3 | 1))))
    flame = rim.copy()
    for tongue in range(1, 5):
        shifted = Image.new("L", subject.size, 0)
        x_shift = round(math.sin(phase * math.tau * (1 + tongue * .13) + tongue * 1.7) * width * .018 * tongue)
        y_shift = -round(height * (.018 + tongue * .012))
        shifted.paste(outer.point(lambda value, t=tongue: round(value * (.28 / t))), (x_shift, y_shift))
        flame = ImageChops.lighter(flame, shifted)
    yy, xx = np.mgrid[:subject.height, :subject.width]
    turbulence = .74 + .26 * np.sin(xx * .045 + yy * .018 - phase * math.tau * 2.3)
    return Image.fromarray(np.clip(np.asarray(flame, dtype=np.float32) * turbulence, 0, 255).astype(np.uint8), mode="L")


def _srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
    r, g, b = np.moveaxis(linear, -1, 0)
    l = .4122214708*r + .5363325363*g + .0514459929*b
    m = .2119034982*r + .6806995451*g + .1073969566*b
    s = .0883024619*r + .2817188376*g + .6299787005*b
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    return np.stack((.2104542553*l_ + .793617785*m_ - .0040720468*s_,
                     1.9779984951*l_ - 2.428592205*m_ + .4505937099*s_,
                     .0259040371*l_ + .7827717662*m_ - .808675766*s_), axis=-1)


def _oklab_to_srgb(lab: np.ndarray) -> np.ndarray:
    light, a, b = np.moveaxis(lab, -1, 0)
    l_ = light + .3963377774*a + .2158037573*b
    m_ = light - .1055613458*a - .0638541728*b
    s_ = light - .0894841775*a - 1.291485548*b
    l, m, s = l_**3, m_**3, s_**3
    linear = np.stack((4.0767416621*l - 3.3077115913*m + .2309699292*s,
                       -1.2684380046*l + 2.6097574011*m - .3413193965*s,
                       -.0041960863*l - .7034186147*m + 1.707614701*s), axis=-1)
    return np.clip(np.where(linear <= .0031308, 12.92*linear, 1.055*np.maximum(linear, 0)**(1/2.4)-.055), 0, 1)


def render_gradient(size: tuple[int, int], gradient: dict, animation_phase: float = 0.0) -> Image.Image:
    """Render the same shape/repeat/ease/stop contract used by the photo gradient editor."""
    width, height = size
    yy, xx = np.mgrid[:height, :width]
    radians = math.radians(float(gradient["angle"]))
    cosine, sine = math.cos(radians), math.sin(radians)
    dx, dy = xx + .5 - float(gradient["cx"]) * width, yy + .5 - float(gradient["cy"]) * height
    u, v = dx * cosine + dy * sine, -dx * sine + dy * cosine
    scale = max(.01, float(gradient["scale"]))
    kind = gradient["kind"]
    if kind == "angular":
        position = (np.arctan2(v, u) / math.tau + .5) / scale
    else:
        along = abs(width / 2 * cosine) + abs(height / 2 * sine)
        across = abs(width / 2 * sine) + abs(height / 2 * cosine)
        if kind == "linear": position = u / (2 * max(1e-6, along) * scale) + .5
        elif kind == "reflected": position = np.abs(u) / (max(1e-6, along) * scale)
        elif kind == "diamond": position = (np.abs(u) / max(1e-6, along) + np.abs(v) / max(1e-6, across)) / (2 * scale)
        else: position = np.hypot(u, v) / (max(1e-6, math.hypot(width / 2, height / 2)) * scale)
    repeat = max(1.0, float(gradient["repeat"]))
    travel = animation_phase if repeat > 1 else math.sin(animation_phase * math.tau) * .15
    position = (position + float(gradient["offset"]) + travel) * repeat
    if bool(gradient["mirror"]):
        position = 1 - np.abs(np.mod(position, 2) - 1)
    elif repeat > 1:
        position = np.mod(position, 1)
    else:
        position = np.clip(position, 0, 1)
    position = np.clip(position, 0, 1) ** max(.05, float(gradient["ease"]))
    stops = sorted(gradient["stops"], key=lambda item: float(item["at"]))
    stop_at = np.array([float(stop["at"]) for stop in stops])
    colors = np.array([parse_color(stop["color"]) for stop in stops], dtype=np.float32) / 255
    if gradient["space"] == "oklab":
        colors = _srgb_to_oklab(colors)
    channels = np.stack([np.interp(position, stop_at, colors[:, channel]) for channel in range(3)], axis=-1)
    if gradient["space"] == "oklab":
        channels = _oklab_to_srgb(channels)
    alpha = np.interp(position, stop_at, np.array([float(stop["opacity"]) for stop in stops]))
    # Match the editor's fixed sub-level ordered dither so broad animated ramps do not band.
    bayer = np.array([
        [0, 32, 8, 40, 2, 34, 10, 42], [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38], [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41], [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37], [63, 31, 55, 23, 61, 29, 53, 21],
    ], dtype=np.float32) / 63 - .5
    noise = np.tile(bayer, (math.ceil(height / 8), math.ceil(width / 8)))[:height, :width]
    rgb = np.clip(channels * 255 + noise[..., None] * float(gradient["dither"]), 0, 255).astype(np.uint8)
    rgba = np.dstack((rgb, np.clip(alpha * 255, 0, 255).astype(np.uint8)))
    return Image.fromarray(rgba, mode="RGBA")


def _audio_samples(args: argparse.Namespace) -> np.ndarray:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
        "-ss", f"{args.start_time:.6f}", "-t", f"{args.end_time - args.start_time:.6f}",
        "-i", args.video_path, "-vn", "-ac", "1", "-ar", "48000",
    ]
    if args.audio_reactive == "bass":
        command.extend(["-af", "lowpass=f=180"])
    command.extend(["-f", "f32le", "pipe:1"])
    output = subprocess.run(command, capture_output=True, check=False)
    if output.returncode != 0 or not output.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(output.stdout, dtype="<f4")


def audio_envelope(args: argparse.Namespace, frame_count: int) -> np.ndarray:
    if args.audio_reactive == "off":
        return np.zeros(frame_count, dtype=np.float32)
    samples = _audio_samples(args)
    if samples.size == 0:
        print("[mask-effects] source has no usable audio; reactivity disabled", file=sys.stderr)
        return np.zeros(frame_count, dtype=np.float32)
    edges = np.linspace(0, samples.size, frame_count + 1, dtype=np.int64)
    rms = np.array([
        math.sqrt(float(np.mean(np.square(samples[edges[i]:edges[i + 1]], dtype=np.float64))))
        if edges[i + 1] > edges[i] else 0.0 for i in range(frame_count)
    ], dtype=np.float32)
    if args.audio_reactive == "transients":
        # numpy's `same` returns the longer operand's length. Keep the analysis window no longer
        # than the clip so very short selections still produce one value per video frame.
        window = min(7, len(rms))
        if window > 1 and window % 2 == 0:
            window -= 1
        baseline = np.convolve(rms, np.ones(window, dtype=np.float32) / window, mode="same")
        rms = np.maximum(0, rms - baseline)
    ceiling = float(np.percentile(rms, 95))
    values = np.clip(rms / max(ceiling, 1e-6), 0, 1)
    # Higher UI smoothing means a longer release. Attacks stay quick enough to follow a beat.
    release = 0.15 + args.reactive_smoothing * 0.8
    for index in range(1, len(values)):
        if values[index] < values[index - 1]:
            values[index] = values[index] * (1 - release) + values[index - 1] * release
    return values


def _alpha(mask: Image.Image, amount: float) -> Image.Image:
    return mask.point(lambda value: int(max(0, min(255, value * amount))))


def _colour_layer(size: tuple[int, int], color: tuple[int, int, int], alpha: Image.Image) -> Image.Image:
    layer = Image.new("RGBA", size, (*color, 0))
    layer.putalpha(alpha)
    return layer


def _subject_over(background: Image.Image, source: Image.Image, mask: Image.Image) -> Image.Image:
    return Image.composite(source.convert("RGB"), background.convert("RGB"), mask).convert("RGBA")


def _mask_bounds(mask: Image.Image) -> tuple[int, int, int, int]:
    return mask.getbbox() or (mask.width // 3, mask.height // 3, mask.width * 2 // 3, mask.height * 2 // 3)


def _edge(mask: Image.Image, radius: int = 5) -> Image.Image:
    radius = max(3, min(63, radius | 1))
    return ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(radius)), mask.filter(ImageFilter.MinFilter(radius)))


def _screen(base: Image.Image, layer: Image.Image) -> Image.Image:
    rgb = ImageChops.screen(base.convert("RGB"), layer.convert("RGB"))
    return Image.composite(rgb, base.convert("RGB"), layer.getchannel("A")).convert("RGBA")


def _blend(base: Image.Image, layer: Image.Image, mode: str) -> Image.Image:
    """Blend an RGBA plate with editor-style alpha and standard RGB blend math."""
    base_rgba, layer_rgba = base.convert("RGBA"), layer.convert("RGBA")
    if mode == "normal":
        return Image.alpha_composite(base_rgba, layer_rgba)
    first = np.asarray(base_rgba.convert("RGB"), dtype=np.float32) / 255
    second = np.asarray(layer_rgba.convert("RGB"), dtype=np.float32) / 255
    if mode == "screen":
        mixed = 1 - (1 - first) * (1 - second)
    elif mode == "add":
        mixed = np.minimum(1, first + second)
    elif mode == "multiply":
        mixed = first * second
    elif mode == "overlay":
        mixed = np.where(first <= .5, 2 * first * second, 1 - 2 * (1 - first) * (1 - second))
    elif mode == "soft-light":
        mixed = (1 - 2 * second) * first * first + 2 * second * first
    else:
        raise ValueError(f"unknown blend mode {mode}")
    alpha = np.asarray(layer_rgba.getchannel("A"), dtype=np.float32)[..., None] / 255
    composed = first * (1 - alpha) + np.clip(mixed, 0, 1) * alpha
    return Image.fromarray(np.clip(composed * 255, 0, 255).astype(np.uint8), mode="RGB").convert("RGBA")


def _extract_clip_matte(layer: Image.Image, args: argparse.Namespace) -> Image.Image:
    """Extract transparency from authored alpha, luminance, or distance from a chroma color."""
    rgba = layer.convert("RGBA")
    if args.key_mode == "alpha":
        return rgba.getchannel("A")
    rgb = np.asarray(rgba.convert("RGB"), dtype=np.float32) / 255
    if args.key_mode == "luma":
        signal = rgb[..., 0] * .2126 + rgb[..., 1] * .7152 + rgb[..., 2] * .0722
        alpha = np.clip((signal - args.key_threshold) / max(.001, args.key_softness) + .5, 0, 1)
    else:
        key = np.asarray(parse_color(args.key_color), dtype=np.float32) / 255
        distance = np.sqrt(np.mean((rgb - key) ** 2, axis=-1))
        alpha = np.clip((distance - args.key_threshold) / max(.001, args.key_softness) + .5, 0, 1)
    alpha = alpha * alpha * (3 - 2 * alpha)
    return Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8), mode="L")


def _displace(image: Image.Image, amount: float, phase: float) -> Image.Image:
    if amount <= 0:
        return image
    values = np.asarray(image)
    shifted = np.empty_like(values)
    for row in range(values.shape[0]):
        offset = round(math.sin(row * .045 + phase * math.tau) * amount)
        shifted[row] = np.roll(values[row], offset, axis=0)
    return Image.fromarray(shifted, mode=image.mode)


def _ease(value: float, easing: str) -> float:
    value = max(0.0, min(1.0, value))
    if easing == "smooth":
        return value * value * (3 - 2 * value)
    if easing == "ease-in":
        return value * value
    if easing == "ease-out":
        return 1 - (1 - value) ** 2
    return value


def sample_animation(args: argparse.Namespace, progress: float) -> dict[str, float]:
    defaults = {
        "effectX": args.effect_x, "effectY": args.effect_y, "effectScale": args.effect_scale,
        "effectRotation": args.effect_rotation, "effectOpacity": args.effect_opacity,
        "intensity": args.intensity,
    }
    keyframes = parse_keyframes(args.keyframes_json)
    if not keyframes:
        return defaults
    if progress <= float(keyframes[0]["at"]):
        return {**defaults, **keyframes[0].get("values", {})}
    if progress >= float(keyframes[-1]["at"]):
        return {**defaults, **keyframes[-1].get("values", {})}
    for left, right in zip(keyframes, keyframes[1:]):
        if float(left["at"]) <= progress <= float(right["at"]):
            span = max(1e-9, float(right["at"]) - float(left["at"]))
            amount = _ease((progress - float(left["at"])) / span, str(right.get("easing", "linear")))
            first, second = {**defaults, **left.get("values", {})}, {**defaults, **right.get("values", {})}
            return {key: float(first[key]) + (float(second[key]) - float(first[key])) * amount for key in defaults}
    return defaults


def _draw_energy_lines(
    size: tuple[int, int], bounds: tuple[int, int, int, int], color: tuple[int, int, int],
    seed: int, amount: float,
) -> Image.Image:
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    rng = np.random.default_rng(seed)
    left, top, right, bottom = bounds
    width, height = max(2, right - left), max(2, bottom - top)
    line_count = max(2, int(4 + amount * 7))
    for line in range(line_count):
        side = line % 4
        points = []
        for step in range(9):
            t = step / 8
            jitter_x = int(rng.normal(0, max(2, width * 0.035)))
            jitter_y = int(rng.normal(0, max(2, height * 0.035)))
            if side == 0: point = (left + int(width * t) + jitter_x, top + jitter_y)
            elif side == 1: point = (right + jitter_x, top + int(height * t) + jitter_y)
            elif side == 2: point = (right - int(width * t) + jitter_x, bottom + jitter_y)
            else: point = (left + jitter_x, bottom - int(height * t) + jitter_y)
            points.append(point)
        draw.line(points, fill=(*color, 230), width=max(1, int(1 + amount)))
    return layer


def _fireball_layer(
    size: tuple[int, int], center: tuple[float, float], radius: float,
    primary: tuple[int, int, int], secondary: tuple[int, int, int], phase: float, amount: float,
) -> Image.Image:
    width, height = size
    yy, xx = np.ogrid[:height, :width]
    dx, dy = xx - center[0], yy - center[1]
    distance = np.sqrt(dx * dx + dy * dy) / max(1, radius)
    angle = np.arctan2(dy, dx)
    turbulence = np.sin(angle * 7 + phase * 8 + distance * 13) * 0.09
    core = np.clip(1 - distance + turbulence, 0, 1)
    alpha = np.clip(core ** 0.62 * 255 * min(1.5, amount), 0, 255)
    mix = np.clip(distance, 0, 1)[..., None]
    first = np.asarray(primary, dtype=np.float32)
    second = np.asarray(secondary, dtype=np.float32)
    rgb = first * (1 - mix) + second * mix
    rgba = np.dstack((np.clip(rgb, 0, 255).astype(np.uint8), alpha.astype(np.uint8)))
    layer = Image.fromarray(rgba, mode="RGBA")
    glow = layer.filter(ImageFilter.GaussianBlur(radius=max(2, radius * 0.15)))
    return Image.alpha_composite(glow, layer)


def _soft_threshold(value: np.ndarray, boundary: np.ndarray | float, softness: float = .045) -> np.ndarray:
    ramp = np.clip((boundary - value) / max(.001, softness) + .5, 0, 1)
    return ramp * ramp * (3 - 2 * ramp)


def transition_alpha(
    size: tuple[int, int], kind: str, progress: float, frame_index: int = 0, speed: float = 1.0,
    anchor: tuple[float, float] = (.5, .5),
) -> Image.Image:
    """A full-frame reveal matte. Running progress backwards gives the corresponding exit."""
    width, height = size
    amount = max(0.0, min(1.0, progress))
    if kind == "none":
        return Image.new("L", size, 255)
    if kind == "fade":
        return Image.new("L", size, round(amount * 255))
    yy, xx = np.mgrid[:height, :width]
    u = xx / max(1, width - 1)
    v = yy / max(1, height - 1)
    if kind == "top-to-bottom":
        alpha = _soft_threshold(v, amount)
    elif kind == "bottom-to-top":
        alpha = _soft_threshold(1 - v, amount)
    elif kind == "edges-to-center":
        dx, dy = u - anchor[0], v - anchor[1]
        maximum = max(
            math.hypot(anchor[0], anchor[1]), math.hypot(1-anchor[0], anchor[1]),
            math.hypot(anchor[0], 1-anchor[1]), math.hypot(1-anchor[0], 1-anchor[1]),
        )
        inward = 1 - np.sqrt(dx * dx + dy * dy) / max(.001, maximum)
        alpha = _soft_threshold(inward, amount, .065)
    else:
        dx, dy = u - anchor[0], v - anchor[1]
        maximum = max(
            math.hypot(anchor[0], anchor[1]), math.hypot(1-anchor[0], anchor[1]),
            math.hypot(anchor[0], 1-anchor[1]), math.hypot(1-anchor[0], 1-anchor[1]),
        )
        radius = np.sqrt(dx * dx + dy * dy) / max(.001, maximum)
        if kind == "explosion":
            angle = np.arctan2(dy, dx)
            turbulence = np.sin(angle * 11 + frame_index * .31 * speed) * .045 * (1 - amount)
            alpha = _soft_threshold(radius, amount + turbulence, .065)
        elif kind == "power-up":
            wave = np.sin(radius * 34 - frame_index * .28 * speed) * .025
            alpha = _soft_threshold(radius, amount + wave, .075) * amount
        else:  # center-to-edges
            alpha = _soft_threshold(radius, amount, .06)
    return Image.fromarray(np.clip(alpha * 255, 0, 255).astype(np.uint8), mode="L")


def transition_energy_overlay(
    size: tuple[int, int], kind: str, progress: float, primary: tuple[int, int, int],
    secondary: tuple[int, int, int], frame_index: int, strength: float,
    anchor: tuple[float, float] = (.5, .5),
) -> Image.Image:
    if kind not in ("explosion", "power-up") or progress <= 0 or progress >= 1:
        return Image.new("RGBA", size, (0, 0, 0, 0))
    width, height = size
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = width * anchor[0], height * anchor[1]
    maximum = max(
        math.hypot(cx, cy), math.hypot(width-cx, cy),
        math.hypot(cx, height-cy), math.hypot(width-cx, height-cy),
    )
    if kind == "explosion":
        radius = maximum * progress
        fade = math.sin(progress * math.pi)
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=(*secondary, int(230 * fade)), width=max(2, int(3 + 7 * strength)))
        return Image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(9)), layer)
    # Several differently colored waves converge into a gradient rather than one flat ring.
    for offset, color in ((0.0, primary), (.11, secondary), (.22, primary)):
        radius = maximum * ((progress + offset + frame_index * .006) % 1.0)
        fade = math.sin(progress * math.pi) * (1 - offset * 1.8)
        draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=(*color, int(175 * fade)), width=max(2, int(2 + 4 * strength)))
    return Image.alpha_composite(layer.filter(ImageFilter.GaussianBlur(12)), layer)


def apply_effect(
    frame: Image.Image,
    mask: Image.Image,
    effect: str,
    color: tuple[int, int, int],
    strength: float,
    secondary_color: tuple[int, int, int] = (255, 107, 26),
    frame_index: int = 0,
    fps: float = 24.0,
    effect_speed: float = 1.0,
    particle_amount: float = 1.0,
) -> Image.Image:
    base = frame.convert("RGBA")
    value = max(0.0, min(3.5, strength))
    phase = frame_index / max(1.0, fps) * effect_speed
    if effect == "glow":
        halo = mask.filter(ImageFilter.GaussianBlur(radius=8 + value * 18))
        halo = ImageChops.subtract(halo, mask.filter(ImageFilter.GaussianBlur(radius=2)))
        return Image.alpha_composite(base, _colour_layer(base.size, color, _alpha(halo, min(2, value * 1.25))))
    if effect == "outline":
        radius = max(1, min(31, int(2 + value * 7)))
        if radius % 2 == 0:
            radius += 1
        edge = ImageChops.subtract(mask.filter(ImageFilter.MaxFilter(radius)), mask.filter(ImageFilter.MinFilter(radius)))
        return Image.alpha_composite(base, _colour_layer(base.size, color, _alpha(edge, min(2, value))))
    if effect == "neon-edge":
        edge = _edge(mask, int(3 + value * 6))
        glow = edge.filter(ImageFilter.GaussianBlur(radius=5 + value * 8))
        result = _screen(base, _colour_layer(base.size, secondary_color, _alpha(glow, value * .9)))
        return _screen(result, _colour_layer(base.size, color, _alpha(edge, min(2, value * 1.3))))
    if effect == "electric":
        edge = _edge(mask, 5)
        result = _screen(base, _colour_layer(base.size, color, _alpha(edge.filter(ImageFilter.GaussianBlur(8)), value)))
        bolts = _draw_energy_lines(base.size, _mask_bounds(mask), secondary_color, frame_index // 2 + 1049, particle_amount * value)
        result = _screen(result, bolts.filter(ImageFilter.GaussianBlur(4)))
        return Image.alpha_composite(result, bolts)
    if effect == "energy-shield":
        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        left, top, right, bottom = _mask_bounds(mask)
        pulse = 1 + math.sin(phase * math.tau) * .035 * value
        cx, cy = (left + right) / 2, (top + bottom) / 2
        width, height = (right - left) * 1.18 * pulse, (bottom - top) * 1.12 * pulse
        box = (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2)
        draw.ellipse(box, outline=(*color, 210), width=max(2, int(2 + value * 3)))
        result = _screen(base, layer.filter(ImageFilter.GaussianBlur(10)))
        return Image.alpha_composite(result, layer)
    if effect == "aura":
        result = base
        for radius, amount in ((38, 0.22), (20, 0.38), (8, 0.65)):
            halo = mask.filter(ImageFilter.GaussianBlur(radius=max(1, radius * value)))
            result = Image.alpha_composite(result, _colour_layer(base.size, color, _alpha(halo, amount * value)))
        return result
    if effect == "spotlight":
        grey = ImageEnhance.Color(base.convert("RGB")).enhance(max(0, 1 - value * 0.8))
        dark = ImageEnhance.Brightness(grey).enhance(max(0.08, 1 - value * 0.62))
        result = _subject_over(dark, base, mask)
        halo = mask.filter(ImageFilter.GaussianBlur(radius=10 + value * 10))
        return Image.alpha_composite(result, _colour_layer(base.size, color, _alpha(halo, value * 0.18)))
    if effect == "background-blur":
        background = base.filter(ImageFilter.GaussianBlur(radius=2 + value * 18))
        return _subject_over(background, base, mask)
    if effect == "background-pixelate":
        divisor = max(3, int(5 + value * 18))
        tiny = base.resize((max(1, base.width // divisor), max(1, base.height // divisor)), Image.Resampling.BILINEAR)
        background = tiny.resize(base.size, Image.Resampling.NEAREST)
        return _subject_over(background, base, mask)
    if effect == "silhouette":
        result = Image.alpha_composite(base, _colour_layer(base.size, color, _alpha(mask, min(1, 0.45 + value * 0.55))))
        halo = mask.filter(ImageFilter.GaussianBlur(radius=5 + value * 12))
        return Image.alpha_composite(result, _colour_layer(base.size, color, _alpha(halo, value * 0.35)))
    if effect == "hologram":
        grey = ImageOps.grayscale(base.convert("RGB"))
        tinted = ImageOps.colorize(grey, black=(0, 18, 30), white=color).convert("RGBA")
        scan = Image.new("L", base.size, 0)
        scan_array = np.zeros((base.height, base.width), dtype=np.uint8)
        scan_array[::max(2, int(6 - min(value, 2) * 2)), :] = int(80 + min(value, 2) * 70)
        scan = Image.fromarray(scan_array, mode="L")
        subject_alpha = ImageChops.multiply(mask, ImageOps.invert(scan))
        result = Image.composite(tinted, base, _alpha(subject_alpha, min(1, 0.45 + value * 0.35)))
        halo = mask.filter(ImageFilter.GaussianBlur(radius=7 + value * 8))
        return Image.alpha_composite(result.convert("RGBA"), _colour_layer(base.size, color, _alpha(halo, value * 0.28)))
    if effect == "heat-vision":
        luminance = np.asarray(ImageOps.grayscale(base.convert("RGB")), dtype=np.float32) / 255
        heat = np.stack([
            np.clip(luminance * 3, 0, 1),
            np.clip((luminance - 0.28) * 2.2, 0, 1),
            np.clip((0.62 - luminance) * 2.4, 0, 1),
        ], axis=-1)
        heat_image = Image.fromarray((heat * 255).astype(np.uint8), mode="RGB").convert("RGBA")
        return Image.composite(heat_image, base, _alpha(mask, min(1, 0.5 + value * 0.5))).convert("RGBA")
    if effect == "rgb-split":
        pixels = np.asarray(base.convert("RGB"), dtype=np.uint8)
        offset = max(1, int(2 + value * 8))
        shifted = pixels.copy()
        shifted[..., 0] = np.roll(pixels[..., 0], offset, axis=1)
        shifted[..., 2] = np.roll(pixels[..., 2], -offset, axis=1)
        split = Image.fromarray(shifted, mode="RGB").convert("RGBA")
        return Image.composite(split, base, _alpha(mask, min(1, .45 + value * .35))).convert("RGBA")
    if effect == "subject-echo":
        subject = Image.new("RGBA", base.size, (0, 0, 0, 0))
        subject.paste(base, (0, 0), mask)
        result = base
        count = max(2, int(2 + particle_amount * 3))
        for echo in range(count, 0, -1):
            shift = int(echo * (4 + value * 5))
            tint = color if echo % 2 else secondary_color
            alpha = _alpha(mask, value * .14 * (1 - echo / (count + 1)))
            coloured = Image.blend(subject, _colour_layer(base.size, tint, mask), .55)
            coloured.putalpha(alpha)
            translated = Image.new("RGBA", base.size, (0, 0, 0, 0))
            translated.alpha_composite(coloured, (-shift, int(math.sin(phase * 4 + echo) * shift * .25)))
            result = Image.alpha_composite(result, translated)
        return result
    if effect == "fireball":
        left, top, right, bottom = _mask_bounds(mask)
        center = ((left + right) * .5, (top + bottom) * .5)
        radius = max(14, min(right - left, bottom - top) * (.16 + value * .12))
        orb = _fireball_layer(base.size, center, radius, color, secondary_color, phase, particle_amount * value)
        result = _screen(base, orb)
        rng = np.random.default_rng(707 + frame_index)
        sparks = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(sparks)
        for _ in range(max(3, int(8 * particle_amount * value))):
            angle = float(rng.random() * math.tau)
            distance = float(radius * (.7 + rng.random() * 1.2))
            x, y = center[0] + math.cos(angle) * distance, center[1] + math.sin(angle) * distance
            r = max(1, int(rng.integers(1, max(2, int(radius * .045)) + 1)))
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(*secondary_color, 220))
        return Image.alpha_composite(result, sparks.filter(ImageFilter.GaussianBlur(1)))
    if effect == "disintegrate":
        result = base.copy()
        rng = np.random.default_rng(1701 + frame_index // 2)
        matte = np.asarray(mask, dtype=np.uint8)
        candidates = np.argwhere((matte > 100) & (rng.random(matte.shape) < .018 * particle_amount * value))
        particles = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(particles)
        source_pixels = np.asarray(base.convert("RGB"))
        drift = int((5 + 18 * value) * (0.35 + phase % 1))
        for y, x in candidates[:4500]:
            size = int(rng.integers(1, 4 + max(1, int(value))))
            rgb = tuple(int(channel) for channel in source_pixels[y, x])
            draw.rectangle((x + drift, y - drift // 3, x + drift + size, y - drift // 3 + size), fill=(*rgb, 210))
        edge = _edge(mask, 5)
        result = _screen(result, _colour_layer(base.size, color, _alpha(edge, value * .75)))
        return Image.alpha_composite(result, particles)
    if effect == "comic-pop":
        subject = ImageOps.posterize(base.convert("RGB"), max(2, 6 - int(min(value, 2) * 2)))
        outlined = ImageOps.invert(ImageOps.grayscale(base.convert("RGB")).filter(ImageFilter.FIND_EDGES))
        subject = ImageEnhance.Contrast(subject).enhance(1.15 + value * .35).convert("RGBA")
        result = Image.composite(subject, base, mask).convert("RGBA")
        ink = ImageChops.multiply(outlined, mask).point(lambda item: 255 if item < 95 else 0)
        result = Image.alpha_composite(result, _colour_layer(base.size, (8, 8, 12), _alpha(ink, .72)))
        dots = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(dots)
        spacing = max(7, int(13 - value * 3))
        for y in range(spacing // 2, base.height, spacing):
            for x in range(spacing // 2, base.width, spacing):
                if mask.getpixel((x, y)) > 128:
                    draw.ellipse((x-1, y-1, x+1, y+1), fill=(*secondary_color, 95))
        return Image.alpha_composite(result, dots)
    if effect == "power-up":
        edge = _edge(mask, max(5, int(7 + value * 6)))
        result = base
        for radius, amount, tint in ((34, .26, secondary_color), (17, .46, color), (5, .92, (255, 255, 255))):
            glow = edge.filter(ImageFilter.GaussianBlur(max(1, radius * max(.35, value))))
            result = _screen(result, _colour_layer(base.size, tint, _alpha(glow, amount * value)))
        bolts = _draw_energy_lines(base.size, _mask_bounds(mask), color, 4200 + frame_index // 2, particle_amount * value)
        result = _screen(result, bolts.filter(ImageFilter.GaussianBlur(5)))
        return Image.alpha_composite(result, bolts)
    if effect == "animated-matte":
        return base
    raise ValueError(f"unknown mask effect {effect}")


def render_frames(
    args: argparse.Namespace,
    masks: list[Image.Image],
    envelope: np.ndarray,
    scratch: Path,
    source_info: dict,
    effect_frames: list[Path] | None = None,
) -> int:
    """Composite rear, subject-lighting, and foreground passes from one immutable source frame."""
    frames = sorted((scratch / "frames").glob("*.jpg"))
    output = scratch / "rendered"
    output.mkdir(parents=True, exist_ok=True)
    color = parse_color(args.color)
    secondary_color = parse_color(args.secondary_color)
    gradient = parse_gradient(args.gradient_json)
    effect_frames = effect_frames or []
    count = min(len(frames), len(masks))
    for index in range(count):
        with Image.open(frames[index]) as source:
            source = source.convert("RGBA")
            # The holdout is deliberately independent of stylised matte growth and inversion.
            subject_matte = shape_mask(
                masks[index], round(args.occlusion_feather), args.occlusion_expand, False,
            )
            selected_matte = shape_mask(masks[index], args.feather, args.expand, args.invert)
            elapsed = index / max(1.0, source_info["fps"])
            timeline_progress = index / max(1, count - 1)
            animation = sample_animation(args, timeline_progress)
            raw_phase = elapsed * args.effect_speed
            phase = raw_phase % 1 if args.loop else min(1.0, raw_phase)
            pulse = 1 + args.pulse_amount * math.sin(elapsed * args.pulse_rate * math.tau)
            center = (animation["effectX"], animation["effectY"])
            clip_layer: Image.Image | None = None
            if args.matte_style == "clip":
                frame_at = min(index, len(effect_frames) - 1)
                with Image.open(effect_frames[frame_at]) as effect_source:
                    clip_layer = _transform_canvas(
                        effect_source.convert("RGBA"), center,
                        animation["effectScale"] * pulse, animation["effectRotation"],
                    )
                matte = _extract_clip_matte(clip_layer, args)
            else:
                matte = animated_matte(selected_matte, args.matte_style, phase)
                matte = _transform_canvas(
                    matte, center, animation["effectScale"] * pulse, animation["effectRotation"],
                )
            if args.placement == "inside":
                matte = ImageChops.multiply(matte, subject_matte)
            matte = _displace(matte, args.displacement, phase)
            if args.motion_blur:
                matte = matte.filter(ImageFilter.GaussianBlur(args.motion_blur))
            if clip_layer is not None and args.displacement:
                clip_layer = _displace(clip_layer, args.displacement, phase)
            strength = animation["intensity"] * pulse * (1 + args.reactive_amount * float(envelope[index]))
            effected = apply_effect(
                source, matte, args.effect, color, strength, secondary_color,
                index, source_info["fps"], args.effect_speed, args.particle_amount,
            )
            opacity = max(0.0, min(1.0, animation["effectOpacity"]))
            effected = Image.blend(source, effected, opacity)
            if clip_layer is not None and args.effect_clip_color == "original":
                if args.key_mode == "chroma" and args.color_spill:
                    # Pull the selected key colour out of semi-transparent edge pixels.
                    keyed = np.asarray(clip_layer.convert("RGBA"), dtype=np.float32)
                    key = np.asarray((*parse_color(args.key_color), 255), dtype=np.float32)
                    keyed[..., :3] = np.clip(keyed[..., :3] + (keyed[..., :3] - key[:3]) * args.color_spill, 0, 255)
                    clip_layer = Image.fromarray(keyed.astype(np.uint8), mode="RGBA")
                clip_layer.putalpha(_alpha(matte, opacity))
                effected = _blend(effected, clip_layer, args.blend_mode)
            use_gradient = args.gradient_enabled \
                and (clip_layer is None or args.effect_clip_color == "gradient")
            if use_gradient:
                gradient_layer = render_gradient(source.size, gradient, phase)
                gradient_layer.putalpha(ImageChops.multiply(gradient_layer.getchannel("A"), _alpha(matte, opacity)))
                effected = _blend(effected, gradient_layer, args.blend_mode)
            if args.grain:
                pixels = np.asarray(effected.convert("RGB"), dtype=np.float32)
                noise = np.random.default_rng(8009 + index).normal(0, 14 * args.grain, pixels.shape[:2])
                grain_mask = np.asarray(matte, dtype=np.float32)[..., None] / 255
                pixels = np.clip(pixels + noise[..., None] * grain_mask, 0, 255)
                effected = Image.fromarray(pixels.astype(np.uint8), mode="RGB").convert("RGBA")
            remaining = max(0.0, (count - 1 - index) / max(1.0, source_info["fps"]))
            in_progress = 1.0 if args.transition_in == "none" else min(1.0, elapsed / args.transition_in_duration)
            out_progress = 1.0 if args.transition_out == "none" else min(1.0, remaining / args.transition_out_duration)
            if args.transition_origin == "subject":
                left, top, right, bottom = _mask_bounds(subject_matte)
                anchor = ((left + right) / (2 * source.width), (top + bottom) / (2 * source.height))
            else:
                anchor = center
            entrance = transition_alpha(source.size, args.transition_in, in_progress, index, args.effect_speed, anchor)
            exit_matte = transition_alpha(source.size, args.transition_out, out_progress, index, args.effect_speed, anchor)
            reveal = ImageChops.darker(entrance, exit_matte)
            revealed = Image.composite(effected, source, reveal).convert("RGBA")
            active_kind, active_progress = (args.transition_in, in_progress) if in_progress < 1 else (args.transition_out, out_progress)
            energy = transition_energy_overlay(
                source.size, active_kind, active_progress, color, secondary_color, index, strength, anchor,
            )
            revealed = _blend(revealed, energy, args.blend_mode)

            # Pass 1: effect behind the clean subject cutout (the sandwich).
            rear = _subject_over(revealed, source, subject_matte)
            result = Image.blend(source, rear, args.rear_amount)

            # Pass 2: light wrap and controlled colour spill affect only the clean subject edge.
            if args.subject_amount and (args.light_wrap or args.color_spill):
                radius = max(3, min(63, (round(5 + args.light_wrap * 34) | 1)))
                inner = ImageChops.subtract(subject_matte, subject_matte.filter(ImageFilter.MinFilter(radius)))
                wrap = inner.filter(ImageFilter.GaussianBlur(2 + args.light_wrap * 8))
                tint = _colour_layer(source.size, color, _alpha(wrap, args.subject_amount * args.light_wrap))
                result = _blend(result, tint, "screen")
                if args.color_spill:
                    spill = _colour_layer(source.size, secondary_color, _alpha(inner, args.subject_amount * args.color_spill * .55))
                    result = _blend(result, spill, "soft-light")

            # Pass 3: a controlled foreground copy can cross the silhouette for sparks/energy.
            if args.front_amount:
                foreground = Image.blend(source, revealed, args.front_amount)
                result = Image.composite(foreground, result, matte).convert("RGBA")
            result.convert("RGB").save(output / f"{index:05d}.png", compress_level=1)
        if index % 24 == 0:
            print(f"[mask-effects] composited {index + 1}/{count}", file=sys.stderr, flush=True)
    return count


def encode(args: argparse.Namespace, source: dict, scratch: Path, frame_count: int) -> None:
    duration = frame_count / source["fps"]
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-framerate", f"{source['fps']:.6f}", "-i", str(scratch / "rendered" / "%05d.png"),
        "-ss", f"{args.start_time:.6f}", "-t", f"{duration:.6f}", "-i", args.video_path,
        "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-preset", "medium", "-crf", "16",
        "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-color_range", "tv", "-c:a", "aac", "-b:a", "256k", "-t", f"{duration:.6f}",
        "-movflags", "+faststart", args.output_path,
    ]
    subprocess.run(command, check=True)


def main() -> int:
    sam2_track.force_utf8_io()
    args = build_parser().parse_args()
    points = validate(args)
    source = sam2_track.probe(args.video_path)
    frame_count = int(round((args.end_time - args.start_time) * source["fps"]))
    if frame_count > sam2_track.MAX_TRACK_FRAMES:
        raise ValueError(
            f"selected range is {frame_count} frames; mask effects currently supports "
            f"at most {sam2_track.MAX_TRACK_FRAMES}"
        )
    if frame_count < 1:
        raise ValueError("selected range contains no video frames")
    if args.dry_run:
        print(json.dumps({"ok": True, "frames": frame_count, "effect": args.effect, "model": args.model}))
        return 0
    if args.preview_width:
        aspect = source["height"] / max(1, source["width"])
        source = {**source, "width": args.preview_width, "height": max(2, round(args.preview_width * aspect / 2) * 2)}
    with tempfile.TemporaryDirectory(prefix="aiv-mask-effects-") as scratch_name:
        scratch = Path(scratch_name)
        produced = sam2_track.extract_frames(args, source, scratch)
        masks = acquire_masks(args, points, source, scratch, produced)
        envelope = audio_envelope(args, produced)
        effect_frames = extract_effect_frames(args, source, scratch, produced)
        rendered = render_frames(args, masks, envelope, scratch, source, effect_frames)
        encode(args, source, scratch, rendered)
    print(json.dumps({
        "ok": True, "outputPath": args.output_path, "frames": rendered,
        "durationSeconds": rendered / source["fps"], "effect": args.effect,
        "model": args.model, "audioReactive": args.audio_reactive,
        "temporalSmooth": args.temporal_smooth, "cleanup": args.cleanup,
        "effectSpeed": args.effect_speed, "particleAmount": args.particle_amount,
        "transitionIn": args.transition_in, "transitionOut": args.transition_out,
        "transitionInDuration": args.transition_in_duration,
        "transitionOutDuration": args.transition_out_duration,
        "placement": args.placement, "matteStyle": args.matte_style,
        "effectOpacity": args.effect_opacity, "effectX": args.effect_x, "effectY": args.effect_y,
        "effectScale": args.effect_scale, "effectRotation": args.effect_rotation,
        "loop": args.loop, "pulseAmount": args.pulse_amount, "pulseRate": args.pulse_rate,
        "gradientEnabled": args.gradient_enabled, "blendMode": args.blend_mode,
        "keyMode": args.key_mode, "transitionOrigin": args.transition_origin,
        "rearAmount": args.rear_amount, "subjectAmount": args.subject_amount,
        "frontAmount": args.front_amount, "previewWidth": args.preview_width,
        "matteCachePath": args.matte_cache_path,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
