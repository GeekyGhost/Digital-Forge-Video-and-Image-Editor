"""One-off introspection of the installed nvvfx VideoSuperRes effect.

Answers the questions the integration actually needs and that the docs do not state:
  - the real QualityLevel members (dir() on an IntEnum leaks int's own methods)
  - whether the effect holds temporal state between run() calls
  - what happens at 3x and 4x, and whether a non-integer target is accepted
  - whether one effect instance can be reused across frames of the same size
  - whether run() takes a BATCH, which is what decides how the video path is written
  - what happens below the documented 360p minimum input

The base frame is 640x360, not the 320x180 this first used. NVIDIA's own filter page states
"the suggested minimum input resolution for VSR is 360p", so probing below it would have measured
the failure mode rather than the feature.
"""
import atexit
import json
import traceback

import torch
from nvvfx import VideoSuperRes

report = {}

# Print whatever was learned even if a later section raises. The first run of this lost every
# answer to a shape mismatch in the fourth check, which is the one thing a diagnostic must not do.
atexit.register(lambda: print(json.dumps(report, indent=2)))


# 1. The real quality levels. VideoSuperRes.QualityLevel is an IntEnum, so dir() on it returns
#    int's methods too -- which is why the setup probe reported 28 rather than the true count.
levels = {member.name: int(member.value) for member in VideoSuperRes.QualityLevel}
report["quality_levels"] = levels
report["quality_level_count"] = len(levels)

# 2. Determinism / temporal state. If the effect keeps state between frames, running the SAME
#    frame twice through one instance gives two different answers -- and a fresh instance would
#    differ from the second run of the reused one. This is the question that decides whether
#    nvvfx is a temporally-coherent video model or just another per-frame image model.
torch.manual_seed(11)
BASE_W, BASE_H = 640, 360
frame = torch.rand(3, BASE_H, BASE_W, device="cuda", dtype=torch.float32)
other = torch.rand(3, BASE_H, BASE_W, device="cuda", dtype=torch.float32)
report["base_input"] = [BASE_W, BASE_H]

effect = VideoSuperRes(quality=VideoSuperRes.QualityLevel.HIGH)
effect.output_width, effect.output_height = BASE_W * 2, BASE_H * 2
effect.load()

try:
    first = torch.from_dlpack(effect.run(frame).image).clone()
    second = torch.from_dlpack(effect.run(frame).image).clone()
    report["repeat_same_frame_identical"] = bool(torch.equal(first, second))
    report["repeat_same_frame_max_delta"] = float((first - second).abs().max())

    # Now push a DIFFERENT frame through, then return to the original. If there is temporal state,
    # the original's result will have been influenced by what came in between.
    _ = torch.from_dlpack(effect.run(other).image).clone()
    after_other = torch.from_dlpack(effect.run(frame).image).clone()
    report["unaffected_by_intervening_frame"] = bool(torch.equal(first, after_other))
    report["intervening_frame_max_delta"] = float((first - after_other).abs().max())

    # And a completely fresh instance, to catch state that lives in the object rather than the
    # stream. Its output size has to match `effect`'s or the comparison is a shape error rather
    # than an answer -- which is exactly how the first run of this died.
    fresh = VideoSuperRes(quality=VideoSuperRes.QualityLevel.HIGH)
    fresh.output_width, fresh.output_height = effect.output_width, effect.output_height
    fresh.load()
    fresh_out = torch.from_dlpack(fresh.run(frame).image).clone()
    report["fresh_instance_identical"] = bool(torch.equal(first, fresh_out))
    report["fresh_instance_max_delta"] = float((first - fresh_out).abs().max())
except Exception as error:  # noqa: BLE001 -- a probe records failures, it does not raise
    report["temporal_state"] = {
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc().splitlines()[-3:],
    }

# 3. Factor range. ESRGAN produces exactly its trained factor; if nvvfx accepts arbitrary output
#    sizes then the chain can hit a target exactly and skip the resample stage entirely.
# 1.33x is in here because NVIDIA's own SDK samples expose it; if it is accepted, the UI can offer
# a factor that lands 1080p on 1440p exactly.
for label, (w, h) in {
    "1.33x": (852, 480), "1.5x": (960, 540), "2x": (1280, 720), "3x": (1920, 1080),
    "4x": (2560, 1440), "5x_over_limit": (3200, 1800),
    "non_uniform": (1600, 720), "odd_target": (1281, 721),
}.items():
    try:
        trial = VideoSuperRes(quality=VideoSuperRes.QualityLevel.HIGH)
        trial.output_width, trial.output_height = w, h
        trial.load()
        out = torch.from_dlpack(trial.run(frame).image).clone()
        report.setdefault("factors", {})[label] = {"requested": [w, h], "got": list(out.shape[-2:][::-1])}
    except Exception as error:
        report.setdefault("factors", {})[label] = {"requested": [w, h], "error": f"{type(error).__name__}: {error}"}

# 4. Denoise at native resolution -- output size equal to input, no upscale at all.
try:
    dn = VideoSuperRes(quality=VideoSuperRes.QualityLevel.DENOISE_HIGH)
    dn.output_width, dn.output_height = BASE_W, BASE_H
    dn.load()
    out = torch.from_dlpack(dn.run(frame).image).clone()
    report["denoise_native"] = {"shape": list(out.shape), "changed_pixels": bool(not torch.equal(out, frame))}
except Exception as error:
    report["denoise_native"] = {"error": f"{type(error).__name__}: {error}"}

# 5. Does one instance handle a different input size than it first saw?
try:
    small = torch.rand(3, BASE_H // 2, BASE_W // 2, device="cuda", dtype=torch.float32)
    out = torch.from_dlpack(effect.run(small).image).clone()
    report["varying_input_size"] = {"ok": True, "shape": list(out.shape)}
except Exception as error:
    report["varying_input_size"] = {"ok": False, "error": f"{type(error).__name__}: {error}"}

# 6. Below the documented 360p minimum. "Suggested" is not "enforced", and which one it is decides
#    whether the editor has to refuse a small source or merely warn about it.
try:
    tiny = torch.rand(3, 240, 426, device="cuda", dtype=torch.float32)
    below = VideoSuperRes(quality=VideoSuperRes.QualityLevel.HIGH)
    below.output_width, below.output_height = 852, 480
    below.load()
    out = torch.from_dlpack(below.run(tiny).image).clone()
    report["below_360p_input"] = {"ok": True, "shape": list(out.shape)}
except Exception as error:
    report["below_360p_input"] = {"ok": False, "error": f"{type(error).__name__}: {error}"}

# 7. THE batching question. The frame loop currently pays a Python round trip and a DLPack handoff
#    per frame, ~2,900 times for two minutes at 24 fps. If run() takes an (N, 3, H, W) batch the
#    loop collapses into one call per batch; if it does not, batching has to be a pipeline of
#    single calls instead, and the answer changes what gets written rather than how fast it is.
for count in (2, 4, 8):
    key = f"batch_{count}"
    try:
        stack = torch.rand(count, 3, BASE_H, BASE_W, device="cuda", dtype=torch.float32)
        batched = VideoSuperRes(quality=VideoSuperRes.QualityLevel.HIGH)
        batched.output_width, batched.output_height = BASE_W * 2, BASE_H * 2
        batched.load()
        out = torch.from_dlpack(batched.run(stack).image).clone()
        # Accepting the call is not the same as processing every frame: a binding that silently
        # takes the first frame would return a 3-D result, or an N of 1.
        report[key] = {"ok": True, "shape": list(out.shape), "kept_all_frames": out.shape[0] == count
                       if out.dim() == 4 else False}
    except Exception as error:
        report[key] = {"ok": False, "error": f"{type(error).__name__}: {error}"}

