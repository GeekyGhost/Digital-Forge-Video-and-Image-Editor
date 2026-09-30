# Digital Forge — Video and Image Editor

#BETA BUILD - This is a beta build of a massive ststem. Keep this in mind if you decide to test it. I have not ran this on a fresh install yet. This is the culmination of a few years worth of work and itterations. After testing cleanup and finalizing a release. 

A local-first, non-destructive video editor built around a single NVIDIA RTX GPU with 16 GB VRAM.
The editor is fully useful with no AI model installed; generation, editing, restoration,
transcription, masking and interpolation all run as replaceable background adapters.

Everything runs on your machine. Nothing is uploaded anywhere.

- **Desktop shell** — Tauri 2
- **Interface** — React + TypeScript
- **Native core** — Rust
- **Media** — FFmpeg / FFprobe
- **AI runtime** — isolated Python JSON-lines workers, one process per job
- **Project format** — versioned `.aivproj` JSON with rational timeline time

---

## Requirements

| | Minimum |
|---|---|
| OS | Windows 10/11 (the launchers and setup scripts are PowerShell) |
| GPU | NVIDIA, **16 GB VRAM** — the bootstrap refuses below 15,360 MiB |
| Driver | Current NVIDIA Studio driver (`nvidia-smi` must be on `PATH`) |
| Python | Managed by `uv`; the canonical application runtime is **3.10.11** |
| Node.js | Current LTS, with `npm` |
| Rust | stable, via [rustup](https://rustup.rs) — only needed to build from source |
| FFmpeg | `ffmpeg` **and** `ffprobe` on `PATH` |
| Disk | ~50 GB for the editor and base packs; **far more** for optional model packs |

The editor itself — import, timeline, trim, transitions, titles, export — needs none of the AI
packs. Install those only for the features you want.

---

## Quick start

1. Download this repository — **Code → Download ZIP**, or:

   ```bat
   git clone https://github.com/GeekyGhost/Digital-Forge-Video-and-Image-Editor.git C:\DigitalForge
   ```

   Keep the path short (for example `C:\DigitalForge`); Python packages nest deeply.

2. Double-click **`Install-Digital-Forge.bat`**.

   The guided installer checks your system and offers to install anything missing with `winget`
   (FFmpeg, Node.js, Git, WebView2 — and Rust plus the Visual C++ build tools only if you build
   from source). It then sets up the editor, gets the desktop app (download the prebuilt one from
   [Releases](https://github.com/GeekyGhost/Digital-Forge-Video-and-Image-Editor/releases) or build
   it), and lets you pick which optional AI features and model packs to install. Every model
   pack's size and licence is shown before anything is downloaded.

3. Double-click **`Launch-AIV-Studio.vbs`** to start the editor (or **`Launch-AIV-Studio.bat`** to
   watch the startup checks). Diagnostics append to `.aiv\launcher.log`.

Everything — Python interpreters, environments, package cache, models, caches and the app — is
installed **inside this folder**. Deleting the folder removes it all. Run the installer again at
any time to resume an interrupted install or add more features.

**[docs/guide/index.html](docs/guide/index.html)** is the interactive user guide: every workspace, tool
and model, step-by-step recipes (including one shot from keyframes to export), shortcuts and limits.
Open the file in any browser; it works offline. With GitHub Pages serving `/docs`, it is also at
<https://geekyghost.github.io/Digital-Forge-Video-and-Image-Editor/guide/>.

**[docs/INSTALL.md](docs/INSTALL.md)** has the full install guide: prerequisites, the feature list with
download sizes, gated models, unattended install options, updating, uninstalling and
troubleshooting.

### Manual setup (developers)

The installer drives the same scripts you can run yourself:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-and-launch.ps1 -NoLaunch
```

That performs an idempotent first-run setup: checks FFmpeg/GPU, installs JavaScript dependencies,
creates `.venv`, installs the pinned **CUDA** build of PyTorch, installs the worker package. If an
existing interpreter cannot start, setup replaces that venv automatically. If `uv` is not
installed, the setup scripts download it into `.aiv\tools`; uv-managed Python interpreters and the
package cache are kept in `.aiv\python` and `.aiv\uv-cache`.

The lower-level development bootstrap:

```powershell
.\scripts\bootstrap.ps1
npm.cmd run tauri dev
```

---

## Vendored sources and isolated environments

The exact Python, Torch and CUDA choices are declared once in `scripts/runtime-matrix.psd1`.
Application, LTX, MOSS, Image Studio/SAM 2, Z-Image training and MiniMax-H3 each have a separate environment; their
setup scripts never install into one another. `uv` installs missing exact Python versions and is
cached under `.aiv/tools` so a broken application venv can still be repaired.

**`vendor/` is not part of this repository** and is excluded from git. Two different things live
there, and only one of them needs fetching.

### One real upstream source: LTX-2

LTX generation, retake and foley are built against a **pinned** revision of Lightricks' LTX-2, and
`setup-ltx.ps1` verifies the checkout and the pristine `uv.lock`, runs `uv sync --frozen`, then
applies the project's documented Windows CUDA overlay from the runtime matrix. The upstream lock is
never edited to carry local wheel choices.

| | |
|---|---|
| Repository | `https://github.com/Lightricks/LTX-2.git` |
| Pinned commit | `4f8905737aac86a554637cac86c178877a39c744` |

Fetch just that commit — a shallow fetch by SHA is a few seconds and about 5 MB:

```powershell
git init vendor\LTX-2
git -C vendor\LTX-2 remote add origin https://github.com/Lightricks/LTX-2.git
git -C vendor\LTX-2 fetch --depth 1 origin 4f8905737aac86a554637cac86c178877a39c744
git -C vendor\LTX-2 checkout FETCH_HEAD
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-ltx.ps1
```

> Upstream has moved on and tagged `v1.2.0`. **Do not silently upgrade.** The validated 16 GB
> profile, the scaled-FP8 streaming path and the retake geometry contract were all measured against
> the commit above, and the adapters read LTX internals directly. Moving forward is a deliberate,
> re-measured change.

To deliberately rebuild the application and LTX runtimes:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-app-env.ps1 -Recreate
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-ltx.ps1 -Recreate
```

### The rest are just environment homes

`vendor\MiniMax-H3` and `vendor\Image-Studio` contain **only a `.venv`**. The optional
`vendor\Z-Image-Trainer` also contains its pinned Diffusers training source. Run the environments
directly on a fresh clone:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-h3.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-image-studio.ps1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup-zimage-trainer.ps1
```

Each feature gets its own environment on purpose: their PyTorch and `diffusers` requirements
conflict and cannot share one.

The **AI clip editor** needs nothing under `vendor/` either — Kiwi-Edit runs in the application
`.venv`, and its `pipeline_kiwi_edit.py` ships inside the `kiwi-edit` model pack.

> `Install-Digital-Forge.bat` offers every one of these environments from a menu. The launcher
> (`setup-and-launch.ps1`) only reports which ones are missing; it does not install them. The
> editor itself runs fine without any of them.

---

## Model packs

Weights are never committed. Install them declaratively — each manifest carries capabilities,
a licence reference and a disk budget, and nothing is pulled as a hidden dependency:

```powershell
.\.venv\Scripts\python.exe .\scripts\models.py list
.\.venv\Scripts\python.exe .\scripts\models.py install utilities --accept-licenses
.\.venv\Scripts\python.exe .\scripts\models.py install ltx-core --accept-licenses
.\.venv\Scripts\python.exe .\scripts\models.py install kiwi-edit --accept-licenses
.\.venv\Scripts\python.exe .\scripts\models.py install image-studio --accept-licenses
.\.venv\Scripts\python.exe .\scripts\models.py install z-image-base --accept-licenses
.\.venv\Scripts\python.exe .\scripts\models.py install minimax-h3 --accept-licenses
.\.venv\Scripts\python.exe .\scripts\models.py install minimax-h3-ref2va --accept-licenses
```

`ltx-core` and the H3 packs are large. The installer checks free disk space first, downloads into
`models/.staging`, verifies the snapshot, then activates it atomically.

**MOSS-SoundEffect v2** uses its own process and Python environment so its Torch pins never mix
with LTX. After placing the model at `models\moss-sfx-v2`, prepare or verify that runtime with:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\install-moss-sfx.ps1
```

**Google's Gemma encoder is gated.** Accept its terms on Hugging Face, authenticate locally, then:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\complete-gemma.ps1
```

**MiniMax-H3** additionally needs its text encoder baked once, using the H3 interpreter — not the
utility one:

```powershell
vendor\MiniMax-H3\.venv\Scripts\python.exe scripts\bake-h3-encoder.py --prune
```

---

## Build

```powershell
npm.cmd run release
```

Type-checks, bundles the interface, compiles the Tauri shell, and copies the result to
`Releases\aiv-studio-<version>.exe`. The launcher always runs the **newest** executable it finds in
`Releases\` or `src-tauri\target\release\`, and warns when that binary is older than your source —
so a stale build can't masquerade as your latest changes.

`Build-AIV-Studio.bat` does the same locked, tested release build. `Test-AIV-Studio.bat` runs the
qualification suite without publishing an executable — 15/24/30 FPS, VFR conforming, trimmed
ranges, long Kiwi windows and seams, clip-extension boundary/audio sync, timeline gaps, overlays,
captions, audio mixing and export duration.

Individual steps:

```powershell
npm.cmd run check
npm.cmd run build
npm.cmd run tauri build
```

---

## What it does

**Editing.** Multi-track timeline (V2 graphics, V1 picture, A1 audio, C1 captions) on exact
rational time. Selection, blade and track-select tools, trim, split, ripple delete, group move,
undo/redo, track lock/mute/visibility, toggleable snapping. Real FFmpeg thumbnails, filmstrips and
decoded waveforms.

**Transitions.** 58 qualified native transitions, each with an incoming and an outgoing side per
clip. At a cut the two fuse — the incoming clip's choice sets the look, the longer duration wins —
and the program monitor composites the result live, so the preview matches the render.

**Aspect and crop.** Per-clip conform with a sequence-wide default: contain, cover, stretch or
none, with black, white or **blur-fill** padding. The monitor mirrors the export's rules in CSS.

**AI clip editor** (Ctrl+K) — Visual Edit offers Kiwi-Edit, LTX Retake, **Mask Effects**, and
**Text Motion**. Mask Effects uses an automatic BiRefNet foreground or click-guided SAM 2 tracking
for glow, outline, aura, focus, privacy and movie looks, optionally driven by loudness, bass or
transients. Text Motion previews Type In/Out, fades, slides, pop, magic, glitch and looping motion,
then places editable text on G1/C1 instead of burning it into the picture. The SFX workspace creates
prompt-designed sounds with MOSS or picture-following Foley with LTX, auditions generated takes
against the clip, then places the chosen WAV on the SFX track with trims, gain, pan and fades.

**LTX 2.3 generation**, six modes on the distilled one-stage path: first+last frame, image→video,
text→video, clip extend, clip to clip, and video→audio foley. Plus **retake** — regenerate a range
of an existing clip.

**MiniMax-H3** — joint video *and* audio generation from frames or from mixed image/video/audio
references. Its Director workspace keeps the cast, shot plan, prompt, sampler settings and take
ledger inside the project. Each take retains its exact recipe and resolved seed for restoration;
aggregate preflight reports all blockers and timing/VRAM advisories before a long render starts.

**Image Studio** — a layer compositor with fast Z-Image Turbo generation, optional Z-Image Base
Creative generation, one- or four-seed candidate grids, ZiT/Base LoRA loading, Qwen Image Edit 2511
for image-conditioned edits and BiRefNet background removal. Its LoRA workspace can auto-caption a
reviewable local dataset with the installed Qwen or Gemma vision encoder and train Z-Image LoRAs in
an isolated runtime. Training saves configurable full-state checkpoints; **Save & pause** releases
the GPU for checkpoint testing and resumes the adapter, optimizer, scheduler and random state from
the exact saved step. A separate **Continue from LoRA** option starts a new training phase from an
existing adapter while intentionally resetting optimizer state. Generated stills feed back to the timeline.

**SAM 2 subject tracking** — click a subject in the monitor and propagate the mask across a range,
producing a matte, a cutout or a clean plate.

**Captions** — Whisper large-v3-turbo auto-captions the selected clip onto C1.

---

## Repository map

```
src/            UI and editor application state
src/domain/     framework-independent timeline/project logic
src-tauri/      native project, media probe, export and job commands
ai-runtime/     isolated model-worker protocol and adapters
models/         manifests only — weights are installed, never committed
scripts/        bootstrap, setup, model installation, diagnostics
docs/           architecture blueprint and implementation notes
```

See the [full blueprint](docs/AI_VIDEO_EDITOR_BLUEPRINT.md) and
[implementation status](docs/IMPLEMENTATION_STATUS.md).

---

## Credits

This editor is wrapped around other people's research. LTX-2, MOSS-SoundEffect, MiniMax-H3, Kiwi-Edit, Z-Image-Turbo,
Qwen Image Edit, BiRefNet, SAM 2, Whisper, Gemma and Demucs were all built and released by others,
and the distilled adapters and quantizations that make them fit on a 16 GB card were too.

**[ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) names everyone**, with links.

---

## Licence

Digital Forge Video Studio is released under the **Functional Source License 1.1 with an Apache 2.0
future licence** (`FSL-1.1-ALv2`). The full text is in [LICENSE.md](LICENSE.md).

In plain terms — this is a summary, not the licence, and the licence text governs:

- **You may** use it personally or commercially, modify it, and redistribute your changes.
- **You may** sell whatever you create with it. The licence covers the software, not your output.
- **You may** use it internally, including at a company, and in professional services you provide.
- **You may not** offer it to others as a commercial product or service that substitutes for it or
  provides substantially similar functionality — a hosted version, or a paid product with this
  editor inside it.
- **Each release becomes Apache-2.0 two years after it is published.** The restriction is
  time-limited, not permanent.

### Third-party licences are separate and may be stricter

This licence covers **this repository's own source code only**. It does not, and cannot, grant you
rights to anything you install alongside it:

- **Model weights** — LTX 2.3, MOSS-SoundEffect, MiniMax-H3, Kiwi-Edit, ZiT, Qwen, BiRefNet, SAM 2, Whisper and
  Gemma each carry their own licence and acceptable-use terms. **Some restrict commercial use.**
  Review the terms for every pack you install; `scripts/models.py` requires you to accept them
  explicitly for exactly this reason.
- **Vendored upstream sources** under `vendor/` remain under their own licences.
- **FFmpeg** is a separate program invoked as a subprocess; its licensing depends on the build you
  install.

If you intend to use this commercially, check the licence of every model pack you enable. That is
where the real constraints live.
