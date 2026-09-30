# Installing Digital Forge Video and Image Editor

This guide covers a fresh install on a new Windows machine. The short version:

1. Download the repository (green **Code → Download ZIP** button, or `git clone`) and extract it to
   a **short path**, for example `C:\DigitalForge`.
2. Double-click **`Install-Digital-Forge.bat`**.
3. When it finishes, double-click **`Launch-AIV-Studio.vbs`**.

Everything the installer creates lives inside that one folder — the Python interpreters, every
virtual environment, the package cache, model weights, caches, logs and the app itself. Deleting
the folder removes the whole installation.

---

## What you need

| | |
|---|---|
| **OS** | Windows 10 or 11, 64-bit |
| **GPU** | An NVIDIA GPU is required (setup installs CUDA PyTorch). The AI model profiles are tuned for **16 GB VRAM** (RTX 4080 / 4070 Ti Super / 5080 class); on smaller cards the editor works but most AI features will not fit. |
| **Driver** | A current NVIDIA Studio or Game Ready driver |
| **Disk** | About 20–40 GB for the editor and its base environment. AI model packs need much more — see the table below. An SSD is strongly recommended. |
| **RAM** | Developed on a 64 GB machine. The large video models memory-map checkpoints of ~28 GB, so RAM plus page file must cover that (see Troubleshooting). |
| **Internet** | Needed during installation only. Once installed, everything runs offline. |

### System prerequisites

The installer checks for each of these and offers to install any that are missing with `winget`
(Windows' built-in package manager). You can also install them yourself beforehand.

| Prerequisite | Needed for | winget id |
|---|---|---|
| NVIDIA driver (`nvidia-smi`) | everything | — install from [nvidia.com](https://www.nvidia.com/Download/index.aspx) |
| FFmpeg + FFprobe | everything | `Gyan.FFmpeg` |
| Node.js 20+ with npm | everything | `OpenJS.NodeJS.LTS` |
| Microsoft Edge WebView2 Runtime | running the app (preinstalled on Windows 11) | `Microsoft.EdgeWebView2Runtime` |
| Git | LTX and several AI runtimes | `Git.Git` |
| Rust (`cargo`) | **only** building the app from source | `Rustlang.Rustup` |
| Visual C++ build tools | **only** building the app from source | `Microsoft.VisualStudio.2022.BuildTools` (with the *Desktop development with C++* workload) |

Python is **not** a prerequisite. The installer downloads [uv](https://docs.astral.sh/uv/) into
`.aiv\tools`, and uv installs the exact Python versions each component needs into `.aiv\python`.

> If the installer installs a prerequisite and then still reports it missing, close the window and
> run `Install-Digital-Forge.bat` again — Windows only refreshes `PATH` for new windows.

---

## What the installer does

1. **Checks the system** — Windows version, prerequisites, GPU and VRAM, free disk space.
2. **Sets up the editor** — installs the JavaScript dependencies (`npm ci`), creates the application
   Python environment in `.venv` with the pinned CUDA build of PyTorch, and installs the AI worker
   package. This is the only required step.
3. **Gets the desktop app** — you choose one of:
   - **Download the prebuilt app** from the project's latest GitHub release into `Releases\`
     (fastest; no compilers needed), or
   - **Build it from source** (needs Rust and the Visual C++ build tools; 5–15 minutes the first
     time), or
   - **Skip** — `Launch-AIV-Studio.bat` then falls back to development mode.
4. **Optional AI features** — pick any from a menu (below). Each gets its own isolated environment.
5. **Model packs** — for the features you picked, it lists every model pack with its size and
   licence, and only downloads after you confirm you accept those licences.

Re-running the installer is always safe. Steps that are already done are skipped, so it resumes an
interrupted install and is also how you add features later.

---

## Optional AI features

| # | Name | What it enables | Model download |
|---|---|---|---|
| 1 | `core-models` | Whisper auto-captions, SAM 2 subject tracking, Real-ESRGAN upscaling | ~4 GB |
| 2 | `kiwi-edit` | AI clip editor (Kiwi-Edit 5B) | ~20 GB |
| 3 | `ltx` | LTX 2.3 generation (first/last frame, image→video, text→video, extend, clip-to-clip), retake and foley | ~103 GB, gated |
| 4 | `h3` | MiniMax-H3 joint video + audio generation, Director workspace, Ref2VA | ~84 GB |
| 5 | `image-studio` | Z-Image Turbo, Qwen Image Edit 2511, BiRefNet background removal, SAM 2 | ~67 GB |
| 6 | `zimage-trainer` | Z-Image Base creative mode and LoRA training (adds `image-studio`) | ~21 GB |
| 7 | `moss` | MOSS prompt-designed sound effects runtime | weights placed by hand |
| 8 | `motion` | CoTracker3 point tracking + video depth (CoTracker3 is **non-commercial**) | downloads its own |
| 9 | `wan-alpha` | Wan-Alpha transparent VFX elements | downloads its own (large) |
| 10 | `wan-animate` | Wan Animate 2 character insertion | downloads its own (large) |
| 11 | `photo` | Photo Edit relight and object removal (depth + LaMa) | ~100 MB |
| 12 | `rtx-upscale` | NVIDIA RTX Video Super Resolution (NVIDIA licence; adds `image-studio`) | from NVIDIA |

Sizes come from `models\manifests\*.json` and are estimates.

### Gated models (LTX)

LTX uses Google's Gemma text encoder and an LTX foley LoRA, both gated on Hugging Face. Before
downloading the `ltx` packs:

1. Create a free account at [huggingface.co](https://huggingface.co) and sign in.
2. Accept the terms on the [Gemma page](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized)
   and on the foley LoRA page (linked in `models\manifests\ltx-loras.json`).
3. Create a *read* token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. When the installer asks, log in and paste the token. (Or later: `.\.venv\Scripts\hf.exe auth login`.)

The Hugging Face token is stored by Hugging Face's own tool in your user profile
(`%USERPROFILE%\.cache\huggingface`), not in this folder.

### MOSS sound effects

The `moss` option prepares the runtime only. Download the MOSS-SoundEffect v2 weights from
[OpenMOSS](https://github.com/OpenMOSS/MOSS-TTS) and place them at `models\moss-sfx-v2`.

### Model licences

This project's own licence (FSL-1.1-ALv2, see `LICENSE.md`) covers **only this repository's source
code**. Every model has its own licence and some **restrict commercial use**. The installer shows
each pack's licence before downloading; read them before you use outputs commercially.
`ACKNOWLEDGEMENTS.md` lists every upstream project.

---

## Running the editor

The **[user guide](guide/index.html)** explains every workspace, tool and model once you are in.

- **`Launch-AIV-Studio.vbs`** — normal start, no console window. Diagnostics go to `.aiv\launcher.log`.
- **`Launch-AIV-Studio.bat`** — same, but shows the startup checks in a console.

Every launch re-checks the environment quickly and repairs it if a dependency file changed.

---

## Installer options

Everything the menu does can be given on the command line, for unattended installs:

```bat
Install-Digital-Forge.bat -Components image-studio,photo -DownloadModels -AcceptLicenses -Prebuilt
Install-Digital-Forge.bat -Components all -Build
Install-Digital-Forge.bat -Components none -NoBuild -NonInteractive
```

| Option | Meaning |
|---|---|
| `-Components a,b,c` | Features to install (names or menu numbers), `all`, or `none` |
| `-DownloadModels` | Also download those features' model packs |
| `-AcceptLicenses` | Confirms you have reviewed the licences of what is downloaded (required with `-NonInteractive`) |
| `-InstallPrerequisites` | Install missing prerequisites with winget without asking |
| `-Prebuilt` / `-Build` / `-NoBuild` | How to get the desktop app |
| `-NonInteractive` | Never prompt |

The individual setup scripts still work on their own — for example `Setup-Wan-Animate.bat`,
`Download-Photo-Models.bat`, `Install-RTX-Upscale.bat`, or any `scripts\setup-*.ps1`.

---

## Where everything lives

```
.aiv\tools\         uv
.aiv\python\        Python interpreters installed by uv
.aiv\uv-cache\      package cache (safe to delete; it is rebuilt on demand)
.aiv\*.log          installer and launcher logs
.venv\              the editor's Python environment
vendor\             LTX source and the isolated H3 / Image Studio / Wan / trainer environments
runtimes\           MOSS and motion environments
models\             model weights (manifests are in models\manifests)
cache\              thumbnails, waveforms, masks and other regenerable caches
projects\           your projects
exports\            your renders
Releases\           the desktop app executable
```

None of these are tracked by git.

---

## Updating

```bat
git pull
Install-Digital-Forge.bat
```

If you downloaded a ZIP, extract the new version over the old folder (your `projects\`, `models\`
and environments are untouched), then run the installer again. Get the matching app from the new
release, or rebuild with `Build-AIV-Studio.bat`.

## Uninstalling

Delete the folder. The only things outside it are the system prerequisites you chose to install
and, if you logged in, the Hugging Face token in `%USERPROFILE%\.cache\huggingface`.

---

## Troubleshooting

**A step failed.** The installer continues with the rest and lists failures at the end. Fix the
reported problem and run it again; finished steps are skipped. The full log is in
`.aiv\install-<date>.log`.

**"Path too long" errors while installing Python packages.** Move the folder to a shorter path
(for example `C:\DigitalForge`), or enable Windows long-path support from an administrator
PowerShell, then reboot:

```powershell
New-ItemProperty -Path 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1 -PropertyType DWORD -Force
```

**"The paging file is too small" (os error 1455) during generation.** The large video models
reserve a lot of Windows commit memory. Make sure the system drive has free space and the page
file is large enough. From an administrator PowerShell, `scripts\fix-pagefile.ps1 -ReportOnly`
shows the current commit limit, and `scripts\fix-pagefile.ps1` pins a 24–32 GB page file on C:
(reboot afterwards).

**The app says an AI feature is "not ready".** Run the installer again and select that feature;
the launcher also prints which components are missing when started with `Launch-AIV-Studio.bat`.

**Antivirus blocks or slows the install.** Downloads of Python wheels and model weights are large;
some antivirus products scan them slowly. Excluding this folder from real-time scanning helps.

---

## For maintainers: publishing a release

1. Bump `version` in `package.json` and `src-tauri\tauri.conf.json`.
2. Run `Build-AIV-Studio.bat`. It tests, builds and writes `Releases\aiv-studio-<version>.exe`.
3. Create a GitHub release and attach that `.exe`. The installer's *Download the prebuilt app*
   option fetches the `aiv-studio-*.exe` asset from the latest release.

The `.exe` is not a standalone installer: it runs from inside the project folder, next to
`ai-runtime\`, `scripts\` and `models\`, and is started by `Launch-AIV-Studio.vbs`.
