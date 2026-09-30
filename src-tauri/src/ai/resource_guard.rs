//! Resource guard: a pre-flight check of Windows commit, RAM and VRAM before a GPU job starts.
//!
//! Why this exists: on 2026-09-24 LTX and MiniMax-H3 died with 0xC0000005 instead of a clean
//! out-of-memory error, because an idle `llama-server` held 23 GB of commit. On Windows the
//! commit limit, not physical RAM, is the real ceiling, and running into it inside a native
//! allocator looks like a random access violation. This module measures the machine, compares it
//! with a per-job requirement, and names the processes holding the memory. It never kills
//! anything; the caller shows the verdict and the user decides.
//!
//! Layout:
//! - [`REQUIREMENTS`] is the one editable table of what each job needs.
//! - [`evaluate`] is pure (numbers and a process list in, a verdict out) and unit-tested.
//! - [`measure`] reads the machine. Windows only; elsewhere it returns an empty snapshot that
//!   always passes, so nothing is ever blocked on a platform we cannot measure.
//! - [`JobMonitor`] samples the machine while a job holds the gate and appends the peaks to
//!   `.aiv/resource-log.jsonl`. That log is how the "estimate" rows in the table become measured.
//!
//! This file deliberately has no Tauri dependency. The gate glue lives in `ai.rs`.

use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    io::Write,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

fn gb(bytes: u64) -> f64 {
    bytes as f64 / GIB
}

// ------------------------------------------------------------------------------------------------
// The requirement table. EDIT HERE.
// ------------------------------------------------------------------------------------------------

/// Every job that takes the AI gate. Its kebab-case serde name is the `id` the frontend and the
/// log use (a test pins the two together).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum JobKind {
    Ltx,
    LtxRetake,
    LtxFoley,
    MinimaxH3,
    MinimaxH3Ref,
    MossSfx,
    KiwiEdit,
    Sam2Track,
    MaskEffect,
    Whisper,
    ZImageTurbo,
    ZImageBase,
    ZImageCaption,
    ZImageTraining,
    QwenEdit,
    BackgroundRemoval,
    PhotoModel,
    WanAlpha,
    PointTrack,
    VideoDepth,
    WanAnimate,
    PromptRewrite,
}

/// What one job needs free before it starts. Units are GiB, which is what Task Manager calls GB.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Requirement {
    pub job: JobKind,
    pub id: &'static str,
    pub label: &'static str,
    /// Free commit (`ullAvailPageFile`) the job needs at its peak.
    pub commit_gb: f64,
    /// Free VRAM the job needs. WDDM, the desktop and this app's own WebView2 hold some, so a 16 GB
    /// card shows about 15 free at best. These catch *another* process holding VRAM (a local LLM
    /// with GPU layers); the block-streaming runtimes need less than their totals suggest.
    pub vram_gb: f64,
    /// `measured <date>` or `estimate`. Shown in the warning so an estimate is never mistaken for
    /// a measurement.
    pub source: &'static str,
}

/// Conservative starting values. Replace an `estimate` with the peak from
/// `.aiv/resource-log.jsonl` (`appTreePeakGb` plus a margin) once a real run has been logged.
pub const REQUIREMENTS: &[Requirement] = &[
    // 27.6 GB mapped FP8 checkpoint plus ~19 GB pinned blocks, plus margin (Phase 0 handoff).
    Requirement { job: JobKind::Ltx, id: "ltx", label: "LTX generate / extend / bridge", commit_gb: 55.0, vram_gb: 12.0, source: "handoff 2026-09-26 (27.6 mapped + ~19 pinned)" },
    Requirement { job: JobKind::LtxRetake, id: "ltx-retake", label: "LTX retake", commit_gb: 55.0, vram_gb: 12.0, source: "same checkpoint as LTX" },
    Requirement { job: JobKind::LtxFoley, id: "ltx-foley", label: "LTX foley", commit_gb: 55.0, vram_gb: 12.0, source: "same checkpoint as LTX" },
    // Two processes: the peak is max(int8 conditioner ~25 GiB, FP8 denoiser 20 GiB), not the sum.
    Requirement { job: JobKind::MinimaxH3, id: "minimax-h3", label: "MiniMax-H3", commit_gb: 40.0, vram_gb: 12.0, source: "estimate" },
    Requirement { job: JobKind::MinimaxH3Ref, id: "minimax-h3-ref", label: "MiniMax-H3 Ref2VA", commit_gb: 40.0, vram_gb: 12.0, source: "estimate" },
    Requirement { job: JobKind::MossSfx, id: "moss-sfx", label: "MOSS sound effects", commit_gb: 16.0, vram_gb: 8.0, source: "estimate" },
    Requirement { job: JobKind::KiwiEdit, id: "kiwi-edit", label: "Kiwi edit", commit_gb: 30.0, vram_gb: 10.0, source: "estimate" },
    Requirement { job: JobKind::Sam2Track, id: "sam2-track", label: "SAM 2 tracking", commit_gb: 8.0, vram_gb: 4.0, source: "estimate" },
    Requirement { job: JobKind::MaskEffect, id: "mask-effect", label: "Mask effect render", commit_gb: 6.0, vram_gb: 2.0, source: "estimate" },
    Requirement { job: JobKind::Whisper, id: "whisper", label: "Whisper transcription", commit_gb: 8.0, vram_gb: 5.0, source: "estimate" },
    Requirement { job: JobKind::ZImageTurbo, id: "z-image-turbo", label: "Z-Image Turbo", commit_gb: 28.0, vram_gb: 10.0, source: "estimate" },
    Requirement { job: JobKind::ZImageBase, id: "z-image-base", label: "Z-Image Base", commit_gb: 28.0, vram_gb: 10.0, source: "estimate" },
    Requirement { job: JobKind::ZImageCaption, id: "z-image-caption", label: "LoRA captioning", commit_gb: 20.0, vram_gb: 10.0, source: "estimate" },
    Requirement { job: JobKind::ZImageTraining, id: "z-image-training", label: "Z-Image LoRA training", commit_gb: 40.0, vram_gb: 12.0, source: "estimate" },
    Requirement { job: JobKind::QwenEdit, id: "qwen-edit", label: "Qwen-Image-Edit", commit_gb: 45.0, vram_gb: 12.0, source: "estimate" },
    Requirement { job: JobKind::BackgroundRemoval, id: "background-removal", label: "Background removal", commit_gb: 8.0, vram_gb: 4.0, source: "estimate" },
    Requirement { job: JobKind::PhotoModel, id: "photo-model", label: "Photo Edit model step", commit_gb: 8.0, vram_gb: 4.0, source: "estimate" },
    // Three processes, peak = max(T5 11.4 GB, FP8 DiT ~14 GB + streaming buffers, VAE). Replace
    // with the Phase 0.6 measurement (appTreePeakGb in .aiv/resource-log.jsonl, plus margin).
    Requirement { job: JobKind::WanAlpha, id: "wan-alpha", label: "Wan Alpha", commit_gb: 32.0, vram_gb: 13.0, source: "measured 2026-09-27: app-tree commit peak 27.4 GB, VRAM 13.1 GB system-wide (81 frames)" },
    // VFX Phase 4. CoTracker3 offline runs the whole range at once (decoded at most 768 wide). Measured
    // on 124 frames; its VRAM grows with the frame count, so a long clip may need more than this.
    Requirement { job: JobKind::PointTrack, id: "point-track", label: "CoTracker3 point tracking", commit_gb: 20.0, vram_gb: 14.0, source: "measured 2026-09-28: 124 frames decoded at 768x436, app-tree commit peak 17.0 GB, VRAM +13.6 GB (14.6 system-wide)" },
    // Video Depth Anything Small (~0.1 GB) in 32-frame windows at 518; frames (<=1280 long edge) and
    // float32 depth for up to 1500 frames are both held in RAM before the 8-bit sequence is written.
    Requirement { job: JobKind::VideoDepth, id: "video-depth", label: "Video Depth Anything", commit_gb: 20.0, vram_gb: 12.0, source: "measured 2026-09-28: 124 frames at 960x544, app-tree commit peak 17.2 GB, VRAM +11.5 GB (12.5 system-wide)" },
    // VFX Phase 5. Two processes: T5 (14.7 GB commit, 10.9 GB VRAM), then the FP8 DiT + CLIP + VAE + the
    // INT8 reference cache in pinned RAM (3.5 GB for 33-frame segments at 672x384).
    Requirement { job: JobKind::WanAnimate, id: "wan-animate", label: "Wan Animate 2", commit_gb: 40.0, vram_gb: 12.0, source: "measured 2026-09-29: 75 frames at 672x384 (3 x 33-frame segments), app-tree commit peak 33.5 GB, VRAM 11.5 GB system-wide" },
    // Qwen2.5-VL-7B at 4-bit (bitsandbytes NF4), loaded from the 16 GB bf16 shards, plus up to
    // three pictures of ~1300 vision tokens each.
    Requirement { job: JobKind::PromptRewrite, id: "prompt-rewrite", label: "Qwen edit prompt rewrite", commit_gb: 20.0, vram_gb: 8.0, source: "estimate" },
];

pub fn requirement(job: JobKind) -> Requirement {
    *REQUIREMENTS.iter().find(|item| item.job == job).expect("every JobKind has a REQUIREMENTS row")
}

// ------------------------------------------------------------------------------------------------
// Snapshot and verdict.
// ------------------------------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessInfo {
    pub pid: u32,
    pub parent_pid: u32,
    pub name: String,
    /// Private bytes: the commit this process is charged for.
    pub private_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VramInfo {
    pub total: u64,
    pub free: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Snapshot {
    /// False when the platform could not be measured; [`evaluate`] then always passes.
    pub measured: bool,
    pub commit_limit: u64,
    pub commit_free: u64,
    pub ram_total: u64,
    pub ram_free: u64,
    pub vram: Option<VramInfo>,
    /// `nvml`, `nvidia-smi` or `unavailable`.
    pub vram_source: String,
    pub processes: Vec<ProcessInfo>,
    pub self_pid: u32,
    pub measure_ms: f64,
}

/// How a holder is classified in the warning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum HolderKind {
    /// `llama-server`, `ollama`, `koboldcpp`, `vmmem`, a Python that is not ours, ...
    KnownOffender,
    /// A descendant of this app (its WebView2, or one of its model workers).
    ThisApp,
    Other,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Holder {
    pub pid: u32,
    pub name: String,
    pub private_gb: f64,
    pub kind: HolderKind,
    /// Why it is flagged, e.g. "local LLM server". Empty for `Other`.
    pub note: String,
    /// True for an app worker whose memory is released when this job starts anyway.
    pub reclaimable: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Verdict {
    pub job: JobKind,
    pub job_id: String,
    pub label: String,
    pub requirement_source: String,
    pub ok: bool,
    pub measured: bool,
    pub commit_needed_gb: f64,
    pub commit_free_gb: f64,
    pub commit_limit_gb: f64,
    /// Commit held by app workers this job will replace; counted as available.
    pub commit_reclaimable_gb: f64,
    pub vram_needed_gb: f64,
    pub vram_free_gb: Option<f64>,
    pub vram_total_gb: Option<f64>,
    pub vram_source: String,
    pub ram_free_gb: f64,
    pub ram_total_gb: f64,
    /// Human-readable reasons the job is short. Empty when `ok`.
    pub shortfalls: Vec<String>,
    /// The processes worth naming, largest first.
    pub holders: Vec<Holder>,
    /// True when one of the holders is a resident app worker the user may release.
    pub has_releasable_app_worker: bool,
    pub measure_ms: f64,
}

impl Verdict {
    /// One line for an error message or a log.
    pub fn summary(&self) -> String {
        if self.ok {
            return format!("{}: resources OK", self.label);
        }
        let mut text = format!("{}: {}", self.label, self.shortfalls.join("; "));
        let named: Vec<String> = self.holders.iter()
            .filter(|holder| holder.kind != HolderKind::Other)
            .take(3)
            .map(|holder| format!("{} (pid {}) {:.1} GB", holder.name, holder.pid, holder.private_gb))
            .collect();
        if !named.is_empty() {
            text.push_str(&format!(". Held by: {}", named.join(", ")));
        }
        text
    }
}

/// Names (lower-case, without `.exe`) worth flagging, and why.
const KNOWN_OFFENDERS: &[(&str, &str)] = &[
    ("llama-server", "local LLM server"),
    ("llama-cli", "local LLM"),
    ("ollama", "Ollama"),
    ("ollama_llama_server", "Ollama model runner"),
    ("ollama app", "Ollama"),
    ("koboldcpp", "local LLM server"),
    ("lm studio", "LM Studio"),
    ("lms", "LM Studio"),
    ("vmmem", "WSL / virtual machine"),
    ("vmmemwsl", "WSL"),
];

const PYTHON_NAMES: &[&str] = &["python", "pythonw", "python3"];

/// Holders smaller than this are not worth a line in the warning.
const MIN_FLAGGED_BYTES: u64 = 512 * 1024 * 1024;
const MIN_OTHER_BYTES: u64 = 1024 * 1024 * 1024;
const MAX_HOLDERS: usize = 10;

pub fn normalise_name(name: &str) -> String {
    let lower = name.trim().to_ascii_lowercase();
    lower.strip_suffix(".exe").map(str::to_owned).unwrap_or(lower)
}

/// The pids descending from `root` (not including it), following `parent_pid`. Windows reuses
/// pids, so a stale parent can point anywhere: the walk is bounded and ignores cycles.
pub fn descendants(processes: &[ProcessInfo], root: u32) -> HashSet<u32> {
    let mut children: HashMap<u32, Vec<u32>> = HashMap::new();
    for process in processes {
        if process.pid != process.parent_pid {
            children.entry(process.parent_pid).or_default().push(process.pid);
        }
    }
    let mut found = HashSet::new();
    let mut stack = vec![(root, 0usize)];
    while let Some((pid, depth)) = stack.pop() {
        if depth > 16 {
            continue;
        }
        for child in children.get(&pid).into_iter().flatten() {
            if *child != root && found.insert(*child) {
                stack.push((*child, depth + 1));
            }
        }
    }
    found
}

fn classify(process: &ProcessInfo, owned: &HashSet<u32>, self_pid: u32) -> (HolderKind, String) {
    if process.pid == self_pid || owned.contains(&process.pid) {
        let name = normalise_name(&process.name);
        let note = if PYTHON_NAMES.contains(&name.as_str()) { "model worker of this app" }
            else if name == "msedgewebview2" { "this app's WebView" }
            else { "this app" };
        return (HolderKind::ThisApp, note.to_owned());
    }
    let name = normalise_name(&process.name);
    if let Some((_, why)) = KNOWN_OFFENDERS.iter().find(|(known, _)| *known == name) {
        return (HolderKind::KnownOffender, (*why).to_owned());
    }
    if PYTHON_NAMES.contains(&name.as_str()) {
        return (HolderKind::KnownOffender, "Python process not started by this app".to_owned());
    }
    (HolderKind::Other, String::new())
}

/// The whole decision. Pure: no clocks, no I/O.
///
/// `reclaimable` lists app-worker pids that this job replaces (the resident LTX retake worker when
/// the job is a retake). Their private bytes count as free commit, and when any of them is alive
/// the VRAM check is skipped, because WDDM does not report per-process VRAM and that worker's VRAM
/// is what the job will reuse.
pub fn evaluate(requirement: &Requirement, snapshot: &Snapshot, reclaimable: &[u32]) -> Verdict {
    let owned = descendants(&snapshot.processes, snapshot.self_pid);
    let reclaim_set: HashSet<u32> = reclaimable.iter().copied().collect();
    let reclaimable_bytes: u64 = snapshot.processes.iter()
        .filter(|process| reclaim_set.contains(&process.pid))
        .map(|process| process.private_bytes)
        .sum();
    let reclaim_alive = snapshot.processes.iter().any(|process| reclaim_set.contains(&process.pid));

    let mut holders: Vec<Holder> = snapshot.processes.iter()
        .filter(|process| process.pid > 4)
        .filter_map(|process| {
            let (kind, note) = classify(process, &owned, snapshot.self_pid);
            let threshold = if kind == HolderKind::Other { MIN_OTHER_BYTES } else { MIN_FLAGGED_BYTES };
            (process.private_bytes >= threshold).then(|| Holder {
                pid: process.pid,
                name: process.name.clone(),
                private_gb: gb(process.private_bytes),
                kind,
                note,
                reclaimable: reclaim_set.contains(&process.pid),
            })
        })
        .collect();
    holders.sort_by(|a, b| b.private_gb.partial_cmp(&a.private_gb).unwrap_or(std::cmp::Ordering::Equal));
    holders.truncate(MAX_HOLDERS);

    let commit_free_gb = gb(snapshot.commit_free);
    let commit_reclaimable_gb = gb(reclaimable_bytes);
    let commit_available = commit_free_gb + commit_reclaimable_gb;
    let vram_free_gb = snapshot.vram.as_ref().map(|vram| gb(vram.free));
    let vram_total_gb = snapshot.vram.as_ref().map(|vram| gb(vram.total));

    let mut shortfalls = Vec::new();
    if snapshot.measured {
        if commit_available < requirement.commit_gb {
            shortfalls.push(format!(
                "needs {:.0} GB of free commit, {:.1} GB is free (commit limit {:.0} GB)",
                requirement.commit_gb, commit_available, gb(snapshot.commit_limit)
            ));
        }
        if let Some(free) = vram_free_gb {
            if !reclaim_alive && free < requirement.vram_gb {
                shortfalls.push(format!(
                    "needs {:.0} GB of free VRAM, {:.1} GB is free of {:.1} GB",
                    requirement.vram_gb, free, vram_total_gb.unwrap_or(0.0)
                ));
            }
        }
    }

    let has_releasable_app_worker = holders.iter()
        .any(|holder| holder.kind == HolderKind::ThisApp && holder.note == "model worker of this app" && !holder.reclaimable);

    Verdict {
        job: requirement.job,
        job_id: requirement.id.to_owned(),
        label: requirement.label.to_owned(),
        requirement_source: requirement.source.to_owned(),
        ok: shortfalls.is_empty(),
        measured: snapshot.measured,
        commit_needed_gb: requirement.commit_gb,
        commit_free_gb,
        commit_limit_gb: gb(snapshot.commit_limit),
        commit_reclaimable_gb,
        vram_needed_gb: requirement.vram_gb,
        vram_free_gb,
        vram_total_gb,
        vram_source: snapshot.vram_source.clone(),
        ram_free_gb: gb(snapshot.ram_free),
        ram_total_gb: gb(snapshot.ram_total),
        shortfalls,
        holders,
        has_releasable_app_worker,
        measure_ms: snapshot.measure_ms,
    }
}

/// Parses `nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits` (MiB).
/// Only the first GPU line is used.
#[cfg_attr(not(windows), allow(dead_code))]
pub fn parse_nvidia_smi(output: &str) -> Option<VramInfo> {
    let line = output.lines().map(str::trim).find(|line| !line.is_empty())?;
    let mut parts = line.split(',').map(|part| part.trim().parse::<u64>());
    let used = parts.next()?.ok()?;
    let total = parts.next()?.ok()?;
    if total == 0 || used > total {
        return None;
    }
    let mib = 1024 * 1024;
    Some(VramInfo { total: total * mib, free: (total - used) * mib })
}

// ------------------------------------------------------------------------------------------------
// Measurement.
// ------------------------------------------------------------------------------------------------

/// Measures the machine. Target cost is well under 200 ms; `measure_ms` records what it took.
pub fn measure() -> Snapshot {
    let started = Instant::now();
    let mut snapshot = platform::measure();
    snapshot.measure_ms = started.elapsed().as_secs_f64() * 1000.0;
    snapshot
}

/// System-wide commit in use, and VRAM in use, for the job monitor. Cheap (no process walk).
fn sample_usage() -> Option<(u64, Option<u64>)> {
    platform::usage()
}

/// Private bytes of this app and every descendant (its workers).
fn sample_app_tree_bytes() -> Option<u64> {
    let processes = platform::processes()?;
    let self_pid = std::process::id();
    let owned = descendants(&processes, self_pid);
    Some(processes.iter()
        .filter(|process| process.pid == self_pid || owned.contains(&process.pid))
        .map(|process| process.private_bytes)
        .sum())
}

#[cfg(any(windows, aiv_check_windows))]
mod platform {
    //! Raw kernel32 FFI plus NVML loaded at runtime, so the app needs no new crates and still
    //! starts on a machine without an NVIDIA driver.
    //!
    //! `aiv_check_windows` lets this compile (not link) on Linux for `cargo check`:
    //! `RUSTFLAGS="--cfg aiv_check_windows" cargo check`.
    #![allow(non_snake_case, clippy::upper_case_acronyms)]
    use super::{ProcessInfo, Snapshot, VramInfo};
    use std::{ffi::c_void, sync::OnceLock};

    type HANDLE = *mut c_void;
    type BOOL = i32;
    const INVALID_HANDLE_VALUE: HANDLE = -1isize as HANDLE;
    const TH32CS_SNAPPROCESS: u32 = 0x0000_0002;
    const PROCESS_QUERY_LIMITED_INFORMATION: u32 = 0x1000;

    #[repr(C)]
    struct MEMORYSTATUSEX {
        dwLength: u32,
        dwMemoryLoad: u32,
        ullTotalPhys: u64,
        ullAvailPhys: u64,
        ullTotalPageFile: u64,
        ullAvailPageFile: u64,
        ullTotalVirtual: u64,
        ullAvailVirtual: u64,
        ullAvailExtendedVirtual: u64,
    }

    #[repr(C)]
    struct PROCESSENTRY32W {
        dwSize: u32,
        cntUsage: u32,
        th32ProcessID: u32,
        th32DefaultHeapID: usize,
        th32ModuleID: u32,
        cntThreads: u32,
        th32ParentProcessID: u32,
        pcPriClassBase: i32,
        dwFlags: u32,
        szExeFile: [u16; 260],
    }

    #[repr(C)]
    struct PROCESS_MEMORY_COUNTERS_EX {
        cb: u32,
        PageFaultCount: u32,
        PeakWorkingSetSize: usize,
        WorkingSetSize: usize,
        QuotaPeakPagedPoolUsage: usize,
        QuotaPagedPoolUsage: usize,
        QuotaPeakNonPagedPoolUsage: usize,
        QuotaNonPagedPoolUsage: usize,
        PagefileUsage: usize,
        PeakPagefileUsage: usize,
        PrivateUsage: usize,
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MEMORYSTATUSEX) -> BOOL;
        fn CreateToolhelp32Snapshot(flags: u32, process_id: u32) -> HANDLE;
        fn Process32FirstW(snapshot: HANDLE, entry: *mut PROCESSENTRY32W) -> BOOL;
        fn Process32NextW(snapshot: HANDLE, entry: *mut PROCESSENTRY32W) -> BOOL;
        fn OpenProcess(access: u32, inherit: BOOL, process_id: u32) -> HANDLE;
        fn CloseHandle(handle: HANDLE) -> BOOL;
        fn K32GetProcessMemoryInfo(process: HANDLE, counters: *mut PROCESS_MEMORY_COUNTERS_EX, cb: u32) -> BOOL;
        fn LoadLibraryW(name: *const u16) -> HANDLE;
        fn GetProcAddress(module: HANDLE, name: *const u8) -> *mut c_void;
    }

    fn memory_status() -> Option<MEMORYSTATUSEX> {
        let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status)
    }

    fn private_bytes(pid: u32) -> Option<u64> {
        let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) };
        if handle.is_null() {
            return None;
        }
        let mut counters: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
        counters.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
        let ok = unsafe { K32GetProcessMemoryInfo(handle, &mut counters, counters.cb) } != 0;
        unsafe { CloseHandle(handle) };
        ok.then_some(counters.PrivateUsage as u64)
    }

    pub fn processes() -> Option<Vec<ProcessInfo>> {
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) };
        if snapshot == INVALID_HANDLE_VALUE || snapshot.is_null() {
            return None;
        }
        let mut entry: PROCESSENTRY32W = unsafe { std::mem::zeroed() };
        entry.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
        let mut list = Vec::with_capacity(400);
        let mut more = unsafe { Process32FirstW(snapshot, &mut entry) } != 0;
        while more {
            let length = entry.szExeFile.iter().position(|unit| *unit == 0).unwrap_or(entry.szExeFile.len());
            let name = String::from_utf16_lossy(&entry.szExeFile[..length]);
            let pid = entry.th32ProcessID;
            // Protected processes (System, Registry, Memory Compression) refuse the query; they
            // are not something the user can act on anyway, so they are simply skipped.
            if let Some(bytes) = private_bytes(pid) {
                list.push(ProcessInfo { pid, parent_pid: entry.th32ParentProcessID, name, private_bytes: bytes });
            }
            more = unsafe { Process32NextW(snapshot, &mut entry) } != 0;
        }
        unsafe { CloseHandle(snapshot) };
        Some(list)
    }

    // ---- NVML, loaded once and kept for the life of the app (its init is the slow part).

    #[repr(C)]
    struct NvmlMemory {
        total: u64,
        free: u64,
        used: u64,
    }

    type NvmlDevice = *mut c_void;
    type NvmlInit = unsafe extern "C" fn() -> i32;
    type NvmlGetHandle = unsafe extern "C" fn(u32, *mut NvmlDevice) -> i32;
    type NvmlGetMemory = unsafe extern "C" fn(NvmlDevice, *mut NvmlMemory) -> i32;

    struct Nvml {
        device: usize,
        get_memory: NvmlGetMemory,
    }

    static NVML: OnceLock<Option<Nvml>> = OnceLock::new();

    fn wide(text: &str) -> Vec<u16> {
        text.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn load_nvml() -> Option<Nvml> {
        let mut library = unsafe { LoadLibraryW(wide("nvml.dll").as_ptr()) };
        if library.is_null() {
            library = unsafe { LoadLibraryW(wide(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvml.dll").as_ptr()) };
        }
        if library.is_null() {
            return None;
        }
        let init = unsafe { GetProcAddress(library, b"nvmlInit_v2\0".as_ptr()) };
        let handle = unsafe { GetProcAddress(library, b"nvmlDeviceGetHandleByIndex_v2\0".as_ptr()) };
        let memory = unsafe { GetProcAddress(library, b"nvmlDeviceGetMemoryInfo\0".as_ptr()) };
        if init.is_null() || handle.is_null() || memory.is_null() {
            return None;
        }
        let init: NvmlInit = unsafe { std::mem::transmute(init) };
        let handle: NvmlGetHandle = unsafe { std::mem::transmute(handle) };
        let get_memory: NvmlGetMemory = unsafe { std::mem::transmute(memory) };
        if unsafe { init() } != 0 {
            return None;
        }
        let mut device: NvmlDevice = std::ptr::null_mut();
        if unsafe { handle(0, &mut device) } != 0 || device.is_null() {
            return None;
        }
        Some(Nvml { device: device as usize, get_memory })
    }

    fn nvml_memory() -> Option<VramInfo> {
        let nvml = NVML.get_or_init(load_nvml).as_ref()?;
        let mut memory = NvmlMemory { total: 0, free: 0, used: 0 };
        let status = unsafe { (nvml.get_memory)(nvml.device as NvmlDevice, &mut memory) };
        (status == 0 && memory.total > 0).then_some(VramInfo { total: memory.total, free: memory.free })
    }

    #[cfg(windows)]
    fn nvidia_smi_memory() -> Option<VramInfo> {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .ok()?;
        super::parse_nvidia_smi(&String::from_utf8_lossy(&output.stdout))
    }

    #[cfg(not(windows))]
    fn nvidia_smi_memory() -> Option<VramInfo> {
        None
    }

    fn vram() -> (Option<VramInfo>, &'static str) {
        if let Some(info) = nvml_memory() {
            return (Some(info), "nvml");
        }
        match nvidia_smi_memory() {
            Some(info) => (Some(info), "nvidia-smi"),
            None => (None, "unavailable"),
        }
    }

    pub fn measure() -> Snapshot {
        let Some(status) = memory_status() else {
            return Snapshot { vram_source: "unavailable".into(), self_pid: std::process::id(), ..Snapshot::default() };
        };
        let (vram, vram_source) = vram();
        Snapshot {
            measured: true,
            commit_limit: status.ullTotalPageFile,
            commit_free: status.ullAvailPageFile,
            ram_total: status.ullTotalPhys,
            ram_free: status.ullAvailPhys,
            vram,
            vram_source: vram_source.to_owned(),
            processes: processes().unwrap_or_default(),
            self_pid: std::process::id(),
            measure_ms: 0.0,
        }
    }

    pub fn usage() -> Option<(u64, Option<u64>)> {
        let status = memory_status()?;
        let vram_used = nvml_memory().map(|info| info.total.saturating_sub(info.free));
        Some((status.ullTotalPageFile.saturating_sub(status.ullAvailPageFile), vram_used))
    }
}

#[cfg(not(any(windows, aiv_check_windows)))]
mod platform {
    use super::{ProcessInfo, Snapshot};

    pub fn measure() -> Snapshot {
        Snapshot { vram_source: "unavailable".into(), self_pid: std::process::id(), ..Snapshot::default() }
    }

    pub fn usage() -> Option<(u64, Option<u64>)> {
        None
    }

    pub fn processes() -> Option<Vec<ProcessInfo>> {
        None
    }
}

// ------------------------------------------------------------------------------------------------
// Job monitor: records what a job really used, so the table can be corrected from evidence.
// ------------------------------------------------------------------------------------------------

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct MonitorRecord {
    job: String,
    started_unix: u64,
    seconds: f64,
    overridden: bool,
    preflight_ok: bool,
    commit_free_start_gb: f64,
    commit_limit_gb: f64,
    /// Peak system-wide commit in use during the job, minus what was in use at the start.
    commit_rise_peak_gb: f64,
    /// Peak private bytes of this app plus all its workers. The best single number for the table.
    app_tree_peak_gb: f64,
    app_tree_start_gb: f64,
    vram_used_start_gb: Option<f64>,
    vram_used_peak_gb: Option<f64>,
    samples: u32,
}

/// Samples while alive; on drop, the sampler thread writes one JSON line and exits. Dropping never
/// blocks the async runtime.
pub struct JobMonitor {
    stop: Arc<AtomicBool>,
}

impl JobMonitor {
    pub fn start(job: JobKind, log_path: PathBuf, verdict: &Verdict, overridden: bool) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let flag = stop.clone();
        let job_id = requirement(job).id.to_owned();
        let commit_free_start_gb = verdict.commit_free_gb;
        let commit_limit_gb = verdict.commit_limit_gb;
        let preflight_ok = verdict.ok;
        let spawned = std::thread::Builder::new().name(format!("aiv-monitor-{job_id}")).spawn(move || {
            let started = Instant::now();
            let started_unix = SystemTime::now().duration_since(UNIX_EPOCH).map(|value| value.as_secs()).unwrap_or(0);
            let Some((commit_start, vram_start)) = sample_usage() else { return };
            let tree_start = sample_app_tree_bytes().unwrap_or(0);
            let (mut commit_peak, mut vram_peak, mut tree_peak) = (commit_start, vram_start, tree_start);
            let mut samples = 0u32;
            let mut tick = 0u32;
            while !flag.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(500));
                tick += 1;
                if let Some((commit, vram)) = sample_usage() {
                    commit_peak = commit_peak.max(commit);
                    vram_peak = match (vram_peak, vram) { (Some(a), Some(b)) => Some(a.max(b)), (a, b) => a.or(b) };
                    samples += 1;
                }
                // The process walk costs a few tens of milliseconds; every 2 s is plenty.
                if tick % 4 == 0 {
                    if let Some(tree) = sample_app_tree_bytes() {
                        tree_peak = tree_peak.max(tree);
                    }
                }
            }
            let record = MonitorRecord {
                job: job_id,
                started_unix,
                seconds: started.elapsed().as_secs_f64(),
                overridden,
                preflight_ok,
                commit_free_start_gb,
                commit_limit_gb,
                commit_rise_peak_gb: gb(commit_peak.saturating_sub(commit_start)),
                app_tree_peak_gb: gb(tree_peak),
                app_tree_start_gb: gb(tree_start),
                vram_used_start_gb: vram_start.map(gb),
                vram_used_peak_gb: vram_peak.map(gb),
                samples,
            };
            if let Some(parent) = log_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            if let (Ok(mut file), Ok(line)) = (
                std::fs::OpenOptions::new().create(true).append(true).open(&log_path),
                serde_json::to_string(&record),
            ) {
                let _ = writeln!(file, "{line}");
            }
        });
        if spawned.is_err() {
            stop.store(true, Ordering::Relaxed);
        }
        Self { stop }
    }
}

impl Drop for JobMonitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

// ------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 1024 * 1024 * 1024;

    fn process(pid: u32, parent_pid: u32, name: &str, gb: f64) -> ProcessInfo {
        ProcessInfo { pid, parent_pid, name: name.into(), private_bytes: (gb * G as f64) as u64 }
    }

    /// The 2026-09-24 machine: 86 GB limit, llama-server holding 23 GB, so 24 GB free.
    fn crash_day() -> Snapshot {
        Snapshot {
            measured: true,
            commit_limit: 86 * G,
            commit_free: 24 * G,
            ram_total: 65 * G,
            ram_free: 30 * G,
            vram: Some(VramInfo { total: 16 * G, free: 15 * G }),
            vram_source: "nvml".into(),
            processes: vec![
                process(4, 0, "System", 0.1),
                process(100, 1, "explorer.exe", 0.3),
                process(200, 100, "aiv-studio.exe", 0.4),
                process(201, 200, "msedgewebview2.exe", 0.8),
                process(900, 100, "llama-server.exe", 23.0),
                process(950, 100, "chrome.exe", 2.5),
                process(960, 100, "python.exe", 1.2),
            ],
            self_pid: 200,
            measure_ms: 12.0,
        }
    }

    #[test]
    fn every_job_has_exactly_one_row() {
        let jobs = [
            JobKind::Ltx, JobKind::LtxRetake, JobKind::LtxFoley, JobKind::MinimaxH3, JobKind::MinimaxH3Ref,
            JobKind::MossSfx, JobKind::KiwiEdit, JobKind::Sam2Track, JobKind::MaskEffect, JobKind::Whisper,
            JobKind::ZImageTurbo, JobKind::ZImageBase, JobKind::ZImageCaption, JobKind::ZImageTraining,
            JobKind::QwenEdit, JobKind::BackgroundRemoval, JobKind::PhotoModel, JobKind::WanAlpha,
            JobKind::PointTrack, JobKind::VideoDepth, JobKind::WanAnimate, JobKind::PromptRewrite,
        ];
        for job in jobs {
            assert_eq!(REQUIREMENTS.iter().filter(|row| row.job == job).count(), 1, "{job:?}");
        }
        assert_eq!(REQUIREMENTS.len(), jobs.len());
        for row in REQUIREMENTS {
            assert_eq!(serde_json::to_value(row.job).unwrap(), row.id, "id must equal the serde name");
            assert_eq!(serde_json::from_value::<JobKind>(serde_json::json!(row.id)).unwrap(), row.job);
        }
        let ids: HashSet<_> = REQUIREMENTS.iter().map(|row| row.id).collect();
        assert_eq!(ids.len(), REQUIREMENTS.len(), "ids must be unique");
        for row in REQUIREMENTS {
            assert!(row.commit_gb > 0.0 && row.vram_gb > 0.0 && row.vram_gb <= 16.0, "{}", row.id);
        }
    }

    #[test]
    fn crash_day_blocks_ltx_and_names_llama_server_first() {
        let verdict = evaluate(&requirement(JobKind::Ltx), &crash_day(), &[]);
        assert!(!verdict.ok);
        assert_eq!(verdict.shortfalls.len(), 1, "{:?}", verdict.shortfalls);
        assert!(verdict.shortfalls[0].contains("commit"));
        let first = &verdict.holders[0];
        assert_eq!(first.name, "llama-server.exe");
        assert_eq!(first.kind, HolderKind::KnownOffender);
        assert!(verdict.summary().contains("llama-server.exe (pid 900) 23.0 GB"), "{}", verdict.summary());
    }

    #[test]
    fn freeing_llama_server_lets_ltx_run() {
        let mut snapshot = crash_day();
        snapshot.processes.retain(|item| item.pid != 900);
        snapshot.commit_free = 62 * G;
        assert!(evaluate(&requirement(JobKind::Ltx), &snapshot, &[]).ok);
    }

    #[test]
    fn small_jobs_pass_on_crash_day() {
        assert!(evaluate(&requirement(JobKind::Sam2Track), &crash_day(), &[]).ok);
    }

    #[test]
    fn classification_of_holders() {
        let verdict = evaluate(&requirement(JobKind::Ltx), &crash_day(), &[]);
        let kind = |name: &str| verdict.holders.iter().find(|item| item.name == name).map(|item| item.kind);
        assert_eq!(kind("python.exe"), Some(HolderKind::KnownOffender), "a Python we did not start");
        assert_eq!(kind("chrome.exe"), Some(HolderKind::Other));
        assert_eq!(kind("msedgewebview2.exe"), Some(HolderKind::ThisApp));
        assert_eq!(kind("explorer.exe"), None, "below the 1 GB threshold for unflagged processes");
        assert_eq!(kind("System"), None, "pid 4 is never listed");
        let sizes: Vec<f64> = verdict.holders.iter().map(|item| item.private_gb).collect();
        assert!(sizes.windows(2).all(|pair| pair[0] >= pair[1]), "largest first: {sizes:?}");
    }

    #[test]
    fn own_python_worker_is_not_an_offender_and_is_releasable() {
        let mut snapshot = crash_day();
        snapshot.processes.push(process(300, 200, "python.exe", 40.0));
        let verdict = evaluate(&requirement(JobKind::MinimaxH3), &snapshot, &[]);
        let worker = verdict.holders.iter().find(|item| item.pid == 300).unwrap();
        assert_eq!(worker.kind, HolderKind::ThisApp);
        assert!(verdict.has_releasable_app_worker);
    }

    #[test]
    fn resident_retake_worker_counts_as_free_for_a_retake() {
        let mut snapshot = crash_day();
        snapshot.processes.retain(|item| item.pid != 900);
        snapshot.processes.push(process(300, 200, "python.exe", 45.0));
        snapshot.commit_free = 20 * G;
        snapshot.vram = Some(VramInfo { total: 16 * G, free: 2 * G });
        let other = evaluate(&requirement(JobKind::Ltx), &snapshot, &[]);
        assert!(!other.ok, "another job sees the worker's memory as used");
        assert_eq!(other.shortfalls.len(), 2);
        let retake = evaluate(&requirement(JobKind::LtxRetake), &snapshot, &[300]);
        assert!(retake.ok, "{:?}", retake.shortfalls);
        assert!((retake.commit_reclaimable_gb - 45.0).abs() < 0.01);
        assert!(!retake.has_releasable_app_worker, "the reclaimable worker is not offered for release");
    }

    #[test]
    fn stale_reclaim_pid_changes_nothing() {
        let base = evaluate(&requirement(JobKind::Ltx), &crash_day(), &[]);
        let stale = evaluate(&requirement(JobKind::Ltx), &crash_day(), &[12345]);
        assert_eq!(base.ok, stale.ok);
        assert_eq!(stale.commit_reclaimable_gb, 0.0);
    }

    #[test]
    fn vram_shortfall_alone_blocks() {
        let mut snapshot = crash_day();
        snapshot.commit_free = 80 * G;
        snapshot.vram = Some(VramInfo { total: 16 * G, free: 6 * G });
        let verdict = evaluate(&requirement(JobKind::QwenEdit), &snapshot, &[]);
        assert!(!verdict.ok);
        assert!(verdict.shortfalls[0].contains("VRAM"));
    }

    #[test]
    fn unknown_vram_does_not_block() {
        let mut snapshot = crash_day();
        snapshot.commit_free = 80 * G;
        snapshot.vram = None;
        assert!(evaluate(&requirement(JobKind::QwenEdit), &snapshot, &[]).ok);
    }

    #[test]
    fn unmeasured_platform_always_passes() {
        let verdict = evaluate(&requirement(JobKind::WanAlpha), &Snapshot::default(), &[]);
        assert!(verdict.ok && !verdict.measured);
    }

    #[test]
    fn descendants_survive_pid_cycles() {
        let processes = vec![process(10, 20, "a", 1.0), process(20, 10, "b", 1.0), process(30, 10, "c", 1.0)];
        let found = descendants(&processes, 10);
        assert!(found.contains(&20) && found.contains(&30));
        assert!(!found.contains(&10));
    }

    #[test]
    fn nvidia_smi_parsing() {
        let info = parse_nvidia_smi("  1234, 16376\n").unwrap();
        assert_eq!(info.total, 16376 * 1024 * 1024);
        assert_eq!(info.free, (16376 - 1234) * 1024 * 1024);
        assert!(parse_nvidia_smi("").is_none());
        assert!(parse_nvidia_smi("[N/A], 16376").is_none());
        assert!(parse_nvidia_smi("20000, 16376").is_none());
    }

    #[test]
    fn names_normalise() {
        assert_eq!(normalise_name("Llama-Server.EXE"), "llama-server");
        assert_eq!(normalise_name("vmmem"), "vmmem");
    }

    #[test]
    fn verdict_serialises_camel_case() {
        let json = serde_json::to_value(evaluate(&requirement(JobKind::Ltx), &crash_day(), &[])).unwrap();
        assert_eq!(json["jobId"], "ltx");
        assert_eq!(json["holders"][0]["kind"], "known-offender");
        assert!(json["commitFreeGb"].as_f64().unwrap() > 23.9);
    }
}
