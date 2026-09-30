use serde::{Deserialize, Serialize};
use std::{path::Path, process::Stdio};
use tokio::process::Command;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PauseAnalysisRequest {
    pub path: String,
    pub source_in_seconds: f64,
    pub duration_seconds: f64,
    pub has_audio: bool,
    pub minimum_duration: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DetectedRange {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub duration_seconds: f64,
    pub kind: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PauseAnalysisResult {
    pub freezes: Vec<DetectedRange>,
    pub silences: Vec<DetectedRange>,
    pub candidates: Vec<DetectedRange>,
}

fn value_after(line: &str, marker: &str) -> Option<f64> {
    line.split_once(marker)?.1.trim().split_whitespace().next()?.parse().ok()
}

fn parse_ranges(log: &str, start_marker: &str, end_marker: &str, kind: &str, total_duration: f64) -> Vec<DetectedRange> {
    let mut start = None;
    let mut ranges = Vec::new();
    for line in log.lines() {
        if let Some(value) = value_after(line, start_marker) {
            start = Some(value.max(0.0));
        }
        if let Some(end) = value_after(line, end_marker) {
            if let Some(begin) = start.take() {
                if end > begin {
                    ranges.push(DetectedRange {
                        start_seconds: begin,
                        end_seconds: end,
                        duration_seconds: end - begin,
                        kind: kind.into(),
                    });
                }
            }
        }
    }
    if let Some(begin) = start {
        if total_duration > begin {
            ranges.push(DetectedRange {
                start_seconds: begin,
                end_seconds: total_duration,
                duration_seconds: total_duration - begin,
                kind: kind.into(),
            });
        }
    }
    ranges
}

fn intersect_ranges(freezes: &[DetectedRange], silences: &[DetectedRange], minimum: f64) -> Vec<DetectedRange> {
    let mut candidates = Vec::new();
    for freeze in freezes {
        for silence in silences {
            let start = freeze.start_seconds.max(silence.start_seconds);
            let end = freeze.end_seconds.min(silence.end_seconds);
            if end - start >= minimum {
                candidates.push(DetectedRange {
                    start_seconds: start,
                    end_seconds: end,
                    duration_seconds: end - start,
                    kind: "freeze+silence".into(),
                });
            }
        }
    }
    candidates
}

pub async fn analyze_pauses(request: PauseAnalysisRequest) -> Result<PauseAnalysisResult, String> {
    if !Path::new(&request.path).is_file() { return Err("analysis source does not exist".into()); }
    if request.duration_seconds <= 0.0 { return Err("analysis range is empty".into()); }
    let minimum = request.minimum_duration.unwrap_or(0.35).clamp(0.1, 10.0);
    let mut args = vec![
        "-hide_banner".into(), "-ss".into(), format!("{:.6}", request.source_in_seconds.max(0.0)),
        "-i".into(), request.path.clone(), "-t".into(), format!("{:.6}", request.duration_seconds),
        "-vf".into(), format!("setpts=PTS-STARTPTS,freezedetect=noise=-50dB:d={minimum:.3}"),
    ];
    if request.has_audio {
        args.extend(["-af".into(), format!("asetpts=PTS-STARTPTS,silencedetect=noise=-45dB:d={minimum:.3}")]);
    }
    args.extend(["-f".into(), "null".into(), "-".into()]);
    let output = Command::new("ffmpeg").args(&args).stdin(Stdio::null()).output().await
        .map_err(|error| format!("failed to launch ffmpeg analysis: {error}"))?;
    if !output.status.success() {
        return Err(String::from_utf8_lossy(&output.stderr).trim().to_owned());
    }
    let log = String::from_utf8_lossy(&output.stderr);
    let freezes = parse_ranges(&log, "freeze_start:", "freeze_end:", "freeze", request.duration_seconds);
    let silences = parse_ranges(&log, "silence_start:", "silence_end:", "silence", request.duration_seconds);
    let candidates = if request.has_audio {
        intersect_ranges(&freezes, &silences, minimum)
    } else {
        freezes.iter().filter(|range| range.duration_seconds >= minimum).cloned().collect()
    };
    Ok(PauseAnalysisResult { freezes, silences, candidates })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_intersects_detection_ranges() {
        let log = "freeze_start: 1.2\nfreeze_end: 2.0\nsilence_start: 1.4\nsilence_end: 2.1 | silence_duration: 0.7";
        let freezes = parse_ranges(log, "freeze_start:", "freeze_end:", "freeze", 3.0);
        let silences = parse_ranges(log, "silence_start:", "silence_end:", "silence", 3.0);
        let candidates = intersect_ranges(&freezes, &silences, 0.3);
        assert_eq!(freezes.len(), 1);
        assert_eq!(silences.len(), 1);
        assert_eq!(candidates[0].start_seconds, 1.4);
        assert_eq!(candidates[0].end_seconds, 2.0);
    }

    #[test]
    fn closes_open_ranges_at_clip_end() {
        let ranges = parse_ranges("freeze_start: 0.5", "freeze_start:", "freeze_end:", "freeze", 2.0);
        assert_eq!(ranges, vec![DetectedRange {
            start_seconds: 0.5, end_seconds: 2.0, duration_seconds: 1.5, kind: "freeze".into(),
        }]);
    }
}
