//! Host-side client for the `flux-ai --json` worker process.
//!
//! This runs on a normal-priority thread, never on the audio thread. The worker
//! prints one JSON event per line; `run_job` spawns it and collects the result.

use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "event", rename_all = "lowercase")]
pub enum Event {
    Started { job: String, input: String },
    Done { job: String, outputs: Vec<PathBuf> },
    Error { job: String, message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Job {
    Separate,
    Transcribe,
}

impl Job {
    fn as_arg(self) -> &'static str {
        match self {
            Job::Separate => "separate",
            Job::Transcribe => "transcribe",
        }
    }
}

#[derive(Debug)]
pub enum WorkerError {
    Spawn(std::io::Error),
    Failed(String),
    NoResult,
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkerError::Spawn(e) => write!(f, "could not start worker: {e}"),
            WorkerError::Failed(m) => write!(f, "worker failed: {m}"),
            WorkerError::NoResult => write!(f, "worker exited without a result"),
        }
    }
}

impl std::error::Error for WorkerError {}

/// Parse one stdout line; non-JSON lines (stray library output) are ignored.
pub fn parse_line(line: &str) -> Option<Event> {
    serde_json::from_str(line.trim()).ok()
}

/// Run `program --json <job> <input> --out <out>` and return the output files.
/// `on_event` is called for every event so a UI can show progress.
pub fn run_job(
    program: &Path,
    job: Job,
    input: &Path,
    out: &Path,
    mut on_event: impl FnMut(&Event),
) -> Result<Vec<PathBuf>, WorkerError> {
    let mut child = Command::new(program)
        .arg("--json")
        .arg(job.as_arg())
        .arg(input)
        .arg("--out")
        .arg(out)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .map_err(WorkerError::Spawn)?;

    let mut result = None;
    for line in BufReader::new(child.stdout.take().expect("piped stdout")).lines() {
        let Ok(line) = line else { break };
        let Some(ev) = parse_line(&line) else { continue };
        on_event(&ev);
        match ev {
            Event::Done { outputs, .. } => result = Some(Ok(outputs)),
            Event::Error { message, .. } => result = Some(Err(WorkerError::Failed(message))),
            Event::Started { .. } => {}
        }
    }
    let _ = child.wait();
    result.unwrap_or(Err(WorkerError::NoResult))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn fake_worker(dir: &Path, body: &str) -> PathBuf {
        let p = dir.join("fake-flux-ai");
        std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        p
    }

    #[test]
    fn parses_events() {
        assert_eq!(
            parse_line(r#"{"event":"done","job":"separate","outputs":["a.wav"]}"#),
            Some(Event::Done { job: "separate".into(), outputs: vec!["a.wav".into()] })
        );
        assert_eq!(parse_line("not json"), None);
    }

    #[test]
    fn parses_protocol_v2_lines() {
        // v2 adds protocol_version/job_id and extra fields; v1 fields still deserialise.
        assert_eq!(
            parse_line(r#"{"event":"started","protocol_version":2,"job_id":"j1","job":"generate","input":"house","backend":"sine-test"}"#),
            Some(Event::Started { job: "generate".into(), input: "house".into() })
        );
        assert_eq!(
            parse_line(r#"{"event":"done","protocol_version":2,"job_id":"j1","job":"generate","outputs":["g.wav","g.wav.provenance.json"],"seconds":2.0}"#),
            Some(Event::Done { job: "generate".into(), outputs: vec!["g.wav".into(), "g.wav.provenance.json".into()] })
        );
        // the new progress event is skipped like any unknown line
        assert_eq!(parse_line(r#"{"event":"progress","protocol_version":2,"job_id":"j1","job":"generate","fraction":0.4,"message":"x"}"#), None);
    }

    #[test]
    fn collects_outputs_and_reports_progress() {
        let d = tempfile::tempdir().unwrap();
        let w = fake_worker(
            d.path(),
            r#"echo noise; echo '{"event":"started","job":"separate","input":"x"}'; echo '{"event":"done","job":"separate","outputs":["d.wav","b.wav"]}'"#,
        );
        let mut seen = 0;
        let out = run_job(&w, Job::Separate, Path::new("x"), d.path(), |_| seen += 1).unwrap();
        assert_eq!(out, vec![PathBuf::from("d.wav"), PathBuf::from("b.wav")]);
        assert_eq!(seen, 2);
    }

    #[test]
    fn surfaces_worker_error() {
        let d = tempfile::tempdir().unwrap();
        let w = fake_worker(d.path(), r#"echo '{"event":"error","job":"separate","message":"boom"}'; exit 1"#);
        let e = run_job(&w, Job::Separate, Path::new("x"), d.path(), |_| {}).unwrap_err();
        assert!(matches!(e, WorkerError::Failed(m) if m == "boom"));
    }

    #[test]
    fn no_result_when_silent() {
        let d = tempfile::tempdir().unwrap();
        let w = fake_worker(d.path(), "exit 0");
        let e = run_job(&w, Job::Transcribe, Path::new("x"), d.path(), |_| {}).unwrap_err();
        assert!(matches!(e, WorkerError::NoResult));
    }

    /// End-to-end against the real worker:
    /// FLUX_AI_BIN=.venv/bin/flux-ai FLUX_AI_WAV=clip.wav cargo test -- --ignored
    #[test]
    #[ignore]
    fn real_separate() {
        let bin = std::env::var("FLUX_AI_BIN").expect("FLUX_AI_BIN");
        let wav = std::env::var("FLUX_AI_WAV").expect("FLUX_AI_WAV");
        let d = tempfile::tempdir().unwrap();
        let out = run_job(Path::new(&bin), Job::Separate, Path::new(&wav), d.path(), |_| {}).unwrap();
        assert_eq!(out.len(), 4, "expected drums/bass/vocals/other, got {out:?}");
    }
}
