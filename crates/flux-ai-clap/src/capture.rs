//! Turn captured audio into a WAV file and run a worker job on it.
//! Pure logic, no plugin types, so it is unit-testable without a host.

use crate::worker::{run_job, Event, Job, WorkerError};
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum CaptureError {
    Empty,
    Wav(hound::Error),
    Io(std::io::Error),
    Worker(WorkerError),
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureError::Empty => write!(f, "nothing was captured"),
            CaptureError::Wav(e) => write!(f, "could not write wav: {e}"),
            CaptureError::Io(e) => write!(f, "io error: {e}"),
            CaptureError::Worker(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CaptureError {}

/// Write interleaved f32 samples as a 32-bit float WAV.
pub fn write_wav(path: &Path, interleaved: &[f32], channels: u16, sample_rate: u32) -> Result<(), CaptureError> {
    let spec = hound::WavSpec {
        channels,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).map_err(CaptureError::Wav)?;
    for s in interleaved {
        w.write_sample(*s).map_err(CaptureError::Wav)?;
    }
    w.finalize().map_err(CaptureError::Wav)
}

/// Save `interleaved` under `job_dir/capture.wav`, run the worker, and return its outputs.
pub fn run_capture_job(
    interleaved: &[f32],
    channels: u16,
    sample_rate: u32,
    job: Job,
    program: &Path,
    job_dir: &Path,
    on_event: impl FnMut(&Event),
) -> Result<Vec<PathBuf>, CaptureError> {
    if interleaved.is_empty() {
        return Err(CaptureError::Empty);
    }
    std::fs::create_dir_all(job_dir).map_err(CaptureError::Io)?;
    let wav = job_dir.join("capture.wav");
    write_wav(&wav, interleaved, channels, sample_rate)?;
    let out = job_dir.join(match job {
        Job::Separate => "stems",
        Job::Transcribe => "midi",
    });
    run_job(program, job, &wav, &out, on_event).map_err(CaptureError::Worker)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn wav_roundtrip() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("a.wav");
        write_wav(&p, &[0.0, 0.5, -0.5, 1.0], 2, 48_000).unwrap();
        let r = hound::WavReader::open(&p).unwrap();
        assert_eq!(r.spec().channels, 2);
        assert_eq!(r.spec().sample_rate, 48_000);
        assert_eq!(r.into_samples::<f32>().map(|s| s.unwrap()).collect::<Vec<_>>(), [0.0, 0.5, -0.5, 1.0]);
    }

    #[test]
    fn empty_capture_is_an_error() {
        let d = tempfile::tempdir().unwrap();
        let e = run_capture_job(&[], 2, 48_000, Job::Separate, Path::new("x"), d.path(), |_| {}).unwrap_err();
        assert!(matches!(e, CaptureError::Empty));
    }

    #[test]
    fn runs_worker_on_saved_wav() {
        let d = tempfile::tempdir().unwrap();
        let w = d.path().join("fake");
        // Fails unless capture.wav exists as the first non-flag argument.
        std::fs::write(
            &w,
            "#!/bin/sh\n[ -f \"$3\" ] || exit 3\necho '{\"event\":\"done\",\"job\":\"separate\",\"outputs\":[\"s.wav\"]}'\n",
        )
        .unwrap();
        std::fs::set_permissions(&w, std::fs::Permissions::from_mode(0o755)).unwrap();
        let out = run_capture_job(&[0.1; 64], 2, 44_100, Job::Separate, &w, &d.path().join("job"), |_| {}).unwrap();
        assert_eq!(out, vec![PathBuf::from("s.wav")]);
        assert!(d.path().join("job/capture.wav").exists());
    }
}
