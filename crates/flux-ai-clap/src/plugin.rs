//! nih-plug CLAP wrapper. Passes audio through untouched; while "Capture" is on it
//! copies the incoming audio into a lock-free ring. When Capture turns off, a
//! background task drains the ring, writes a WAV and runs the flux-ai worker.

use crate::capture::run_capture_job;
use crate::worker::Job;
use nih_plug::prelude::*;
use ringbuf::traits::{Consumer, Observer, Producer, Split};
use ringbuf::{HeapCons, HeapProd, HeapRb};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Max capture length in seconds (stereo, at the host sample rate).
const MAX_CAPTURE_SECS: usize = 120;
const CHANNELS: usize = 2;

#[derive(Enum, Debug, PartialEq, Clone, Copy)]
pub enum JobKind {
    #[id = "separate"]
    #[name = "Separate stems"]
    Separate,
    #[id = "transcribe"]
    #[name = "Audio to MIDI"]
    Transcribe,
}

#[derive(Params)]
struct FluxAiParams {
    /// Turn on to record the track's audio; turn off to start the selected job.
    #[id = "capture"]
    capture: BoolParam,
    #[id = "job"]
    job: EnumParam<JobKind>,
}

impl Default for FluxAiParams {
    fn default() -> Self {
        Self {
            capture: BoolParam::new("Capture", false),
            job: EnumParam::new("Job", JobKind::Separate),
        }
    }
}

/// Work done off the audio thread.
pub enum Task {
    Run { sample_rate: u32, job: JobKind },
}

pub struct FluxAi {
    params: Arc<FluxAiParams>,
    producer: Option<HeapProd<f32>>,
    consumer: Arc<Mutex<Option<HeapCons<f32>>>>,
    capturing: bool,
    sample_rate: f32,
}

impl Default for FluxAi {
    fn default() -> Self {
        Self {
            params: Arc::new(FluxAiParams::default()),
            producer: None,
            consumer: Arc::new(Mutex::new(None)),
            capturing: false,
            sample_rate: 44_100.0,
        }
    }
}

fn worker_program() -> PathBuf {
    std::env::var_os("FLUX_AI_BIN").map(PathBuf::from).unwrap_or_else(|| "flux-ai".into())
}

fn output_root() -> PathBuf {
    if let Some(p) = std::env::var_os("FLUX_AI_OUT") {
        return PathBuf::from(p);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    home.join("Music").join("Flux")
}

impl Plugin for FluxAi {
    const NAME: &'static str = "Flux AI Bridge";
    const VENDOR: &'static str = "Flux";
    const URL: &'static str = "https://github.com/flow-verse/flux-ai-bridge";
    const EMAIL: &'static str = "";
    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[AudioIOLayout {
        main_input_channels: NonZeroU32::new(2),
        main_output_channels: NonZeroU32::new(2),
        ..AudioIOLayout::const_default()
    }];

    type SysExMessage = ();
    type BackgroundTask = Task;

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn task_executor(&mut self) -> TaskExecutor<Self> {
        let consumer = self.consumer.clone();
        Box::new(move |task| {
            let Task::Run { sample_rate, job } = task;
            // Drain everything captured so far. This is the only place the consumer is used.
            let samples: Vec<f32> = {
                let mut guard = consumer.lock().unwrap();
                let Some(cons) = guard.as_mut() else { return };
                let mut v = vec![0.0; cons.occupied_len()];
                let n = cons.pop_slice(&mut v);
                v.truncate(n);
                v
            };
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let dir = output_root().join(format!("job-{stamp}"));
            let job = match job {
                JobKind::Separate => Job::Separate,
                JobKind::Transcribe => Job::Transcribe,
            };
            match run_capture_job(&samples, CHANNELS as u16, sample_rate, job, &worker_program(), &dir, |e| {
                nih_log!("flux-ai: {e:?}");
            }) {
                Ok(outputs) => nih_log!("flux-ai: {} file(s) in {}", outputs.len(), dir.display()),
                Err(e) => nih_error!("flux-ai: {e}"),
            }
        })
    }

    fn initialize(
        &mut self,
        _layout: &AudioIOLayout,
        config: &BufferConfig,
        _context: &mut impl InitContext<Self>,
    ) -> bool {
        self.sample_rate = config.sample_rate;
        // Allocate once here, never in process().
        let cap = MAX_CAPTURE_SECS * config.sample_rate as usize * CHANNELS;
        let (prod, cons) = HeapRb::<f32>::new(cap).split();
        self.producer = Some(prod);
        *self.consumer.lock().unwrap() = Some(cons);
        true
    }

    fn reset(&mut self) {
        self.capturing = false;
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        let on = self.params.capture.value();
        if on && !self.capturing {
            self.capturing = true;
        } else if !on && self.capturing {
            self.capturing = false;
            context.execute_background(Task::Run {
                sample_rate: self.sample_rate as u32,
                job: self.params.job.value(),
            });
        }

        if self.capturing {
            if let Some(prod) = self.producer.as_mut() {
                for mut frame in buffer.iter_samples() {
                    let l = *frame.get_mut(0).unwrap_or(&mut 0.0);
                    let r = *frame.get_mut(1).unwrap_or(&mut 0.0);
                    // Drops samples if the ring is full; never blocks.
                    let _ = prod.try_push(l);
                    let _ = prod.try_push(r);
                }
            }
        }
        ProcessStatus::Normal
    }
}

impl ClapPlugin for FluxAi {
    const CLAP_ID: &'static str = "com.flow-verse.flux-ai-bridge";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("Capture audio and run stem separation or audio-to-MIDI in the background");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[ClapFeature::AudioEffect, ClapFeature::Utility, ClapFeature::Stereo];
}

nih_export_clap!(FluxAi);
