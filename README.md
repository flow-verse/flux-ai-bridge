# flux-ai-bridge

Brings offline/online neural audio models onto the DAW timeline.

```
DAW (CLAP host) ── flux-ai-clap (Rust, nih-plug) ── IPC/PyO3 ── python/flux_ai (Demucs, Basic Pitch, madmom, RAVE)
```

| Capability | Model | Mode |
|---|---|---|
| Stem separation | Demucs | offline job, results placed on new tracks |
| Audio → MIDI | Basic Pitch | offline job, notes + pitch-bend |
| Beat / downbeat tracking | madmom | near-real-time |
| Timbre transfer | RAVE | real-time (streaming, PQMF) |

## Rules
- The audio thread never calls Python. Inference runs on a normal-priority worker and hands results over a lock-free SPSC ring (see `flux-host/rt/spsc`).
- Python is isolated for ML only.

## MVP order
1. `python/flux_ai`: CLI `flux-ai separate in.wav --out stems/` works standalone  ← start here
2. Same for `transcribe` (Basic Pitch)
3. Rust CLAP shell that spawns the worker and reports progress
4. Return stems/MIDI to the host
5. madmom downbeats → cue-point heuristics (SSM: novelty / metrical / salience rules)
6. RAVE real-time timbre transfer

## Install
```bash
python3 -m venv .venv && .venv/bin/pip install -e python          # separate (Demucs)
# optional: audio-to-MIDI (Basic Pitch via ONNX runtime, no TensorFlow needed)
.venv/bin/pip install --no-deps basic-pitch
.venv/bin/pip install -e "python[transcribe]"
```
`basic-pitch` is installed with `--no-deps` because its `resampy<0.4.3` pin does not build on Python 3.12+.

## Usage
```bash
flux-ai separate  clip.wav --out stems/    # drums, bass, vocals, other
flux-ai transcribe clip.wav --out midi/    # notes + pitch bends
flux-ai --json separate clip.wav --out stems/   # JSON-lines events for the plugin
```

## Long-form sets (`flux-ai longform`)
Plans and renders hours of one genre (deep house, techno, lo-fi hip hop, ambient, maqam groove) into one mastered file (−14 LUFS, ≤ −1 dBTP) with chapters, a cue sheet and `provenance.json`. Runs without flux-host for now (built-in placeholder synth). See [docs/longform.md](docs/longform.md).
```bash
flux-ai longform --genre "deep house" --hours 1 --out runs/dh-1h
```

## Verified so far
- `separate`: Demucs `htdemucs`, 4 stems (macOS arm64, Python 3.13).
- `transcribe`: three-note sine melody gave A4, C#5, E5 at the right times (Python 3.12, ONNX backend).
- Rust `worker` module runs `flux-ai --json` end to end (`cargo test -- --ignored` with `FLUX_AI_BIN` and `FLUX_AI_WAV`).

## CLAP plugin ("Flux AI Bridge")
Passes audio through unchanged. Turn **Capture** on to record the track (up to 120 s, stereo) into a pre-allocated lock-free ring; turn it off and a background task writes `capture.wav` and runs `flux-ai --json` (the **Job** parameter picks *Separate stems* or *Audio to MIDI*). Results land in `$FLUX_AI_OUT/job-<time>/` (default `~/Music/Flux`); the worker binary comes from `$FLUX_AI_BIN` (default `flux-ai` on PATH). Returning stems to the host as tracks is still open (issue #3).

```bash
cargo xtask bundle flux-ai-clap --release     # -> target/bundled/Flux AI Bridge.clap
cargo xtask bundle flux-ai-clap --features rt-assert   # debug build that aborts if process() allocates
cargo test -p flux-ai-clap --no-default-features        # logic tests without nih-plug
tests/host/run.sh                              # end-to-end test in a minimal CLAP host (C)
```

### Test status
- Unit tests (worker parsing, WAV writing, job runner): pass.
- `tests/host/run.sh`: a small C CLAP host loads the bundle, sets Capture on/off through parameter events, feeds 3-8 s of audio and waits; Demucs stems appear (4 wavs). The captured WAV length matched the audio fed (8.02 s for 8 s). Also passes with the `rt-assert` debug build (no allocation on the audio thread).
- `clap-validator`: 31 passed, 10 skipped (no presets/notes/double precision), 2 failed: `param-set-events` and `param-set-no-cookies`. Those two fail only because the validator's fixed random seed leaves both of this plugin's 2-value parameters unchanged; with a float parameter added they pass. (That float variant instead fails three `state-reproducibility-*` tests, which the unmodified nih-plug `gain` example also fails at this revision.) Not tested in a commercial DAW.

## Design constraints from the research reports (see flux-docs/docs/research)
- Production-quality stem separation is a **batch** job, not a live effect. Pre-process, then load stems as clips.
- Live neural audio (RAVE) must run **causal** with real-time factor < 1.0 under DAW CPU contention; expect tens of ms jitter. Plan for a "slim" fallback and dry passthrough.
- Report the generate-ahead buffer to the host as fixed latency (Plugin Delay Compensation), as the MRT2-Accompany plugin does.
- Stems from copyrighted recordings need clearance for commercial release.
