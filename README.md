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

## Verified so far
- `separate`: Demucs `htdemucs`, 4 stems (macOS arm64, Python 3.13).
- `transcribe`: three-note sine melody gave A4, C#5, E5 at the right times (Python 3.12, ONNX backend).
- Rust `worker` module runs `flux-ai --json` end to end (`cargo test -- --ignored` with `FLUX_AI_BIN` and `FLUX_AI_WAV`).

## Design constraints from the research reports (see flux-docs/docs/research)
- Production-quality stem separation is a **batch** job, not a live effect. Pre-process, then load stems as clips.
- Live neural audio (RAVE) must run **causal** with real-time factor < 1.0 under DAW CPU contention; expect tens of ms jitter. Plan for a "slim" fallback and dry passthrough.
- Report the generate-ahead buffer to the host as fixed latency (Plugin Delay Compensation), as the MRT2-Accompany plugin does.
- Stems from copyrighted recordings need clearance for commercial release.
