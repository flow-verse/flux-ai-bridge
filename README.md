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

## Verified so far
- `flux-ai separate` runs end-to-end on macOS arm64 / Python 3.13 (Demucs `htdemucs`, 4 stems).
- `transcribe` (Basic Pitch) is untested: it needs TensorFlow, so it is an optional extra (`pip install -e "python[transcribe]"`) and may need Python ≤3.12.

## Design constraints from the research reports (see flux-docs/docs/research)
- Production-quality stem separation is a **batch** job, not a live effect. Pre-process, then load stems as clips.
- Live neural audio (RAVE) must run **causal** with real-time factor < 1.0 under DAW CPU contention; expect tens of ms jitter. Plan for a "slim" fallback and dry passthrough.
- Report the generate-ahead buffer to the host as fixed latency (Plugin Delay Compensation), as the MRT2-Accompany plugin does.
- Stems from copyrighted recordings need clearance for commercial release.
