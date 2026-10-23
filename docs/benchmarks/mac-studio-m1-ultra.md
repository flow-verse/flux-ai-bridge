# Generation backends on a Mac Studio (M1 Ultra, 64 GB)

Measured 2026-10-23 with `flux-ai bench` and `flux-ai generate` from this repo
(flux-ai 0.1.0). Every run asked for **120 s of 122 BPM, A minor, instrumental**
with the prompt "deep house, warm analog pads, punchy four-on-the-floor kick,
rolling bassline, instrumental", seeds 0-4 (5 seeds per tier; the P4 plan
suggests 20 for the acceptance threshold, which was not run).

**Nobody listened to these files.** Everything below is an objective measure
(estimators, timers, memory counters). The "listening notes" are what the
estimators and levels say, not judgements of musical quality.

## Summary

| backend / tier | s compute per audio minute | real-time factor | peak RSS | MLX peak | BPM within ±1 (5 seeds) | key = A minor (exact / incl. C major) |
|---|---|---|---|---|---|---|
| ACE-Step 1.5 turbo (2B DiT + 1.7B LM) | **34.7** | 0.58 | 14.0 GB | 21.3 GB | **5/5** | 3/5 / 3/5 |
| ACE-Step 1.5 xl-turbo (4B DiT + 1.7B LM) | **42.6** | 0.71 | 16.0 GB | 30.7 GB | **5/5** | **4/5 / 4/5** |
| Magenta RT 2 base (2.4B), note guide on | 83.3 | 1.39 | 4.1 GB | 2.7 GB | 2/5 | 2/5 / **5/5** |
| Magenta RT 2 small (230M), note guide on | 51.4 | 0.86 | 1.9 GB | 0.5 GB | 1/5 | 3/5 / **5/5** |
| Magenta RT 2 base, note guide **off** (BPM/key only in the prompt text) | 101.6 | 1.69 | 4.1 GB | 2.7 GB | 1/5 | 0/5 / 0/5 |

- *Compute per audio minute* and *RTF* use the median of the 4 warm seeds
  (seeds 1-4; the model is already loaded). Cold first calls, including model
  load from the local cache: turbo 95 s, xl-turbo 194 s, MRT base 217 s, MRT
  small 89 s (each for 120 s of audio).
- *Peak RSS* is `ru_maxrss` of the worker process. Apple Silicon unified memory
  used by MLX is not all counted in RSS, so *MLX peak* (`mx.get_peak_memory()`)
  is listed too; the two overlap and must not be added. `/usr/bin/time -l`
  reported a peak memory footprint of about 51 GB for the very first ACE-Step
  turbo run, so budget for most of the 64 GB when running the XL tier next to
  other work.
- *BPM* is the least-squares slope of beat times from librosa's beat tracker
  (librosa's own global tempo is quantised to about ±1.4 BPM at 122 BPM, too
  coarse for a ±1 test; the code is `python/flux_ai/analysis.py`). A
  flat-prior variant is also stored in `bench.json` because librosa's default
  prior centres on 120 BPM.
- *Key* is Krumhansl-Kessler profile correlation on the mean CQT chroma. "incl.
  C major" also accepts the relative major, which shares every pitch class with
  A minor and is the usual estimator confusion.

### Acceptance runs (P4)

Both default backends generated a 2-minute 122 BPM A-minor instrumental
(seed 42) with the plain CLI:

```bash
flux-ai --json generate --backend ace-step --prompt "$PROMPT" --seconds 120 --bpm 122 --key "A minor" \
        --seed 42 --out accept_ace_122_Am.wav --accept-licence ace-step-1.5
flux-ai --json generate --backend magenta-rt --model base --prompt "$PROMPT" --seconds 120 --bpm 122 \
        --key "A minor" --seed 42 --out accept_mrt_122_Am.wav --accept-licence magenta-rt-2
```

| | wall time (cold, incl. load) | peak RSS | measured BPM | estimated key |
|---|---|---|---|---|
| ACE-Step turbo | 128 s | 15.1 GB | 122.00 | G minor (miss) |
| Magenta RT 2 base | 210 s | 4.4 GB | 125.75 (miss) | C major (relative of A minor) |

The P4 acceptance suggestion (BPM within ±1 and key matching in ≥80% of 20
seeds) is met by **ACE-Step on tempo** (10/10 across both tiers, 5 seeds each)
but **not on key** (turbo 60%, xl-turbo 80% on 5 seeds; 20 seeds still to be
run). **Magenta RT 2 meets neither**: it has no tempo or key input, and even
with the MIDI note guide its tempo wanders (estimates 114-130 BPM).

## Per-seed results

ACE-Step turbo: BPM 122.32, 122.00, 122.60, 121.99, 121.99; key A minor, E minor, F# minor, A minor, A minor.
ACE-Step xl-turbo: BPM 122.00 ×5; key A major, A minor ×4 (seed 0's runner-up was A minor, r 0.71 vs 0.86).
MRT base (guide): BPM 120.04, 116.05, 122.15, 126.09, 122.31; key A minor, C major, C major, C major, A minor.
MRT small (guide): BPM 121.63, 119.99, 129.80, 125.24, 114.25; key C major, A minor, A minor, C major, A minor.
MRT base (no guide): BPM 125.39, 124.01, 121.99, 124.00, 123.27; key G minor, D# major, F# minor, F# minor, C# major.

ACE-Step stage timings for 120 s (warm seeds, from the provenance sidecars):

| tier | 5 Hz LM (1.7B, MLX) | DiT diffusion (8 steps, MLX) | VAE decode (MLX) | pipeline total |
|---|---|---|---|---|
| turbo | 14-22 s | 7-14 s | 28-39 s | 53-74 s |
| xl-turbo | 17-19 s | 21-25 s | 20-24 s | 59-65 s |

The VAE decode (fp32 on MLX) costs as much as the rest put together; it is the
first thing to optimise for ACE-Step on this Mac.

Magenta RT 2 speed is not an adapter overhead: the upstream CLI
(`mrt mlx generate --duration 20`) measured 43.8 ms per 40 ms frame for small
(RTF 1.10) and 64.8 ms for base (RTF 1.62) on the same machine. The Python MLX
path is therefore **not real-time on an M1 Ultra** for either tier; upstream
lists real-time streaming for its C++ engine and apps, which this adapter does
not use.

## Objective listening notes (no one listened)

- ACE-Step turbo: RMS -15.4 to -16.4 dBFS, peak normalised to -1 dBFS by
  ACE-Step; 5-9% of samples below -60 dBFS (quiet intros/outros). 167-221
  beats tracked out of about 244 expected, i.e. some sections without a clear
  pulse. Key estimates scatter (E minor, F# minor) on 2 of 5 seeds.
- ACE-Step xl-turbo: 213-232 beats tracked, 2-6% near-silence, key correlation
  0.63-0.86 (stronger tonal centre than turbo).
- Magenta RT 2: quieter (RMS -22 to -27 dBFS with the guide, -18 to -25 without), 1-8% near-silence, 196-261
  beats tracked; the beat tracker often prefers half tempo with a flat prior
  (57-64 BPM), which points to a weak or drifting pulse.
- The MIDI note guide (A-minor chord loop with an onset on every beat at the
  requested tempo, drums on) is what gives Magenta RT its key: 5/5 A minor or
  C major with the guide, 0/5 without it.

## Other checks on this machine

- `flux-ai serve --json`, 4 jobs of 30 s in one process: Magenta RT small 34.2 s
  then 23.2 s (warm); ACE-Step turbo 54.4 s then 19.5 s (warm). 0 non-JSON
  lines on stdout.
- ACE-Step `repaint` of 10-20 s in a 30 s clip: correlation with the source is
  1.000 outside the region and 0.76 inside it. `continue` by 20 s: the first
  29 s match the context exactly (correlation 1.000), output 50 s.
- AMT small (MPS): 20 s of MIDI in 52 s (170 notes, 14 tracks); `infill` of
  5-10 s in 24 s; `continue` by 8 s in 12 s.
- Stable Audio 3: installs on this Mac in a separate venv (`.venv-sa3`:
  stable-audio-3 0.1.0, torch 2.7.1, transformers 5.20.2). It cannot share a venv
  with ACE-Step (torch 2.7.1 and transformers>=5.8 against ACE-Step's
  torch>=2.9.1 and transformers<4.58). It **did not run**: the weights are
  gated on Hugging Face and this machine's HF account gets HTTP 403 for
  `stabilityai/stable-audio-3-small-music` and `-medium`. To run it, the owner
  must accept Stability AI's terms on those model pages with the HF account,
  then pass `--accept-licence stable-audio-3`. Upstream marks Medium as
  CUDA-targeted (Flash Attention 2); Small-Music is listed for CPU/Mac.

## Setup and exact versions

Hardware: Mac Studio, Apple M1 Ultra, 64 GB, macOS 15.7.4 (arm64).

```bash
uv venv .venv-gen --python 3.12                       # Python 3.12.9
uv pip install -p .venv-gen <ACE-Step-1.5 source> "magenta-rt[mlx]==2.0.3" librosa pytest psutil
uv pip install -p .venv-gen "mlx==0.31.2"             # see below
uv pip install -p .venv-gen --no-deps -e python
uv pip install -p .venv-gen <anticipation source> mido   # AMT
flux-ai bench --backend ace-step --model turbo --seeds 5 --seconds 120 --bpm 122 --key "A minor" --prompt "$PROMPT"
flux-ai bench --backend ace-step --model xl-turbo --seeds 5 --seconds 120 --bpm 122 --key "A minor" --prompt "$PROMPT"
flux-ai bench --backend magenta-rt --model base --model small --seeds 5 --seconds 120 --bpm 122 --key "A minor" --prompt "$PROMPT"
flux-ai bench --backend magenta-rt --model base --seeds 5 --seconds 120 --bpm 122 --key "A minor" \
        --prompt "$PROMPT, 122 BPM, A minor" --extra note_guide=false
```

| component | version |
|---|---|
| ACE-Step 1.5 code | `ace-step` 1.5.0, commit ca1e85fe9430179831e6bc6be790c332190a3866 |
| ACE-Step weights | HF `ACE-Step/Ace-Step1.5` @ 19671f40 (turbo DiT, 1.7B LM, VAE, Qwen3-Embedding-0.6B; 10.1 GB); `ACE-Step/acestep-v15-xl-turbo` @ d4a0b288 (20.0 GB) |
| Magenta RT 2 | `magenta-rt` 2.0.3; HF `google/magenta-realtime-2` @ 010aa0dc (mlxfn models: base 2.8 GB, small 0.5 GB, plus resources) |
| AMT | `anticipation` @ af37397; HF `stanford-crfm/music-small-800k` |
| MLX | 0.31.2 (0.32.4 fails to import the published MRT2 `.mlxfn` files: "[import_function] Invalid string size.") |
| torch / transformers | 2.12.1 / 4.57.6 |
| librosa / numpy | 1.0.0 / 2.3.5 |

ACE-Step ran with its Apple Silicon defaults: MLX DiT and MLX VAE (fp32), 5 Hz
LM on MLX, PyTorch parts on MPS. Weights live in `~/.cache/flux-ai/models/`
(ACE-Step 28 GB with XL, Magenta RT 4.3 GB). Licence acceptances for
`ace-step-1.5`, `magenta-rt-2` and `amt` were recorded in
`~/.cache/flux-ai/licences.json` by these runs. Raw `bench.json` files and
audio were kept out of the repo.
