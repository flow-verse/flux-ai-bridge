# Long-form orchestrator (`flux-ai longform`, phase P5)

Generates hours of music in one genre as a single mastered file with chapters,
a cue sheet and an AI-disclosure provenance record. It does not need
flux-host, a DAW or plugins: it plans the set, writes every section as MIDI on
an exact beat grid and renders it with a small built-in numpy/scipy synth. When
flux-host gains a headless render command (P1), `FluxHostRenderer` takes the
synth's place and nothing else changes.

## Install and run

```bash
uv venv -p 3.12 .venv-longform
uv pip install -p .venv-longform --no-deps -e python      # skip demucs/torch
uv pip install -p .venv-longform numpy scipy soundfile pyloudnorm mido jsonschema pyyaml mutagen pytest
# or, when the ML extras are wanted too:  pip install -e "python[longform]"

flux-ai longform --genre "deep house" --hours 1 --seed 1 --out runs/dh-1h
flux-ai longform --genre techno --hours 4 --plan-only --out runs/t4     # write plan.json only
flux-ai longform --plan runs/t4/plan.json --out runs/t4 --json          # execute an (edited) plan
flux-ai longform --genre lofi --hours 2 --neural-backend ace-step       # optional neural layers
```

Options: `--seed N`, `--format auto|wav|flac|rf64`, `--workers N`,
`--neural-sections breakdown,peak|all`, `--renderer builtin|flux-host`.
Genres shipped: `deep house`, `techno`, `lo-fi hip hop` (`lofi`), `ambient`,
`maqam groove` (`maqam`, `bayati`, `hijaz`). Extra or overriding tables can be
put in a directory listed in `$FLUX_LONGFORM_GENRES`.

### JSON events (job protocol v2)

With `--json` (before or after `longform`) stdout carries only JSON lines:

```json
{"event": "started",  "protocol_version": 2, "job": "longform", "job_id": "3f0c…", "input": "deep house", "hours": 1.0, "seed": 1, "out": "runs/dh-1h"}
{"event": "progress", "protocol_version": 2, "job": "longform", "job_id": "3f0c…", "stage": "render", "fraction": 0.41, "message": "s031"}
{"event": "done",     "protocol_version": 2, "job": "longform", "job_id": "3f0c…", "outputs": ["…/1-h-of-deep-house.wav", "….cue", "chapters.txt", "provenance.json", "report.json"], "report": {…}}
{"event": "error",    "protocol_version": 2, "job": "longform", "job_id": "3f0c…", "message": "KeyError: unknown genre …"}
```

Stages: `compose`, `neural`, `render`, `master`, `analysis`, `done`. The
`event`/`outputs`/`message` fields are the v1 ones, so the Rust worker parser
keeps working.

## Outputs (run directory)

| File | Content |
|---|---|
| `<title>.wav` / `.flac` | Master, 48 kHz stereo 24-bit. WAV below 4 h; FLAC (default) or RF64 (`--format rf64`) at 4 h and above, because a 24-bit stereo WAV hits the 4 GiB RIFF limit at about 4.1 h. |
| `<title>.cue` | Cue sheet, one track per segment. |
| `chapters.txt` | YouTube-style chapter list (`0:00:00 Part 1 - A minor (8A) - 122 BPM`). Chapter times sit at the bass swap of each transition, where the new part is heard. |
| `provenance.json` | Full plan, all seeds, genre-table hashes, renderer and generator versions, neural backends/models/licences with their verification results, mastering figures, a DDEX-style AI field block, and `human_decisions` (copied from the plan). |
| `report.json` | Wall clock per stage, speed factor, loudness, true peak, limiter reduction, join-click count, similarity summary, file sizes, peak RSS. |
| `similarity.json` | Self-similarity report (see below). |
| `plan.json`, `midi/`, `sections/`, `neural/` | Executed plan, section MIDI, checkpoint cache. |

Embedded disclosure: FLAC gets Vorbis comments `AI_GENERATED=1`,
`AI_DISCLOSURE`, `AI_SYSTEM`, `AI_PLAN_SEED`, `PROVENANCE_SHA256`,
`DDEX_AI_STYLE`; WAV/RF64 get a LIST-INFO comment starting `AI_GENERATED=1;`
plus an iXML chunk carrying the same fields. No watermark (owner decision, P6).

## Plan schema

`python/flux_ai/longform/schema/set_plan.schema.json` (JSON Schema 2020-12).
Three levels:

- **Set**: `genre`, `seed`, `hours`, `sample_rate`, `target` (LUFS, tolerance,
  true peak), `energy_curve` (points 0–1), `motifs` (set-level motif library:
  `[step, length, scale_degree, velocity]` in sixteenths), `anti_repetition`
  (no-repeat windows), `human_decisions`, `sections`.
- **Section**: `id`, `segment` (sections of one segment share key, tempo and
  material, like one track in a DJ set), `type` (intro, groove, build, peak,
  breakdown, outro), `genre`, `bpm`, `key` (`{tonic, mode}`, mode is a Western
  mode or a maqam from the genre table), `camelot`, `energy`/`energy_end`,
  `bars`, `layers`, `seeds`, `patterns` (drum/bass/progression/chord-rhythm
  IDs from the genre table), `motif` (`{id, transform}`), `transition_out`
  (`continue` | `blend` | `fade`, `bars`, `eq`, `bass_swap`), optional
  `neural` (`{backend, prompt, gain_db, seed}`).
- **Bar**: `bar_plan[]` with `chord` (scale degree), `energy`, `phrase`,
  `variation`, `fill`.

An agent can write a plan with only the required fields; `normalise()` fills
layers, patterns, seeds and the bar plan deterministically. Unknown keys are
allowed for annotations.

## Pipeline

1. **Plan** (`planner.py`). Energy arc from a genre template; tempo follows
   energy inside the genre range with at most `bpm_step_max` change between
   segments; key path by Camelot moves (same, ±1, relative, +2) for Western
   genres or by the maqam modulation table (e.g. bayati → hijaz on the same
   tonic, bayati on D → rast on C) for the maqam style. Form, drum pattern,
   bass pattern, progression, chord rhythm and motif are drawn per segment
   from no-repeat windows (window = min(configured, vocabulary − 1)).
2. **Compose** (`symbolic.py`). One standard MIDI file per section: drums
   (GM notes), bass, chords, pad, lead (the segment's motif, transformed:
   up2/down3/invert/retrograde/shift, varied per phrase), phrase-level
   arrangement mutations every 8 bars, CC74 brightness = energy + slow drift.
   Quarter tones are pitch bends (±2 st range set by RPN) with notes rotated
   over a channel pool so chords can carry different bends.
3. **Neural layer** (`neural.py`, optional). Calls
   `get_backend(name).generate(GenerateRequest(...))` with the section's BPM,
   key and seed, covering the constant-tempo body of the section. The result
   is resampled to 48 kHz, tempo-checked (onset autocorrelation within ±4 %,
   octave errors allowed), phase-aligned to the beat grid, faded at bar
   boundaries, high-passed at 120 Hz and mixed at `gain_db` under the
   symbolic layers. Unavailable backends or failed beat checks skip the layer
   and the reason is logged in provenance. Key detection is recorded but does
   not gate.
4. **Render** (`render.py`, `synth.py`). `Renderer.render(section_midi,
   section_plan, out_wav)` writes the section mix and a `low` stem (kick +
   bass), bar 0 to end of body plus a release/reverb tail. Sections render in
   parallel processes.
5. **Transitions** (`transitions.py`). Tempo is stored per beat as integer
   microseconds, so overlapping sections are sample-aligned. `blend`: high
   band (mix − low) equal-power crossfade over the overlap, low band swapped
   over one beat at the middle bar (bass swap); tempo changes become a linear
   per-beat ramp across the overlap in both sections' MIDI. `fade`:
   equal-power full-band crossfade. `continue`: next section starts on the
   bar line while the previous one's release tail plays out.
6. **Master** (`master.py`). Pass 1 streams the assembled mix through a
   BS.1770 meter (pyloudnorm's K-weighting coefficients, filter state carried
   between chunks; matches `pyloudnorm` within 0.05 LU in tests). Pass 2
   applies the gain and a look-ahead limiter driven by a 4× oversampled peak
   estimate (gain ≤ ceiling/peak by construction; 5 ms look-ahead, 40 ms
   hold), measures loudness and true peak again while writing, and repeats
   with a corrected gain if loudness misses by more than 0.1 LU. Memory is
   bounded by the 5.5 s chunk size, not the set length.
7. **Checks** (`analysis.py`). Join click detector at every section boundary
   and bass swap; self-similarity over 8-bar windows aligned to the plan's bar
   grid (standardised chroma + MFCC sequences, cosine), flagging non-adjacent
   pairs above 0.95.

### Checkpoint and resume

Each section is cached as `sections/<id>-<hash>/` where the hash covers the
section plan, its MIDI bytes, the genre table, the renderer identity
(including a digest of the synth code) and the neural layer audio. A section
directory appears only after its render completes (atomic rename). Rerunning
the same command after a crash or kill re-renders only missing sections, then
re-masters (mastering is a whole-file pass and is not checkpointed).

## Measured results

Machine: Apple M1 Ultra (20 cores), 64 GB, macOS 15, Python 3.12, builtin
renderer, 8 worker processes.

Each run was one unattended `flux-ai longform … --json` command from an empty
output directory (measured on 2026-10-23 at commit `9c2649e`).

| Run | Audio | Wall clock | Speed | Render / master / analysis (s) | LUFS | True peak | Join clicks | Similarity max (flags > 0.95) | Master file | Section cache | Peak RSS main / worker |
|---|---|---|---|---|---|---|---|---|---|---|---|
| deep house 1 h, seed 1 | 0:59:47, 88 sections, 12 segments | 125 s | 28.6× | 21 / 95 / 7 | −14.01 | −1.30 dBTP | 0 of 87 joins | 0.910 (0) | WAV 1.03 GB | 0.86 GB | 1.17 / 1.74 GB |
| deep house 4 h, seed 4 | 4:00:18, 347 sections, 49 segments | 488 s | 29.6× | 71 / 376 / 32 | −14.01 | −1.30 dBTP | 0 of 346 | 0.978 (2) | FLAC 2.90 GB | 3.38 GB | 1.07 / 2.71 GB |
| techno 1 h, seed 2 | 1:00:36 | 116 s | 31.3× | 18 / 90 / 6 | −14.00 | −1.30 | 0 | 0.969 (1) | WAV 1.05 GB | 0.91 GB | 0.75 / 1.94 GB |
| lo-fi hip hop 1 h, seed 2 | 1:00:07 | 112 s | 32.3× | 16 / 88 / 6 | −14.02 | −1.30 | 0 | 0.937 (0) | WAV 1.04 GB | 0.70 GB | 0.68 / 1.49 GB |
| ambient 1 h, seed 2 | 0:59:44 | 129 s | 27.8× | 39 / 84 / 6 | −14.00 | −1.30 | 0 | 0.910 (0) | WAV 1.03 GB | 0.66 GB | 0.81 / 2.08 GB |
| maqam groove 1 h, seed 2 | 1:00:11 | 115 s | 31.4× | 20 / 88 / 6 | −14.07 | −1.30 | 0 | 0.863 (0) | WAV 1.04 GB | 0.82 GB | 0.66 / 1.61 GB |

- Independent check of the 1 h deep house master: `pyloudnorm` on the whole
  file gives −14.008 LUFS, a 4× oversampled peak of −1.30 dBTP, and the click
  detector finds 0 discontinuities over all 172 M samples.
- Memory stays bounded: the main process peaks at about 1 GB for both 1 h and
  4 h (it holds one 5.5 s chunk plus filter context). Workers peak at 1.5–2.7
  GB while rendering their longest section; 8 workers fit easily in 64 GB.
- Mastering dominates wall clock (about 75 %): two streaming passes with 4×
  resampling for the limiter and true-peak meter, single-threaded. Rendering
  the symbolic core in 8 processes runs at roughly 170–200× real time.
- Compared with the worked example in the report (target ≥ 10× real time for
  the symbolic core, about 1 h total for 4 h symbolic-only): the built-in
  synth path takes about 8 minutes for 4 h. This says nothing yet about
  flux-host + Surge XT render speed, which P1 still has to measure.
- **Resume after SIGKILL** (techno 1 h, 71 sections): the run was killed with
  31 sections cached and 8 half-written `.tmp` section directories. Rerunning
  the same command removed the partial directories, rendered only the
  remaining 40 sections and produced an in-spec master (−14.00 LUFS,
  −1.30 dBTP, 0 join clicks) in 114 s. Separately, deleting 3 of the 88
  cached sections of the 1 h run and rerunning rendered exactly those 3.
- **Self-similarity**: three of the six runs have flags at 0.95. The 4 h run
  has 2 flagged pairs out of about 424,000 (0.978 and 0.956), both between
  sparse 3–4-layer outro/intro overlaps of neighbouring transitions. The
  techno run has 1 flagged pair (0.969) inside its opening intro, which has
  only two layers. Before the phrase-level variation work, the 4 h run had 7
  flags up to 0.999 (bare-kick intro phrases). Sparse DJ-tool passages are
  where repetition remains.
- RF64 output is covered by a unit test (write, iXML append, read back) but
  was not used for a full 4 h run; the 4 h run used the default FLAC path.

## Limitations

- **The built-in synth is a placeholder** for flux-host + Surge XT rendering
  (P1/P2). It is serviceable for testing the pipeline and checking structure,
  not a release-quality sound. `FluxHostRenderer` is a stub that raises until
  `flux-host render` exists.
- Genre tables are starting values for owner curation, not sourced facts; the
  maqam cents are a 24-EDO-style approximation.
- `flux_ai.backends` is not in this branch; `_backend_shim.py` provides the
  agreed names (`GenerateRequest`, `GenerateResult`, `get_backend`) and a
  `sine-test` backend. The code imports the real package first, so the shim
  can be deleted at merge.
- The similarity check flags near-exact repeats of 8-bar windows; it does not
  measure whether hour 2 is musically interesting. Human review gates are
  still needed.
- Mastering is loudness normalisation plus a safety limiter, not creative
  mastering. Loudness is measured on the whole set, so the energy arc is kept.
- Sparse intro/outro passages (two to four layers) can still exceed the 0.95
  similarity threshold in long sets; see the measured results above. Likely
  fixes are timbre drift on hats and percussion, or more layer orders.
- The master stage is single-threaded and takes about 75 % of wall clock.
  Splitting pass 2 across processes, with overlapping limiter context, would
  speed it up.
- Only 4/4. No vocals. No watermark.
- DDEX field names are DDEX-*style* placeholders pending verification (P6).
