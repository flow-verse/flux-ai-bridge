# Long-form orchestrator (`flux-ai longform`, phase P5)

Generates hours of music in one genre as a single mastered file with chapters,
a cue sheet and an AI-disclosure provenance record. It plans the set, writes
every section as MIDI on an exact beat grid and renders it either with a small
built-in numpy/scipy synth (default; needs no DAW or plugins) or, with
`--renderer flux-host`, through flux-host's headless engine with plugin
instruments (see [flux-host renderer](#flux-host-renderer)). Transitions,
mastering, checks and outputs are the same for both.

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
flux-ai longform --genre "deep house" --hours 1 --renderer flux-host \
    --flux-plugin default="Surge XT"                                     # plugin instruments via flux-engine
```

Options: `--seed N`, `--format auto|wav|flac|rf64`, `--workers N`,
`--neural-sections breakdown,peak|all`, `--renderer builtin|flux-host`,
`--flux-engine PATH`, `--flux-plugin LAYER=ID` (repeatable), `--flux-block N`.
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
| `provenance.json` | Full plan, all seeds, genre-table hashes, renderer identity (for flux-host: engine API version, binary, plugin ids), per-section instruments (`section_rendering`), generator version, neural backends/models/licences with their verification results, mastering figures, a DDEX-style AI field block, `human_decisions` (copied from the plan) and the `decision_log` (see below). |
| `report.json` | Wall clock per stage, speed factor, loudness, true peak, limiter reduction, join clicks (`join_clicks`: clicks the stitching introduced; `section_clicks_near_joins`: clicks near a join that are already in a section render, i.e. made by the instrument), similarity summary, file sizes, peak RSS. |
| `similarity.json` | Self-similarity report (see below). |
| `plan.json`, `midi/`, `sections/`, `neural/` | Executed plan, section MIDI, checkpoint cache. With flux-host, each `sections/<id>-<hash>/project/` is the saved engine project of that section, which opens in Flux for review. |

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
  (`continue` | `blend` | `fade`, `bars`, `eq`, `bass_swap`, `swap_at`), optional
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
4. **Render** (`render.py`, `synth.py`, `fluxhost.py`). `Renderer.render(section_midi,
   section_plan, out_wav)` writes the section mix and a `low` stem (kick +
   bass), bar 0 to end of body plus a release/reverb tail. Built-in sections
   render in parallel processes; flux-host sections render one at a time
   through one warm engine.
5. **Transitions** (`transitions.py`). Tempo is stored per beat as integer
   microseconds, so overlapping sections are sample-aligned. `blend`: high
   band (mix − low) equal-power crossfade over the overlap, low band swapped
   over one beat at the bar given by `swap_at` (fraction of the overlap;
   genre table `segment.transition.swap_at`, default 0.5 = middle, 0 = the
   first bar so the outgoing bass never plays under the incoming key); tempo
   changes become a linear per-beat ramp across the overlap in both sections'
   MIDI. The generator lets only one section's hats, claps and percussion
   play in each half of the overlap (outgoing first), so two sections' timing
   jitter never flams. `fade`: equal-power full-band crossfade. `continue`:
   next section starts on the bar line while the previous one's release tail
   plays out; layers new to a groove, build or peak enter one at a time over
   its first bars, quiet layers first and kick and bass last. A build right
   after a breakdown holds kick and bass for the following peak (the drop),
   ends with a drum fill and, when the table marks the section type
   `riser: true`, carries a swept noise riser from the built-in synth
   (`riser_db`, default −26). A peak after a build still drops at bar 0. The
   set ends with a 4-bar fade.
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
(including a digest of the synth code; for flux-host also the engine binary's
size and mtime, its API version, block length and the resolved plugin ids and
versions) and the neural layer audio. A section
directory appears only after its render completes (atomic rename). Rerunning
the same command after a crash or kill re-renders only missing sections, then
re-masters (mastering is a whole-file pass and is not checkpointed).

## flux-host renderer

`--renderer flux-host` (`fluxhost.py`, `FluxHostRenderer`) plays the pitched
layers (bass, chords, pad, lead) with plugin instruments in flux-host's headless
engine, `flux-engine serve` (JSON-RPC over stdio; flux-host
`flux/engine/API.md`). The client is `flux_ai.engine_client.EngineClient`, the
same supervised client flux-mcp uses (`flux_mcp.engine` re-exports it). One
engine process stays up for the whole run. If it crashes, it is restarted and
the section is rendered again from scratch. The engine binary comes from
`--flux-engine`, else `$FLUX_ENGINE`, else `PATH`; plugins are found through
the engine's usual folders plus `$FLUX_CLAP_PATH`.

Per section:

1. `create_project` in `sections/<id>-<hash>/project` at the section tempo.
2. Agent turn `longform <id>: instruments`: one `add_track` per layer that has
   notes, with its instrument.
3. Agent turn `longform <id>: notes`: one `write_notes` clip per layer from
   bar 1, covering body + tail; `set_range` to the same bars.
4. `get_history`, `save_project`, then `render` of exactly body + tail seconds
   (the body already includes the transition overlaps) as a 48 kHz 32-bit float
   WAV with per-track stems. The stems are mixed in Python, faded at the tail
   end and written as `mix.flac`/`mix.low.flac` exactly like the built-in
   renderer's, so transitions and mastering consume them unchanged.

**Timing.** The project runs at the section's constant tempo. Note times come
from the section MIDI in seconds (its tempo map carries the per-beat ramps in
transitions) and are converted to beats at that tempo, rounded to the engine's
960 PPQ. The audio therefore sits on the plan's microsecond grid, just like the
built-in synth's. Tempo-synced plugin features follow the constant tempo, so
they drift slightly inside a ramp.

**Instruments.** Plugins are named by id or exact name (as `list_plugins`
reports them). Precedence per layer: `--flux-plugin LAYER=ID`, then the genre
table's `flux_host.plugins.LAYER`, then `--flux-plugin default=ID`, then the
genre table's `flux_host.plugins.default`, then `$FLUX_LONGFORM_PLUGIN`, then
`Surge XT` (the intended instrument; flux-installer issue #5). A missing plugin
fails before rendering, with the list of instruments the engine found. Layer
levels: the genre table's optional `flux_host.gain_db.LAYER`, else the timbre
table's `level_db` relative to the bass. This keeps the table's balance between
layers and lets the plugin's own output set the absolute level. No shipped
genre table has a `flux_host` block yet.

**Drums.** With `flux_host.drum_presets` in the genre table (deep house and
maqam groove have one, measured 2026-10-25), every listed drum voice gets its
own engine track with its own preset of the default instrument (Surge XT's
factory and third-party percussion patches: a 909-style kick, a noise-layer
clap, hats, a cymbal; a membranophone model for doum and tek). Unless a drum
instrument is configured (`--flux-plugin drums=ID` or
`flux_host.plugins.drums`), drums stay on the built-in synth (with its drum
reverb and the genre's texture, e.g. lo-fi vinyl) and are mixed with the engine
stems. The built-in kick also drives the genre's sidechain ducking of the
engine layers. With a drum instrument, the kick (GM 36) gets its own track so
the `low` stem can still be formed, and no built-in audio is used.

**Not carried yet** (counted in each section's `meta.json` →
`stats.conversion`): CC74 brightness automation (the engine accepts CC events
now; mapping CC74 to each patch's filter is still to do). Quarter tones and
presets are carried; see the sections below.

### Decision log (provenance)

`provenance.json` → `decision_log` (`flux.decision_log/0.1`) is the start of
the human-decision log. It holds, in order:

- `actor: human` entries copied from the plan's `human_decisions` (genre,
  duration, seed and plan edits that the caller recorded);
- `actor: agent` entries for every flux-engine agent turn the renderer made,
  per section (`Agent: longform s012: instruments`, `… notes`), plus each
  section's `get_history` result and the path of its saved project
  (`engine_history`). These are automated steps, not human decisions;
- `human_review: null`.

What it does not hold yet: any human listening or approval. A human-review step
would fill `human_review` with who reviewed, when, the verdict per section
(approve, redo or edited) and any edits, recorded as further engine agent turns
on the saved section projects (so `get_history` shows them) or as plan changes
with a new plan hash. The disclosure text counts only `human_decisions`.

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
  synth path takes about 8 minutes for 4 h. The flux-host path with Test Synth
  and with Surge XT are measured below.
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

### flux-host renderer, 10 minutes of deep house

Same machine, measured on 2026-10-23 (load average about 3). Each run was one
unattended command from an empty directory: `--genre "deep house" --hours
0.1666667 --seed 1` (0:10:11, 14 sections, 13 joins). flux-host used **Test
Synth** for every pitched layer (`--flux-plugin default="Test Synth"`), drums on
the built-in synth, engine block 1024.

| Renderer | Wall clock | Speed | Render / master / analysis (s) | LUFS | True peak | Join clicks | Peak RSS main |
|---|---|---|---|---|---|---|---|
| builtin (8 workers), 2 runs | 23.3 / 23.8 s | 26.3× / 25.7× | 7.1–7.3 / 14.8–14.9 / 1.1 | −14.01 | −1.30 dBTP | 0 of 13 | 0.54 GB |
| flux-host (1 engine), 2 runs | 45.2 / 44.8 s | 13.6× / 13.7× | 28.0 / 15.5–15.7 / 1.1 | −14.01 | −1.30 dBTP | 0 of 13 (6 instrument clicks near joins) | 1.9 GB |
| flux-host, rerun on a finished directory | 16.4 s | | 0 (14 cached) / 14.9 / 1.1 | | | | |

- flux-host takes about 1.9× the wall clock of the builtin renderer for 10
  minutes. Its render stage (28 s) splits into about 4.0 s of project building
  over JSON-RPC (8,659 notes), 17.6 s of engine rendering (about 35× real time
  for the mix plus four stems per section) and about 6 s in Python (built-in
  drums, stem mixing, FLAC writes). Sections render one after another; the
  builtin renderer spreads them over 8 processes.
- **Test Synth is a test instrument, not release sound.** It is a bare 8-voice
  sine with no envelope, so every note-off is a hard step, and it applies note
  events at the start of each processing block (up to 21 ms early at block
  1024). The flux-host master has 1,047 such sample discontinuities (the
  builtin master: 0). Six of them fall within 50 ms of a join and are reported
  as `section_clicks_near_joins`, because the same click is in the section
  render; the stitching itself added none. Surge XT (flux-installer issue #5)
  is the intended instrument; see the next section.
- Peak RSS of the main process rises to 1.9 GB because the drum synth and the
  stems of the longest section are held there (the builtin renderer does that
  in worker processes).
- The integration test (`test_longform_fluxhost.py`, skipped without
  `flux-engine` or Test Synth) renders a 2:54 two-section plan with an 8-bar
  blend, bass swap and +3 BPM ramp in about 15 s. It checks loudness, true
  peak, join clicks, non-silent stems, provenance, engine restart after a
  crash, and that a rerun renders nothing.

### flux-host renderer with Surge XT, 10 minutes of deep house

Surge XT 1.3.4 CLAP (official notarised release) on the M1 Ultra, seed 1,
`--renderer flux-host --flux-plugin "default=CLAP-Surge XT-a8f7ec6e-22a0f1f2"`
(bass, lead, pad and chords on Surge XT; drums on the built-in synth):

| Audio | Wall clock | Speed | Render stage | Master stage | LUFS / true peak | Join clicks | Section clicks near joins |
|---|---|---|---|---|---|---|---|
| 10:12 (14 sections) | 69.7 s | 8.8× | 52.1 s | 16.1 s | −14.01 / −1.30 dBTP | 0 | 0 |

- Unlike Test Synth, Surge XT has envelopes, so there are no section clicks.
- Rendering section 6 again with `flux-engine render --stems`: bass −28.1,
  lead −26.9, pad −18.7, chords −22.1 dBFS RMS, so every Surge track sounds.
  The section measures 122.48 BPM (plan 122.5) and A minor (plan A minor) with
  `flux_ai.analysis.analyse`.
- That run used Surge XT's default "Init Saw" patch on every layer. Presets
  came next (below).
- Installing Surge XT without admin rights: expand the official `.pkg`
  (`pkgutil --expand-full`), copy `Surge XT.clap` to
  `~/Library/Audio/Plug-Ins/CLAP/` and the resources payload to
  `~/Library/Application Support/Surge XT/`.

### Per-layer Surge XT presets

flux-engine now has `list_presets`/`load_preset` (flux-host #11). Each genre
table names a Surge XT factory patch per layer under `flux_host.presets`, plus
two measured corrections for that plugin:

- `transpose`: several factory basses and organs sound one or two octaves
  below the written note (measured with a single held note and pYIN: "Deep
  end" plays A0 for a written A2, "House Organ" A1 for A3). The renderer adds
  the listed semitones so every layer sounds where the planner wrote it. All
  patches were in tune to within 0.05 semitone.
- `gain_db`: patch output levels differ by up to 19 dB; the offsets keep the
  timbre table's balance relative to the bass (measured on a test phrase per
  layer after transposing).

Override per run with `--flux-preset LAYER=NAME` (MCP: `flux_presets`).

### Genre-table knobs added after the first listening review (2026-10-24)

Two Surge XT renders (`assets/renders/`) were reviewed by an audio-capable model
with the prompt in `assets/renders/LISTENING-PROMPT.md`; the measurements and
the review's verdicts are next to them. Table keys that came out of it:

| Key | Meaning |
|---|---|
| `segment.transition.swap_at` | where the kick + bass swap sits in a blend (0 = first bar) |
| `pad.tones` | scale steps of the pad chord (default `[0, 2, 4]`; maqam uses `[0, 4, 7]`, no third) |
| `lead.degree_weights` | melodic preference per scale degree (the jins of a maqam) |
| `lead.rest_degrees` | the last note of every motif snaps to the nearest of these |
| `timbre.kick.tune_to_key` | built-in kick fundamental on the section tonic nearest `freq` |
| `flux_host.eq.<plugin>.<layer>` | `{highpass, lowpass}` Hz applied to the engine stem |
| `offbeat_push_ms` | offbeat-eighth hats, percussion and ride land this late (and softer); odd sixteenths softer still |
| `pad.octave` | register of the pad (default: the chords' octave) |
| `lead.rest_below_energy` | below this energy the lead rests every other phrase (default 0.75) |
| `lead.ornament_prob` | chance of a neighbour-note grace per phrase (default 0.3) |
| `lead.start_degrees` | degrees a motif may start on (default `[0, 2, 4]`) |
| `flux_host.drum_presets.<plugin>.<voice>` | `{preset, note, gain_db}`: that drum voice on its own engine track with this preset of the default instrument, played at `note`; voices not listed stay on the built-in synth |

Round 2 (2026-10-25, two independent listeners on the v2 renders) added the
groove keys above, kept kick and bass playing through an incoming overlap, let
the set's final outro keep its harmony, made stepwise melodies treat the degree
below the tonic as a lower neighbour only and approach their resting note by
step, and cut the maqam riser and ghammaz drone.

Generator version 2 (staggered entries, drum hand-off in overlaps, pad tones
from the table) changes every section hash, so old run directories re-render.
The planner now gives each segment two themes from the motif library: the
grooves and breakdowns play a B theme, the other sections the A theme.

### Maqam quarter tones through flux-host

The section MIDI carries quarter tones as per-channel pitch bends (MPE style).
Surge XT applies a pitch bend to all its voices whatever the channel, so the
renderer routes each distinct cent offset of a layer to its own extra
instrument track (`lead~-50c`), whose clip starts with one pitch bend
(`write_notes` events, flux-host #10). `flux_host.bend_range` gives the
instrument's range (Surge XT: ±2 semitones, measured). Offsets beyond the range
are rounded and counted (`microtones_out_of_range`).

3:25 of maqam bayati on D, seed 3, Surge XT: all 142 microtonal notes carried
on 18 extra tracks, none out of range; −14.00 LUFS, −1.30 dBTP, no clicks, 51 s
wall clock. Rendering section s000 with stems and tracking pitch (pYIN): the
`lead~-50c` and `chords~-50c` stems sit on pitch class 3.5 (E half-flat, 150
cents above D, bayati's second degree) for 80% and 49% of voiced frames, and
on no semitone; the unbent stems sit on D, A and C.

| Genre | bass | chords | pad | lead |
|---|---|---|---|---|
| deep house | Deep end (+24) | House Organ (+24) | Winter warmer | Nice pluck 1 |
| techno | Rubber bass (+12) | FM pluck (+12) | Distant | Acidofil (+24) |
| lo-fi hip hop | Mellow (+24) | EP2 | Still | Lofipluck |
| ambient | Sub 1 (+12) | Sunday | Verb Pad (−12) | Fantasy Bell |
| maqam groove | Pianobass (+24) | Organ 1 | Distant | East |

Chosen by name and measurement, not by listening; they are starting points
for curation.

10 minutes of deep house, seed 1, with these presets:

| Audio | Wall clock | Speed | LUFS / true peak | Join clicks | Section clicks |
|---|---|---|---|---|---|
| 10:12 | 199.8 s | 3.1× | −14.00 / −1.30 dBTP | 0 | 0 |

- The richer patches are about 3× slower to render than Init Saw (8.8×).
- Tempo estimates match the plan (122.5 / 124 BPM). Key estimates are mixed:
  section s010 reads B minor as planned; s003 and s006 read D minor (the
  plan's iv chord dominates those sections); the full part 1 reads A major
  (right tonic; the organ's drawbar harmonics add a major third). The same
  plan rendered with the built-in synth also misses part 2 (G major), so the
  estimator, not the instruments, is the limit here.

## Limitations

- **The built-in synth is a placeholder** for flux-host + Surge XT rendering.
  It is serviceable for testing the pipeline and checking structure, not a
  release-quality sound. The flux-host renderer with per-layer Surge XT
  presets is the intended path (measured above).
- flux-host renders sections sequentially through one engine; several engines
  in parallel would bring the render stage closer to the builtin renderer's.
- Genre tables are starting values for owner curation, not sourced facts; the
  maqam cents are a 24-EDO-style approximation. Each extra cent offset costs one
  more instrument instance per section.
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
