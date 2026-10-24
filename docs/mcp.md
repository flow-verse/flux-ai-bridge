# flux-mcp: the Flux MCP server

`flux-mcp` lets an MCP client such as Claude Code compose, arrange and render music in Flux. It is
the "thin MCP adapter" from phase P3 of the agent report: the core contract stays flux-engine's own
JSON-RPC API ([flux-host/flux/engine/API.md](https://github.com/flow-verse/flux-host/blob/main/flux/engine/API.md)),
and flux-mcp translates a small, curated tool set onto it and onto `flux-ai` jobs.

```
MCP client ──stdio (MCP)──▶ flux-mcp ──stdio (JSON-RPC 2.0, NDJSON)──▶ flux-engine serve
                                    └──stdio (flux-ai protocol v2)───▶ flux-ai serve (warm worker)
                                    └──one-shot `flux-ai longform --json` (when available)
```

- **stdio only.** There is no network listener: only the process that starts flux-mcp can talk to
  it, and only flux-mcp can talk to the engine. This is the report's "localhost only" decision; it
  also keeps the AGPL engine off the network.
- **Supervision.** flux-mcp starts one `flux-engine serve` child on first use and one `flux-ai serve`
  worker on the first AI job. If the engine crashes or does not answer within the timeout, the call
  fails with a clear error (including the tail of the engine's stderr), the engine is stopped, and the
  next call starts a fresh engine and reopens the last saved project. That result carries a `notice`
  saying unsaved edits and the undo history were lost.
- **Stdout is protocol only.** The SDK's stdio transport points file descriptor 1 at stderr while
  serving, the children's stdout are pipes owned by flux-mcp, and diagnostics go to stderr.
  `test_stdio_server_speaks_only_mcp` runs the real entry point over stdio.

## Install

```bash
cd flux-ai-bridge
uv venv .venv-mcp --python 3.12
uv pip install --python .venv-mcp/bin/python -e "python[mcp]"          # server
uv pip install --python .venv-mcp/bin/python -e "python[mcp-evals,dev]" # + LLM eval runner and pytest
```

`flux-ai` itself installs `demucs` (and torch) as a base dependency. For the MCP server alone you can
skip that with `uv pip install --no-deps -e python` plus `mcp==2.3.0 numpy`; stem separation then
reports that demucs is missing, everything else works.

You also need a `flux-engine` binary (build flux-host, target `flux-engine`; see
[flux-host/flux/engine/README.md](https://github.com/flow-verse/flux-host/blob/main/flux/engine/README.md)).
flux-mcp finds it via `--engine PATH`, then `$FLUX_ENGINE`, then `PATH`.

### Claude Code

```bash
claude mcp add flux -- /path/to/flux-ai-bridge/.venv-mcp/bin/flux-mcp \
    --engine /path/to/flux-host/conanbuild/Release/products/bin/flux-engine \
    --root ~/Music/flux
```

With the bundled test plugin (a sine "Test Synth") instead of installed plugins:

```bash
claude mcp add flux -e FLUX_CLAP_PATH=/path/to/flux-host/conanbuild/Release/tests/test_plugins/CLAP/Release \
    -- /path/to/.venv-mcp/bin/flux-mcp --engine /path/to/flux-engine
```

Other clients use the same command line (`"command": ".../flux-mcp", "args": ["--engine", "..."]`).

### Options

| Option | Meaning |
|---|---|
| `--engine PATH` | flux-engine binary (default `$FLUX_ENGINE`, then `PATH`) |
| `--plugin-path DIR`, `--vst3-path DIR` | extra plugin folders passed to the engine (repeatable); `CLAP_PATH`/`FLUX_CLAP_PATH`/`FLUX_VST3_PATH` also work |
| `--sr RATE` | engine sample rate (default 48000) |
| `--flux-ai CMD` | how to run flux-ai (default: this Python's `flux_ai`, then `flux-ai` on `PATH`) |
| `--root DIR` | refuse any project, render or import path outside DIR |
| `--workdir DIR` | base for relative paths (default: the current directory); render paths are relative to the project |
| `--timeout S`, `--render-timeout S` | per-call engine timeout (60 s) and render timeout (1800 s) |

## Tools

24 tools. Every description states units and conventions; every schema carries ranges.
Annotations follow the MCP spec: `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`
(the Python SDK spells them `read_only_hint` and so on).

| Tool | What it does | Annotations |
|---|---|---|
| `get_project` | Full `describe_project` (optionally with notes) plus undo history and `unsaved_changes` | read-only |
| `find_plugins` | Search installed CLAP/VST3 plugins | read-only |
| `get_plugin_params` | Parameters with type, normalised and plain value, range; `db` for gain parameters | read-only |
| `list_backends` | flux-ai generation backends: availability, modes, tiers, licence and acceptance | read-only |
| `new_project` | Create and open a project (bpm, time signature) | destructive (closes the open project) |
| `open_project` | Open a saved project | destructive |
| `save_project` | Save in place or "save as" | destructive, idempotent |
| `add_track` | Instrument, MIDI or audio track, optionally with a plugin | additive edit |
| `load_plugin` | Add an instrument or effect to a track | additive edit |
| `set_plugin_params` | Set parameters in `normalized`, `plain`, `db` (gain) or `percent` units | edit, idempotent |
| `write_midi_clip` | One MIDI clip at a bar with notes in beats; pitches as numbers or names (C4 = 60) | additive edit |
| `copy_clip` | Copy a MIDI clip to a bar/track, optionally transposed | additive edit |
| `import_file` | Import audio or MIDI at a bar | additive edit |
| `set_tempo` | Tempo from a bar onwards | edit, idempotent |
| `add_marker` | Named marker at a bar | additive edit |
| `set_render_range` | Song start/end markers (default render range) | edit, idempotent |
| `render_mix` | Offline render to WAV/FLAC, bar range, bit depth, optional stems | destructive (writes files), idempotent |
| `undo_last_turn` | Undo the previous tool call (one agent turn) | destructive |
| `redo` | Redo | destructive |
| `generate_audio` | flux-ai `generate` (audio, or MIDI for `amt`), then import | open-world (downloads weights) |
| `separate_stems` | flux-ai `separate` (Demucs), optional import | open-world |
| `transcribe_to_midi` | flux-ai `transcribe` (Basic Pitch), then import | open-world |
| `plan_longform` | `flux-ai longform --plan-only`; result also at `flux://plan` | idempotent |
| `run_longform` | `flux-ai longform` from a plan (or genre and hours); `renderer` `builtin` (default) or `flux-host` with optional `flux_plugins` per layer (passed as `--renderer flux-host --flux-plugin LAYER=ID`, plus `--flux-engine` when the server was given an engine path) | open-world |

Not exposed on purpose: `begin_agent_turn`/`end_agent_turn` (flux-mcp wraps every edit itself),
`get_history` (part of `get_project` and every edit result), `ping`, `shutdown`, `set_loop_range`.

### Conventions the tools enforce

These come from the DAWZY study (state before edits, units in tools, atomic undoable edits, few tools):

- **Fresh state.** Every edit result contains `project`: the compact summary read *after* the edit
  (track, plugin and clip ids and extents, tempo, range, markers, history, unsaved flag).
- **One tool call, one undo step.** Each edit tool runs `begin_agent_turn`, its engine calls, then
  `end_agent_turn`, so the undo history shows `Agent: write clip on Bass, bars 1-3`. If any step fails
  the turn is ended with `discard: true`: a failed tool call leaves the project exactly as it was.
  `generate_audio`/`separate_stems`/`transcribe_to_midi` import their results as one turn too.
- **Musical time.** Bars are 1-based, range ends exclusive; note times are beats from the clip start.
  Pitches accept names (`"F#2"`); gain accepts dB (`-6` → 0.501 plain). Conversions are code, not the model.
- **Validation before sending.** Schemas reject out-of-range values; pitches are parsed and notes that
  overrun their clip are rejected with the exact beat before the engine is called.
- **`dry_run`** on every edit tool returns the engine's `changes` list without applying anything.

## Resources

| URI | Content |
|---|---|
| `flux://project` | `describe_project` JSON of the open project (`{"open": false}` if none) |
| `flux://plan` | the latest long-form plan from `plan_longform`/`run_longform` in this session |
| `flux://backends` | the `list_backends` result |

## Progress

`render_mix`, `generate_audio`, `separate_stems`, `transcribe_to_midi`, `plan_longform` and
`run_longform` send MCP progress notifications when the client passes a progress token:

- flux-ai jobs forward the worker's `progress` events as `progress = fraction × 100`, `total = 100`,
  with the event's message (the worker throttles them to 1 % steps);
- `render_mix` forwards flux-engine's render `progress` notifications once a second as `progress = percent of
  frames`, `total = 100` (elapsed seconds only until the first notification arrives).

## Safety model

- **Destructive hints** on everything that discards or overwrites: `new_project`, `open_project`,
  `save_project`, `render_mix`, `undo_last_turn`, `redo`. Clients that confirm destructive tools ask
  the user first.
- **Explicit consent flags, defaulting to safe.** `new_project`/`open_project` refuse to drop unsaved
  edits unless `discard_unsaved: true`; `render_mix` refuses to replace any existing mix or stem file
  unless `overwrite: true`; `save_project` refuses to "save as" into a non-empty directory unless
  `overwrite: true`; AI outputs never overwrite. Descriptions tell the model these need the user's
  agreement.
- **Licences are never accepted implicitly.** `generate_audio` checks the backend's licence first;
  if it is not accepted it fails with `LicenceNotAccepted`, the licence name, URL and commercial-use
  terms, and the instruction to ask the user. Only an explicit `accept_licence` equal to that backend's
  licence id is passed on (flux-ai then records the acceptance once, as with `--accept-licence`); a
  mismatched id is refused. flux-mcp never starts the worker with `--accept-licence`.
  `run_longform` accepts nothing.
- **Provenance.** Every generated file gets flux-ai's `<out>.provenance.json` sidecar.
- **Paths.** `--root` confines all file paths. No tool deletes files.
- Annotations are hints; the consent flags and checks above are what actually protect the user.

## Evaluation

`python/flux_mcp/evals/` holds 22 DAWZY-style tasks (composition, arrangement, mixing, render, files,
AI). Each has a natural-language prompt, a setup done through the tools, a reference tool-call
sequence, and a checker that reads `describe_project` straight from the engine and the rendered files
(FLAC STREAMINFO and WAV headers for exact durations). A pytest negative control asserts that every
checker *fails* on the project as set up, so no task passes by doing nothing.

| id | prompt (abridged) |
|---|---|
| drum_bass_groove_124 | 4-bar drum and bass groove at 124 BPM |
| pad_c_minor_progression | Pad with a 4-bar C minor progression, 3+ note chords |
| transpose_bass_octave | Transpose the bassline up an octave (new clip) |
| render_bars_1_8_flac | Render bars 1-8 to FLAC (checks 16.000 s) |
| set_tempo_90 / tempo_change_bar_17 | Whole-song tempo; tempo change from bar 17 |
| markers_song_form / song_range_32_bars | Intro/Drop/Outro markers; 32-bar song range |
| lead_melody_a_minor | 8-note A minor melody at bar 5 |
| bass_level_half / bass_gain_minus_6db | Plain-unit parameter; -6 dB gain |
| undo_last_change | Undo only the last change |
| render_stems | Bars 1-4 mix plus one stem per track |
| four_on_the_floor / arpeggio_16ths / hihat_velocity_accents | Rhythm, 16th arpeggio, velocity accents |
| two_sections_keys | Intro and chorus clips at bars 1 and 9 |
| save_as / new_project_6_8 | Save as; new 6/8 project at 100 BPM |
| import_midi_bar_5 / import_audio_bar_3 | File imports at bars |
| generate_sine_and_import | Generate with the `sine-test` backend and import (flux-ai worker) |

Runners:

```bash
export FLUX_ENGINE=/path/to/flux-engine FLUX_CLAP_PATH=/path/to/test_plugins/CLAP/Release
.venv-mcp/bin/python -m flux_mcp.evals --list
.venv-mcp/bin/python -m flux_mcp.evals --json scripted.json          # reference sequences (CI)
ANTHROPIC_API_KEY=… .venv-mcp/bin/python -m flux_mcp.evals --llm --json llm.json [--model claude-opus-5-5]
```

The LLM runner is a manual tool-use loop over the Anthropic Messages API (`anthropic` Python SDK):
the MCP tool list becomes the Claude tool list, each `tool_use` block is executed through the MCP
client and returned as `tool_result` (with `is_error`), up to 25 turns. Default model
`claude-opus-5-5` (override with `--model` or `FLUX_MCP_EVAL_MODEL`), effort `high`. It only sends
the task prompt; the model must discover ids with the tools. No server-side model fallback is
enabled, so a score always belongs to the model named; the report records the model that served
each task. Without `ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN` the runner (and its pytest smoke test)
is skipped; keys are never stored.

### Results

| Runner | Date | Result |
|---|---|---|
| scripted (reference sequences) | 2026-10-23, Mac Studio M1 Ultra, flux-engine Release, Test Synth | **22/22 (100 %)**, about 2 s for the suite |
| LLM (`claude-opus-5-5`, effort high) | 2026-10-24, same machine and engine | **22/22 (100 %)**, 1-6 tool calls per task, 3-25 s per task |

The scripted pass rate shows every task is solvable with the tool set and that checkers and tools
agree; it is not a model score. The LLM run is the model score: Claude drove the MCP tools from the
task prompt alone and passed every task, above the P3 bar (≥ 80 %, against DAWZY's 44 %). The suite is
small (22 tasks) and the checkers verify the project state, not musical quality.

API keys that are not scoped to a workspace also need `ANTHROPIC_WORKSPACE_ID`; the runner sends it
as the `anthropic-workspace-id` header.

## Tests

```bash
cd python && ../.venv-mcp/bin/python -m pytest flux_mcp -q
```

- `test_tools.py`: tool count, descriptions, annotations, schema ranges and units, resources,
  pre-engine validation, note-name and dB conversions (no engine needed).
- `test_supervision.py`: a fake engine for crash, timeout, stray stdout and restart; the real
  `flux-mcp` entry point over stdio.
- `test_session.py`: a full session against the real flux-engine (create, tracks, clips, copy and
  transpose, dB/percent params, tempo, markers, range, render with stems, save, resource), one tool
  call = one undo step restoring `describe_project` exactly, failed multi-step edits discarded,
  `dry_run`, error paths, `--root`, crash restart with project reopen, `sine-test` generation with
  monotonic progress, licence refusal, long-form availability.
- `test_evals.py`: the scripted suite must pass 100 %, the negative control, the LLM loop with a
  fake client, and an LLM smoke test when a key is set.

Engine tests skip with a reason when `flux-engine` or the Test Synth is not found (`FLUX_ENGINE`,
`FLUX_CLAP_PATH`, `PATH`, or a sibling `flux-host/conanbuild/Release` build).

## Spec and SDK

- MCP Python SDK **`mcp==2.3.0`** (pinned in the `mcp` extra), high-level `MCPServer` (FastMCP's v2
  name) over `stdio_server`. The SDK negotiates protocol revision 2026-07-28 and still serves
  2025-11-25 clients.
- Relied on: tools with JSON Schema `inputSchema` generated from typed signatures (pydantic `Field`
  ranges and descriptions); `ToolAnnotations` hints (`readOnlyHint`, `destructiveHint` (default true
  when not read-only), `idempotentHint`, `openWorldHint`), which are hints only; tool failures returned
  as results with `isError: true`; static resources with a MIME type; progress via
  `notifications/progress` (`Context.report_progress(progress, total, message)`, absolute and
  monotonic, a no-op when the client sent no progress token); the stdio transport rule that stdout
  carries only MCP messages.

## Deferred

- Edit-in-place, delete and move of clips/tracks, automation, time-signature changes: not in the
  engine API yet (see its README). The tools say so, and `copy_clip` covers transposed variations.
- `notifications/resources/updated` for `flux://project` after edits.
- `run_longform` and `plan_longform` are written against the CLI flags of the long-form branch
  (`flux-ai longform --genre --hours [--plan-only] [--plan] [--seed] [--neural-backend] [--out] --json`)
  and report "unavailable" until that subcommand exists; the plan is read from a `plan`/`plan_path`
  field of the `done` event or from a `.json` output.
- Cancellation of long jobs (an MCP cancel stops waiting but not the flux-ai job).
