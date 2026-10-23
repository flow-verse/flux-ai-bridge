# flux-ai job protocol v2

`flux-ai --json <command> …` and `flux-ai serve` write **one JSON object per
line on stdout and nothing else**. Library chatter (progress bars, logging,
C-level prints from MLX/TFLite) is redirected to stderr at the file-descriptor
level, so a host can parse stdout line by line. Lines that do not parse as JSON
should still be ignored, as `worker.rs` does.

## Events

Every event carries:

| field | type | meaning |
|---|---|---|
| `event` | string | `started`, `progress`, `done`, `error` (plus `ready`, `pong`, `bye` in serve mode) |
| `protocol_version` | int | always `2` |
| `job_id` | string | caller-supplied (`--job-id`, or `job_id` in a serve request) or random |
| `job` | string | `separate`, `transcribe`, `generate`, `backends`, `serve` |

Per-event fields:

| event | fields (v1 fields in **bold**) |
|---|---|
| `started` | **`job`**, **`input`** (input file; the prompt for `generate`; `""` for `backends`); `generate` adds `backend`, `out` |
| `progress` | `fraction` (0..1, non-decreasing within a job), `message` |
| `done` | **`job`**, **`outputs`** (list of paths); `generate` adds `backend`, `model`, `sample_rate`, `seconds`, `elapsed_s`, `provenance` (sidecar path; also the last entry of `outputs`); `backends` adds `backends` (list, see below) |
| `error` | **`job`**, **`message`**, `error_type` (Python exception class, e.g. `LicenceNotAccepted`) |

### Backwards compatibility

v1 had exactly `started {job,input}`, `done {job,outputs}` and
`error {job,message}`. v2 only *adds* fields and the new `progress` event, so a
v1 parser that ignores unknown fields and unknown event types keeps working.
`crates/flux-ai-clap/src/worker.rs` deserialises with serde (unknown fields are
ignored; an unknown `event` tag fails to parse and the line is skipped like any
non-JSON line), so it needs no change.

Example (`flux-ai --json --job-id j1 generate --backend sine-test --prompt house --seconds 2 --bpm 122 --key "A minor" --out g.wav`):

```json
{"event": "started", "protocol_version": 2, "job_id": "j1", "job": "generate", "input": "house", "backend": "sine-test", "out": "g.wav"}
{"event": "progress", "protocol_version": 2, "job_id": "j1", "job": "generate", "fraction": 0.0, "message": "rendering"}
{"event": "progress", "protocol_version": 2, "job_id": "j1", "job": "generate", "fraction": 0.8, "message": "writing wav"}
{"event": "progress", "protocol_version": 2, "job_id": "j1", "job": "generate", "fraction": 1.0, "message": "done"}
{"event": "done", "protocol_version": 2, "job_id": "j1", "job": "generate", "outputs": ["g.wav", "g.wav.provenance.json"], "backend": "sine-test", "model": "sine-test-1", "sample_rate": 44100, "seconds": 2.0, "elapsed_s": 0.02, "provenance": "g.wav.provenance.json"}
```

## Commands

```bash
flux-ai --json separate in.wav --out stems/
flux-ai --json transcribe in.wav --out midi/
flux-ai --json generate --backend ace-step --prompt "deep house" --seconds 120 --bpm 122 --key "A minor" \
        --seed 1 --out song.wav --accept-licence ace-step-1.5
flux-ai --json generate --backend ace-step --mode repaint --context song.wav --start 30 --end 45 --prompt "breakdown" --out song2.wav
flux-ai --json generate --backend ace-step --mode continue --context song.wav --seconds 60 --prompt "…" --out longer.wav
flux-ai --json backends            # availability, licences, modes, tiers
flux-ai --json serve               # warm worker, see below
flux-ai bench --backend ace-step --model turbo --backend magenta-rt --model base --model small --seeds 5
```

`generate` options: `--backend` (ace-step | magenta-rt | stable-audio-3 | amt | sine-test),
`--prompt`, `--seconds`, `--bpm`, `--key`, `--seed`, `--mode`
(generate | continue | repaint | infill), `--context`, `--start`/`--end`
(repaint/infill region), `--lyrics`, `--vocals`, `--model` (tier),
`--extra KEY=VALUE` (backend-specific, value parsed as JSON when possible),
`--out`, `--accept-licence ID` (repeatable).

## Licence acceptance

Each weights-bearing backend has a licence id (`ace-step-1.5`, `magenta-rt-2`,
`stable-audio-3`, `amt`). Weights are downloaded on demand, never bundled.
Before first use the licence must be accepted once: interactively on a
terminal, or non-interactively with `--accept-licence ID` (agents, CI, serve
requests via `accept_licences`). Acceptances are recorded in
`$FLUX_AI_CACHE/licences.json` (default `~/.cache/flux-ai/licences.json`).
Without it a job fails with `error_type: "LicenceNotAccepted"` and a message
naming the licence, its URL and the flag to pass.

## Provenance sidecar

Every output `<out>` gets `<out>.provenance.json`
(schema `flux-ai/provenance@1`): `ai_generated: true`, `flux_ai_version`,
`backend`, `model` (`id`, `version`, `weights_licence`, `licence_acceptance`),
`prompt`, `lyrics`, `seed`, `conditioning` (mode, seconds, bpm, key,
instrumental, context file and its sha256, extra), `backend_params` (the exact
parameters passed to the model), `started_at`/`finished_at` (UTC ISO 8601),
`output` (path, sha256, sample rate, seconds) and `host`.

## Warm worker: `flux-ai serve`

Reads newline-delimited JSON requests on stdin and keeps backend instances and
their loaded weights in memory between jobs. Jobs run one at a time in
arrival order; a failing job emits `error` and the server continues. EOF or a
`shutdown` request ends the process with exit code 0.

On start it emits `{"event": "ready", "job": "serve", "jobs": [...], …}`.

Requests:

```json
{"job_id": "a1", "job": "generate", "params": {"backend": "ace-step", "prompt": "deep house", "seconds": 120,
  "bpm": 122, "key": "A minor", "seed": 1, "out": "/abs/song.wav", "extra": {"model": "turbo"}},
 "accept_licences": ["ace-step-1.5"]}
{"job_id": "a2", "job": "separate", "params": {"input": "/abs/in.wav", "out": "/abs/stems"}}
{"job_id": "a3", "job": "backends"}
{"job": "ping"}
{"job": "shutdown"}
```

`generate` params mirror the CLI flags: `backend`, `prompt`, `seconds`, `bpm`,
`key`, `seed`, `mode`, `context`, `lyrics`, `instrumental`, `extra`, `out`.
Replies are the same events as the one-shot CLI, tagged with the request's
`job_id`; `ping` answers `pong` and `shutdown` answers `bye`. A line that is
not a JSON object yields an `error` event with `job: "serve"`.
