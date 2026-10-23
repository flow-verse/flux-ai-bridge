"""Anticipatory Music Transformer adapter (Apache-2.0 code and weights): symbolic MIDI.

Install: ``pip install -e "python[amt]"``. Output is a Standard MIDI File: the
request's ``out`` suffix is replaced by ``.mid`` and ``sample_rate`` is 0.

AMT works in seconds and has no tempo, key or text input; ``prompt``, ``bpm``
and ``key`` are recorded in provenance but do not condition the model.

Modes (context is a MIDI file):
  generate  unconditional, 0..seconds
  continue  continue after the context for ``seconds`` (output = context + continuation)
  infill    regenerate ``extra["start"]..extra["end"]`` using the music on both sides
            (anticipation of the future events), or, with
            ``extra["accompany"] = [GM programs]``, treat those instruments as a
            fixed melody and generate an accompaniment for ``seconds``.

``extra``: ``model`` ("medium" default, "small", "large"), ``top_p`` (0.98),
``device`` (mps/cpu/cuda; default mps when available).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable

from flux_ai import licences, provenance
from flux_ai.backends import GenerateRequest, GenerateResult, check_request, finalize, report

TIERS = {"small": "stanford-crfm/music-small-800k", "medium": "stanford-crfm/music-medium-800k", "large": "stanford-crfm/music-large-800k"}


class AMTBackend:
    name = "amt"
    licence_id = "amt"
    modes = frozenset({"generate", "continue", "infill"})
    models = list(TIERS)
    output_kind = "midi"

    def __init__(self):
        self._models: dict[tuple[str, str], object] = {}

    def available(self) -> tuple[bool, str]:
        missing = [m for m in ("anticipation", "torch", "transformers", "mido") if importlib.util.find_spec(m) is None]
        if missing:
            return False, f'missing {", ".join(missing)}: pip install -e "python[amt]"'
        return True, "installed; weights download from Hugging Face on first use"

    def _model(self, tier: str, device: str, progress):
        key = (tier, device)
        if key not in self._models:
            from transformers import AutoModelForCausalLM

            report(progress, 0.02, f"loading {TIERS[tier]} on {device}")
            self._models[key] = AutoModelForCausalLM.from_pretrained(TIERS[tier]).to(device).eval()
        return self._models[key]

    def generate(self, req: GenerateRequest, progress: Callable[[float, str], None] | None = None) -> GenerateResult:
        check_request(self, req)
        licences.require(self.licence_id)
        tier = str(req.extra.get("model", "medium"))
        if tier not in TIERS:
            raise ValueError(f"unknown amt model '{tier}'; choose from {', '.join(TIERS)}")
        import torch
        from anticipation import ops
        from anticipation.convert import events_to_midi, midi_to_events
        from anticipation.sample import generate
        from anticipation.tokenize import extract_instruments

        device = req.extra.get("device") or ("mps" if torch.backends.mps.is_available() else "cpu")
        started = provenance.now()
        model = self._model(tier, device, progress)
        torch.manual_seed(int(req.seed))
        top_p = float(req.extra.get("top_p", 0.98))
        out = Path(req.out).with_suffix(".mid")
        report(progress, 0.05, f"sampling ({req.mode})")
        params = {"top_p": top_p, "device": device}
        if req.mode == "generate":
            events = generate(model, start_time=0, end_time=float(req.seconds), top_p=top_p)
        else:
            ctx = midi_to_events(str(req.context_audio))
            ctx_end = ops.max_time(ctx, seconds=True)
            if req.mode == "continue":
                events = generate(model, ctx_end, ctx_end + float(req.seconds), inputs=ctx, top_p=top_p)
                params["start"], params["end"] = ctx_end, ctx_end + float(req.seconds)
            elif req.extra.get("accompany"):
                instruments = [int(i) for i in req.extra["accompany"]]
                _, controls = extract_instruments(ctx, instruments)
                acc = generate(model, 0, float(req.seconds), controls=controls, top_p=top_p)
                events = ops.combine(acc, controls)
                params["accompany"] = instruments
            else:
                start = float(req.extra.get("start", 0.0))
                end = float(req.extra.get("end", min(ctx_end, start + req.seconds)))
                events = generate(model, start, end, inputs=ops.mask(ctx, start, end), top_p=top_p)
                params["start"], params["end"] = start, end
        out.parent.mkdir(parents=True, exist_ok=True)
        events_to_midi(events).save(str(out))
        report(progress, 1.0, "done")
        return finalize(
            self, req, output=out, model=TIERS[tier], model_version="anticipation af37397 / " + TIERS[tier],
            sample_rate=0, seconds=float(ops.max_time(events, seconds=True)), started_at=started, backend_params=params,
        )
