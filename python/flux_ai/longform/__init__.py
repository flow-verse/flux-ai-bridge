# SPDX-License-Identifier: GPL-3.0-or-later
"""Long-form "N hours of <genre>" orchestrator (phase P5).

Host-independent: plans a set, writes symbolic MIDI per section, renders it
(built-in synth now, flux-host later), stitches beat-aligned transitions and
masters to -14 LUFS / -1 dBTP with provenance and AI-disclosure tags.
"""
from .planner import SetPlan, plan  # noqa: F401
