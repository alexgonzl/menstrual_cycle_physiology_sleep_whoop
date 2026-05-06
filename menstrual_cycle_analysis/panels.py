"""Figure-panel composers.

One function per (sub)panel in the paper, each returning a `matplotlib.Figure`.
Notebooks call these and handle savefig themselves.

Real implementations are added per phase as their notebook lands. Phase A
intentionally leaves this empty so the public-facing API stays small until
Phase B exercises Fig 1.
"""
from __future__ import annotations
