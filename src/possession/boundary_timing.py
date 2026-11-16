"""Refine accepted H/A boundary timing without changing the spell sequence."""

import numpy as np

from .data import AWAY, HOME
from . import restart_state, spell_filter

RADIUS = 36
MIN_GAIN = 2.0


def ownership_evidence(trk, control=None, smooth=2):
    """Signed per-frame evidence: positive home, negative away."""
    players = np.asarray(trk["players"])
    ball = np.asarray(trk["ball"])
    teams = np.asarray(trk["team"])
    distance = np.linalg.norm(players[..., :2] - ball[:, None, :2], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    home = distance[:, teams == HOME].min(axis=1)
    away = distance[:, teams == AWAY].min(axis=1)
    both = np.isfinite(home) & np.isfinite(away) & np.isfinite(ball[:, :2]).all(axis=1)
    geometric = np.zeros(len(ball), np.float32)
    geometric[both] = np.clip((away[both] - home[both]) / 4.0, -1.0, 1.0)
    control = spell_filter.control_team(trk) if control is None else np.asarray(control)
    signed_control = np.where(control == HOME, 1.0, np.where(control == AWAY, -1.0, 0.0))
    evidence = geometric * 0.25 + signed_control
    if smooth:
        evidence = restart_state._window_mean(evidence, smooth, smooth)
    return np.asarray(evidence, dtype=np.float32)


def refine(labels, trk, radius=RADIUS, min_gain=MIN_GAIN, evidence=None):
    """Move H/A transitions to the best nearby control split; keep spell count and D fixed."""
    out = np.asarray(labels).copy()
    evidence = ownership_evidence(trk) if evidence is None else np.asarray(evidence, dtype=np.float32)
    if evidence.shape != out.shape:
        raise ValueError("ownership evidence must have one value per label")
    changes = np.flatnonzero(out[1:] != out[:-1]) + 1
    cumulative = np.r_[0.0, np.cumsum(evidence, dtype=np.float64)]
    for index, at in enumerate(changes):
        old, new = out[at - 1], out[at]
        if old not in "HA" or new not in "HA" or old == new:
            continue
        previous = changes[index - 1] if index else 0
        following = changes[index + 1] if index + 1 < len(changes) else len(out)
        lo = max(previous + 1, at - radius)
        hi = min(following - 1, at + radius)
        candidates = np.arange(lo, hi + 1)
        if not len(candidates):
            continue
        old_sign = 1.0 if old == "H" else -1.0
        new_sign = 1.0 if new == "H" else -1.0
        score = (old_sign * (cumulative[candidates] - cumulative[lo]) +
                 new_sign * (cumulative[hi + 1] - cumulative[candidates]))
        original = int(np.flatnonzero(candidates == at)[0])
        best = int(np.argmax(score))
        if score[best] <= score[original] + min_gain:
            continue
        boundary = int(candidates[best])
        if boundary < at:
            out[boundary:at] = new
        elif boundary > at:
            out[at:boundary] = old
    return out
