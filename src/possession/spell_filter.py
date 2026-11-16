"""Control-aware removal of short, unsupported H/A possession blips."""

import numpy as np

from .data import AWAY, HOME

FPS = 25
MAX_SHORT_FRAMES = 100         # 4.0 s, selected on complete public matches
MIN_CONTROL_FRAMES = 5
CONTROL_DISTANCE = 2.5
CONTROL_MARGIN = 0.45
MAX_CONTROL_SPEED = 10.0


def control_team(trk, distance=CONTROL_DISTANCE, margin=CONTROL_MARGIN,
                 max_speed=MAX_CONTROL_SPEED):
    """Strong controlling team per frame, or -1 when tracking shows no clear control."""
    players = np.asarray(trk["players"])
    ball = np.asarray(trk["ball"])
    teams = np.asarray(trk["team"])
    separation = np.linalg.norm(players[..., :2] - ball[:, None, :2], axis=2)
    separation[~np.isfinite(separation)] = np.inf
    home = separation[:, teams == HOME].min(axis=1)
    away = separation[:, teams == AWAY].min(axis=1)
    nearest = np.minimum(home, away)
    both = np.isfinite(home) & np.isfinite(away)
    advantage = np.zeros(len(ball), np.float32)
    advantage[both] = np.abs(home[both] - away[both])
    visible = np.isfinite(ball[:, :2]).all(axis=1) & np.isfinite(nearest)
    speed = ball[:, 2] if ball.shape[1] > 2 else np.full(len(ball), np.nan)
    controlled = (visible & (nearest <= distance) & (advantage >= margin) &
                  (~np.isfinite(speed) | (speed <= max_speed)))
    owner = np.where(home <= away, HOME, AWAY).astype(np.int8)
    return np.where(controlled, owner, -1).astype(np.int8)


def merge_short_spells(labels, trk, max_frames=MAX_SHORT_FRAMES,
                       min_control_frames=MIN_CONTROL_FRAMES, control=None):
    """Merge a short sandwiched H/A spell unless its team demonstrates strong control."""
    out = np.asarray(labels).copy()
    control = control_team(trk) if control is None else np.asarray(control)
    if control.shape != out.shape:
        raise ValueError("control evidence must have one value per label")
    while True:
        starts = np.flatnonzero(np.r_[True, out[1:] != out[:-1]])
        stops = np.r_[starts[1:], len(out)]
        merged = False
        for i, (start, stop) in enumerate(zip(starts, stops)):
            if i == 0 or i + 1 == len(starts) or out[start] not in "HA":
                continue
            if stop - start > max_frames:
                continue
            previous, following = out[starts[i - 1]], out[starts[i + 1]]
            if previous != following or previous not in "HA":
                continue
            team = HOME if out[start] == "H" else AWAY
            if np.sum(control[start:stop] == team) >= min_control_frames:
                continue
            out[start:stop] = previous
            merged = True
            break
        if not merged:
            return out
