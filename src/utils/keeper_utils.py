"""Find each team's goalkeeper slot on every frame.

1. For each half, find the goal each team defends.
2. For each frame, within a window of `window_s` back and forward (inside the half), score each of
   the team's slots by the share of frames it is the deepest player and the share it is in its
   own penalty area. The best slot is that frame's keeper. Frames within one window of a half's
   start or end, where the window is cut short, take the pick of the nearest full-window frame.

Slots are 0-based: ATD "Track N" is slot N - 1.
"""

import warnings

import numpy as np

HOME, AWAY = 0, 1
LEFT, RIGHT = 0, 1
KEYS = ("keeper_BAR", "keeper_ALT")

DEFAULTS = {
    "window_s": 180.0,
    "fps": 25,
    "w_deepest": 0.5,
    "w_box": 0.5,
    "box_depth_m": 16.5,
    "box_half_width_m": 20.16,
    "off_pitch_m": 2.0,         # players further off the pitch are ignored
    "pitch": (105.3, 68.0),
}


def clean_params(params=None):
    params = dict(params or {})
    unknown = sorted(set(params) - set(DEFAULTS))
    if unknown:
        raise ValueError("unknown keeper_utils settings: {}".format(", ".join(unknown)))
    return {**DEFAULTS, **params}


def on_pitch(xy, margin, pitch):
    """True where the point is at most `margin` beyond any line; False for NaN."""
    x, y = xy[..., 0], xy[..., 1]
    with np.errstate(invalid="ignore"):
        return (x >= -margin) & (x <= pitch[0] + margin) & (y >= -margin) & (y <= pitch[1] + margin)


def defended_sides(players_xy, team, halves):
    """(n_halves, 2): the goal (LEFT / RIGHT) HOME and AWAY defend in each half, -1 in a half with
    no frame where both teams are tracked. The team whose median x is lower on most frames defends
    the left goal."""
    team = np.asarray(team)
    sides = np.empty((len(halves), 2), dtype=np.int8)
    for h, (first, last) in enumerate(halves):
        x = players_xy[first:last + 1, :, 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            home, away = (np.nanmedian(x[:, team == t], axis=1) for t in (HOME, AWAY))
        both = ~np.isnan(home) & ~np.isnan(away)
        if not both.any():
            sides[h] = -1
            continue
        sides[h] = (LEFT, RIGHT) if np.mean(home[both] < away[both]) >= 0.5 else (RIGHT, LEFT)
    return sides


def window_counts(mask, half_window):
    """Count of True in rows [i - half_window, i + half_window] of mask, per row and column."""
    mask = np.asarray(mask)
    csum = np.zeros((len(mask) + 1,) + mask.shape[1:], dtype=np.int32)
    np.cumsum(mask, axis=0, dtype=np.int32, out=csum[1:])
    rows = np.arange(len(mask))
    hi, lo = np.minimum(rows + half_window + 1, len(mask)), np.maximum(rows - half_window, 0)
    return csum[hi] - csum[lo]


def full_window_rows(n, half_window):
    """Per row, the nearest row whose window fits inside [0, n); the middle row if none does."""
    lo = min(half_window, (n - 1) // 2)
    return np.clip(np.arange(n), lo, max(n - 1 - half_window, lo))


def _best(score, slots):
    """Per row, the slot with the highest score; -1 where all scores are 0."""
    i = np.argmax(score, axis=1)
    return np.where(score[np.arange(len(score)), i] > 0, slots[i], -1)


def find_keeper_slots(players_xy, team, halves, params=None):
    """Each team's keeper slot on every frame.

    players_xy: (frames, slots, >= 2) x, y in metres, NaN where empty.
    team:       (slots,) HOME / AWAY.
    halves:     [(first_frame, last_frame), ...], inclusive.

    Returns {"keeper_BAR": (frames,), "keeper_ALT": (frames,)}: slot per frame, -1 outside the
    halves or where no slot of the team is seen.
    """
    p = clean_params(params)
    xy = np.asarray(players_xy)[..., :2]
    team = np.asarray(team)
    length, width = p["pitch"]
    half_window = int(round(p["window_s"] * p["fps"]))
    sides = defended_sides(xy, team, halves)
    out = {key: np.full(len(xy), -1, dtype=np.int64) for key in KEYS}
    for h, (first, last) in enumerate(halves):
        if sides[h, 0] < 0:
            continue
        block = xy[first:last + 1]
        present = on_pitch(block, p["off_pitch_m"], p["pitch"])
        n_window = window_counts(np.ones(len(block), dtype=bool), half_window)[:, None]
        src = full_window_rows(len(block), half_window)
        for t, key in zip((HOME, AWAY), KEYS):
            slots = np.flatnonzero(team == t)
            pts, ok = block[:, slots], present[:, slots]
            depth = pts[..., 0] if sides[h, t] == LEFT else length - pts[..., 0]
            depth = np.where(ok, depth, np.inf)
            seen = np.isfinite(depth).any(axis=1)
            deepest = np.zeros(ok.shape, dtype=bool)
            deepest[seen, np.argmin(depth[seen], axis=1)] = True
            with np.errstate(invalid="ignore"):
                in_box = ((depth <= p["box_depth_m"])
                          & (np.abs(pts[..., 1] - width / 2) <= p["box_half_width_m"]))
            n_seen = np.maximum(window_counts(seen, half_window), 1)[:, None]
            deep_share = window_counts(deepest, half_window) / n_seen
            box_share = window_counts(in_box, half_window) / n_window
            score = p["w_deepest"] * deep_share + p["w_box"] * box_share
            out[key][first:last + 1] = _best(score, slots)[src]
    return out
