"""Start long tracking gaps by carrying on the last motion (step 4 of clean_data.pipeline, after
the bounded interpolation of clean_data.interpolate).

A long gap is a run of frames where a slot (or the ball) has no x, y for more than min_gap frames
(the pipeline uses 15 s, the longest gap the interpolation fills). Its first n_fill frames (the
pipeline uses FILL_FRAMES = 20 = 0.8 s, ball only) are filled by dead reckoning: the object keeps
the direction and speed it had over the last `lookback` frames (LOOKBACK_FRAMES = 5) before the
gap. The rest of the gap stays empty.

A long gap is left alone when
- the object is not tracked on the lookback + 1 frames just before it (gaps at the start of
  the file, tracks too short to give a direction), or
- one of its frames was blanked on purpose by the keeper repair (a cell tracked in the original
  tracking but NaN here): moving that slot on would bring the ghost keeper back.
A gap that runs to the end of the file counts: the object is still missing for > min_gap.
On a filled frame speed is that constant speed (m/s), worked out from the positions.
"""

import numpy as np

FILL_FRAMES = 20    
LOOKBACK_FRAMES = 5


def extrapolate_gaps(xys, min_gap, n_fill, lookback, fps, protect=None):
    """Dead-reckon the start of every gap longer than min_gap frames, column by column.

    xys       (frames, cols, 3) x, y, speed; a cell is missing where x is NaN
    min_gap   only gaps longer than this many frames are filled
    n_fill    how many frames at the start of such a gap are filled
    lookback  direction and speed = mean step over the last lookback frames before the gap
    protect   (frames, cols) bool, cells that must stay NaN: a gap holding one is not filled

    Returns (a filled copy, filled: (frames, cols) bool, long gaps per column, filled gaps per
    column)."""
    n, cols = xys.shape[:2]
    tracked = ~np.isnan(xys[..., 0])
    out = xys.copy()
    filled = np.zeros((n, cols), dtype=bool)
    n_long = np.zeros(cols, dtype=np.int64)
    n_done = np.zeros(cols, dtype=np.int64)
    for c in range(cols):
        edges = np.diff(np.r_[0, (~tracked[:, c]).astype(np.int8), 0])
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        long = stops - starts > min_gap
        n_long[c] = long.sum()
        for s, e in zip(starts[long], stops[long]):
            first = s - lookback - 1              # frames first .. s-1: lookback steps
            if first < 0 or not tracked[first:s, c].all():
                continue
            if protect is not None and protect[s:e, c].any():
                continue
            last = xys[s - 1, c, :2].astype(np.float64)
            step = (last - xys[first, c, :2].astype(np.float64)) / lookback   # metres per frame
            k = min(n_fill, e - s)
            out[s:s + k, c, :2] = last + step * np.arange(1, k + 1)[:, None]
            out[s:s + k, c, 2] = np.hypot(*step) * fps
            filled[s:s + k, c] = True
            n_done[c] += 1
    return out, filled, n_long, n_done


def extrapolate(players, ball, min_gap, n_fill, lookback, fps, original=None, ball_only=False):
    """extrapolate_gaps() on the players and the ball together (ball_only: the ball alone, the
    players are left as they are). original = the raw (players, ball): cells tracked there but
    NaN here were blanked on purpose and are protected."""
    xys = np.concatenate([players, ball[:, None]], axis=1)
    protect = None
    if original is not None:
        raw = np.concatenate([original[0], original[1][:, None]], axis=1)
        protect = np.isnan(xys[..., 0]) & ~np.isnan(raw[..., 0])
    cols = slice(-1, None) if ball_only else slice(None)
    out, filled = xys.copy(), np.zeros(xys.shape[:2], dtype=bool)
    n_long, n_done = np.zeros(xys.shape[1], np.int64), np.zeros(xys.shape[1], np.int64)
    out[:, cols], filled[:, cols], n_long[cols], n_done[cols] = extrapolate_gaps(
        xys[:, cols], min_gap, n_fill, lookback, fps, None if protect is None else protect[:, cols])
    return {"players": out[:, :-1], "ball": out[:, -1],
            "filled_players": filled[:, :-1], "filled_ball": filled[:, -1],
            "long_players": int(n_long[:-1].sum()), "done_players": int(n_done[:-1].sum()),
            "long_ball": int(n_long[-1]), "done_ball": int(n_done[-1]),
            "protect": protect}


def off_pitch(xy, pitch):
    """(...) bool: the point is outside the pitch lines."""
    return (xy[..., 0] < 0) | (xy[..., 0] > pitch[0]) | (xy[..., 1] < 0) | (xy[..., 1] > pitch[1])
