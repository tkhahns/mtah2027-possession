"""Replays on the broadcast that still carry tracking. Frames without a homography are blanked
already; a replay with one shows an earlier moment, so its players line up with the tactical
players of that moment rather than of now.
"""

import numpy as np

FPS = 25
CHUNK = 1000


def find(tactical, broadcast, halves, shifts, every=25, near_m=1.0, min_seen=4, min_lag_s=2.0,
         max_lag_s=120.0, lag_step=10, margin=0.25):
    """(broadcast frames,) bool, True on the replays found. tactical, broadcast: {players, ...};
    halves: [(first, last)] tactical frames; shifts: tactical - broadcast frame per half."""
    tp, bp = tactical["players"], broadcast["players"]
    out = np.zeros(len(bp), bool)
    lags = np.arange(int(min_lag_s * FPS), int(max_lag_s * FPS) + 1, lag_step)
    for (first, last), shift in zip(halves, shifts):
        t = np.arange(int(first), int(last) + 1, every)
        b = t - int(shift)
        ok = (b >= 0) & (b < len(bp))
        t, b = t[ok], b[ok]
        xy, keep = _compact(bp, b, min_seen)
        t, b = t[keep], b[keep]
        if not len(t):
            continue
        now = _share(tp, xy, t, near_m)
        earlier = np.zeros(len(t))
        for lag in lags:
            ok = t - lag >= first
            if not ok.any():
                break
            earlier[ok] = np.maximum(earlier[ok], _share(tp, xy[ok], t[ok] - lag, near_m))
        for f in b[earlier - now >= margin]:
            out[max(0, f - every // 2):f + every - every // 2] = True
    return out


def _compact(players, frames, min_seen):
    """((n, m, 2) the seen players first on each kept frame, (len(frames),) kept: at least
    min_seen seen)."""
    xy = np.asarray(players[frames, :, :2], dtype=np.float32)
    seen = ~np.isnan(xy[..., 0])
    keep = seen.sum(1) >= min_seen
    xy, seen = xy[keep], seen[keep]
    m = int(seen.sum(1).max()) if len(seen) else 0
    order = np.argsort(~seen, axis=1, kind="stable")[:, :m]
    return np.take_along_axis(xy, order[..., None], 1), keep


def _share(t_players, xy, frames, near_m):
    """(n,) share of the broadcast players xy (n, m, 2) with a tactical player within near_m at
    the tactical frames."""
    out = np.zeros(len(frames))
    for a in range(0, len(frames), CHUNK):
        tp = np.asarray(t_players[frames[a:a + CHUNK], :, :2], dtype=np.float32)
        b = xy[a:a + CHUNK]
        d = np.linalg.norm(b[:, :, None] - tp[:, None], axis=-1)
        close = (np.nan_to_num(d, nan=np.inf) <= near_m).any(2)
        out[a:a + CHUNK] = close.sum(1) / (~np.isnan(b[..., 0])).sum(1)
    return out
