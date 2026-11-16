"""Team-role evidence at several time scales: does a track fit its own team's shape or the other
team's, over windows of 3 s to 1 min?

1. grid: per scale, windows every third of their length (at least 1 s). In each, every team's
   players seen on at least half the window go to its 11 roles (Hungarian
   on mean distances, the team's position fitted along). delta[slot] = cost as labelled - cost
   with that slot moved to the other team (team positions kept from the labelled fit): above 0
   the slot fits the other team better. Only when it takes one of their roles (within lam_m):
   otherwise the head count alone would say so, e.g. for a ghost copy of a teammate.
2. thresholds (src/models/team_role_scales.json): per scale, the delta that real players on clean
   Metrica exceed 1% of the time. A wrong-team track falls below -threshold as rarely: moving it
   back is the same comparison turned round.
3. stretch: the windows inside a stretch; z = their median / threshold at the longest scale that
   fits. "other": z >= 1 and every shorter scale's median is above 0 too; "own": z <= -1 and every
   shorter scale's median is below 0.
"""

import json
import multiprocessing
import os
import warnings

import numpy as np

from .team_fit import attack_signs, formations, oriented

FPS = 25
EVERY = 5
SCALES = (3, 5, 10, 20, 30, 60)
RESTARTS = 5                       # k-means starts for the formations (one can merge two players)
MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
                     "team_role_scales.json")
_JOB = {}


def hungarian(cost):
    """Column assigned to each row of a square cost matrix, minimising the total (O(n^3))."""
    cost = np.asarray(cost, dtype=float)
    n = len(cost)
    u, v = np.zeros(n + 1), np.zeros(n + 1)
    p, way = np.zeros(n + 1, dtype=int), np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(n + 1, np.inf), np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            free = ~used[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            j1 = 1 + int(np.argmin(np.where(free, minv[1:], np.inf)))
            delta = minv[j1]
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    out = np.zeros(n, dtype=int)
    out[p[1:] - 1] = np.arange(n)
    return out


def assign(rows, k, lam_m, empty_m):
    """(cost, role of each row, k = none): rows (P, k) mean distances; an extra player costs
    lam_m, an empty role empty_m."""
    p = len(rows)
    n = max(p, k)
    square = np.full((n, n), lam_m)
    square[p:, :] = empty_m
    square[:p, :k] = rows
    cols = hungarian(square)
    return float(square[np.arange(n), cols].sum()), cols[:p]


def fit(paths, roles, lam_m, empty_m, iters=3):
    """(cost, centre (frames, 2), rows (P, K)): the assignment's total cost, also giving the team's
    fitted position per frame and each player's mean distance to each role (at most lam_m)."""
    k = len(roles)
    if not len(paths):
        return empty_m * k, None, np.zeros((0, k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmedian(paths, axis=0)
        for step in range(iters):
            d = np.linalg.norm((paths - centre)[:, :, None, :] - roles[None, None], axis=-1)
            rows = np.minimum(np.nanmean(d, axis=1), lam_m)
            cost, cols = assign(rows, k, lam_m, empty_m)
            if step == iters - 1:
                break
            hit = np.flatnonzero(cols < k)
            hit = hit[rows[hit, cols[hit]] < lam_m]
            if not len(hit):
                break
            offset = np.nanmean(paths[hit] - roles[cols[hit]][:, None, :], axis=0)
            centre = np.where(np.isnan(offset), centre, offset)
    return cost, centre, rows


def window_deltas(xy, team, roles, signs, pitch, lam_m=10.0, empty_m=5.0, min_seen=0.5):
    """(slots,) delta of each slot seen on min_seen of the frames of xy (frames, slots, 2); NaN
    otherwise. roles (2, K, 2) per team; signs: (sign team 0, sign team 1)."""
    team = np.asarray(team)
    out = np.full(xy.shape[1], np.nan)
    if np.isnan(roles).any():
        return out
    ok = (~np.isnan(xy[..., 0])).mean(axis=0) >= min_seen
    k = roles.shape[1]
    turned = [oriented(xy, signs[c], pitch) for c in (0, 1)]
    fits = []
    for c in (0, 1):
        members = np.flatnonzero(ok & (team == c))
        paths = np.moveaxis(turned[c][:, members], 1, 0)
        fits.append((members,) + fit(paths, roles[c], lam_m, empty_m))
    for s in np.flatnonzero(ok):
        c, o = int(team[s]), 1 - int(team[s])
        members, cost_c, _, rows_c = fits[c]
        _, cost_o, centre_o, rows_o = fits[o]
        if centre_o is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            d = np.linalg.norm((turned[o][:, s] - centre_o)[:, None] - roles[o][None], axis=-1)
            row = np.minimum(np.nanmean(d, axis=0), lam_m)
        if not np.isfinite(row).all():
            continue
        minus = assign(rows_c[members != s], k, lam_m, empty_m)[0]
        plus, cols = assign(np.vstack([rows_o, row]), k, lam_m, empty_m)
        delta = cost_c + cost_o - minus - plus
        took = cols[-1] < k and row[min(cols[-1], k - 1)] < lam_m
        out[s] = delta if took else min(delta, 0.0)
    return out


def windows(halves, scales=SCALES):
    """{scale: first frames of its windows}, a third of the window apart (at least 1 s), inside
    the halves ((first, last) frames, inclusive)."""
    out = {}
    for L in scales:
        size, step = L * FPS, max(FPS, L * FPS // 3)
        out[L] = np.concatenate([np.arange(a, z + 2 - size, step) for a, z in halves]
                                or [np.zeros(0, int)]).astype(np.int64)
    return out


def grid(players, team, halves, pitch, scales=SCALES, forms=None, signs=None, workers=1, **kw):
    """{scale: (first frames (W,), delta (W, slots))} over the halves. forms, signs: from
    clean_data.team_fit (worked out here when not given, RESTARTS k-means starts); workers > 1
    runs the windows in that many processes where the platform can fork."""
    team = np.asarray(team)
    if forms is None:
        forms = formations(players, team, halves, pitch, 120, restarts=RESTARTS)
    signs = signs or attack_signs(players, team, halves, pitch)
    starts = windows(halves, scales)
    jobs = [(L, int(t)) for L in scales for t in starts[L]]
    _JOB.update(players=players, team=team, halves=halves, pitch=pitch, forms=forms, signs=signs,
                kw=kw)
    try:
        if workers > 1 and "fork" in multiprocessing.get_all_start_methods():
            chunks = [jobs[i::workers * 4] for i in range(workers * 4)]
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                parts = pool.map(_windows, chunks)
            done = dict(zip([j for c in chunks for j in c], [d for p in parts for d in p]))
        else:
            done = dict(zip(jobs, _windows(jobs)))
    finally:
        _JOB.clear()
    n = players.shape[1]
    return {L: (starts[L], np.array([done[(L, int(t))] for t in starts[L]]).reshape(-1, n))
            for L in scales}


def _windows(jobs):
    j = _JOB
    out = []
    for L, t in jobs:
        frames = np.arange(t, t + L * FPS, EVERY)
        h = next(h for h, (a, z) in enumerate(j["halves"]) if a <= t <= z)
        grid_frames, roles = j["forms"][h]
        g = int(np.argmin(np.abs(grid_frames - frames[len(frames) // 2])))
        xy = np.asarray(j["players"][frames, :, :2], dtype=np.float64)
        out.append(window_deltas(xy, j["team"], roles[g], (j["signs"][(h, 0)], j["signs"][(h, 1)]),
                                 j["pitch"], **j["kw"]))
    return out


def load_thresholds(path=MODEL):
    """{scale: threshold in metres}."""
    with open(path) as handle:
        return {int(k): v for k, v in json.load(handle)["thresholds"].items()}


def stretch(g, thresholds, slot, first, stop):
    """Evidence that slot follows a player of the other team on frames first..stop-1: {"verdict":
    "other" / "own" / None, "z", "scale" (longest with a window inside), "per_scale": {scale:
    (windows, median delta)}}."""
    per = {}
    for L, (starts, delta) in g.items():
        if L not in thresholds:
            continue
        lo, hi = np.searchsorted(starts, [first, stop - L * FPS + 1])
        v = delta[lo:hi, slot]
        v = v[np.isfinite(v)]
        if len(v):
            per[L] = (len(v), float(np.median(v)))
    if not per:
        return {"verdict": None, "z": np.nan, "scale": None, "per_scale": per}
    top = max(per)
    z = per[top][1] / thresholds[top]
    signs = [np.sign(m) for L, (_, m) in per.items() if L < top]
    verdict = ("other" if z >= 1 and all(s > 0 for s in signs) else
               "own" if z <= -1 and all(s < 0 for s in signs) else None)
    return {"verdict": verdict, "z": float(z), "scale": top, "per_scale": per}
