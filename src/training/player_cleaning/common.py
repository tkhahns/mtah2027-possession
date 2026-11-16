"""Shared helpers for the player-team cleaning experiments on the Metrica games (external data
only): loading, injected wrong-team stretches, cached role grids, scoring moves."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from clean_data import role_scales, team_fit  # noqa: E402
from possession import metrica, testsets  # noqa: E402

FPS = 25
CACHE = os.path.join(metrica.METRICA_DIR, "cache")
GRID_VERSION = 2
CONTACT_M = 1.5
LENGTHS = (3, 5, 10, 20, 30, 60)
BINS = ((3, 5), (5, 10), (10, 20), (20, 30), (30, 60), (60, 10 ** 6))


def load(game, kind):
    trk, _ = testsets.load(game, kind)
    halves = [(a, z) for _, a, z in metrica.halves(metrica.read_events(game))]
    return trk, halves


def inject(trk, halves, rng, spacing_s=90, lengths=LENGTHS):
    """(players, team, rows): every spacing_s, at the first contact with an opponent in the next
    60 s, one of the two players' next L seconds (L cycling through lengths, cut where his track
    breaks) go to a new slot of the other team. rows: (new slot, first, stop, L). Keepers (median
    oriented x under 20 m) are never picked; stretches under 3 s are dropped."""
    players, team = trk["players"], np.asarray(trk["team"])
    signs = team_fit.attack_signs(players, team, halves, trk["pitch"])
    picks, n = [], 0
    for h, (first, last) in enumerate(halves):
        for t0 in range(first, last - 60 * FPS, spacing_s * FPS):
            L = lengths[n % len(lengths)]
            n += 1
            hit = _contact(players, team, t0, t0 + 60 * FPS)
            if hit is None:
                continue
            f, pair = hit
            s = pair[rng.integers(2)]
            stop = min(last + 1, f + L * FPS)
            seen = ~np.isnan(players[f:stop, s, 0])
            stop = f + (int(np.argmin(seen)) if not seen.all() else len(seen))
            if stop - f < 3 * FPS or (picks and f < picks[-1][2] + FPS):
                continue
            x = team_fit.oriented(players[f:stop, s, :2], signs[(h, team[s])], trk["pitch"])[:, 0]
            if np.nanmedian(x) >= 20:
                picks.append((s, f, stop, L))
    extra = np.full((len(players), len(picks), players.shape[2]), np.nan, dtype=players.dtype)
    moved = players.copy()
    rows = []
    for i, (s, f, stop, L) in enumerate(picks):
        extra[f:stop, i] = players[f:stop, s]
        moved[f:stop, s] = np.nan
        rows.append((players.shape[1] + i, f, stop, L))
    new_team = np.r_[team, [1 - team[s] for s, _, _, _ in picks]].astype(np.int8)
    return np.concatenate([moved, extra], axis=1), new_team, rows


def _contact(players, team, f0, f1):
    xy = np.asarray(players[f0:f1, :, :2], dtype=np.float64)
    d = np.linalg.norm(xy[:, team == 0][:, :, None] - xy[:, team == 1][:, None], axis=-1)
    hit = np.argwhere(np.nan_to_num(d, nan=np.inf) <= CONTACT_M)
    if not len(hit):
        return None
    f, i, j = hit[0]
    return f0 + int(f), (np.flatnonzero(team == 0)[i], np.flatnonzero(team == 1)[j])


def cached_grid(name, players, team, halves, pitch, workers=18):
    """role_scales.grid, kept in external/metrica/cache/ (public data, gitignored)."""
    path = os.path.join(CACHE, "role_scales_{}_v{}.npz".format(name, GRID_VERSION))
    if os.path.exists(path):
        z = np.load(path)
        return {L: (z["s{}".format(L)], z["d{}".format(L)]) for L in role_scales.SCALES}
    g = role_scales.grid(players, team, halves, pitch, workers=workers)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, **{"s{}".format(L): g[L][0] for L in g},
                        **{"d{}".format(L): g[L][1] for L in g})
    return g


def score(moves, rows, n_frames, n_slots):
    """(caught per row (bool), false-move frames, false moves): a row is caught when moves cover
    half its frames; a move is false when under half of it is injected."""
    moved = np.zeros((n_frames, n_slots), bool)
    for m in moves:
        moved[m["first"]:m["last"] + 1, m["slot"]] = True
    injected = np.zeros_like(moved)
    for s, a, z, _ in rows:
        injected[a:z, s] = True
    caught = np.array([moved[a:z, s].mean() >= 0.5 for s, a, z, _ in rows], bool)
    false_moves = sum(injected[m["first"]:m["last"] + 1, m["slot"]].mean() < 0.5 for m in moves)
    return caught, int((moved & ~injected).sum()), int(false_moves)


def length_bin(n_frames):
    s = n_frames / FPS
    return next(i for i, (a, b) in enumerate(BINS) if a <= s < b) if s >= BINS[0][0] else None
