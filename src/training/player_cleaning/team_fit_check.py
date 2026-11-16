"""Calibrate clean_data.team_fit on the Metrica games: how often the team-shape test fires on
players who are on the right team (false alarms), and how many injected wrong-team stretches it
catches. An injected stretch starts at a real contact with an opponent (as our tracker's errors
do) and moves that player's next 60 s onto a new slot of the other team.

    python src/training/player_cleaning/team_fit_check.py
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from clean_data import team_fit  # noqa: E402
from possession import metrica, testsets  # noqa: E402

FPS, EVERY = 25, 5
SCALES = (60, 120, 300)
LENGTHS = (10, 30, 60)
CONTACT_M = 1.5


def contacts(players, team, f0, f1):
    """(frame, slot) of the first opposite-team pair within CONTACT_M in f0..f1, or None."""
    xy = np.asarray(players[f0:f1, :, :2], dtype=np.float64)
    home, away = xy[:, team == 0], xy[:, team == 1]
    d = np.linalg.norm(home[:, :, None] - away[:, None], axis=-1)
    hit = np.argwhere(np.nan_to_num(d, nan=np.inf) <= CONTACT_M)
    if not len(hit):
        return None
    f, i, j = hit[0]
    return f0 + int(f), (np.flatnonzero(team == 0)[i], np.flatnonzero(team == 1)[j])


def inject(trk, halves, rng, spacing_s=180, length_s=60):
    """(players, team, [(new slot, first, stop, original team)]): one injected stretch per
    spacing_s, never a keeper (median oriented x under 20 m)."""
    players, team = trk["players"], np.asarray(trk["team"])
    signs = team_fit.attack_signs(players, team, halves, trk["pitch"])
    picks = []
    for h, (first, last) in enumerate(halves):
        for t0 in range(first, last - length_s * FPS, spacing_s * FPS):
            hit = contacts(players, team, t0, t0 + 60 * FPS)
            if hit is None:
                continue
            f, pair = hit
            s = pair[rng.integers(2)]
            stop = f + length_s * FPS
            seen = ~np.isnan(players[f:stop, s, 0])
            stop = f + (int(np.argmin(seen)) if not seen.all() else len(seen))
            x = team_fit.oriented(players[f:stop, s, :2], signs[(h, team[s])], trk["pitch"])[:, 0]
            if stop - f >= 10 * FPS and np.nanmedian(x) >= 20:
                picks.append((s, f, stop))
    extra = np.full((len(players), len(picks), players.shape[2]), np.nan, dtype=players.dtype)
    moved = players.copy()
    rows = []
    for i, (s, f, stop) in enumerate(picks):
        extra[f:stop, i] = players[f:stop, s]
        moved[f:stop, s] = np.nan
        rows.append((players.shape[1] + i, f, stop, int(team[s])))
    new_team = np.r_[team, [1 - t for _, _, _, t in rows]].astype(np.int8)
    return np.concatenate([moved, extra], axis=1), new_team, rows


def segment_means(sup, frames, slots_runs, length_s):
    """Mean support over each run's first length_s (runs long enough, half its samples known)."""
    out = []
    for s, a, z in slots_runs:
        if z - a < length_s * FPS:
            continue
        sel = (frames >= a) & (frames < a + length_s * FPS)
        v = sup[sel, s]
        if np.isfinite(v).mean() >= 0.5:
            out.append(float(np.nanmean(v)))
    return np.array(out)


def real_runs(players, halves, length_s):
    """Real stretches to compare with: each slot chopped into length_s pieces."""
    runs = []
    for first, last in halves:
        for a in range(first, last - length_s * FPS, length_s * FPS):
            for s in range(players.shape[1]):
                if not np.isnan(players[a:a + length_s * FPS, s, 0]).any():
                    runs.append((s, a, a + length_s * FPS))
    return runs


def main():
    rng = np.random.default_rng(7)
    found = {(k, w, L): [] for k in ("clean", "messy") for w in SCALES for L in LENGTHS}
    alarms = {key: [] for key in found}
    start = time.time()
    for kind in ("clean", "messy"):
        for game in metrica.GAMES:
            trk, _ = testsets.load(game, kind)
            halves = [(a, z) for _, a, z in metrica.halves(metrica.read_events(game))]
            frames = np.concatenate([np.arange(a, z + 1, EVERY) for a, z in halves])
            moved, new_team, rows = inject(trk, halves, rng)
            for w in SCALES:
                forms = team_fit.formations(trk["players"], trk["team"], halves, trk["pitch"], w)
                own, other = team_fit.support(trk["players"], trk["team"], halves, trk["pitch"],
                                              forms, frames)
                forms_m = team_fit.formations(moved, new_team, halves, trk["pitch"], w)
                own_m, other_m = team_fit.support(moved, new_team, halves, trk["pitch"], forms_m,
                                                  frames)
                for L in LENGTHS:
                    alarms[(kind, w, L)].extend(segment_means(
                        own - other, frames, real_runs(trk["players"], halves, L), L))
                    found[(kind, w, L)].extend(segment_means(
                        own_m - other_m, frames, [(s, a, z) for s, a, z, _ in rows], L))
            print("{} game {}: {} injected ({:.0f} s)".format(kind, game, len(rows), time.time() - start),
                  flush=True)
    print("\nthreshold = 99th percentile of real stretches on clean (1% false alarms)")
    print("{:5} {:>7} {:>6} {:>9} {:>11} {:>10} {:>9}".format(
        "kind", "window", "length", "threshold", "false alarm", "caught", "injected"))
    for w in SCALES:
        for L in LENGTHS:
            thr = float(np.percentile(alarms[("clean", w, L)], 99))
            for kind in ("clean", "messy"):
                real, inj = np.array(alarms[(kind, w, L)]), np.array(found[(kind, w, L)])
                print("{:5} {:>6}s {:>5}s {:>9.2f} {:>11.1%} {:>10.1%} {:>9}".format(
                    kind, w, L, thr, np.mean(real > thr), np.mean(inj > thr), len(inj)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
