"""Three test sets built from the Metrica sample games, to choose pipeline variants on.

    from possession import testsets
    trk, truth = testsets.load(1, "messy")      # kind: "clean", "hidden" or "messy"

- clean   Metrica as it is: near-perfect tracking, 22 players, ball lost mostly when dead.
          A diagnostic only: our match is nothing like this.
- hidden  clean, but the ball is blanked on 40% of live play, in stretches, more often in
          crowds and on fast balls (our ball goes missing mostly *during* play; Metrica's
          doesn't). 
- messy   hidden, plus the tracking faults docs/DATA_SCHEMA.md describes for our match: players
          split over 46 tracklet slots (23 per team) that appear and vanish, ~20.5 of 22
          players seen per frame, a few duplicate tracks, position jitter, and ID swaps between
          opponents in tackles near the ball, which put a player under the wrong team.

The truth is the same in all three (built from the events and the *clean* ball). Each set is
deterministic (fixed seed per game) and cached in external/metrica/cache/ (gitignored); bump
VERSION after changing anything here so stale caches aren't used. The messy rates are set from
the aggregate facts in DATA_SCHEMA.md where there is one; the swap and duplicate rates are
guesses (MESSY below), not fitted to our match.
"""

import os

import numpy as np

from . import metrica

KINDS = ("clean", "hidden", "messy")
VERSION = 2                           # in the cache file names (see above)
FPS = 25
SLOTS_PER_TEAM = 23                   # 46 slots for 22 players, like our ATD file

MESSY = {
    "tracklet_s": 30.0,               # mean tracklet length before the track breaks
    "gap_s": 1.6,                     # mean gap after a break: ~20.5 of 22 players seen
    "swap_m": 1.2,                    # opponents this close at the ball may swap identities
    "swap_p": 0.3,                    # share of those contacts that swap
    "swap_s": (1.0, 4.0),             # how long a swap lasts
    "duplicates_per_min": 1.0,        # ghost copies of a player in another slot of his team
    "duplicate_s": (1.0, 5.0),
    "jitter_m": 0.25,                 # position noise, drifting slowly (smoothed over 1 s)
    "ball_jitter_m": 0.15,
}


def load(game, kind="clean", root=metrica.METRICA_DIR, refresh=False):
    """(trk, truth) for one game and test set. trk has the same keys as metrica.load_game(),
    with speeds filled in; messy has 46 player slots instead of 22."""
    if kind not in KINDS:
        raise ValueError("kind must be one of {}".format(KINDS))
    trk = metrica.load_game(game, root)
    truth = metrica.truth_from_events(metrica.read_events(game, root), len(trk["ball"]), trk["ball"])
    if kind == "clean":
        return with_speed(trk), truth
    cache = os.path.join(root, "cache", "game{}_{}_v{}.npz".format(game, kind, VERSION))
    if os.path.exists(cache) and not refresh:
        return {**np.load(cache), "pitch": trk["pitch"], "always_dead": trk["always_dead"]}, truth
    seed = 1729 + game
    out = hide_ball(trk, truth, seed=seed)
    if kind == "messy":
        out = make_messy(out, seed=seed)
    out = with_speed(out)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, players=out["players"], ball=out["ball"], team=out["team"])
    return out, truth


def speed(xy, step=2):
    """m/s per frame from positions (frames, ..., 2): distance over +-step frames; NaN where
    either end is missing."""
    out = np.full(xy.shape[:-1], np.nan, dtype=np.float32)
    if len(xy) > 2 * step:
        out[step:-step] = np.linalg.norm(xy[2 * step:] - xy[:-2 * step], axis=-1) * FPS / (2 * step)
    return out


def with_speed(trk):
    """Copy of trk with the speed column (index 2) of players and ball computed from positions."""
    players, ball = trk["players"].copy(), trk["ball"].copy()
    players[..., 2] = speed(players[..., :2])
    ball[:, 2] = speed(ball[:, :2])
    return {**trk, "players": players, "ball": ball}


# -- hidden ball ------------------------------------------------------------------------------

def hide_ball(trk, truth, share=0.40, mean_gap_frames=35, seed=1729):
    """Copy with the ball blanked on `share` of live (H/A) frames, in contiguous stretches.

    Gap starts are picked more often around fast balls and crowded areas; lengths are log-normal 
    (many short occlusions, a few long crowded ones). Only live frames are blanked; players, truth 
    and dead frames are left alone. A validation perturbation, not training on our match."""
    truth = np.asarray(truth)
    ball = np.asarray(trk["ball"]).copy()
    players = np.asarray(trk["players"])
    live = np.isin(truth, ["H", "A"])
    visible = np.isfinite(ball[:, :2]).all(axis=1)
    live_n = int(live.sum())
    need = max(0, int(round(np.clip(share, 0.0, 1.0) * live_n)) - int((live & ~visible).sum()))
    hidden = np.zeros(len(ball), dtype=bool)
    if not need or not live_n:
        return {**trk, "ball": ball}

    distance = np.linalg.norm(players[:, :, :2] - ball[:, None, :2], axis=2)
    crowd = np.sum(distance < 8.0, axis=1).astype(np.float32)
    fast = np.nan_to_num(speed(ball[:, :2], step=1))
    weight = 1.0 + np.clip(crowd / 4.0, 0.0, 2.0) + np.clip(fast / 12.0, 0.0, 2.0)
    rng = np.random.default_rng(seed)
    attempts = 0
    while need > 0 and attempts < live_n * 2:
        choices = np.flatnonzero(live & visible & ~hidden)
        if not len(choices):
            break
        p = weight[choices].astype(np.float64)
        start = int(rng.choice(choices, p=p / p.sum()))
        length = int(np.clip(rng.lognormal(np.log(max(2, mean_gap_frames)), 0.65), 5, 5 * mean_gap_frames))
        frames = np.arange(start, min(len(ball), start + min(length, need)))
        frames = frames[live[frames] & visible[frames] & ~hidden[frames]]
        hidden[frames] = True
        need -= len(frames)
        attempts += 1
    ball[hidden] = np.nan
    return {**trk, "ball": ball}


# -- messy tracking ---------------------------------------------------------------------------

def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _swap_at_tackles(players, team, ball, rng, cfg):
    """Swap the positions of the nearest home and away player to the ball for a few seconds when
    they are within swap_m of each other there, for a share swap_p of such contacts."""
    players = players.copy()
    d = np.linalg.norm(players[:, :, :2] - ball[:, None, :2], axis=2)
    d[~np.isfinite(d)] = np.inf
    home = np.flatnonzero(team == team.min())
    away = np.flatnonzero(team != team.min())
    h = home[np.argmin(d[:, home], axis=1)]
    a = away[np.argmin(d[:, away], axis=1)]
    rows = np.arange(len(players))
    gap = np.linalg.norm(players[rows, h, :2] - players[rows, a, :2], axis=1)
    near = np.isfinite(gap) & (gap < cfg["swap_m"]) & (np.minimum(d[rows, h], d[rows, a]) < 3.0)
    for start, _ in _runs(near):
        if rng.random() >= cfg["swap_p"]:
            continue
        stop = min(len(players), start + int(rng.uniform(*cfg["swap_s"]) * FPS))
        i, j = h[start], a[start]
        players[start:stop, [i, j]] = players[start:stop, [j, i]]
    return players


def _to_slots(players, team, rng, cfg):
    """Spread each player over tracklets in the 23 slots of his column's team: the track breaks
    every ~tracklet_s and comes back after ~gap_s in a free slot."""
    n = len(players)
    teams = np.unique(team)
    slot_team = np.repeat(teams, SLOTS_PER_TEAM).astype(team.dtype)
    out = np.full((n, len(slot_team), players.shape[2]), np.nan, dtype=np.float32)
    used = np.zeros((len(slot_team), n), dtype=bool)
    for p in range(players.shape[1]):
        mine = np.flatnonzero(slot_team == team[p])
        for start, stop in _runs(np.isfinite(players[:, p, 0])):
            s = start
            while s < stop:
                e = min(stop, s + max(FPS, int(rng.exponential(cfg["tracklet_s"] * FPS))))
                free = [k for k in mine if not used[k, s:e].any()]
                if free:
                    k = free[int(rng.integers(len(free)))]
                    out[s:e, k] = players[s:e, p]
                    used[k, s:e] = True
                s = e + int(rng.exponential(cfg["gap_s"] * FPS))
    return out, slot_team, used


def _add_duplicates(slots, slot_team, used, rng, cfg):
    """Ghost copies: a player's track repeated, ~1 m off, in another free slot of his team."""
    n = len(slots)
    for _ in range(int(cfg["duplicates_per_min"] * n / (60 * FPS))):
        s = int(rng.integers(n))
        e = min(n, s + int(rng.uniform(*cfg["duplicate_s"]) * FPS))
        src = np.flatnonzero(used[:, s:e].all(axis=1))
        if not len(src):
            continue
        k = src[int(rng.integers(len(src)))]
        free = [j for j in np.flatnonzero(slot_team == slot_team[k]) if not used[j, s:e].any()]
        if not free:
            continue
        j = free[int(rng.integers(len(free)))]
        slots[s:e, j, :2] = slots[s:e, k, :2] + rng.normal(0, 0.7, 2).astype(np.float32)
        used[j, s:e] = True
    return slots


def _jitter(xy, sigma, rng, width=FPS):
    """Noise of about sigma metres, smoothed over `width` frames so it wanders like real error."""
    noise = rng.normal(0, sigma * np.sqrt(width), xy.shape).astype(np.float32)
    c = np.cumsum(np.concatenate([np.zeros((1,) + xy.shape[1:], np.float32), noise]), axis=0)
    smooth = np.empty_like(noise)
    smooth[width - 1:] = (c[width:] - c[:-width]) / width
    smooth[:width - 1] = noise[:width - 1] / np.sqrt(width)
    return xy + smooth


def make_messy(trk, seed=1729, cfg=None):
    """Copy of trk with our match's tracking faults (see the module docstring). The ball is left
    as given (hide it first with hide_ball) apart from jitter."""
    cfg = {**MESSY, **(cfg or {})}
    rng = np.random.default_rng(seed + 100)
    team = np.asarray(trk["team"])
    players = _swap_at_tackles(np.asarray(trk["players"]), team, np.asarray(trk["ball"]), rng, cfg)
    slots, slot_team, used = _to_slots(players, team, rng, cfg)
    slots = _add_duplicates(slots, slot_team, used, rng, cfg)
    slots[..., :2] = _jitter(slots[..., :2], cfg["jitter_m"], rng)
    ball = np.asarray(trk["ball"]).copy()
    ball[:, :2] = _jitter(ball[:, :2], cfg["ball_jitter_m"], rng)
    return {**trk, "players": slots, "ball": ball, "team": slot_team}
