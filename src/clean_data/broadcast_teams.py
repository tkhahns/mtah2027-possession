"""Where the tactical and broadcast trackers disagree on a player's team. Counts only, no fix.

Both ATDs are in metres on the same pitch; broadcast frame = tactical frame - shift, one shift per
half (utils.find_broadcast_blocks).
1. pair: on each frame, a tactical and a broadcast player are paired when each is the other's
   nearest, within pair_m, and no other player of either tracker is within clear_m (contacts,
   where a team label means little, are skipped);
2. tracks: each slot's runs of frames, gaps up to gap_s bridged, on each camera;
3. per tactical track with at least min_paired_s of pairs: agree, disagree or mixed; how many
   broadcast tracks back each side; how the track started (after which track ended nearby); the
   head count; time as the player on the ball. The same per broadcast track, the other way round.
"""

import numpy as np

from utils.series_utils import runs

FPS = 25
CHUNK = 2000
DEFAULTS = {
    "pair_m": 1.0,          # a tactical and a broadcast player this close can be paired
    "clear_m": 2.0,         # ...when nobody else of either tracker is this close
    "gap_s": 0.5,           # a slot's gaps up to this long stay one track
    "min_paired_s": 2.0,    # a track needs this much paired time to get a verdict
    "share": 0.8,           # agree / disagree: at least this share of its paired frames
    "min_backer_s": 1.0,    # a track on the other camera counts as a vote with this much pairing
    "handover_s": 2.0,      # a new track starting this soon after another ended nearby...
    "overlap_s": 1.0,       # ...or this long before, follows it
    "reach_m": 2.0,         # nearby: within reach_m + run_mps x the gap
    "run_mps": 7.0,
    "touchline_m": 1.0,     # tracks this close to a side line (or off the pitch)...
    "touchline_share": 0.8,  # ...for this share of their frames are referees or ball boys
    "ball_m": 2.0,          # on the ball: the nearest player, within ball_m
    "full": 12,             # head count: a team with this many tracked players has one too many
}


def check(tactical, broadcast, halves, shifts, params=None):
    """{"coverage", "tactical": [track rows], "broadcast": [track rows]}. tactical, broadcast:
    {players (N, K, >=2), ball (N, >=2), team (K,), pitch}; halves: [(first, last)] tactical
    frames; shifts: tactical - broadcast frame per half."""
    p = {**DEFAULTS, **(params or {})}
    m = pairing(tactical, broadcast, halves, shifts, p)
    tp, t_team = tactical["players"], np.asarray(tactical["team"])
    t_seen = ~np.isnan(tp[..., 0])
    counts = np.stack([(t_seen & (t_team == c)).sum(1) for c in (0, 1)], 1)
    on_ball = _on_ball(tp, tactical["ball"], p["ball_m"])
    off_line = _touchline(tp, tactical["pitch"], p["touchline_m"])
    paired = m["partner"] >= 0
    coverage = {"player_frames": int((t_seen & m["graded"][:, None]).sum()),
                "paired": int(paired.sum()), "agree": int(m["agree"].sum())}
    t_rows = _tactical_rows(m["t_table"], m["t_ids"], m["partner"], m["agree"], m["graded"],
                            counts, on_ball, off_line, t_team, tp, p)
    b_rows = _broadcast_rows(m["b_table"], m["partner"], m["agree"], m["t_ids"],
                             np.asarray(broadcast["team"]), p)
    return {"coverage": coverage, "tactical": t_rows, "broadcast": b_rows}


def pairing(tactical, broadcast, halves, shifts, params=None, skip=None):
    """Per tactical frame and slot: partner (the broadcast track paired, -1 if none) and agree
    (same team); with each camera's tracks (t_ids, t_table, b_ids, b_table) and graded (frames
    inside the halves). skip: (broadcast frames,) bool, frames paired with nobody (replays); the
    broadcast tracks are not cut there."""
    p = {**DEFAULTS, **(params or {})}
    tp, bp = tactical["players"], broadcast["players"]
    t_team, b_team = np.asarray(tactical["team"]), np.asarray(broadcast["team"])
    t_seen, b_seen = ~np.isnan(tp[..., 0]), ~np.isnan(bp[..., 0])
    gap = int(p["gap_s"] * FPS)
    t_ids, t_table = tracks(t_seen, gap)
    b_ids, b_table = tracks(b_seen, gap)
    partner = np.full(t_seen.shape, -1, np.int32)
    graded = np.zeros(len(tp), bool)
    for (first, last), shift in zip(halves, shifts):
        t = np.arange(int(first), int(last) + 1)
        graded[t] = True
        b = t - int(shift)
        ok = (b >= 0) & (b < len(bp))
        if skip is not None:
            ok[ok] = ~np.asarray(skip, bool)[b[ok]]
        t, b = t[ok], b[ok]
        slot = pair(tp, bp, t, b, p["pair_m"], p["clear_m"])
        partner[t] = np.where(slot >= 0, b_ids[b[:, None], np.maximum(slot, 0)], -1)
    b_track_team = np.array([b_team[s] for s, _, _ in b_table])
    agree = np.zeros(t_seen.shape, bool)
    paired = partner >= 0
    agree[paired] = np.broadcast_to(t_team, t_seen.shape)[paired] == b_track_team[partner[paired]]
    return {"partner": partner, "agree": agree, "t_ids": t_ids, "t_table": t_table,
            "b_ids": b_ids, "b_table": b_table, "graded": graded}


def pair(t_players, b_players, t_frames, b_frames, pair_m=1.0, clear_m=2.0):
    """(len(t_frames), K_t) broadcast slot paired with each tactical slot, -1 where none: each
    the other's nearest, within pair_m, nobody else of either tracker within clear_m."""
    out = np.full((len(t_frames), t_players.shape[1]), -1, np.int32)
    for a in range(0, len(t_frames), CHUNK):
        tp = np.asarray(t_players[t_frames[a:a + CHUNK], :, :2], dtype=np.float32)
        bp = np.asarray(b_players[b_frames[a:a + CHUNK], :, :2], dtype=np.float32)
        d = np.linalg.norm(tp[:, :, None] - bp[:, None], axis=-1)
        d = np.where(np.isnan(d), np.inf, d)
        j, i = d.argmin(2), d.argmin(1)
        best = np.take_along_axis(d, j[..., None], 2)[..., 0]
        mutual = np.take_along_axis(i, j, 1) == np.arange(d.shape[1])[None]
        other_b = _second(d, 2)
        other_t = np.take_along_axis(_second(d, 1), j, 1)
        ok = mutual & (best <= pair_m) & (other_b > clear_m) & (other_t > clear_m)
        out[a:a + CHUNK] = np.where(ok, j, -1)
    return out


def tracks(seen, gap):
    """((frames, slots) track id, -1 where unseen; [(slot, first, stop)]): each slot's runs of
    seen frames, gaps up to `gap` frames bridged."""
    ids = np.full(seen.shape, -1, np.int32)
    table = []
    for s in range(seen.shape[1]):
        merged = []
        for a, z in runs(seen[:, s]):
            if merged and a - merged[-1][1] <= gap:
                merged[-1][1] = z
            else:
                merged.append([a, z])
        for a, z in merged:
            ids[a:z, s] = np.where(seen[a:z, s], len(table), -1)
            table.append((s, a, z))
    return ids, table


def _tactical_rows(table, ids, partner, agree, graded, counts, on_ball, off_line, team, tp, p):
    min_paired, min_backer = int(p["min_paired_s"] * FPS), int(p["min_backer_s"] * FPS)
    ends = np.array([z for _, _, z in table])
    order = np.argsort(ends)
    rows = []
    for k, (s, a, z) in enumerate(table):
        f = np.arange(a, z)
        f = f[graded[f] & (ids[f, s] == k)]
        if not len(f):
            continue
        row = {"track": k, "slot": int(s), "team": int(team[s]), "first": int(a), "stop": int(z),
               "seconds": round(len(f) / FPS, 1), "verdict": "unchecked"}
        if off_line[f, s].mean() >= p["touchline_share"]:
            row["verdict"] = "touchline"
            rows.append(row)
            continue
        on = f[partner[f, s] >= 0]
        row["paired_s"] = round(len(on) / FPS, 1)
        if len(on) >= min_paired:
            share = 1.0 - agree[on, s].mean()
            row["disagree"] = round(float(share), 3)
            row["verdict"] = ("disagree" if share >= p["share"] else
                              "agree" if share <= 1 - p["share"] else "mixed")
            row["backers_against"] = _votes(partner[on, s][~agree[on, s]], min_backer)
            row["backers_for"] = _votes(partner[on, s][agree[on, s]], min_backer)
            bad = on[~agree[on, s]]
            if len(bad):
                own, other = counts[bad, team[s]], counts[bad, 1 - team[s]]
                row["full_share"] = round(float(np.mean((own >= p["full"]) & (other < p["full"] - 1))), 3)
        row["start"], row["after_team"] = _start(k, table, ends, order, team, tp, p)
        row["ball_s"] = round(float((on_ball[f] == s).sum()) / FPS, 1)
        rows.append(row)
    return rows


def _broadcast_rows(table, partner, agree, t_ids, b_team, p):
    """Per broadcast track with pairs: paired seconds, disagree share and the tactical tracks
    behind each side."""
    min_paired, min_backer = int(p["min_paired_s"] * FPS), int(p["min_backer_s"] * FPS)
    f, s = np.nonzero(partner >= 0)
    b_track = partner[f, s]
    order = np.argsort(b_track, kind="stable")
    f, s, b_track = f[order], s[order], b_track[order]
    bounds = np.flatnonzero(np.diff(np.r_[-1, b_track, len(table)])) if len(b_track) else []
    rows = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        k = int(b_track[lo])
        ff, ss = f[lo:hi], s[lo:hi]
        ok = agree[ff, ss]
        slot, a, z = table[k]
        row = {"track": k, "slot": int(slot), "team": int(b_team[slot]),
               "paired_s": round(len(ff) / FPS, 1), "first_tactical": int(ff.min()),
               "stop_tactical": int(ff.max()) + 1, "verdict": "unchecked"}
        if len(ff) >= min_paired:
            share = 1.0 - ok.mean()
            row["disagree"] = round(float(share), 3)
            row["verdict"] = ("disagree" if share >= p["share"] else
                              "agree" if share <= 1 - p["share"] else "mixed")
            row["backers_against"] = _votes(t_ids[ff[~ok], ss[~ok]], min_backer)
            row["backers_for"] = _votes(t_ids[ff[ok], ss[ok]], min_backer)
        rows.append(row)
    return rows


def _votes(track_ids, minimum):
    """How many distinct tracks appear at least `minimum` times."""
    if not len(track_ids):
        return 0
    _, n = np.unique(track_ids, return_counts=True)
    return int((n >= minimum).sum())


def _start(k, table, ends, order, team, tp, p):
    """("half start" / "same slot" / "after teammate" / "after opponent" / "fresh", that track's
    team or None): the nearest other track that ended within handover_s before this one started
    (or overlap_s after) close enough to have been the same player."""
    s, a, _ = table[k]
    lo = np.searchsorted(ends[order], a - int(p["handover_s"] * FPS))
    hi = np.searchsorted(ends[order], a + int(p["overlap_s"] * FPS), side="right")
    start = tp[a, s, :2]
    best, best_d = None, np.inf
    for j in order[lo:hi]:
        js, ja, jz = table[j]
        if j == k or ja >= a:
            continue
        gap = max(0, a - jz) / FPS
        d = float(np.linalg.norm(tp[jz - 1, js, :2] - start))
        if d <= p["reach_m"] + p["run_mps"] * gap and d < best_d:
            best, best_d = j, d
    if best is None:
        before = a == 0 or bool(np.isnan(tp[max(0, a - FPS):a, :, 0]).all())
        return ("half start" if before else "fresh"), None
    other = int(team[table[best][0]])
    if table[best][0] == s:
        return "same slot", other
    return ("after teammate" if other == team[s] else "after opponent"), other


def _on_ball(players, ball, ball_m):
    """Per frame: the slot nearest the ball if within ball_m, else -1."""
    out = np.full(len(players), -1, np.int32)
    for a in range(0, len(players), 20000):
        d = np.linalg.norm(np.asarray(players[a:a + 20000, :, :2], dtype=np.float32)
                           - np.asarray(ball[a:a + 20000, None, :2], dtype=np.float32), axis=-1)
        d = np.where(np.isnan(d), np.inf, d)
        near = d.argmin(1)
        out[a:a + 20000] = np.where(d.min(1) <= ball_m, near, -1)
    return out


def _touchline(players, pitch, margin):
    """(frames, slots): within margin of a side line, or off the pitch."""
    x, y = players[..., 0], players[..., 1]
    with np.errstate(invalid="ignore"):
        return ((y <= margin) | (y >= pitch[1] - margin) | (x < 0) | (x > pitch[0]))


def _second(d, axis):
    """The second smallest value along axis (inf when there is only one)."""
    if d.shape[axis] < 2:
        return np.full(np.delete(d.shape, axis), np.inf)
    return np.take(np.partition(d, 1, axis=axis), 1, axis=axis)
