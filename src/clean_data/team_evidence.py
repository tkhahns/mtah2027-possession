"""Cut each tactical track at its contacts and gather, per stretch between contacts, the evidence
that it follows a player of the other team.

1. contacts: a track is in contact when an opponent (or a slot marked `neither`) is within near_m;
   an identity can only jump there, so each track is cut into stretches between contacts;
2. per stretch: the broadcast's verdict and votes (clean_data.broadcast_teams.pairing), the mean
   team-shape support per window, the head count (own team 12+, other 10 or fewer) and the time
   as the player on the ball.
"""

import numpy as np

from utils.series_utils import runs

FPS = 25
CHUNK = 5000


def contacts(players, team, neither=None, near_m=1.5):
    """(frames, slots) True where an opponent, or a `neither` slot (frames, slots) bool, is within
    near_m; a `neither` slot itself is never in contact."""
    team = np.asarray(team)
    n, k = players.shape[:2]
    neither = np.zeros((n, k), bool) if neither is None else neither
    opposite = team[:, None] != team[None, :]
    out = np.zeros((n, k), bool)
    for a in range(0, n, CHUNK):
        xy = np.asarray(players[a:a + CHUNK, :, :2], dtype=np.float32)
        d = np.linalg.norm(xy[:, :, None] - xy[:, None], axis=-1)
        close = np.nan_to_num(d, nan=np.inf) <= near_m
        close &= ~np.eye(k, dtype=bool)[None]
        hit = close & (opposite[None] | neither[a:a + CHUNK, None, :])
        out[a:a + CHUNK] = hit.any(axis=2) & ~neither[a:a + CHUNK]
    return out


def stretches(t_table, t_ids, contact, graded, min_s=2.0):
    """[(track, slot, first, stop)]: each track's runs of graded frames without contact, at least
    min_s long."""
    out = []
    for k, (s, a, z) in enumerate(t_table):
        free = (t_ids[a:z, s] == k) & graded[a:z] & ~contact[a:z, s]
        for lo, hi in runs(free):
            if hi - lo >= min_s * FPS:
                out.append((k, int(s), int(a + lo), int(a + hi)))
    return out


def evidence(rows, pairing, supports, frames, counts, on_ball, team, full=12, min_backer_s=1.0):
    """One dict per stretch: seconds, paired_s, disagree (share of pairs saying the other team,
    None without pairs), votes_against / votes_for (broadcast tracks, at least min_backer_s
    each), against_tracks ([[broadcast track, frames], ...] saying the other team, most first, at
    most 3), support {window: mean or None}, full_share, ball_s."""
    team = np.asarray(team)
    out = []
    for k, s, a, z in rows:
        f = np.arange(a, z)
        partner, agree = pairing["partner"][f, s], pairing["agree"][f, s]
        on = partner >= 0
        row = {"track": k, "slot": s, "team": int(team[s]), "first": a, "stop": z,
               "seconds": round((z - a) / FPS, 1), "paired_s": round(on.sum() / FPS, 1),
               "disagree": round(float(1 - agree[on].mean()), 3) if on.any() else None,
               "votes_against": _votes(partner[on & ~agree], min_backer_s),
               "votes_for": _votes(partner[on & agree], min_backer_s),
               "against_tracks": _most(partner[on & ~agree], 3)}
        sel = (frames >= a) & (frames < z)
        row["support"] = {}
        for w, sup in supports.items():
            v = sup[sel, s]
            row["support"][w] = (round(float(np.nanmean(v)), 2)
                                 if sel.sum() >= 2 and np.isfinite(v).mean() >= 0.5 else None)
        own, opp = counts[f, team[s]], counts[f, 1 - team[s]]
        row["full_share"] = round(float(np.mean((own >= full) & (opp <= full - 2))), 3)
        row["ball_s"] = round(float((on_ball[f] == s).sum()) / FPS, 1)
        out.append(row)
    return out


def candidates(rows, min_paired_s=1.0, share=0.8, min_votes=2, head=0.5):
    """The stretches (evidence rows) the broadcast puts on the other team: at least min_paired_s of
    pairs, `share` of them saying so. strong: min_votes broadcast tracks say so, or one does and
    the head count agrees on at least `head` of the stretch."""
    out = []
    for r in rows:
        if r["paired_s"] < min_paired_s or r["disagree"] is None or r["disagree"] < share:
            continue
        votes = r["votes_against"]
        out.append({**r, "strong": bool(votes >= min_votes or (votes >= 1 and r["full_share"] >= head))})
    return out


def broadcast_candidates(tactical, broadcast, halves, shifts, touchline_share=0.8):
    """(candidates, pairing): candidates() on every stretch between contacts of every tactical
    track, side-line tracks left out, the broadcast lined up by `shifts` (tactical - broadcast
    frame, one per half). tactical, broadcast: {players, ball, team, pitch}, the broadcast blanked
    where it has no homography; nobody is paired on its replays (clean_data.broadcast_replays).
    pairing: clean_data.broadcast_teams.pairing's result, with replay_frames."""
    from . import broadcast_replays, broadcast_teams as bt
    replay = broadcast_replays.find(tactical, broadcast, halves, shifts)
    pairing = bt.pairing(tactical, broadcast, halves, shifts, skip=replay)
    players, team = tactical["players"], np.asarray(tactical["team"])
    t_table, t_ids = pairing["t_table"], pairing["t_ids"]
    line = bt._touchline(players, tactical["pitch"], bt.DEFAULTS["touchline_m"])
    side = {k for k, (s, a, z) in enumerate(t_table)
            if line[a:z, s][t_ids[a:z, s] == k].mean() >= touchline_share}
    rows = [r for r in stretches(t_table, t_ids, contacts(players, team), pairing["graded"])
            if r[0] not in side]
    seen = ~np.isnan(players[..., 0])
    counts = np.stack([(seen & (team == c)).sum(1) for c in (0, 1)], 1)
    on_ball = bt._on_ball(players, tactical["ball"], bt.DEFAULTS["ball_m"])
    rows = evidence(rows, pairing, {}, np.zeros(0, dtype=int), counts, on_ball, team)
    return candidates(rows), {**pairing, "stretches": len(rows), "replay_frames": int(replay.sum())}


def _most(track_ids, n):
    """[[track, count], ...]: the n most frequent, most first."""
    if not len(track_ids):
        return []
    ids, counts = np.unique(track_ids, return_counts=True)
    order = np.argsort(-counts, kind="stable")[:n]
    return [[int(ids[i]), int(counts[i])] for i in order]


def _votes(track_ids, min_s):
    if not len(track_ids):
        return 0
    _, n = np.unique(track_ids, return_counts=True)
    return int((n >= min_s * FPS).sum())
