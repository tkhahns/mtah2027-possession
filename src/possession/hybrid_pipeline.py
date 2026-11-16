"""Dead-ball/carry decoder: the nearest observed team owns the ball, an observed ball outside the
pitch is dead, configured intervals are always dead, and an owner is carried through missing-ball
frames only while play remains live.
"""

import numpy as np

from games import GAME, GAMES, half_video_blocks

from .data import AWAY, HOME

NOBODY = -1
KICKOFF_2ND = 113742


def _last_seen(valid):
    """Index of the latest true element, or -1 before the first one."""
    valid = np.asarray(valid, dtype=bool)
    return np.maximum.accumulate(np.where(valid, np.arange(len(valid)), -1))


def _always_dead(trk):
    if "always_dead" in trk:
        return trk["always_dead"]
    return [(half_video_blocks(GAME)[1][1], KICKOFF_2ND)]


def nearest_team(trk):
    """Nearest observed team, with NOBODY where ball or every player is absent."""
    players = np.asarray(trk["players"])
    ball = np.asarray(trk["ball"])
    teams = np.asarray(trk["team"])
    distance = np.linalg.norm(players[..., :2] - ball[:, None, :2], axis=2)
    valid = np.isfinite(distance).any(axis=1) & np.isfinite(ball[:, :2]).all(axis=1)
    out = np.full(len(ball), NOBODY, np.int8)
    out[valid] = teams[np.nanargmin(distance[valid], axis=1)]
    return out


def smooth_labels(codes, window=50):
    home_c = np.r_[0, np.cumsum(codes == HOME)]
    away_c = np.r_[0, np.cumsum(codes == AWAY)]
    index = np.arange(len(codes))
    lo, hi = np.maximum(0, index - window), np.minimum(len(codes), index + window + 1)
    home, away = home_c[hi] - home_c[lo], away_c[hi] - away_c[lo]
    labels = np.full(len(codes), "D", dtype="<U1")
    visible = home + away > 0
    labels[visible] = np.where(home[visible] >= away[visible], "H", "A")
    return labels


def dead_ball(trk):
    """Dead frames from last-observed out-of-pitch ball state and fixed match intervals."""
    ball = np.asarray(trk["ball"])
    observed = np.isfinite(ball[:, :2]).all(axis=1)
    last = _last_seen(observed)
    length, width = trk.get("pitch", GAMES[GAME]["pitch"])
    dead = np.zeros(len(ball), dtype=bool)
    known = last >= 0
    xy = ball[last[known], :2]
    dead[known] = ((xy[:, 0] < 0) | (xy[:, 0] > length) |
                   (xy[:, 1] < 0) | (xy[:, 1] > width))
    for first, stop in _always_dead(trk):
        dead[max(0, int(first)):min(len(dead), int(stop))] = True
    return dead


def carry_owner(codes, dead):
    """Carry the latest live owner through missing evidence, never across dead ball."""
    codes = np.asarray(codes)
    dead = np.asarray(dead, dtype=bool)
    known = (codes != NOBODY) | dead
    source = _last_seen(known)
    out = codes.astype(np.int8, copy=True)
    fill = (out == NOBODY) & ~dead & (source >= 0)
    out[fill] = np.where(dead[source[fill]], NOBODY, out[source[fill]])
    out[dead] = NOBODY
    return out


def labels_from_codes(codes, dead, window=50):
    labels = smooth_labels(carry_owner(codes, dead), window)
    labels[dead] = "D"
    return labels


def predict_deadball(trk, window=50):
    """Dead-ball/carry baseline: the nearest team, carried through missing ball."""
    return labels_from_codes(nearest_team(trk), dead_ball(trk), window)
