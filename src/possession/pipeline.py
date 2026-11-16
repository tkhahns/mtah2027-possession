"""The pipeline that builds our submission. It starts as the official smooth50 baseline,
re-implemented on the cached arrays, so main produces a valid file from day one. Replace
stages, keep the shape:

    tracking -> per-frame evidence -> dead-ball mask -> team per frame -> temporal model -> labels
"""

import numpy as np

from .data import AWAY, HOME, load_tracking
from .labels import graded_mask

NOBODY = -1


def nearest_team(trk):
    """Team of the slot nearest the ball, NOBODY where the ball is lost. Workstream: team signal."""
    ball_xy = trk["ball"][:, None, :2]
    dist = np.linalg.norm(trk["players"][:, :, :2] - ball_xy, axis=2)    # (frames, slots)
    has_ball = ~np.isnan(trk["ball"][:, 0])
    has_player = ~np.all(np.isnan(dist), axis=1)
    ok = has_ball & has_player
    out = np.full(len(dist), NOBODY, dtype=np.int8)
    out[ok] = trk["team"][np.nanargmin(dist[ok], axis=1)]
    return out


def dead_ball(trk, codes):
    """True where the ball is dead: wherever nobody was found. Workstream: dead ball."""
    return codes == NOBODY


def smooth(codes, window=50):
    """Centred majority vote, +/- window frames. Workstream: temporal model (HMM/Viterbi next)."""
    home_c = np.concatenate(([0], np.cumsum(codes == HOME)))
    away_c = np.concatenate(([0], np.cumsum(codes == AWAY)))
    idx = np.arange(len(codes))
    lo, hi = np.maximum(0, idx - window), np.minimum(len(codes), idx + window + 1)
    home, away = home_c[hi] - home_c[lo], away_c[hi] - away_c[lo]
    labels = np.full(len(codes), "D", dtype="<U1")
    seen = (home + away) > 0
    labels[seen] = np.where(home[seen] >= away[seen], "H", "A")
    return labels


def predict(trk, window=50):
    """Labels for every frame of a tracking dict (no graded mask): the stages wired together."""
    codes = nearest_team(trk)
    codes[dead_ball(trk, codes)] = NOBODY
    return smooth(codes, window)


def run(window=50):
    labels = predict(load_tracking(), window)
    labels[~graded_mask()] = "X"   # ungraded frames cost nothing either way
    return labels
