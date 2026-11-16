"""Do two players play for the same team? A small learned model of how a pair moves together over
a window of frames, and a vote of every other player on a track's team.

1. velocities: central differences over +-2 frames (m/s), NaN where a position is missing;
2. pair features over the window's frames where both are seen (FEATURES): how often they are
   within 2 m and 5 m, their median distance, how alike their velocities are (cosine when both
   run, speed difference) and how often they run together;
3. model: logistic regression P(same team | features), trained on external games only (Metrica);
4. vote: every other player j says "same team as i" with weight logit P(same | i, j), signed by
   whether the labels agree; support(i) = minus the mean, so above 0 the others say i is
   labelled with the wrong team.
"""

import json

import numpy as np

FPS = 25
RUN_MPS = 1.5          # a player is running above this speed
MIN_SHARED = 10        # a pair needs this many shared samples to vote
FEATURES = ("within_2m", "within_5m", "median_m", "cosine", "speed_diff", "together")


def velocities(players, half=2, fps=FPS):
    """(N, K, 2) m/s by central differences over +-half frames, NaN where unknown."""
    xy = np.asarray(players[..., :2], dtype=np.float32)
    out = np.full(xy.shape, np.nan, np.float32)
    out[half:-half] = (xy[2 * half:] - xy[:-2 * half]) * (fps / (2 * half))
    return out


def pair_features(xi, vi, xj, vj):
    """(len(FEATURES),) for two players over the same frames (positions and velocities (n, 2)),
    or None with fewer than MIN_SHARED frames where both are seen."""
    ok = ~(np.isnan(xi[:, 0]) | np.isnan(xj[:, 0]))
    if ok.sum() < MIN_SHARED:
        return None
    d = np.linalg.norm(xi[ok] - xj[ok], axis=1)
    a, b = vi[ok], vj[ok]
    known = ~(np.isnan(a[:, 0]) | np.isnan(b[:, 0]))
    sa, sb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    run = known & (sa > RUN_MPS) & (sb > RUN_MPS)
    cos = (a[run] * b[run]).sum(1) / (sa[run] * sb[run]) if run.any() else np.zeros(0)
    return np.array([
        np.mean(d <= 2.0), np.mean(d <= 5.0), min(np.median(d), 40.0) / 40.0,
        cos.mean() if len(cos) else 0.0,
        min(np.mean(np.abs(sa[known] - sb[known])), 5.0) / 5.0 if known.any() else 0.0,
        np.sum(cos > 0.8) / max(1, known.sum()),
    ])


def track_pairs(players, vel, frames, slot):
    """([other slots], (n, len(FEATURES))) for slot against every other slot over `frames`."""
    xi, vi = players[frames, slot, :2], vel[frames, slot]
    others, rows = [], []
    for j in range(players.shape[1]):
        if j == slot:
            continue
        f = pair_features(xi, vi, players[frames, j, :2], vel[frames, j])
        if f is not None:
            others.append(j)
            rows.append(f)
    return others, np.array(rows).reshape(-1, len(FEATURES))


class PairModel:
    """Logistic regression on FEATURES: P(same team)."""

    def __init__(self, weights, bias):
        self.weights, self.bias = np.asarray(weights, dtype=float), float(bias)

    @classmethod
    def fit(cls, X, y, l2=1e-3, iters=50):
        """Newton steps on the L2-penalised log loss; X (n, len(FEATURES)), y 1 = same team."""
        A = np.c_[np.asarray(X, dtype=float), np.ones(len(X))]
        y = np.asarray(y, dtype=float)
        w = np.zeros(A.shape[1])
        reg = l2 * np.eye(A.shape[1])
        reg[-1, -1] = 0.0
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-A @ w))
            grad = A.T @ (p - y) / len(y) + reg @ w
            hess = (A * (p * (1 - p))[:, None]).T @ A / len(y) + reg
            step = np.linalg.solve(hess, grad)
            w -= step
            if np.abs(step).max() < 1e-8:
                break
        return cls(w[:-1], w[-1])

    def logit(self, X):
        return np.asarray(X, dtype=float) @ self.weights + self.bias

    def to_json(self):
        return {"features": list(FEATURES), "weights": self.weights.tolist(), "bias": self.bias}

    @classmethod
    def from_json(cls, data):
        if tuple(data["features"]) != FEATURES:
            raise ValueError("the saved pair model uses other features: {}".format(data["features"]))
        return cls(data["weights"], data["bias"])

    def save(self, path):
        with open(path, "w") as handle:
            json.dump(self.to_json(), handle, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as handle:
            return cls.from_json(json.load(handle))


def support(model, players, vel, team, frames, slot):
    """(support for the other team, number of voters) for slot over `frames`: minus the mean of
    the label-signed logits of every other player with enough shared frames."""
    others, X = track_pairs(players, vel, frames, slot)
    if not others:
        return np.nan, 0
    team = np.asarray(team)
    sign = np.where(team[others] == team[slot], 1.0, -1.0)
    return float(-np.mean(sign * model.logit(X))), len(others)
