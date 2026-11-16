"""Slot-invariant player-shape features for validating Smart Tagging.

The tactical tracking uses short-lived slots rather than player identities.  These features only
depend on the set of visible points in each frame: global and per-team centroids/covariances, plus
the same moments for players close to the ball.  They therefore survive substitutions, tracklet
breaks and slot permutations.
"""

import warnings

import numpy as np

from .data import AWAY, HOME

NEAR_BALL_M = 18.0


def _moments(xy, valid):
    """Centroid, diagonal variance, covariance and line-likeness for point sets per frame."""
    xy = np.asarray(xy, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    count = valid.sum(axis=1).astype(np.float32)
    denominator = np.maximum(count, 1.0)[:, None]
    safe = np.where(valid[:, :, None], xy, 0.0)
    centre = safe.sum(axis=1) / denominator
    delta = np.where(valid[:, :, None], xy - centre[:, None], 0.0)
    var = (delta * delta).sum(axis=1) / denominator
    cov = (delta[..., 0] * delta[..., 1]).sum(axis=1) / denominator[:, 0]
    trace = var.sum(axis=1)
    root = np.sqrt(np.maximum((var[:, 0] - var[:, 1]) ** 2 + 4 * cov ** 2, 0.0))
    major = (trace + root) / 2
    minor = (trace - root) / 2
    linearity = np.zeros_like(major)
    np.divide(major - minor, major, out=linearity, where=major > 1e-6)
    missing = count == 0
    centre[missing] = np.nan
    var[missing] = np.nan
    cov[missing] = np.nan
    linearity[missing] = np.nan
    return centre, var, cov, linearity.astype(np.float32), count


def _mean(values, valid):
    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(values)
    count = valid.sum(axis=1)
    total = np.where(valid, values, 0.0).sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)


def frame_features(trk, near_ball_m=NEAR_BALL_M):
    """Return spatial descriptors as arrays whose first dimension is frame.

    Variances are in square metres. ``linearity`` is zero for an isotropic cloud and approaches
    one for players arranged on a line. Local values use visible players within ``near_ball_m``.
    """
    players = np.asarray(trk["players"], dtype=np.float32)
    ball = np.asarray(trk["ball"], dtype=np.float32)
    teams = np.asarray(trk["team"])
    xy = players[..., :2]
    valid = np.isfinite(xy).all(axis=2)
    centre, var, cov, linearity, count = _moments(xy, valid)
    speed = players[..., 2] if players.shape[2] > 2 else np.full(valid.shape, np.nan)

    team_centre = np.full((len(players), 2, 2), np.nan, np.float32)
    team_var = np.full_like(team_centre, np.nan)
    team_cov = np.full((len(players), 2), np.nan, np.float32)
    team_linearity = np.full((len(players), 2), np.nan, np.float32)
    team_count = np.zeros((len(players), 2), np.float32)
    team_speed = np.full((len(players), 2), np.nan, np.float32)
    for team in (HOME, AWAY):
        slots = np.flatnonzero(teams == team)
        values = _moments(xy[:, slots], valid[:, slots])
        team_centre[:, team], team_var[:, team], team_cov[:, team], \
            team_linearity[:, team], team_count[:, team] = values
        team_speed[:, team] = _mean(speed[:, slots], valid[:, slots])

    ball_xy = ball[:, :2]
    ball_visible = np.isfinite(ball_xy).all(axis=1)
    distance = np.linalg.norm(xy - ball_xy[:, None], axis=2)
    local_valid = valid & ball_visible[:, None] & (distance <= float(near_ball_m))
    local_centre, local_var, local_cov, local_linearity, local_count = _moments(xy, local_valid)
    local_team_count = np.stack(
        [local_valid[:, teams == team].sum(axis=1) for team in (HOME, AWAY)], axis=1
    ).astype(np.float32)

    return {
        "centroid": centre, "variance": var, "covariance": cov, "linearity": linearity,
        "visible_count": count, "mean_speed": _mean(speed, valid),
        "team_centroid": team_centre, "team_variance": team_var,
        "team_covariance": team_cov, "team_linearity": team_linearity,
        "team_visible_count": team_count, "team_mean_speed": team_speed,
        "local_centroid": local_centre, "local_variance": local_var,
        "local_covariance": local_cov, "local_linearity": local_linearity,
        "local_count": local_count, "local_team_count": local_team_count,
        "ball": ball_xy.copy(), "ball_visible": ball_visible,
    }


def interval_summary(features, start, stop):
    """Robust median summary of frame features over a half-open interval."""
    n = len(features["ball"])
    start, stop = max(0, int(start)), min(n, int(stop))
    if stop <= start:
        raise ValueError("empty Smart Tagging interval")
    out = {}
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # expected for an entirely absent feature
        for name, values in features.items():
            values = np.asarray(values)
            if values.dtype == bool:
                out[name] = values[start:stop].mean(axis=0)
            else:
                out[name] = np.nanmedian(values[start:stop], axis=0)
    out["start_frame"], out["end_frame"] = start, stop
    return out
