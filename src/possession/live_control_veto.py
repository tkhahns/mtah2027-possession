"""Conservative field-wide veto for false dead intervals during controlled live play.

This is deliberately a postprocessor rather than another dead/live classifier.  It may only
remove a short D island already surrounded by the same team, and only when tactical tracking
shows sustained, moving control by that team.  Hard out-of-pitch evidence is never overridden.
"""

import numpy as np

from . import hybrid_pipeline, spell_filter
from .data import AWAY, HOME

FPS = 25
MAX_DEAD_FRAMES = 200
HARD_LOOKBACK_FRAMES = 250
MIN_VISIBLE_SHARE = 0.80
MIN_CONTROL_SHARE = 0.50
MAX_OPPONENT_CONTROL_SHARE = 0.05
MIN_ACTIVE_CONTROL_SHARE = 0.15
MIN_ACTIVE_SPEED = 1.0
MIN_NET_DISPLACEMENT = 1.5
MIN_PATH_LENGTH = 3.0


def runs(mask):
    """Return half-open true runs from a one-dimensional boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    change = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def ball_speed(ball, fps=FPS):
    """Use supplied ball speed where finite, with position differences as a fallback."""
    ball = np.asarray(ball)
    speed = (ball[:, 2].astype(np.float32, copy=True) if ball.shape[1] > 2
             else np.full(len(ball), np.nan, np.float32))
    xy = ball[:, :2]
    valid = np.isfinite(xy).all(axis=1)
    derived = np.full(len(ball), np.nan, np.float32)
    consecutive = valid[1:] & valid[:-1]
    derived[1:][consecutive] = (
        np.linalg.norm(xy[1:][consecutive] - xy[:-1][consecutive], axis=1) * float(fps))
    missing = ~np.isfinite(speed)
    speed[missing] = derived[missing]
    return speed


def _recent_hard(hard, frames):
    cumulative = np.r_[0, np.cumsum(np.asarray(hard, dtype=np.int64))]
    index = np.arange(len(hard))
    first = np.maximum(0, index - int(frames))
    return (cumulative[index + 1] - cumulative[first]) > 0


def restore_controlled_live(labels, trk, max_dead_frames=MAX_DEAD_FRAMES,
                            hard_lookback_frames=HARD_LOOKBACK_FRAMES,
                            min_visible_share=MIN_VISIBLE_SHARE,
                            min_control_share=MIN_CONTROL_SHARE,
                            max_opponent_control_share=MAX_OPPONENT_CONTROL_SHARE,
                            min_active_control_share=MIN_ACTIVE_CONTROL_SHARE,
                            min_active_speed=MIN_ACTIVE_SPEED,
                            min_net_displacement=MIN_NET_DISPLACEMENT,
                            min_path_length=MIN_PATH_LENGTH,
                            hard_dead=None, control=None):
    """Restore strongly supported live-control islands without inventing a new owner.

    The target owner must be identical on both sides of the D run.  A candidate is rejected when
    the ball was recently outside the pitch, is mostly invisible, is stationary, or has meaningful
    opposing-team control.  ``hard_dead`` and ``control`` are injectable for deterministic audits.
    """
    output = np.asarray(labels).copy()
    ball = np.asarray(trk["ball"])
    if output.shape != (len(ball),):
        raise ValueError("labels and tactical tracking must have the same length")
    hard = (hybrid_pipeline.dead_ball(trk) if hard_dead is None
            else np.asarray(hard_dead, dtype=bool))
    owner = (spell_filter.control_team(trk) if control is None
             else np.asarray(control, dtype=np.int8))
    if hard.shape != output.shape or owner.shape != output.shape:
        raise ValueError("hard-dead and control evidence must have one value per label")

    recent_hard = _recent_hard(hard, hard_lookback_frames)
    speed = ball_speed(ball)
    visible = np.isfinite(ball[:, :2]).all(axis=1)

    for start, stop in runs(output == "D"):
        length = stop - start
        if start == 0 or stop == len(output) or length > int(max_dead_frames):
            continue
        label = output[start - 1]
        if label not in "HA" or output[stop] != label:
            continue
        interval = slice(start, stop)
        if np.any(recent_hard[interval]):
            continue
        if np.mean(visible[interval]) < float(min_visible_share):
            continue

        target = HOME if label == "H" else AWAY
        opposing = AWAY if target == HOME else HOME
        target_control = owner[interval] == target
        if np.mean(target_control) < float(min_control_share):
            continue
        if np.mean(owner[interval] == opposing) > float(max_opponent_control_share):
            continue
        active_control = target_control & np.isfinite(speed[interval]) & (
            speed[interval] >= float(min_active_speed))
        if np.mean(active_control) < float(min_active_control_share):
            continue

        xy = ball[start:stop, :2]
        valid_xy = xy[np.isfinite(xy).all(axis=1)]
        if len(valid_xy) < 2:
            continue
        net = float(np.linalg.norm(valid_xy[-1] - valid_xy[0]))
        path = float(np.linalg.norm(np.diff(valid_xy, axis=0), axis=1).sum())
        if net < float(min_net_displacement) and path < float(min_path_length):
            continue
        output[start:stop] = label
    return output
