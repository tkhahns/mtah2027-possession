"""Refine existing dead-ball interval edges without changing interval identity."""

import numpy as np

from . import constrained_restart, restart_state


RADIUS = 50
DEAD_CENTER = 0.50
MIN_GAIN = 2.0
TRANSITION_WEIGHT = 0.0


def stop_evidence(features):
    """0..1 moving-to-stopped evidence derived from contextual feature transitions."""
    features = np.asarray(features)
    column = {name: features[:, i] for i, name in enumerate(restart_state.FEATURE_NAMES)}
    ball_before = column["ball_speed_before"] * 15.0
    ball_change = column["ball_speed_change"] * 15.0
    player_before = column["player_speed_before"] * 5.0
    player_change = column["player_speed_change"] * 5.0
    ball = np.clip(-ball_change / 3.0, 0.0, 1.0) * np.clip(ball_before / 3.0, 0.0, 1.0)
    player = (np.clip(-player_change / 1.0, 0.0, 1.0) *
              np.clip(player_before / 1.5, 0.0, 1.0))
    return np.maximum(ball, player).astype(np.float32)


def dead_evidence(probability, center=DEAD_CENTER, smooth=restart_state.WINDOW):
    """Signed state evidence: positive values support D and negative values support live play."""
    probability = np.asarray(probability, dtype=np.float32)
    if smooth:
        probability = restart_state._window_mean(probability, smooth, smooth)
    return np.asarray(probability - center, dtype=np.float32)


def _runs(mask):
    change = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1))


def refine(labels, evidence, hard=None, stop=None, restart=None, radius=RADIUS,
           min_gain=MIN_GAIN, transition_weight=TRANSITION_WEIGHT):
    """Jointly move both edges of each D run while preserving every hard-dead frame.

    Only dead intervals with live H/A labels on both sides are eligible. Each interval remains
    non-empty and retains all hard evidence that it contained before refinement. The number and
    order of D, H and A runs therefore cannot change.
    """
    labels = np.asarray(labels)
    out = labels.copy()
    evidence = np.asarray(evidence, dtype=np.float32)
    if evidence.shape != labels.shape:
        raise ValueError("dead evidence must have one value per label")
    hard = np.zeros(len(labels), dtype=bool) if hard is None else np.asarray(hard, dtype=bool)
    stop = np.zeros(len(labels), dtype=np.float32) if stop is None else np.asarray(stop, np.float32)
    restart = (np.zeros(len(labels), dtype=np.float32) if restart is None
               else np.asarray(restart, np.float32))
    if hard.shape != labels.shape or stop.shape != labels.shape or restart.shape != labels.shape:
        raise ValueError("hard and transition evidence must have one value per label")

    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    boundaries = np.r_[0, changes, len(labels)]
    cumulative = np.r_[0.0, np.cumsum(evidence, dtype=np.float64)]
    for run_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        if labels[start] != "D" or run_index == 0 or run_index + 1 >= len(boundaries) - 1:
            continue
        previous = boundaries[run_index - 1]
        following = boundaries[run_index + 2]
        if labels[start - 1] not in "HA" or labels[end] not in "HA":
            continue

        # Reserve the midpoint of each neighboring live run. The same anchor is shared when a
        # short live run sits between two D intervals, so independently refined intervals cannot
        # expand over one another and erase that possession.
        left_anchor = (previous + start - 1) // 2
        right_anchor = (end + following - 1) // 2
        start_lo = max(left_anchor + 1, start - radius)
        start_hi = min(end - 1, start + radius)
        end_lo = max(start + 1, end - radius)
        end_hi = min(right_anchor, end + radius)
        contained_hard = np.flatnonzero(hard[start:end]) + start
        if len(contained_hard):
            start_hi = min(start_hi, int(contained_hard[0]))
            end_lo = max(end_lo, int(contained_hard[-1]) + 1)
        starts = np.arange(start_lo, start_hi + 1)
        ends = np.arange(end_lo, end_hi + 1)
        if not len(starts) or not len(ends):
            continue

        # Across a fixed local window, maximizing signed live/dead/live agreement is equivalent
        # to twice the evidence inside the candidate D interval. Transition terms break ties and
        # favor an actual stop at entry and stopped-to-moving evidence at the restart.
        score = 2.0 * (cumulative[ends][None, :] - cumulative[starts][:, None])
        score += transition_weight * (stop[starts][:, None] + restart[ends][None, :])
        score[starts[:, None] >= ends[None, :]] = -np.inf
        original_score = (2.0 * (cumulative[end] - cumulative[start]) +
                          transition_weight * (stop[start] + restart[end]))
        best_flat = int(np.argmax(score))
        best_start, best_end = np.unravel_index(best_flat, score.shape)
        if score[best_start, best_end] <= original_score + min_gain:
            continue
        new_start, new_end = int(starts[best_start]), int(ends[best_end])
        out[start:end] = labels[start - 1]
        out[new_start:new_end] = "D"
        if new_end < end:
            out[new_end:end] = labels[end]
        elif new_end > end:
            out[end:new_end] = "D"
    return out


def evidence_from_model(probability, features, center=DEAD_CENTER):
    """Return state, stop and restart evidence used by the frozen refiner."""
    return (dead_evidence(probability, center), stop_evidence(features),
            constrained_restart.motion_evidence(features))
