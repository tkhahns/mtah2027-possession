"""Small frame-indexed time-series helpers shared by signal-processing modules."""

import numpy as np


def runs(mask):
    """Return half-open ``(start, stop)`` runs where a boolean mask is true."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _window_sum(values, before, after):
    total = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    index = np.arange(len(values))
    lo = np.clip(index - before, 0, len(values))
    hi = np.clip(index + after + 1, 0, len(values))
    return total[hi] - total[lo]


def rolling_count(mask, n):
    """Number of true values in the last ``n`` frames, including the current frame."""
    return np.rint(_window_sum(np.asarray(mask, dtype=np.float64), n - 1, 0)).astype(np.int64)


def _nan_window_mean(values, before, after):
    values = np.asarray(values, dtype=np.float64)
    valid = ~np.isnan(values)
    total = _window_sum(np.where(valid, values, 0.0), before, after)
    count = _window_sum(valid.astype(np.float64), before, after)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def past_mean(values, n):
    """Mean of the last ``n`` frames, ignoring missing values."""
    return _nan_window_mean(values, n - 1, 0)


def centred_mean(values, half_width):
    """Mean in a centred window, ignoring missing values."""
    return _nan_window_mean(values, half_width, half_width)


def last_index(valid):
    """Most recent valid frame at or before each frame, or -1."""
    valid = np.asarray(valid, dtype=bool)
    return np.maximum.accumulate(np.where(valid, np.arange(len(valid)), -1))


def next_index(valid):
    """Next valid frame at or after each frame, or ``len(valid)``."""
    valid = np.asarray(valid, dtype=bool)
    n = len(valid)
    flipped = np.where(valid[::-1], np.arange(n), -1)
    return (n - 1 - np.maximum.accumulate(flipped))[::-1]
