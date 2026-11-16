"""Neural semi-Markov decoding for complete H/A/D segments.

The decoder combines set-temporal state emissions, transition-specific boundary probabilities,
duration priors learned from external matches, the current rule-based sequence as a soft prior,
and hard out-of-pitch anchors.  Dynamic programming operates on a sparse boundary lattice so a
complete match remains practical on CPU.
"""

import numpy as np

from . import hybrid_pipeline, set_temporal

LABELS = np.asarray(list("HAD"))


def _runs(mask):
    change = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(change == 1), np.flatnonzero(change == -1)))


def duration_statistics(truths, stride=set_temporal.STRIDE):
    """Log-duration mean/std in sampled frames for H, A and D segments."""
    values = {label: [] for label in LABELS}
    for truth in truths:
        sampled = np.asarray(truth)[::int(stride)]
        starts = np.flatnonzero(np.r_[True, sampled[1:] != sampled[:-1]])
        stops = np.r_[starts[1:], len(sampled)]
        for start, stop in zip(starts, stops):
            label = sampled[start]
            if label in values:
                values[label].append(max(1, int(stop - start)))
    result = []
    for label in LABELS:
        logged = np.log(np.asarray(values[label] or [1], dtype=np.float64))
        result.append((float(logged.mean()), float(max(logged.std(), 0.50))))
    return np.asarray(result, dtype=np.float32)


def _transition_scores(boundary, at, weight, penalty):
    probability = np.clip(boundary[min(at, len(boundary) - 1)], 1e-5, 1.0)
    score = np.full((3, 3), -np.inf, dtype=np.float64)
    for previous, old in enumerate(LABELS):
        for state, new in enumerate(LABELS):
            kind = set_temporal.boundary_type(old, new)
            if kind is not None:
                score[previous, state] = float(weight) * np.log(probability[kind]) - float(penalty)
    return score


def _candidate_positions(base, boundary, grid, threshold):
    n = len(base)
    positions = {0, n}
    positions.update(range(int(grid), n, int(grid)))
    positions.update((np.flatnonzero(base[1:] != base[:-1]) + 1).tolist())
    strength = np.max(boundary, axis=1)
    peaks = np.flatnonzero(
        (strength >= float(threshold))
        & (strength >= np.r_[-np.inf, strength[:-1]])
        & (strength >= np.r_[strength[1:], -np.inf]))
    positions.update(int(value) for value in peaks if 0 < value < n)
    return np.asarray(sorted(positions), dtype=np.int32)


def _decode_block(base, state_probability, boundary_probability, hard, duration,
                  base_weight, boundary_weight, duration_weight, transition_penalty,
                  candidate_grid, candidate_threshold, max_duration):
    n = len(base)
    positions = _candidate_positions(
        base, boundary_probability, candidate_grid, candidate_threshold)
    emission = np.log(np.clip(state_probability, 1e-6, 1.0)).astype(np.float64)
    for state, label in enumerate(LABELS):
        emission[:, state] += float(base_weight) * (base == label)
    emission[hard, :2] -= 1e4
    cumulative = np.vstack([np.zeros(3), np.cumsum(emission, axis=0)])

    count = len(positions)
    score = np.full((count, 3), -np.inf, dtype=np.float64)
    back_position = np.full((count, 3), -1, dtype=np.int32)
    back_state = np.full((count, 3), -1, dtype=np.int8)
    for end_i in range(1, count):
        end = int(positions[end_i])
        first = int(np.searchsorted(positions, end - int(max_duration), side="left"))
        start_i = np.arange(first, end_i, dtype=np.int32)
        starts = positions[start_i]
        lengths = end - starts
        segment = cumulative[end] - cumulative[starts]
        logged = np.log(lengths)[:, None]
        segment += -float(duration_weight) * 0.5 * (
            (logged - duration[None, :, 0]) / duration[None, :, 1]) ** 2

        # A segment beginning at zero does not pay a transition cost.
        if first == 0:
            score[end_i] = segment[0]
            back_position[end_i] = 0
            start_i = start_i[1:]
            starts = starts[1:]
            segment = segment[1:]
        if not len(start_i):
            continue

        probability = np.clip(boundary_probability[starts], 1e-5, 1.0)
        transition = np.full((len(starts), 3, 3), -np.inf, dtype=np.float64)
        value = -float(transition_penalty)
        transfer = float(boundary_weight) * np.log(probability[:, 0]) + value
        stop = float(boundary_weight) * np.log(probability[:, 1]) + value
        restart = float(boundary_weight) * np.log(probability[:, 2]) + value
        transition[:, 0, 1] = transfer
        transition[:, 1, 0] = transfer
        transition[:, 0, 2] = stop
        transition[:, 1, 2] = stop
        transition[:, 2, 0] = restart
        transition[:, 2, 1] = restart
        candidate = score[start_i, :, None] + transition + segment[:, None, :]
        flat = candidate.reshape(-1, 3)
        best_flat = np.argmax(flat, axis=0)
        best = flat[best_flat, np.arange(3)]
        better = best > score[end_i]
        score[end_i, better] = best[better]
        back_position[end_i, better] = start_i[best_flat[better] // 3]
        back_state[end_i, better] = (best_flat[better] % 3).astype(np.int8)

    output = np.empty(n, dtype="<U1")
    end_i = count - 1
    state = int(np.argmax(score[end_i]))
    if not np.isfinite(score[end_i, state]):
        return np.asarray(base).copy()
    while end_i > 0:
        start_i = int(back_position[end_i, state])
        if start_i < 0:
            return np.asarray(base).copy()
        output[int(positions[start_i]):int(positions[end_i])] = LABELS[state]
        previous = int(back_state[end_i, state])
        end_i = start_i
        if end_i:
            state = previous
    return output


def decode(base_labels, trk, state_probability, boundary_probability, duration,
           stride=set_temporal.STRIDE, base_weight=1.0, boundary_weight=2.0,
           duration_weight=0.5, transition_penalty=4.0, candidate_grid=5,
           candidate_threshold=0.35, max_duration=900):
    """Decode complete segments while preserving the base pipeline's X regions."""
    base_labels = np.asarray(base_labels)
    state_probability = np.asarray(state_probability)
    boundary_probability = np.asarray(boundary_probability)
    if state_probability.shape != (len(base_labels), 3):
        raise ValueError("state probabilities must have shape (frames, 3)")
    if boundary_probability.shape != (len(base_labels), 3):
        raise ValueError("boundary probabilities must have shape (frames, 3)")
    stride = int(stride)
    sample = np.arange(0, len(base_labels), stride)
    sampled_base = base_labels[sample]
    sampled_state = state_probability[sample]
    sampled_boundary = boundary_probability[sample]
    sampled_hard = hybrid_pipeline.dead_ball(trk)[sample]
    sampled_output = sampled_base.copy()
    for start, stop in _runs(np.isin(sampled_base, LABELS)):
        sampled_output[start:stop] = _decode_block(
            sampled_base[start:stop], sampled_state[start:stop],
            sampled_boundary[start:stop], sampled_hard[start:stop], np.asarray(duration),
            base_weight, boundary_weight, duration_weight, transition_penalty,
            candidate_grid, candidate_threshold, max_duration)

    # A sampled state owns frames until the following sample. X is always inherited from base.
    full = sampled_output[np.minimum(np.arange(len(base_labels)) // stride,
                                     len(sampled_output) - 1)]
    full[base_labels == "X"] = "X"
    return full
