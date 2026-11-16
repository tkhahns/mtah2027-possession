"""Permutation-invariant player-set encoder and temporal boundary refinement.

Training lives in ``src/training/possession_models/train_set_temporal.py`` and uses Torch.  This
module keeps feature construction, target construction and constrained decoding importable for
synthetic tests and external-game evaluation.
"""

import numpy as np

from . import restart_state

FPS = 25
STRIDE = 5
PLAYER_FEATURES = (
    "x", "y", "speed", "ball_dx", "ball_dy", "ball_distance", "home", "away")
BOUNDARY_TYPES = ("transfer", "stop", "restart")
LABELS = np.asarray(list("HAD"))


def _player_inputs(trk):
    players = np.asarray(trk["players"], dtype=np.float32)
    ball = np.asarray(trk["ball"], dtype=np.float32)
    team = np.asarray(trk["team"])
    length, width = map(float, trk.get("pitch", (105.0, 68.0)))
    last_ball, _, _ = restart_state._last_observed(ball)
    ball_xy = np.where(np.isfinite(last_ball), last_ball, [length / 2.0, width / 2.0])
    xy = players[..., :2]
    visible = np.isfinite(xy).all(axis=2)
    relative = xy - ball_xy[:, None, :]
    distance = np.linalg.norm(relative, axis=2)
    speed = players[..., 2] if players.shape[2] > 2 else np.full(visible.shape, np.nan)
    home = np.broadcast_to(team == 0, visible.shape)
    away = np.broadcast_to(team == 1, visible.shape)
    values = np.stack([
        xy[..., 0] / length,
        xy[..., 1] / width,
        speed / 5.0,
        relative[..., 0] / length,
        relative[..., 1] / width,
        distance / 30.0,
        home.astype(np.float32),
        away.astype(np.float32),
    ], axis=2)
    return np.nan_to_num(values, nan=0.0, posinf=4.0, neginf=-4.0), visible


def frame_inputs(trk, stride=STRIDE):
    """Return sampled player sets, masks, scalar context and original frame indices."""
    stride = int(stride)
    if stride <= 0:
        raise ValueError("stride must be positive")
    player, mask = _player_inputs(trk)
    context = restart_state.frame_features(trk)
    index = np.arange(0, len(player), stride, dtype=np.int32)
    return (player[index].astype(np.float32), mask[index],
            context[index].astype(np.float32), index)


def target_arrays(truth, index, radius=2):
    """Sample H/A/D states and soft transfer/stop/restart boundary targets."""
    truth = np.asarray(truth)
    index = np.asarray(index, dtype=np.int64)
    sampled = truth[index]
    state = np.full(len(index), -1, dtype=np.int64)
    for code, label in enumerate(LABELS):
        state[sampled == label] = code
    boundary = np.zeros((len(index), len(BOUNDARY_TYPES)), dtype=np.float32)
    changes = np.flatnonzero(sampled[1:] != sampled[:-1]) + 1
    radius = int(radius)
    for at in changes:
        old, new = sampled[at - 1], sampled[at]
        if old in "HA" and new in "HA" and old != new:
            kind = 0
        elif old in "HA" and new == "D":
            kind = 1
        elif old == "D" and new in "HA":
            kind = 2
        else:
            continue
        lo, hi = max(0, at - radius), min(len(index), at + radius + 1)
        distance = np.abs(np.arange(lo, hi) - at)
        boundary[lo:hi, kind] = np.maximum(
            boundary[lo:hi, kind], 1.0 - distance / float(radius + 1))
    return state, boundary


def build_model(player_features=len(PLAYER_FEATURES), context_features=len(restart_state.FEATURE_NAMES),
                hidden=48, dropout=0.10):
    """Construct the Torch set encoder + dilated temporal convolution model."""
    import torch
    from torch import nn

    class ResidualTemporal(nn.Module):
        def __init__(self, channels, dilation):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
                nn.GroupNorm(4, channels), nn.ReLU(), nn.Dropout(dropout),
                nn.Conv1d(channels, channels, 1), nn.ReLU())

        def forward(self, values):
            return values + self.net(values)

    class SetTemporalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.player = nn.Sequential(
                nn.Linear(player_features, 24), nn.ReLU(), nn.Linear(24, 24), nn.ReLU())
            self.frame = nn.Sequential(
                nn.Linear(24 * 4 + context_features, hidden), nn.ReLU(), nn.Dropout(dropout))
            self.temporal = nn.ModuleList(
                [ResidualTemporal(hidden, dilation) for dilation in (1, 2, 4, 8)])
            self.state = nn.Conv1d(hidden, 3, 1)
            self.boundary = nn.Conv1d(hidden, len(BOUNDARY_TYPES), 1)

        @staticmethod
        def _pool(encoded, select):
            select = select.unsqueeze(-1)
            raw_count = select.sum(dim=2)
            count = raw_count.clamp_min(1)
            mean = (encoded * select).sum(dim=2) / count
            maximum = encoded.masked_fill(~select, -1e4).max(dim=2).values
            maximum = torch.where(raw_count > 0, maximum, torch.zeros_like(maximum))
            return mean, maximum

        def forward(self, player, mask, context):
            encoded = self.player(player)
            home = mask & (player[..., 6] > 0.5)
            away = mask & (player[..., 7] > 0.5)
            home_mean, home_max = self._pool(encoded, home)
            away_mean, away_max = self._pool(encoded, away)
            frame = self.frame(torch.cat(
                [home_mean, home_max, away_mean, away_max, context], dim=-1))
            temporal = frame.transpose(1, 2)
            for block in self.temporal:
                temporal = block(temporal)
            return self.state(temporal).transpose(1, 2), self.boundary(temporal).transpose(1, 2)

    return SetTemporalModel()


def predict_probabilities(model, trk, device="cpu", stride=STRIDE):
    """Run a Torch checkpoint over one complete match and return sampled probabilities."""
    import torch

    player, mask, context, index = frame_inputs(trk, stride)
    model = model.to(device).eval()
    with torch.no_grad():
        state, boundary = model(
            torch.from_numpy(player[None]).to(device),
            torch.from_numpy(mask[None]).to(device),
            torch.from_numpy(context[None]).to(device))
        state = torch.softmax(state[0], dim=1).cpu().numpy()
        boundary = torch.sigmoid(boundary[0]).cpu().numpy()
    return state.astype(np.float32), boundary.astype(np.float32), index


def upsample(values, index, length):
    """Linearly map sampled probabilities back to the 25 Hz tracking timeline."""
    values = np.asarray(values)
    index = np.asarray(index)
    target = np.arange(int(length))
    return np.stack([
        np.interp(target, index, values[:, column], left=values[0, column],
                  right=values[-1, column])
        for column in range(values.shape[1])
    ], axis=1).astype(np.float32)


def boundary_type(old, new):
    if old in "HA" and new in "HA" and old != new:
        return 0
    if old in "HA" and new == "D":
        return 1
    if old == "D" and new in "HA":
        return 2
    return None


def refine(labels, state_probability, boundary_probability, radius=50,
           boundary_weight=4.0, min_gain=1.0):
    """Move existing boundaries using neural state agreement without changing spell order."""
    labels = np.asarray(labels)
    output = labels.copy()
    state_probability = np.asarray(state_probability, dtype=np.float64)
    boundary_probability = np.asarray(boundary_probability, dtype=np.float64)
    if state_probability.shape != (len(labels), 3):
        raise ValueError("state probabilities must have shape (frames, 3)")
    if boundary_probability.shape != (len(labels), len(BOUNDARY_TYPES)):
        raise ValueError("boundary probabilities must have shape (frames, 3)")
    log_state = np.log(np.clip(state_probability, 1e-6, 1.0))
    cumulative = np.vstack([np.zeros(3), np.cumsum(log_state, axis=0)])
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    boundaries = np.r_[0, changes, len(labels)]
    label_index = {label: i for i, label in enumerate(LABELS)}
    for run in range(1, len(boundaries) - 1):
        at = int(boundaries[run])
        old, new = labels[at - 1], labels[at]
        kind = boundary_type(old, new)
        if kind is None:
            continue
        previous, following = int(boundaries[run - 1]), int(boundaries[run + 1])
        lo = max(previous + 1, at - int(radius))
        hi = min(following - 1, at + int(radius))
        candidates = np.arange(lo, hi + 1)
        if not len(candidates):
            continue
        old_state, new_state = label_index[old], label_index[new]
        score = ((cumulative[candidates, old_state] - cumulative[lo, old_state])
                 + (cumulative[hi + 1, new_state] - cumulative[candidates, new_state])
                 + float(boundary_weight) * boundary_probability[candidates, kind])
        original = int(np.flatnonzero(candidates == at)[0])
        best = int(np.argmax(score))
        if score[best] <= score[original] + float(min_gain):
            continue
        boundary = int(candidates[best])
        if boundary < at:
            output[boundary:at] = new
        elif boundary > at:
            output[at:boundary] = old
    return output
