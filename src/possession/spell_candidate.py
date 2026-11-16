"""Learned keep-versus-merge decisions for short sandwiched possession spells."""

import json
import os

import numpy as np

from . import boundary_timing, restart_state, spell_filter
from .data import AWAY, HOME

FPS = 25
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_MODEL = os.path.join(ROOT, "src", "models", "spell_candidate.json")
MAX_FRAMES = 150
CONTEXT_FRAMES = 25

FEATURE_NAMES = (
    "duration_s", "previous_duration_s", "following_duration_s",
    "control_count", "control_share", "control_longest_s", "opponent_control_share",
    "ownership_mean", "ownership_positive_share", "ownership_max", "ownership_min",
    "ownership_first_half", "ownership_second_half", "ownership_before", "ownership_after",
    "ball_visible_share", "ball_speed_mean", "ball_speed_p90",
    "nearest_distance_mean", "nearest_distance_min", "distance_margin_mean",
    "distance_margin_positive_share", "dead_probability_mean", "dead_probability_max",
    "player_speed_mean", "global_spread_mean", "local_count_mean",
    "team_separation_mean", "boundary_distance_mean",
)


def candidate_runs(labels, max_frames=MAX_FRAMES):
    """Return short H/A runs sandwiched by the same opposing live label."""
    labels = np.asarray(labels)
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    stops = np.r_[starts[1:], len(labels)]
    candidates = []
    for i, (start, stop) in enumerate(zip(starts, stops)):
        if i == 0 or i + 1 == len(starts) or labels[start] not in "HA":
            continue
        if stop - start > max_frames:
            continue
        previous, following = labels[starts[i - 1]], labels[starts[i + 1]]
        if previous == following and previous in "HA":
            team = HOME if labels[start] == "H" else AWAY
            candidates.append((start, stop, team, start - starts[i - 1],
                               stops[i + 1] - starts[i + 1]))
    if not candidates:
        empty = np.empty(0, np.int32)
        return empty, empty.copy(), np.empty(0, np.int8), empty.copy(), empty.copy()
    values = np.asarray(candidates, dtype=np.int32)
    return (values[:, 0], values[:, 1], values[:, 2].astype(np.int8),
            values[:, 3], values[:, 4])


def _longest_run(values):
    values = np.asarray(values, dtype=bool)
    if not values.any():
        return 0
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return int(np.max(stops - starts))


def _summary(values, function=np.nanmean, default=0.0):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return float(function(finite)) if len(finite) else float(default)


def candidate_features(labels, trk, probability, restart_features=None, control=None,
                       ownership=None, max_frames=MAX_FRAMES, context=CONTEXT_FRAMES):
    """Return candidate intervals, teams and finite spell-level feature rows."""
    starts, stops, teams, previous_length, following_length = candidate_runs(labels, max_frames)
    if not len(starts):
        return starts, stops, teams, np.empty((0, len(FEATURE_NAMES)), np.float32)
    restart_features = (restart_state.frame_features(trk) if restart_features is None
                        else np.asarray(restart_features))
    control = spell_filter.control_team(trk) if control is None else np.asarray(control)
    ownership = (boundary_timing.ownership_evidence(trk, control=control) if ownership is None
                 else np.asarray(ownership))
    probability = restart_state._window_mean(
        np.asarray(probability, np.float32), restart_state.WINDOW, restart_state.WINDOW)
    column = {name: restart_features[:, i]
              for i, name in enumerate(restart_state.FEATURE_NAMES)}

    players = np.asarray(trk["players"])
    ball = np.asarray(trk["ball"])
    slots = np.asarray(trk["team"])
    distance = np.linalg.norm(players[..., :2] - ball[:, None, :2], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    home = distance[:, slots == HOME].min(axis=1)
    away = distance[:, slots == AWAY].min(axis=1)
    rows = []
    for start, stop, team, previous, following in zip(
            starts, stops, teams, previous_length, following_length):
        sign = 1.0 if team == HOME else -1.0
        owner = ownership[start:stop] * sign
        team_control = control[start:stop] == team
        opponent_control = control[start:stop] == 1 - team
        candidate_distance = (home if team == HOME else away)[start:stop]
        opponent_distance = (away if team == HOME else home)[start:stop]
        length = stop - start
        valid_margin = np.isfinite(opponent_distance) & np.isfinite(candidate_distance)
        margin = np.full(length, np.nan, np.float32)
        np.subtract(opponent_distance, candidate_distance, out=margin, where=valid_margin)
        before = ownership[max(0, start - context):start] * sign
        after = ownership[stop:min(len(labels), stop + context)] * sign
        middle = start + (stop - start) // 2
        first, second = owner[:middle - start], owner[middle - start:]
        interval = slice(start, stop)
        global_spread = np.sqrt(
            (column["spread_x"][interval] * float(trk.get("pitch", (105.0, 68.0))[0])) ** 2 +
            (column["spread_y"][interval] * float(trk.get("pitch", (105.0, 68.0))[1])) ** 2)
        rows.append([
            length / FPS, previous / FPS, following / FPS,
            float(team_control.sum()), float(team_control.mean()),
            _longest_run(team_control) / FPS, float(opponent_control.mean()),
            _summary(owner), float(np.mean(owner > 0)), _summary(owner, np.max),
            _summary(owner, np.min), _summary(first), _summary(second),
            _summary(before), _summary(after),
            _summary(column["ball_visible"][interval]),
            _summary(column["ball_speed"][interval]) * 15.0,
            _summary(column["ball_speed"][interval], lambda x: np.percentile(x, 90)) * 15.0,
            _summary(candidate_distance), _summary(candidate_distance, np.min),
            _summary(margin[valid_margin]),
            float(np.mean(margin[valid_margin] > 0)) if valid_margin.any() else 0.0,
            _summary(probability[interval]), _summary(probability[interval], np.max),
            _summary(column["player_speed"][interval]) * 5.0,
            _summary(global_spread), _summary(column["local_count_share"][interval]) * 10.0,
            _summary(column["team_separation"][interval]) * 30.0,
            _summary(column["ball_boundary_distance"][interval]) * 15.0,
        ])
    return starts, stops, teams, np.nan_to_num(
        np.asarray(rows, np.float32), nan=0.0, posinf=100.0, neginf=-100.0)


def keep_targets(starts, stops, teams, truth, minimum_iou=0.30):
    """Target one when a candidate can match a same-team truth spell at the lowest official IoU."""
    truth = np.asarray(truth)
    truth_starts = np.flatnonzero(np.r_[True, truth[1:] != truth[:-1]])
    truth_stops = np.r_[truth_starts[1:], len(truth)]
    by_team = {
        team: [(int(start), int(stop)) for start, stop in zip(truth_starts, truth_stops)
               if truth[start] == ("H" if team == HOME else "A")]
        for team in (HOME, AWAY)
    }
    target = np.zeros(len(starts), np.float32)
    for i, (start, stop, team) in enumerate(zip(starts, stops, teams)):
        best = 0.0
        for true_start, true_stop in by_team[int(team)]:
            overlap = max(0, min(int(stop), true_stop) - max(int(start), true_start))
            if not overlap:
                continue
            union = max(int(stop), true_stop) - min(int(start), true_start)
            best = max(best, overlap / union)
        target[i] = best >= minimum_iou
    return target


def load_model(path=DEFAULT_MODEL):
    with open(path, encoding="utf-8") as handle:
        model = json.load(handle)
    if model.get("feature_names") != list(FEATURE_NAMES):
        raise ValueError("spell-candidate checkpoint feature layout does not match the code")
    return model


def probabilities(features, model):
    """NumPy inference for the Torch-trained two-layer candidate classifier."""
    x = ((np.asarray(features, np.float32) - np.asarray(model["mean"], np.float32)) /
         np.asarray(model["std"], np.float32))
    hidden = np.maximum(0.0, x @ np.asarray(model["weight1"], np.float32).T +
                        np.asarray(model["bias1"], np.float32))
    logits = hidden @ np.asarray(model["weight2"], np.float32) + float(model["bias2"])
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def merge_from_probabilities(labels, starts, stops, probability, threshold):
    """Merge low-probability candidates, skipping intervals invalidated by an earlier merge."""
    labels = np.asarray(labels)
    out = labels.copy()
    for start, stop, keep in zip(starts, stops, probability):
        if keep >= threshold or start == 0 or stop >= len(out):
            continue
        if not np.all(out[start:stop] == labels[start]):
            continue
        if out[start - 1] == out[stop] and out[start - 1] in "HA":
            out[start:stop] = out[start - 1]
    return out


def mode_probabilities(features, probability, mode):
    """Combine learned decisions with the frozen rule for guarded/extended variants."""
    features = np.asarray(features)
    probability = np.asarray(probability).copy()
    duration = features[:, FEATURE_NAMES.index("duration_s")]
    control_count = features[:, FEATURE_NAMES.index("control_count")]
    hard_merge = ((duration * FPS <= spell_filter.MAX_SHORT_FRAMES + 1e-4) &
                  (control_count < spell_filter.MIN_CONTROL_FRAMES))
    if mode == "replace":
        return probability
    if mode == "guard":
        probability[~hard_merge] = 1.0
        return probability
    if mode == "extend":
        probability[hard_merge] = 0.0
        return probability
    raise ValueError("unknown spell-candidate mode: {}".format(mode))


def merge(labels, trk, probability, model=DEFAULT_MODEL, threshold=None, **features):
    model = load_model(model) if isinstance(model, (str, os.PathLike)) else model
    starts, stops, _, values = candidate_features(labels, trk, probability, **features)
    threshold = float(model["threshold"] if threshold is None else threshold)
    keep_probability = mode_probabilities(
        values, probabilities(values, model), model.get("mode", "replace"))
    return merge_from_probabilities(
        labels, starts, stops, keep_probability, threshold)
