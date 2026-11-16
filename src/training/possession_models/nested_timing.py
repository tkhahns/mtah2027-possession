"""Strictly match-nested restart and keep-versus-merge base predictions.

The temporal model must not be evaluated on base labels produced by checkpoints that saw the
held-out match.  This module fits every upstream checkpoint on the outer training matches only.
Candidate training rows use restart predictions from an additional leave-one-match-out layer.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]

from training.possession_models import train_restart_state as restart_trainer  # noqa: E402
from training.possession_models import train_spell_candidate as candidate_trainer  # noqa: E402
from possession import hybrid_pipeline, spell_candidate  # noqa: E402

MIN_POSSESSION_RATIO = 0.90


def add_restart_fields(games):
    """Add the arrays needed by the restart/candidate trainers to prepared temporal rows."""
    for views in games.values():
        for row in views.values():
            if "features" in row:
                continue
            features = restart_trainer.restart_state.frame_features(row["trk"])
            target = (row["truth"] == "D").astype(np.float32)
            graded = np.isin(row["truth"], ["H", "A", "D"])
            hard = hybrid_pipeline.dead_ball(row["trk"])
            row.update({
                "features": features, "target": target, "graded": graded, "hard": hard,
                "train_mask": graded & ~hard,
                "weight": restart_trainer._boundary_weight(target),
            })
    return games


def _candidate_configuration(rows):
    baseline = candidate_trainer.score_rows(rows)
    best = None
    for mode in candidate_trainer.MODES:
        for threshold in candidate_trainer.THRESHOLDS:
            result = candidate_trainer.score_rows(rows, mode, float(threshold))
            non_regression = (result["hidden"] >= baseline["hidden"] - 0.002 and
                              result["messy"] >= baseline["messy"] - 0.002 and
                              result["macro"] >= baseline["macro"] - 0.002)
            count_ok = (result["hidden_ratio"] >= MIN_POSSESSION_RATIO and
                        result["messy_ratio"] >= MIN_POSSESSION_RATIO)
            key = (count_ok and non_regression, count_ok, non_regression,
                   result["choose"], result["macro"],
                   -abs(result["ratio"] - 1.0))
            if best is None or key > best[0]:
                best = (key, mode, float(threshold), result)
    return best[1], best[2], best[3]


def _selection_rows(rows):
    """Use control-semantic matches for merge/count calibration.

    IDSSE contributes frame-exact dead/live supervision and classifier examples, but its provider
    possession changes at more contested touches than the challenge definition.  Metrica's event
    truth explicitly removes touches without control, so it is the appropriate threshold domain.
    """
    control = [row for row in rows if not str(row.get("game", "")).startswith("idsse:")]
    return control or rows


def fit(games, train_ids, restart_epochs, candidate_epochs, device, seed, inner_folds=3):
    """Fit outer-training-only restart and candidate models plus candidate configuration."""
    train_ids = tuple(sorted(train_ids, key=str))
    if len(train_ids) < 2:
        raise ValueError("nested timing requires at least two outer-training matches")
    add_restart_fields(games)
    outer_restart = restart_trainer.train(
        games, list(train_ids), restart_epochs, device,
        restart_trainer.stable_seed(seed, "outer:" + ",".join(map(str, train_ids))))
    # Retain the public checkpoint contract for standalone restart_pipeline consumers.  The
    # production constrained decoder uses its own hard-anchor thresholds, but this metadata is
    # still part of restart_state.json's documented interface.
    outer_restart.update({"threshold": 0.80, "min_dead_frames": 75})
    candidate_rows = []
    # Match-grouped inner folds keep every candidate row out of its restart checkpoint while
    # avoiding O(n^2) restart fits once the seven IDSSE matches are included.
    folds = np.array_split(np.asarray(train_ids, dtype=object), min(int(inner_folds), len(train_ids)))
    for validation_fold in folds:
        validation_ids = tuple(validation_fold.tolist())
        inner_ids = tuple(game for game in train_ids if game not in validation_ids)
        if not inner_ids:
            raise ValueError("an inner restart fold cannot be empty")
        inner_restart = restart_trainer.train(
            games, list(inner_ids), restart_epochs, device,
            restart_trainer.stable_seed(seed, "inner:{}".format(
                ",".join(map(str, validation_ids)))))
        candidate_rows.extend(candidate_trainer.probability_rows(
            games, validation_ids, inner_restart))
    candidate = candidate_trainer.train_classifier(
        candidate_rows, candidate_epochs, device,
        restart_trainer.stable_seed(seed, "candidate:" + ",".join(map(str, train_ids))))
    for row in candidate_rows:
        row["candidate_probability"] = spell_candidate.probabilities(
            row["candidate_features"], candidate)
    mode, threshold, result = _candidate_configuration(_selection_rows(candidate_rows))
    candidate.update({
        "mode": mode, "threshold": threshold, "train_games": list(train_ids),
        "restart_generation": "outer-training match-grouped cross-fit",
    })
    print("nested base selected {} {:.2f}: spell {:.4f}, macro {:.4f}, ratio {:.3f}".format(
        mode, threshold, result["choose"], result["macro"], result["ratio"]), flush=True)
    return outer_restart, candidate


def predict(games, ids, restart_model, candidate_model):
    """Return {(game, kind): labels} from the supplied fold-specific checkpoints."""
    rows = candidate_trainer.probability_rows(games, ids, restart_model)
    output = {}
    for row in rows:
        row["candidate_probability"] = spell_candidate.probabilities(
            row["candidate_features"], candidate_model)
        output[(row["game"], row["kind"])] = candidate_trainer.labels_for(
            row, candidate_model["mode"], float(candidate_model["threshold"]))
    return output
