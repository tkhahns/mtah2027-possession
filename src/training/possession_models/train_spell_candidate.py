"""Train and select a learned keep-versus-merge classifier on complete public matches."""

import argparse
import itertools
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.dirname(os.path.abspath(__file__))]

from possession import scoring  # noqa: E402
import train_restart_state as restart_trainer  # noqa: E402
from possession import (boundary_timing, constrained_restart, dead_boundary_timing,  # noqa: E402
                        spell_candidate, spell_filter)

THRESHOLDS = np.arange(0.10, 0.91, 0.05)
MODES = ("replace", "guard", "extend")


def train_classifier(rows, epochs, device, seed):
    import torch
    from torch import nn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    features = np.concatenate([row["candidate_features"] for row in rows])
    target = np.concatenate([row["candidate_target"] for row in rows])
    if not len(features) or target.min() == target.max():
        raise ValueError("candidate training fold needs both keep and merge examples")
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0), 1e-3).astype(np.float32)
    x = torch.from_numpy(((features - mean) / std).astype(np.float32)).to(device)
    y = torch.from_numpy(target.astype(np.float32)).to(device)
    model = nn.Sequential(nn.Linear(features.shape[1], 16), nn.ReLU(), nn.Linear(16, 1)).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=3e-3)
    positives = max(float(target.sum()), 1.0)
    pos_weight = torch.tensor((len(target) - positives) / positives, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    for epoch in range(1, epochs + 1):
        optimiser.zero_grad(set_to_none=True)
        loss = loss_fn(model(x)[:, 0], y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        if epoch == 1 or epoch % 100 == 0 or epoch == epochs:
            print("candidate epoch {}/{} loss {:.5f}".format(
                epoch, epochs, float(loss.detach().cpu())), flush=True)
    first, second = model[0], model[2]
    exported = {
        "feature_names": list(spell_candidate.FEATURE_NAMES),
        "mean": mean.tolist(), "std": std.tolist(),
        "weight1": first.weight.detach().cpu().numpy().tolist(),
        "bias1": first.bias.detach().cpu().numpy().tolist(),
        "weight2": second.weight.detach().cpu().numpy()[0].tolist(),
        "bias2": float(second.bias.detach().cpu().numpy()[0]),
        "max_frames": spell_candidate.MAX_FRAMES,
        "candidate_count": int(len(target)), "positive_count": int(target.sum()),
        "seed": seed,
    }
    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(
            ((features - mean) / std).astype(np.float32)).to(device))[:, 0]).cpu().numpy()
    actual = spell_candidate.probabilities(features, exported)
    if not np.allclose(actual, expected, atol=1e-5):
        raise RuntimeError("NumPy candidate inference does not match Torch")
    return exported


def decode(game, kind, row, probability):
    labels = constrained_restart.predict_from_probabilities(
        row["trk"], probability, row["features"])
    control = spell_filter.control_team(row["trk"])
    ownership = boundary_timing.ownership_evidence(row["trk"], control=control)
    starts, stops, teams, features = spell_candidate.candidate_features(
        labels, row["trk"], probability, row["features"], control, ownership)
    target = spell_candidate.keep_targets(starts, stops, teams, row["truth"])
    print("decoded game {} {:6s}: {} candidates, {} keep ({:.1f}%)".format(
        game, kind, len(target), int(target.sum()), 100 * target.mean() if len(target) else 0.0),
        flush=True)
    return {
        "game": game, "kind": kind, "row": row, "probability": probability,
        "labels": labels, "control": control, "ownership": ownership,
        "candidate_starts": starts, "candidate_stops": stops,
        "candidate_features": features, "candidate_target": target,
    }


def probability_rows(games, ids, model):
    return [decode(game, kind, games[game][kind],
                   restart_trainer.restart_state.probabilities(
                       games[game][kind]["features"], model))
            for game in ids for kind in restart_trainer.KINDS]


def finish(row, labels):
    labels = boundary_timing.refine(
        labels, row["row"]["trk"], evidence=row["ownership"])
    dead = dead_boundary_timing.dead_evidence(row["probability"])
    return dead_boundary_timing.refine(
        labels, dead, row["row"]["hard"], radius=dead_boundary_timing.RADIUS,
        min_gain=dead_boundary_timing.MIN_GAIN)


def labels_for(row, mode=None, threshold=None):
    if mode is None:
        labels = spell_filter.merge_short_spells(
            row["labels"], row["row"]["trk"], control=row["control"])
    else:
        keep = spell_candidate.mode_probabilities(
            row["candidate_features"], row["candidate_probability"], mode)
        labels = spell_candidate.merge_from_probabilities(
            row["labels"], row["candidate_starts"], row["candidate_stops"], keep, threshold)
    return finish(row, labels)


def score_rows(rows, mode=None, threshold=None):
    spell = {kind: [] for kind in restart_trainer.KINDS}
    macro = {kind: [] for kind in restart_trainer.KINDS}
    ratios = {kind: [] for kind in restart_trainer.KINDS}
    for row in rows:
        result = scoring.score(labels_for(row, mode, threshold), row["row"]["truth"])
        spell[row["kind"]].append(result["spellF1"])
        macro[row["kind"]].append(result["macroF1"])
        ratios[row["kind"]].append(result["pred_poss"] / result["true_poss"])
    result = {kind: float(np.mean(values)) for kind, values in spell.items()}
    result["choose"] = float(np.mean(list(result.values())))
    result["macro"] = float(np.mean([v for values in macro.values() for v in values]))
    for kind, values in ratios.items():
        result[kind + "_ratio"] = float(np.mean(values))
    result["ratio"] = float(np.mean([value for values in ratios.values() for value in values]))
    return result


def show(name, result):
    print("{}: spell {:.4f}, hidden {:.4f}, messy {:.4f}, macro {:.4f}, ratio {:.3f}".format(
        name, result["choose"], result["hidden"], result["messy"],
        result["macro"], result["ratio"]))


def subset_models(games, epochs, device, seed):
    ids = sorted(games)
    models = {}
    for size in (1, 2):
        for subset in itertools.combinations(ids, size):
            print("\nrestart training subset", subset)
            subset_seed = seed + size * 1000 + sum((i + 1) * game * 37
                                                   for i, game in enumerate(subset))
            models[subset] = restart_trainer.train(
                games, list(subset), epochs, device, subset_seed)
    return models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--restart-epochs", type=int, default=250)
    parser.add_argument("--candidate-epochs", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", default=spell_candidate.DEFAULT_MODEL)
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--frozen-only", action="store_true",
                        help="score the existing checkpoint's frozen mode and threshold")
    args = parser.parse_args(argv)

    games = restart_trainer.prepare()
    ids = sorted(games)
    models = subset_models(games, args.restart_epochs, args.device, args.seed)
    held_out = []
    final_training_rows = []
    for held_out_game in ids:
        train_ids = tuple(game for game in ids if game != held_out_game)
        print("\nouter candidate fold: hold out game", held_out_game)
        nested_rows = []
        for train_game in train_ids:
            singleton = tuple(game for game in train_ids if game != train_game)
            nested_rows.extend(probability_rows(games, [train_game], models[singleton]))
        classifier = train_classifier(
            nested_rows, args.candidate_epochs, args.device, args.seed + held_out_game * 101)
        test_rows = probability_rows(games, [held_out_game], models[train_ids])
        for row in test_rows:
            row["candidate_probability"] = spell_candidate.probabilities(
                row["candidate_features"], classifier)
        held_out.extend(test_rows)
        final_training_rows.extend(test_rows)

    base = score_rows(held_out)
    show("fixed control-aware timing pipeline", base)
    if args.frozen_only:
        frozen = spell_candidate.load_model(args.output)
        result = score_rows(held_out, frozen["mode"], float(frozen["threshold"]))
        print("frozen mode {}, threshold {:.2f}".format(
            frozen["mode"], float(frozen["threshold"])))
        show("frozen learned candidate pipeline", result)
        print("spell delta {:+.4f}, macro delta {:+.4f}".format(
            result["choose"] - base["choose"], result["macro"] - base["macro"]))
        return
    best = None
    for mode in MODES:
        for threshold in THRESHOLDS:
            result = score_rows(held_out, mode, float(threshold))
            show("{} threshold {:.2f}".format(mode, threshold), result)
            acceptable = (result["hidden"] >= base["hidden"] - 0.002 and
                          result["messy"] >= base["messy"] - 0.002 and
                          result["macro"] >= base["macro"] - 0.002)
            key = (acceptable, result["choose"], result["macro"],
                   -abs(result["ratio"] - 1.0))
            if best is None or key > best[0]:
                best = (key, mode, float(threshold), result)
    print("\nselected mode {}, threshold {:.2f}".format(best[1], best[2]))
    show("selected learned candidate pipeline", best[3])
    print("spell delta {:+.4f}, macro delta {:+.4f}".format(
        best[3]["choose"] - base["choose"], best[3]["macro"] - base["macro"]))

    if args.cv_only:
        return
    final = train_classifier(
        final_training_rows, args.candidate_epochs, args.device, args.seed)
    final.update({
        "threshold": best[2], "mode": best[1], "train_games": ids,
        "restart_generation": "complete-match leave-one-game-out",
    })
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(final, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("saved", args.output)


if __name__ == "__main__":
    main()
