"""Train and validate a spatial dead/live restart-state model on public Metrica matches.

Complete games are held out. The model learns only on frames not already covered by the precise
out-of-pitch rule, so capacity is directed at fouls, offsides and central restarts rather than at
relearning the hard rule. The challenge match and its labels are never opened.
"""

import argparse
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
MODELS = os.path.join(ROOT, "src", "models")

from possession import scoring  # noqa: E402
from possession import hybrid_pipeline, idsse, metrica, restart_pipeline, restart_state, testsets  # noqa: E402

KINDS = ("hidden", "messy")
THRESHOLDS = np.r_[np.arange(0.20, 0.96, 0.05), 0.975, 0.99, 1.0]
MIN_DEAD_FRAMES = (0, 25, 50, 75)
STRIDE = 5


def _boundary_weight(target, radius=25):
    change = np.flatnonzero(target[1:] != target[:-1]) + 1
    weight = np.ones(len(target), np.float32)
    for frame in change:
        weight[max(0, frame - radius):min(len(target), frame + radius + 1)] = 3.0
    return weight


def stable_seed(seed, value):
    """Deterministic seed offset for integer or named external games."""
    text = str(value)
    return int(seed) + sum((index + 1) * ord(char) for index, char in enumerate(text))


def prepare(idsse_matches=()):
    games = {}
    sources = [(game, testsets.load) for game in metrica.GAMES]
    sources.extend(("idsse:{}".format(match),
                    lambda _game, kind, match=match: idsse.load(match, kind))
                   for match in idsse_matches)
    for game, loader in sources:
        games[game] = {}
        for kind in KINDS:
            trk, truth = loader(game, kind)
            hard = hybrid_pipeline.dead_ball(trk)
            features = restart_state.frame_features(trk)
            target = (truth == "D").astype(np.float32)
            graded = np.isin(truth, ["H", "A", "D"])
            train_mask = graded & ~hard
            games[game][kind] = {
                "trk": trk, "truth": truth, "features": features, "target": target,
                "graded": graded, "hard": hard, "train_mask": train_mask,
                "weight": _boundary_weight(target),
            }
            print("prepared game {} {:6s}: {} frames, {:.1f}% dead, {:.1f}% hard-covered".format(
                game, kind, int(graded.sum()), 100 * target[graded].mean(),
                100 * (hard & (target > 0) & graded).sum() / max(1, (target[graded] > 0).sum())),
                flush=True)
    return games


def arrays(games, ids):
    xs, ys, ws = [], [], []
    for game in ids:
        for kind in KINDS:
            row = games[game][kind]
            sample = row["train_mask"] & (np.arange(len(row["target"])) % STRIDE == 0)
            xs.append(row["features"][sample])
            ys.append(row["target"][sample])
            ws.append(row["weight"][sample])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)


def train(games, ids, epochs, device, seed):
    import torch
    from torch import nn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    features, target, sample_weight = arrays(games, ids)
    mean = features.mean(axis=0).astype(np.float32)
    std = np.maximum(features.std(axis=0), 1e-3).astype(np.float32)
    x = torch.from_numpy(((features - mean) / std).astype(np.float32)).to(device)
    y = torch.from_numpy(target.astype(np.float32)).to(device)
    weight = torch.from_numpy(sample_weight).to(device)
    model = nn.Sequential(
        nn.Linear(features.shape[1], 32), nn.ReLU(),
        nn.Linear(32, 16), nn.ReLU(),
        nn.Linear(16, 1),
    ).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-3)
    positives = max(float(target.sum()), 1.0)
    pos_weight = torch.tensor((len(target) - positives) / positives, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    for epoch in range(1, epochs + 1):
        model.train()
        optimiser.zero_grad(set_to_none=True)
        loss = (loss_fn(model(x)[:, 0], y) * weight).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()
        if epoch == 1 or epoch % 50 == 0 or epoch == epochs:
            print("epoch {}/{} loss {:.5f}".format(
                epoch, epochs, float(loss.detach().cpu())), flush=True)
    first, second, third = model[0], model[2], model[4]
    return {
        "feature_names": list(restart_state.FEATURE_NAMES),
        "mean": mean.tolist(), "std": std.tolist(),
        "weight1": first.weight.detach().cpu().numpy().tolist(),
        "bias1": first.bias.detach().cpu().numpy().tolist(),
        "weight2": second.weight.detach().cpu().numpy().tolist(),
        "bias2": second.bias.detach().cpu().numpy().tolist(),
        "weight3": third.weight.detach().cpu().numpy()[0].tolist(),
        "bias3": float(third.bias.detach().cpu().numpy()[0]),
        "threshold": 1.0, "train_games": list(ids), "train_kinds": list(KINDS),
        "stride": STRIDE, "seed": seed,
    }


def probability_rows(games, ids, model):
    return [(game, kind, games[game][kind],
             restart_state.probabilities(games[game][kind]["features"], model))
            for game in ids for kind in KINDS]


def score_rows(rows, threshold, min_dead_frames=0):
    scores = {kind: [] for kind in KINDS}
    ratios = {kind: [] for kind in KINDS}
    dead_stats = []
    for _, kind, row, probability in rows:
        labels = restart_pipeline.predict_from_probabilities(
            row["trk"], probability, threshold, min_dead_frames=min_dead_frames)
        result = scoring.score(labels, row["truth"])
        scores[kind].append(result["spellF1"])
        ratios[kind].append(result["pred_poss"] / result["true_poss"])
        smooth = restart_state._window_mean(probability, restart_state.WINDOW, restart_state.WINDOW)
        predicted = restart_pipeline.combine_dead(
            row["hard"], smooth, threshold, min_dead_frames)
        actual, graded = row["target"] > 0, row["graded"]
        tp = int(np.sum(predicted & actual & graded))
        dead_stats.append((tp, int(np.sum(predicted & graded)), int(np.sum(actual & graded))))
    means = {kind: float(np.mean(values)) for kind, values in scores.items()}
    means["choose"] = float(np.mean(list(means.values())))
    means["ratio"] = float(np.mean([value for values in ratios.values() for value in values]))
    tp, predicted_n, actual_n = np.sum(dead_stats, axis=0)
    means["dead_precision"] = float(tp / max(1, predicted_n))
    means["dead_recall"] = float(tp / max(1, actual_n))
    return means


def baseline(games):
    rows = [(game, kind, games[game][kind], np.zeros(len(games[game][kind]["truth"])))
            for game in sorted(games, key=str) for kind in KINDS]
    return score_rows(rows, 1.0)


def show(name, result):
    print("{}: choose {:.4f}, hidden {:.4f}, messy {:.4f}, ratio {:.3f}, dead P/R {:.3f}/{:.3f}".format(
        name, result["choose"], result["hidden"], result["messy"], result["ratio"],
        result["dead_precision"], result["dead_recall"]))


def cross_validate(games, epochs, device, seed):
    held_out = []
    ids = sorted(games, key=str)
    for game in ids:
        print("\nfold: hold out game", game)
        model = train(games, [item for item in ids if item != game], epochs, device,
                      stable_seed(seed, game))
        held_out.extend(probability_rows(games, [game], model))
    base = baseline(games)
    show("dead-ball/carry baseline", base)
    best = None
    for min_dead_frames in MIN_DEAD_FRAMES:
        for threshold in THRESHOLDS:
            result = score_rows(held_out, float(threshold), min_dead_frames)
            show("threshold {:.3f}, min {:d}".format(threshold, min_dead_frames), result)
            acceptable = (result["hidden"] >= base["hidden"] - 0.002 and
                          result["messy"] >= base["messy"] - 0.002)
            key = (acceptable, result["choose"])
            if best is None or key > best[0]:
                best = (key, float(threshold), int(min_dead_frames), result)
    print("selected threshold {:.3f}, min dead frames {}".format(best[1], best[2]))
    show("selected", best[3])
    print("delta vs baseline {:+.4f}".format(best[3]["choose"] - base["choose"]))
    return best[1], best[2], best[3], base


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", default=os.path.join(MODELS, "restart_state.json"))
    parser.add_argument("--cross-validate", action="store_true")
    parser.add_argument("--cv-only", action="store_true",
                        help="run complete-match validation without fitting a final checkpoint")
    parser.add_argument("--include-idsse", action="store_true",
                        help="add every downloaded public IDSSE match")
    args = parser.parse_args(argv)
    idsse_matches = idsse.available_matches() if args.include_idsse else ()
    games = prepare(idsse_matches)
    threshold = 1.0
    min_dead_frames = 0
    if args.cross_validate or args.cv_only:
        threshold, min_dead_frames, _, _ = cross_validate(
            games, args.epochs, args.device, args.seed)
    if args.cv_only:
        return
    model = train(games, sorted(games, key=str), args.epochs, args.device, args.seed)
    model["threshold"] = threshold
    model["min_dead_frames"] = min_dead_frames
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("saved", args.output)


if __name__ == "__main__":
    main()
