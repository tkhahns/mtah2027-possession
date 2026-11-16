"""Train and cross-validate the slot-invariant temporal boundary model on Metrica games.

Each fold holds out one complete match.  Training pairs the clean view with hidden and messy
tracking views of the same frames and adds a consistency loss, so slot count, tracklet swaps and
ball occlusion cannot become shortcuts.  The challenge match is never opened.
"""

import argparse
import copy
import json
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src")]
MODELS = os.path.join(ROOT, "src", "models")

from possession import scoring  # noqa: E402
from training.possession_models import nested_timing  # noqa: E402
from possession import (hybrid_pipeline, idsse, learned_timing_pipeline, live_control_veto,
                        metrica, segmental_decoder, set_temporal, spell_filter, testsets)  # noqa: E402

KINDS = ("hidden", "messy")
MIN_POSSESSION_RATIO = 0.90


def prepare(stride=set_temporal.STRIDE, idsse_matches=()):
    games = {}
    sources = [(game, testsets.load) for game in metrica.GAMES]
    sources.extend(("idsse:{}".format(match),
                    lambda _game, kind, match=match: idsse.load(match, kind))
                   for match in idsse_matches)
    for game, loader in sources:
        games[game] = {}
        for kind in ("clean",) + KINDS:
            trk, truth = loader(game, kind)
            player, mask, context, index = set_temporal.frame_inputs(trk, stride)
            state, boundary = set_temporal.target_arrays(truth, index)
            games[game][kind] = {
                "trk": trk, "truth": truth, "player": player, "mask": mask,
                "context": context, "index": index, "state": state, "boundary": boundary,
            }
            positives = (boundary > 0).sum(axis=0)
            print("prepared game {} {:6s}: samples={} slots={} boundaries={}".format(
                game, kind, len(index), player.shape[1], positives.tolist()), flush=True)
    return games


def _examples(games, ids, window, step):
    result = []
    for game in ids:
        length = len(games[game]["clean"]["state"])
        starts = list(range(0, max(1, length - window + 1), step))
        if not starts or starts[-1] != max(0, length - window):
            starts.append(max(0, length - window))
        for kind in KINDS:
            result.extend((game, kind, start) for start in starts)
    return result


def _batch(rows, view, window):
    slots = max(row[view]["player"].shape[1] for row in rows)
    batch = len(rows)
    features = len(set_temporal.PLAYER_FEATURES)
    context_features = len(rows[0][view]["context"][0])
    player = np.zeros((batch, window, slots, features), np.float32)
    mask = np.zeros((batch, window, slots), bool)
    context = np.zeros((batch, window, context_features), np.float32)
    for i, row in enumerate(rows):
        values = row[view]
        count = values["player"].shape[1]
        start = row["start"]
        stop = min(len(values["player"]), start + window)
        n = stop - start
        player[i, :n, :count] = values["player"][start:stop]
        mask[i, :n, :count] = values["mask"][start:stop]
        context[i, :n] = values["context"][start:stop]
    return player, mask, context


def _targets(rows, window):
    state = np.full((len(rows), window), -1, np.int64)
    boundary = np.zeros((len(rows), window, len(set_temporal.BOUNDARY_TYPES)), np.float32)
    valid = np.zeros((len(rows), window), bool)
    for i, row in enumerate(rows):
        values = row["clean"]
        start = row["start"]
        stop = min(len(values["state"]), start + window)
        n = stop - start
        state[i, :n] = values["state"][start:stop]
        boundary[i, :n] = values["boundary"][start:stop]
        valid[i, :n] = True
    return state, boundary, valid


def _training_rows(games, examples):
    return [{"clean": games[game]["clean"], "noisy": games[game][kind], "start": start}
            for game, kind, start in examples]


def train(games, ids, epochs=6, batch_size=8, window=128, device="cpu", seed=2027):
    import torch
    from torch import nn

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = set_temporal.build_model().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-3)
    examples = _examples(games, ids, window, max(16, window // 2))

    # Estimate boundary imbalance from complete training matches, then cap it to keep soft
    # neighborhoods from dominating the state objective.
    targets = np.concatenate([games[game]["clean"]["boundary"] for game in ids], axis=0)
    positive = np.maximum((targets > 0).sum(axis=0), 1)
    boundary_weight = np.minimum(30.0, (len(targets) - positive) / positive).astype(np.float32)
    boundary_loss = nn.BCEWithLogitsLoss(
        pos_weight=torch.from_numpy(boundary_weight).to(device), reduction="none")
    state_loss = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(1, int(epochs) + 1):
        random.Random(seed + epoch).shuffle(examples)
        model.train()
        losses = []
        for offset in range(0, len(examples), int(batch_size)):
            selected = examples[offset:offset + int(batch_size)]
            rows = _training_rows(games, selected)
            clean = _batch(rows, "clean", window)
            noisy = _batch(rows, "noisy", window)
            target_state, target_boundary, valid = _targets(rows, window)

            def tensors(values):
                return (torch.from_numpy(values[0]).to(device),
                        torch.from_numpy(values[1]).to(device),
                        torch.from_numpy(values[2]).to(device))

            clean_state, clean_boundary = model(*tensors(clean))
            noisy_state, noisy_boundary = model(*tensors(noisy))
            y_state = torch.from_numpy(target_state).to(device)
            y_boundary = torch.from_numpy(target_boundary).to(device)
            valid_tensor = torch.from_numpy(valid).to(device)
            supervised = (
                state_loss(clean_state.reshape(-1, 3), y_state.reshape(-1))
                + state_loss(noisy_state.reshape(-1, 3), y_state.reshape(-1)))
            boundary_values = (boundary_loss(clean_boundary, y_boundary)
                               + boundary_loss(noisy_boundary, y_boundary)).mean(dim=2)
            supervised = supervised + boundary_values[valid_tensor].mean()
            consistency = (
                (torch.softmax(noisy_state, dim=2)
                 - torch.softmax(clean_state.detach(), dim=2)).square().mean()
                + (torch.sigmoid(noisy_boundary)
                   - torch.sigmoid(clean_boundary.detach())).square().mean())
            loss = supervised + 0.25 * consistency
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            losses.append(float(loss.detach().cpu()))
        print("epoch {}/{} loss {:.5f}".format(epoch, epochs, float(np.mean(losses))), flush=True)
    return model.eval()


def probability_rows(games, ids, model, device, duration, base_labels=None):
    rows = []
    for game in ids:
        for kind in KINDS:
            row = games[game][kind]
            state, boundary, index = set_temporal.predict_probabilities(
                model, row["trk"], device=device)
            rows.append({"game": game, "kind": kind, "row": row,
                         "state": set_temporal.upsample(state, index, len(row["truth"])),
                         "boundary": set_temporal.upsample(boundary, index, len(row["truth"])),
                         "base": (base_labels[(game, kind)] if base_labels is not None
                                  else learned_timing_pipeline.predict(row["trk"])),
                         "duration": duration,
                         "hard": hybrid_pipeline.dead_ball(row["trk"]),
                         "control": spell_filter.control_team(row["trk"])})
            print("predicted held-out game {} {}".format(game, kind), flush=True)
    return rows


def score_rows(rows, configuration=None):
    by_kind = {kind: [] for kind in KINDS}
    macro = {kind: [] for kind in KINDS}
    ratios = {kind: [] for kind in KINDS}
    for item in rows:
        labels = item["base"]
        if configuration is not None:
            labels = set_temporal.refine(
                labels, item["state"], item["boundary"], **configuration)
        result = scoring.score(labels, item["row"]["truth"])
        by_kind[item["kind"]].append(result["spellF1"])
        macro[item["kind"]].append(result["macroF1"])
        ratios[item["kind"]].append(result["pred_poss"] / max(1, result["true_poss"]))
    result = {kind: float(np.mean(values)) for kind, values in by_kind.items()}
    result["choose"] = float(np.mean(list(result.values())))
    result["macro"] = float(np.mean([value for values in macro.values() for value in values]))
    for kind, values in ratios.items():
        result[kind + "_ratio"] = float(np.mean(values))
    result["ratio"] = float(np.mean([value for values in ratios.values() for value in values]))
    return result


def score_segmental_rows(rows, configuration, controlled_live_veto=False):
    """Score free segment decoding, using duration statistics from each row's train fold."""
    by_kind = {kind: [] for kind in KINDS}
    macro = {kind: [] for kind in KINDS}
    ratios = {kind: [] for kind in KINDS}
    for item in rows:
        labels = segmental_decoder.decode(
            item["base"], item["row"]["trk"], item["state"], item["boundary"],
            item["duration"], **configuration)
        if controlled_live_veto:
            labels = live_control_veto.restore_controlled_live(
                labels, item["row"]["trk"], hard_dead=item["hard"],
                control=item["control"])
        result = scoring.score(labels, item["row"]["truth"])
        by_kind[item["kind"]].append(result["spellF1"])
        macro[item["kind"]].append(result["macroF1"])
        ratios[item["kind"]].append(result["pred_poss"] / max(1, result["true_poss"]))
    result = {kind: float(np.mean(values)) for kind, values in by_kind.items()}
    result["choose"] = float(np.mean(list(result.values())))
    result["macro"] = float(np.mean([value for values in macro.values() for value in values]))
    for kind, values in ratios.items():
        result[kind + "_ratio"] = float(np.mean(values))
    result["ratio"] = float(np.mean([value for values in ratios.values() for value in values]))
    return result


def show(name, result):
    print("{}: choose {:.4f}, hidden {:.4f}, messy {:.4f}, macro {:.4f}, ratio {:.3f}".format(
        name, result["choose"], result["hidden"], result["messy"],
        result["macro"], result["ratio"]), flush=True)


def boundary_configurations():
    for radius in (25, 50, 75):
        for boundary_weight in (0.0, 2.0, 4.0, 8.0):
            for min_gain in (0.0, 0.5, 1.0, 2.0):
                yield {"radius": radius, "boundary_weight": boundary_weight,
                       "min_gain": min_gain}


def segmental_configurations():
    # Sparse 2-second lattice points keep complete-match CPU decoding practical;
    # all existing and high-confidence neural boundaries are also candidates.
    common = {"candidate_grid": 10, "candidate_threshold": 0.35,
              "max_duration": 900}
    seen = set()
    for duration_weight, bases, boundaries, penalties in (
            (0.5, (1.25, 1.5, 1.75, 2.0, 2.25), (4.0, 5.0),
             (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0)),
            (0.5, (1.5, 1.75), (3.0,), (-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0))):
        for base_weight in bases:
            for boundary_weight in boundaries:
                for transition_penalty in penalties:
                    configuration = dict(
                        common, base_weight=base_weight, boundary_weight=boundary_weight,
                        duration_weight=duration_weight,
                        transition_penalty=transition_penalty)
                    key = tuple(sorted(configuration.items()))
                    if key not in seen:
                        seen.add(key)
                        yield configuration


def acceptable_segmental(result, boundary_result, minimum_ratio=MIN_POSSESSION_RATIO):
    """Require robustness on each corruption view and guard against under-segmentation."""
    return (result["hidden"] >= boundary_result["hidden"] - 0.002
            and result["messy"] >= boundary_result["messy"] - 0.002
            and result["hidden_ratio"] >= float(minimum_ratio)
            and result["messy_ratio"] >= float(minimum_ratio))


def cross_validate(games, epochs, batch_size, window, device, seed,
                   restart_epochs=250, candidate_epochs=500, nested_base=True,
                   held_ids=None):
    ids = sorted(games, key=str)
    held_ids = ids if held_ids is None else list(held_ids)
    held_out = []
    for held in held_ids:
        print("\nfold: hold out game {}".format(held), flush=True)
        train_ids = [game for game in ids if game != held]
        duration = segmental_decoder.duration_statistics(
            [games[game]["clean"]["truth"] for game in train_ids])
        base_labels = None
        if nested_base:
            restart_model, candidate_model = nested_timing.fit(
                games, train_ids, restart_epochs, candidate_epochs, device,
                nested_timing.restart_trainer.stable_seed(seed, held))
            base_labels = nested_timing.predict(
                games, [held], restart_model, candidate_model)
        model = train(games, train_ids, epochs, batch_size, window, device,
                      nested_timing.restart_trainer.stable_seed(seed, held))
        held_out.extend(probability_rows(
            games, [held], model, device, duration, base_labels=base_labels))
        del model

    baseline = score_rows(held_out)
    show("learned timing baseline", baseline)
    best = None
    for configuration in boundary_configurations():
        result = score_rows(held_out, configuration)
        acceptable = (result["hidden"] >= baseline["hidden"] - 0.002
                      and result["messy"] >= baseline["messy"] - 0.002)
        key = (acceptable, result["choose"], result["macro"])
        if best is None or key > best[0]:
            best = (key, configuration, result)
    if not best[0][0]:
        print("no configuration passed hidden/messy non-regression; keeping baseline", flush=True)
        boundary_configuration = None
        boundary_result = baseline
    else:
        boundary_configuration = best[1]
        boundary_result = best[2]
        print("selected boundary", boundary_configuration, flush=True)
        show("set-temporal boundary refiner", boundary_result)
        print("delta choose {:+.4f}, macro {:+.4f}".format(
            boundary_result["choose"] - baseline["choose"],
            boundary_result["macro"] - baseline["macro"]), flush=True)

    segment_best = None
    for configuration in segmental_configurations():
        result = score_segmental_rows(held_out, configuration)
        acceptable = acceptable_segmental(result, boundary_result)
        key = (acceptable, result["choose"], result["macro"],
               -abs(result["ratio"] - 1.0))
        if segment_best is None or key > segment_best[0]:
            segment_best = (key, configuration, result)
    if segment_best[0][0] and segment_best[2]["choose"] > boundary_result["choose"]:
        print("selected segmental", segment_best[1], flush=True)
        show("semi-Markov segment decoder", segment_best[2])
        print("segmental delta choose {:+.4f}, macro {:+.4f}".format(
            segment_best[2]["choose"] - boundary_result["choose"],
            segment_best[2]["macro"] - boundary_result["macro"]), flush=True)
        veto_result = score_segmental_rows(
            held_out, segment_best[1], controlled_live_veto=True)
        show("semi-Markov + controlled-live veto", veto_result)
        print("veto delta choose {:+.4f}, macro {:+.4f}".format(
            veto_result["choose"] - segment_best[2]["choose"],
            veto_result["macro"] - segment_best[2]["macro"]), flush=True)
        selected = {"mode": "segmental", "parameters": segment_best[1]}
        return selected, segment_best[2], baseline
    print("semi-Markov decoder rejected; constrained boundary refiner remains selected", flush=True)
    show("best rejected semi-Markov candidate", segment_best[2])
    selected = ({"mode": "boundary", "parameters": boundary_configuration}
                if boundary_configuration is not None else {"mode": "baseline", "parameters": {}})
    return selected, boundary_result, baseline


def resolve_device(requested):
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU", flush=True)
        return "cpu"
    return requested


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--restart-epochs", type=int, default=250)
    parser.add_argument("--candidate-epochs", type=int, default=500)
    parser.add_argument("--include-idsse", action="store_true",
                        help="add every downloaded public IDSSE match")
    parser.add_argument("--frozen-base", action="store_true",
                        help="diagnostic only: use the non-nested frozen base checkpoints")
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--output", default=os.path.join(MODELS, "set_temporal.pt"))
    parser.add_argument("--restart-output", default=os.path.join(MODELS, "restart_state.json"))
    parser.add_argument("--candidate-output", default=os.path.join(MODELS, "spell_candidate.json"))
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    print("device", device, flush=True)
    idsse_matches = idsse.available_matches() if args.include_idsse else ()
    if args.include_idsse and len(idsse_matches) != len(idsse.MATCHES):
        parser.error("--include-idsse requires all seven matches; run src/training/fetch_idsse.py")
    games = prepare(idsse_matches=idsse_matches)
    held_ids = list(metrica.GAMES) if args.include_idsse else None
    configuration, _, _ = cross_validate(
        games, args.epochs, args.batch_size, args.window, device, args.seed,
        args.restart_epochs, args.candidate_epochs, nested_base=not args.frozen_base,
        held_ids=held_ids)
    if args.cv_only:
        return
    model = train(games, sorted(games, key=str), args.epochs, args.batch_size,
                  args.window, device, args.seed)
    restart_model, candidate_model = nested_timing.fit(
        games, sorted(games, key=str), args.restart_epochs, args.candidate_epochs,
        device, args.seed)
    for path, checkpoint in ((args.restart_output, restart_model),
                             (args.candidate_output, candidate_model)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(checkpoint, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print("saved", path, flush=True)
    duration = segmental_decoder.duration_statistics(
        [games[game]["clean"]["truth"] for game in sorted(games, key=str)])
    import torch
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"state_dict": copy.deepcopy(model).cpu().state_dict(),
                "configuration": configuration,
                "duration": duration.tolist(),
                "stride": set_temporal.STRIDE,
                "seed": args.seed,
                "train_games": list(sorted(games, key=str)),
                "train_kinds": list(KINDS)}, args.output)
    print("saved", args.output, flush=True)


if __name__ == "__main__":
    main()
