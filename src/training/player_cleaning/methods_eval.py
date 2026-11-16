"""The player-team cleaning methods (clean_data.players) on the Metrica games with injected
wrong-team stretches (common.inject: 3 s to 1 min, starting at a real contact, put on a new
slot of the other team). No broadcast on Metrica, so handovers_checked and appear_vanish run on
the roles and the head count only. Role thresholds nested: from the messy set of the other two
games (src/models/team_role_scales.json, "nested"). Prints rates only.

Caught: a method moves half an injected stretch. False: moved frames outside injected stretches
(seconds and moves per 90 min); on messy these include its own 1-4 s swaps at the ball.

    python src/training/player_cleaning/methods_eval.py [--variant handovers_checked --variant appear_vanish ...]
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BINS, FPS, ROOT, cached_grid, inject, length_bin, load, score  # noqa: E402
from clean_data import players as cleaning  # noqa: E402
from possession import metrica  # noqa: E402

MODEL = os.path.join(ROOT, "src", "models", "team_role_scales.json")
VARIANTS = {
    "handovers_checked": lambda t, h, sh: cleaning.handovers_checked(t, None, h, shared=sh),
    "appear_vanish": lambda t, h, sh: cleaning.appear_vanish(t, None, h, shared=sh),
    "roles_scales cut 1": lambda t, h, sh: cleaning.roles_scales(t, None, h, shared=sh, cut=1.0),
    "roles_scales cut 1.5": lambda t, h, sh: cleaning.roles_scales(t, None, h, shared=sh,
                                                                   cut=1.5),
    "roles_scales (cut 2)": lambda t, h, sh: cleaning.roles_scales(t, None, h, shared=sh),
    "roles_scales min 5 s": lambda t, h, sh: cleaning.roles_scales(t, None, h, shared=sh,
                                                                   cut=1.0, min_s=5.0),
    "roles_scales min 10 s": lambda t, h, sh: cleaning.roles_scales(t, None, h, shared=sh,
                                                                    cut=1.0, min_s=10.0),
    "events_checked roles 1x": lambda t, h, sh: cleaning.events_checked(t, None, h, shared=sh,
                                                                        role_cut=1.0),
    "events_checked roles 1.5x": lambda t, h, sh: cleaning.events_checked(t, None, h, shared=sh,
                                                                          role_cut=1.5),
    "events_checked roles 2x": lambda t, h, sh: cleaning.events_checked(t, None, h, shared=sh,
                                                                        role_cut=2.0),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", action="append", choices=sorted(VARIANTS),
                        help="only this variant (repeat for several)")
    names = parser.parse_args().variant or list(VARIANTS)
    with open(MODEL) as handle:
        nested = {int(g): {int(L): v for L, v in t.items()}
                  for g, t in json.load(handle)["nested"].items()}
    caught = {(k, v): [] for k in ("clean", "messy") for v in names}
    bins = {k: [] for k in ("clean", "messy")}
    false = {(k, v): [0, 0] for k in ("clean", "messy") for v in names}
    minutes = {k: 0.0 for k in ("clean", "messy")}
    start = time.time()
    for kind in ("clean", "messy"):
        for game in metrica.GAMES:
            trk, halves = load(game, kind)
            players, team, rows = inject(trk, halves, np.random.default_rng(100 * game + len(kind)))
            pitch = trk["pitch"]
            grid = cached_grid("game{}_{}_injected".format(game, kind), players, team, halves, pitch)
            tactical = {"players": players, "ball": trk["ball"], "team": team, "pitch": pitch}
            shared = {"grid": grid, "thresholds": nested[game]}
            bins[kind].extend(length_bin(z - a) for _, a, z, _ in rows)
            minutes[kind] += sum(z - a + 1 for a, z in halves) / FPS / 60
            for name in names:
                moves = VARIANTS[name](tactical, halves, shared)
                hit, bad_frames, bad_moves = score(moves, rows, len(players), players.shape[1])
                caught[(kind, name)].extend(hit)
                false[(kind, name)][0] += bad_frames
                false[(kind, name)][1] += bad_moves
            print("{} game {}: {} injected ({:.0f} s)".format(kind, game, len(rows),
                                                              time.time() - start), flush=True)

    head = ["{}-{} s".format(a, b) if b < 10 ** 6 else "{}+ s".format(a) for a, b in BINS]
    for kind in ("clean", "messy"):
        b = np.array(bins[kind])
        print("\n{} ({:.0f} min; injected per length: {})".format(
            kind, minutes[kind], " ".join(str((b == i).sum()) for i in range(len(BINS)))))
        print("{:26}".format("caught") + "".join("{:>9}".format(h) for h in head)
              + "{:>8}{:>15}{:>14}".format("all", "false s / 90", "false / 90"))
        for name in names:
            hit = np.array(caught[(kind, name)])
            cells = ["{:>9}".format("{:.0%}".format(hit[b == i].mean()) if (b == i).any() else "-")
                     for i in range(len(BINS))]
            per90 = 90 / minutes[kind]
            print("{:26}".format(name) + "".join(cells) + "{:>8}{:>15.0f}{:>14.1f}".format(
                "{:.0%}".format(hit.mean()), false[(kind, name)][0] / FPS * per90,
                false[(kind, name)][1] * per90))
    return 0


if __name__ == "__main__":
    sys.exit(main())
