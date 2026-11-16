"""Thresholds for clean_data.role_scales: per scale, the 99th percentile of delta over every
(window, slot) of the Metrica games, as they are (no injected errors). Written from the messy set
(players drop out as on our match, which widens the spread); clean is printed for comparison.
Saved to src/models/team_role_scales.json. External data only.

    python src/training/player_cleaning/role_scales_thresholds.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, cached_grid, load  # noqa: E402
from clean_data import role_scales  # noqa: E402
from possession import metrica  # noqa: E402

OUT = os.path.join(ROOT, "src", "models", "team_role_scales.json")
FLOOR_M = 0.5


def null_deltas(kind):
    """{game: {scale: finite deltas}}."""
    out = {}
    for game in metrica.GAMES:
        trk, halves = load(game, kind)
        g = cached_grid("game{}_{}".format(game, kind), trk["players"], trk["team"], halves,
                        trk["pitch"])
        out[game] = {L: d[np.isfinite(d)] for L, (_, d) in g.items()}
        print("{} game {} done".format(kind, game), flush=True)
    return out


def cuts(deltas, games):
    """{scale: (q99, cut)}: cut = max(|q99|, FLOOR_M), pooled over games."""
    out = {}
    for L in role_scales.SCALES:
        q = float(np.percentile(np.concatenate([deltas[g][L] for g in games]), 99))
        out[L] = (q, max(abs(q), FLOOR_M))
    return out


def main():
    messy, clean = null_deltas("messy"), null_deltas("clean")
    print("\n{:>5} {:>12} {:>12} {:>9} {:>10}".format("scale", "messy q99", "clean q99", "cut",
                                                    "windows"))
    final = cuts(messy, metrica.GAMES)
    ref = cuts(clean, metrica.GAMES)
    for L in role_scales.SCALES:
        print("{:>4}s {:>12.2f} {:>12.2f} {:>9.2f} {:>10}".format(
            L, final[L][0], ref[L][0], final[L][1], sum(len(messy[g][L]) for g in metrica.GAMES)))
    out = {"source": "Metrica games 1-3, messy set, every window and slot (no injected errors)",
           "rule": "cut = max(|99th percentile|, {} m); other if median >= cut, own if <= -cut"
                   .format(FLOOR_M),
           "q99": {str(L): round(final[L][0], 3) for L in role_scales.SCALES},
           "q99_clean": {str(L): round(ref[L][0], 3) for L in role_scales.SCALES},
           "thresholds": {str(L): round(final[L][1], 3) for L in role_scales.SCALES},
           "nested": {str(g): {str(L): round(c, 3) for L, (_, c) in
                               cuts(messy, [h for h in metrica.GAMES if h != g]).items()}
                      for g in metrica.GAMES}}
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=1)
    print("written", os.path.relpath(OUT, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
