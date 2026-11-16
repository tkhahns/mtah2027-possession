"""The keeper holds the ball, so his team has it (final_pipeline: after the live-control veto,
before the rule clean-up).

The data pipeline gives the ball a goalkeeper's track over a long ball gap that starts next to him
(clean_data.keeper_ball; the cleaned tracking's "keeper_hold" lists (first, until, team) per hold).
Unless the referee's whistle stopped play in the whistle_before_s before the hold starts, every
frame of the hold becomes the keeper's team: H for the home keeper, A for the away one. The rule
clean-up runs after this, so a set-piece tag window (a goal kick) can still make it dead.

    labels = apply(labels, cleaned.get("keeper_hold", ()), whistles)
"""

from collections import Counter

import numpy as np

from .data import HOME

WHISTLE_BEFORE_S = 20.0     # a whistle this long before the hold starts: play was stopped
FPS = 25


def apply(labels, holds, whistles, fps=FPS, whistle_before_s=None, report=None):
    """labels with each hold given to the keeper's team (a copy). holds: (first, until, team)
    frames inclusive, team 0 home / 1 away; whistles: the referee's whistle start frames.
    whistle_before_s None = WHISTLE_BEFORE_S."""
    if whistle_before_s is None:
        whistle_before_s = WHISTLE_BEFORE_S
    counts = Counter()
    out = np.array(labels, copy=True)
    w = np.sort(np.asarray(list(whistles), dtype=np.int64))
    window = int(round(whistle_before_s * fps))
    for first, until, team in holds:
        counts["holds"] += 1
        k = np.searchsorted(w, first - window)          # the first whistle at or after the window
        if k < len(w) and w[k] < first:
            counts["whistle before: left alone"] += 1
            continue
        label = "H" if team == HOME else "A"
        counts["frames changed"] += int(np.sum(out[first:until + 1] != label))
        out[first:until + 1] = label
        counts["given to the keeper's team"] += 1
    if report is not None:                              # a plain dict or a Counter
        for key, value in counts.items():
            report[key] = report.get(key, 0) + value
    return out
