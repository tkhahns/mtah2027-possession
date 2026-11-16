"""Label helpers shared by every model: graded mask, segments, and the submission writer."""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, source_frames  # noqa: E402


def graded_mask(game=GAME):
    """True on the frames the ground truth grades (the two halves); X everywhere else is free."""
    mask = np.zeros(source_frames("tactical", game), dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def segments(labels):
    """[(label, first_frame, last_frame), ...] -- maximal runs. Handy for possession counts."""
    labels = np.asarray(labels)
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change - 1, [len(labels) - 1]))
    return [(labels[s], int(s), int(e)) for s, e in zip(starts, ends)]


def possession_summary(labels, fps=25):
    """Possession count and mean duration (s) on graded frames -- the full-match sanity check."""
    graded = np.where(graded_mask(), np.asarray(labels), "X")
    spells = [(e - s + 1) / fps for lab, s, e in segments(graded) if lab in ("H", "A")]
    return {"possessions": len(spells), "mean_s": float(np.mean(spells)) if spells else 0.0}


def write_submission(path, labels):
    labels = np.asarray(labels)
    assert len(labels) == source_frames("tactical"), "wrong row count: {}".format(len(labels))
    assert set(np.unique(labels)) <= {"H", "A", "D", "X"}, np.unique(labels)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))
    return path
