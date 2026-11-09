"""Build the possession submission from the organisers' data bundle.

    python src/main.py --data data --team <team_name>            # -> output/<team_name>.csv
    python src/main.py --data data --team <team_name> --whistles # also use the referee's whistles

For every frame of the tactical video it writes one label: H Barca has the ball, A Athletic,
D dead ball, X not graded. The stages:

    tracking  -> team_shape.frame_features   who is where, how stretched, how fast, near the ball
              -> evidence.evidence           log-odds: home (not away) has the ball / ball dead
              -> tag_cleaning                Smart Tagging set pieces checked, soft anchors
              -> whistles (optional)         referee whistles from the sound: dead from there
              -> decoder.decode              Viterbi: H / A / D with long, stable possessions
"""

import argparse
import os
import sys

import numpy as np

import decoder
import evidence
import loaders
import tag_cleaning
import team_shape
import whistles
from config import ALWAYS_DEAD, FPS, HALVES, LABELS, N_FRAMES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def graded_mask(n=N_FRAMES):
    """True on the frames the ground truth grades (the two halves)."""
    mask = np.zeros(n, dtype=bool)
    for _, first, last in HALVES:
        mask[first:last + 1] = True
    return mask


def force_dead(ll, stretches):
    """Emissions with live play ruled out on fixed [first, stop) stretches."""
    ll = np.array(ll, dtype=np.float64, copy=True)
    for first, stop in stretches:
        ll[max(0, int(first)):min(len(ll), int(stop)), [decoder.H, decoder.A]] = -1e4
    return ll


def predict(trk, tags=None, whistle_events=None, halves=HALVES, always_dead=ALWAYS_DEAD):
    """H / A / D for every frame of a tracking dict (players, ball, team, pitch)."""
    feats = team_shape.frame_features(trk, halves)
    ev = evidence.evidence(trk, feats)
    ll = decoder.emissions(ev)
    if tags:
        ll = tag_cleaning.anchor(ll, tag_cleaning.clean_set_pieces(tags, feats, ev))
    if whistle_events:
        ll = whistles.anchor(ll, whistle_events, halves)
    ll = force_dead(ll, always_dead)
    return decoder.decode(ll, halves)


def write_submission(path, labels):
    """frame,label CSV, one row per video frame (organisers' SUBMISSION.md)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="\n") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def check_submission(path, n=N_FRAMES):
    """Read the file back and check the organisers' contract; returns the labels."""
    with open(path) as handle:
        lines = handle.read().splitlines()
    assert lines[0] == "frame,label", "bad header: {!r}".format(lines[0])
    rows = [line.split(",") for line in lines[1:]]
    assert len(rows) == n, "{} rows, expected {}".format(len(rows), n)
    assert all(int(f) == i for i, (f, _) in enumerate(rows)), "frames are not 0..{}".format(n - 1)
    labels = np.array([lab for _, lab in rows])
    assert set(labels) <= set(LABELS), "unknown labels: {}".format(set(labels) - set(LABELS))
    return labels


def possessions(labels, fps=FPS):
    """(count, mean seconds) of the H / A runs: a real match has roughly 250-500."""
    labels = np.asarray(labels)
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts, ends = np.r_[0, change], np.r_[change, len(labels)]
    spells = [(e - s) / fps for s, e in zip(starts, ends) if labels[s] in ("H", "A")]
    return len(spells), float(np.mean(spells)) if spells else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data"),
                        help="folder holding the bundle's tactical/ folder (default: ./data)")
    parser.add_argument("--team", default="team_name", help="team name: the file is <team>.csv")
    parser.add_argument("--out", default=os.path.join(ROOT, "output"), help="output folder")
    parser.add_argument("--no-tags", action="store_true", help="ignore the Smart Tagging")
    parser.add_argument("--whistles", action="store_true",
                        help="add referee whistles from the video's sound (needs ffmpeg)")
    parser.add_argument("--ffmpeg", help="path to ffmpeg (default: on PATH)")
    args = parser.parse_args()

    print("reading tracking from", args.data)
    trk = loaders.load_tracking(args.data)
    tags = [] if args.no_tags else loaders.load_smart_tags(args.data)
    print("  smart tags  {}".format(len(tags)))
    events = []
    if args.whistles:
        events = whistles.from_video(loaders.path(args.data, ".mp4"), args.ffmpeg)
        print("  whistles    {}".format(len(events)))

    labels = predict(trk, tags, events)
    labels[~graded_mask()] = "X"           # ungraded frames cost nothing either way

    path = os.path.join(args.out, args.team + ".csv")
    write_submission(path, labels)
    labels = check_submission(path)
    count, mean_s = possessions(labels)
    print("wrote {}: {} rows, valid".format(path, len(labels)))
    print("  labels      " + "  ".join("{} {}".format(k, int(np.sum(labels == k))) for k in LABELS))
    print("  possessions {} of mean {:.1f} s{}".format(
        count, mean_s, "" if 250 <= count <= 500 else "   CHECK THIS: expected ~250-500"))


if __name__ == "__main__":
    sys.exit(main())
