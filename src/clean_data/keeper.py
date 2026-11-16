"""Keeper slots that stray into the other team's half (see the tracking quirks: keeper tracks
jump into the other box).

    slots = find_slots(data["players"], data["team"], data["halves"])
    players, removed, moved = fix_keepers(data["players"], data["halves"], slots)

    from clean_data.keeper import load_input
    data = load_input()
    data["players"]   # (frames, slots, 3) float32: x_m, y_m, speed -- NaN where the slot is empty
    data["ball"]      # (frames, 3)        float32: x_m, y_m, speed -- NaN where the ball is lost
    data["team"]      # (slots,)           int8:    0 = home (Barca), 1 = away (Athletic)
    data["halves"]    # [(half, first_video_frame, last_video_frame)]

Run it to check the input loads (prints only shapes and counts, never the data):

    python src/clean_data/keeper.py [--data-dir <bundle>]

--frame N also prints that one frame's values. That is match data: keep it in your terminal,
never paste it into an AI chat, issue or PR.
"""

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from possession.data import load_tracking, read_tracking  # noqa: E402  also puts the repo root on sys.path
from games import DATA_DIR, GAME, GAMES, asset_paths, half_video_blocks, require  # noqa: E402
from utils import keeper_utils  # noqa: E402
from utils.series_utils import runs  # noqa: E402

HOME, AWAY = 0, 1
PITCH = GAMES[GAME]["pitch"]
LENGTH = PITCH[0]
FPS = GAMES[GAME]["fps"]


def load_input(game=GAME, data_dir=None):
    """Players, ball, team per slot and the half blocks the keeper fix needs. With the default data
    folder the tracking comes from the cache; with another data_dir it is read straight from the
    files, so a different bundle never picks up the cache of the original."""
    trk = load_tracking(game) if data_dir is None else read_tracking(game, data_dir)
    return {"players": trk["players"], "ball": trk["ball"], "team": trk["team"],
            "halves": half_video_blocks(game)}


def home_on_left(n, halves):
    """(frames,) 1 where home defends the left goal, 0 where it defends the right one, -1 outside
    the halves (before kick-off, half-time, after the end). Home starts on the left and the teams
    change ends at half-time."""
    side = np.full(n, -1, dtype=np.int8)
    for half, first, last in halves:
        side[first:last + 1] = 1 if half == 1 else 0
    return side


def find_slots(players, team, halves, pitch=PITCH, fps=FPS):
    """(frames, 2) home / away keeper slot per frame, found by utils.keeper_utils; -1 outside the
    halves."""
    found = keeper_utils.find_keeper_slots(
        players, team, [(first, last) for _, first, last in halves],
        {"pitch": tuple(map(float, pitch)), "fps": fps})
    return np.stack([found[key] for key in keeper_utils.KEYS], axis=1)


def per_frame(slots, n):
    """(n, 2) view of a (home, away) slot pair or of per-frame slots."""
    return np.broadcast_to(np.asarray(slots, dtype=np.int64), (n, 2))


def fix_keepers(players, halves, slots, length=LENGTH):
    """Where both keeper slots are in the same half, only one of them is that half's keeper. The
    slot in the other team's half is the wrong one; the slot whose own half it is, the right one.

    - Only the wrong slot has a speed: it is the real track. Copy it (x, y, speed) to the right
      slot, then blank (NaN) the wrong slot.
    - Otherwise (only the right slot has a speed, both have one, or neither): blank the wrong slot.

    slots: (frames, 2) slots per frame (find_slots()), -1 where not known, or one (home, away)
    pair. Frames outside the halves, and frames where either slot is not known or not tracked,
    are left alone.

    Returns (a cleaned copy of players,
             removed: (frames, 2) bool, home / away keeper slot blanked,
             moved:   (frames, 2) bool, home / away keeper slot overwritten by the other slot)."""
    n = len(players)
    side = home_on_left(n, halves)
    slots = per_frame(slots, n)
    rows = np.arange(n)
    keepers = players[rows[:, None], np.maximum(slots, 0)]   # (frames, 2, 3) home / away keeper
    tracked = ~np.isnan(keepers[..., 0]) & (slots >= 0)
    left = keepers[..., 0] < length / 2
    same_half = tracked.all(axis=1) & (left[:, HOME] == left[:, AWAY]) & (side >= 0)
    # own half: home on the left when side == 1, away on the other side
    own_left = np.stack([side == 1, side == 0], axis=1)
    removed = same_half[:, None] & (left != own_left)   # one True per same-half frame: the wrong slot
    has_speed = ~np.isnan(keepers[..., 2])
    only_wrong_has_speed = (has_speed[:, HOME] != has_speed[:, AWAY]) & (has_speed & removed).any(axis=1)
    moved = removed[:, ::-1] & only_wrong_has_speed[:, None]   # the right slot of those frames

    out = players.copy()
    for t in (HOME, AWAY):
        out[rows[moved[:, t]], slots[moved[:, t], t]] = keepers[moved[:, t], 1 - t]
        out[rows[removed[:, t]], slots[removed[:, t], t]] = np.nan
    return out, removed, moved


def write_atd(raw_in, raw_out, removed, moved, slots):
    """Copy the ATD text file, rewriting only the keeper groups fix_keepers() changed: a moved slot
    gets the other keeper's group text as it is, a removed slot gets NaN,NaN,NaN. Every other
    character, line endings included, is copied unchanged."""
    changed = removed.any(axis=1) | moved.any(axis=1)
    slots = per_frame(slots, len(removed))
    with open(raw_in, newline="") as src, open(raw_out, "w", newline="") as dst:
        for line in src:
            body = line.rstrip("\r\n")
            parts = body.split(":")
            if len(parts) == 3 and changed[int(parts[0]) - 1]:   # ATD frame_id - 1 = video frame
                row = int(parts[0]) - 1
                groups = parts[1].split(";")
                before = list(groups)
                for t in (HOME, AWAY):
                    if moved[row, t]:
                        groups[slots[row, t]] = before[slots[row, 1 - t]]
                    if removed[row, t]:
                        groups[slots[row, t]] = "NaN,NaN,NaN"
                parts[1] = ";".join(groups)
                line = ":".join(parts) + line[len(body):]
            dst.write(line)


def save_clean(game, data_dir, out_dir, removed, moved, slots):
    """Write the cleaned ATD (.txt) and an unchanged copy of the metadata (.xml) into out_dir, in
    the same layout as the bundle, so read_tracking(data_dir=out_dir) reads it."""
    meta_in, raw_in = require(["atd_meta", "atd_raw"], game, data_dir)
    out = asset_paths(game, out_dir)
    os.makedirs(os.path.dirname(out["atd_raw"]), exist_ok=True)
    shutil.copyfile(meta_in, out["atd_meta"])
    write_atd(raw_in, out["atd_raw"], removed, moved, slots)
    return out["atd_raw"]


def changed_stretches(removed, moved, slots):
    """[(first_frame, last_frame, slot, what)] in frame order, one per stretch of frames where a
    keeper slot was blanked or copied over; a stretch also ends where the keeper slots change."""
    slots = per_frame(slots, len(removed))
    out = []
    for t in (HOME, AWAY):
        for mask, copied in ((moved[:, t], True), (removed[:, t], False)):
            for start, stop in runs(mask):
                pairs = slots[start:stop]
                cuts = (np.flatnonzero((pairs[1:] != pairs[:-1]).any(axis=1)) + 1).tolist()
                for a, b in zip([0] + cuts, cuts + [stop - start]):
                    what = "copied from slot {}".format(pairs[a, 1 - t]) if copied else "blanked"
                    out.append((start + a, start + b - 1, int(pairs[a, t]), what))
    return sorted(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--game", default=GAME)
    parser.add_argument("--data-dir", help="another bundle, read without the cache")
    parser.add_argument("--frame", type=int,
                        help="also print this video frame's values, before and after the clean")
    parser.add_argument("--keeper-slots", type=int, nargs=2, metavar=("HOME", "AWAY"),
                        help="use these two slots all match instead of finding them per frame")
    parser.add_argument("--list-changes", action="store_true",
                        help="print every stretch of frames where a keeper slot changed")
    parser.add_argument("--save", action="store_true",
                        help="write the cleaned ATD to --out (default <data dir>/clean)")
    parser.add_argument("--out", help="folder for the cleaned bundle")
    parser.add_argument("--force", action="store_true", help="overwrite an existing cleaned ATD")
    args = parser.parse_args()

    data = load_input(args.game, args.data_dir)
    players, ball, team = data["players"], data["ball"], data["team"]
    print("players", players.shape, players.dtype)
    print("ball   ", ball.shape, ball.dtype)
    print("slots   {} (home {}, away {})".format(len(team), (team == 0).sum(), (team == 1).sum()))
    for half, first, last in data["halves"]:
        print("half {}  video frames {}..{}".format(half, first, last))

    if args.keeper_slots:
        slots = per_frame(args.keeper_slots, len(players))
        if team[args.keeper_slots[HOME]] != HOME or team[args.keeper_slots[AWAY]] != AWAY:
            print("WARNING: slot {} is not home or slot {} is not away in the metadata".format(
                *args.keeper_slots))
    else:
        slots = find_slots(players, team, data["halves"])
    inside = home_on_left(len(players), data["halves"]) >= 0
    cleaned, removed, moved = fix_keepers(players, data["halves"], slots)
    for t, name in ((HOME, "home"), (AWAY, "away")):
        picked, frames = np.unique(slots[inside, t], return_counts=True)
        print("{} keeper: {}".format(name, ", ".join(
            "{} on {} frames".format("slot {}".format(s) if s >= 0 else "no slot", n)
            for s, n in zip(picked, frames))))
        print("  blanked on {} frames (on {} of them first copied to the {} keeper slot)".format(
            removed[:, t].sum(), moved[:, 1 - t].sum(), "away" if t == HOME else "home"))

    if args.list_changes:
        stretches = changed_stretches(removed, moved, slots)
        print("\n{} stretches, {} frames changed".format(
            len(stretches), (removed.any(axis=1) | moved.any(axis=1)).sum()))
        for first, last, slot, what in stretches:
            print("frames {:6d}-{:6d} ({:4d})  slot {:2d} {}".format(
                first, last, last - first + 1, slot, what))

    if args.save:
        out_dir = args.out or os.path.join(args.data_dir or DATA_DIR, "clean")
        target = asset_paths(args.game, out_dir)["atd_raw"]
        if os.path.exists(target) and not args.force:
            sys.exit("{} exists; pass --force to overwrite it".format(target))
        save_clean(args.game, args.data_dir, out_dir, removed, moved, slots)
        check = read_tracking(args.game, out_dir)["players"]          # never cached
        print("\nsaved {}  (reads back the same as the clean: {})".format(
            target, np.array_equal(check, cleaned, equal_nan=True)))

    if args.frame is not None:
        f = args.frame
        pair = slots[f].tolist()
        print("\nframe {}  (x_m, y_m, speed; nan = not tracked)".format(f))
        print("{:13} {:>22}   {:>22}".format("", "before", "after"))
        print("ball          {:7.2f} {:6.2f} {:7.2f}   {:7.2f} {:6.2f} {:7.2f}".format(
            *ball[f], *ball[f]))
        for s in range(len(team)):
            mark = ""
            if s in pair and removed[f, pair.index(s)]:
                mark = "  <- blanked"
            elif s in pair and moved[f, pair.index(s)]:
                mark = "  <- copied from slot {}".format(pair[1 - pair.index(s)])
            print("slot {:2d} {}{} {:7.2f} {:6.2f} {:7.2f}   {:7.2f} {:6.2f} {:7.2f}{}".format(
                s, "home" if team[s] == 0 else "away", "*" if s in pair else " ",
                *players[f, s], *cleaned[f, s], mark))
