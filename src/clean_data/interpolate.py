"""Fill short tracking gaps by straight-line interpolation (runs after keeper.py).

A gap is a run of frames where a slot (or the ball) has no x, y. It is filled only when
- it is at most --max-gap seconds long (default 15 s),
- the slot is tracked on the frame just before AND just after it (gaps at the start or end of the
  file stay empty), and
- none of its frames was blanked on purpose by keeper.py (a cell tracked in the original ATD but
  NaN in data/clean): filling those would bring the ghost keeper back.
Longer gaps stay NaN. On a filled frame x, y move at constant speed from one end to the other and
speed is that constant speed (distance / time across the gap).

    python src/clean_data/interpolate.py              # data/clean -> data/clean_interpolation
    python src/clean_data/interpolate.py --max-gap 10 --max-ball-gap 5 --force

Prints only counts, never the data.
"""

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from possession.data import load_tracking, read_tracking  # noqa: E402  also puts the repo root on sys.path
from games import DATA_DIR, GAME, GAMES, asset_paths, require  # noqa: E402

MAX_GAP_S = 15.0


def fill_gaps(xys, max_gap, fps, protect=None):
    """Linear fill of the NaN gaps in each column of xys.

    xys      (frames, cols, 3) x, y, speed; a cell is missing where x is NaN
    max_gap  longest gap filled, in frames; a scalar or one value per column
    protect  (frames, cols) bool, cells that must stay NaN: a gap holding one is not filled

    Returns (a filled copy, filled: (frames, cols) bool)."""
    n, cols = xys.shape[:2]
    tracked = ~np.isnan(xys[..., 0])
    idx = np.arange(n)[:, None]
    # last tracked frame at or before f, first tracked frame at or after f (-1 / n when none)
    prev = np.maximum.accumulate(np.where(tracked, idx, -1), axis=0)
    nxt = np.minimum.accumulate(np.where(tracked, idx, n)[::-1], axis=0)[::-1]
    length = nxt - prev - 1
    fill = ~tracked & (prev >= 0) & (nxt < n) & (length <= np.broadcast_to(max_gap, (cols,)))
    if protect is not None:
        count = np.vstack([np.zeros((1, cols), np.int64), np.cumsum(protect, axis=0)])
        col = np.arange(cols)[None, :]
        # protected cells strictly between prev and nxt
        inside = count[np.clip(nxt, 0, n), col] - count[np.clip(prev + 1, 0, n), col]
        fill &= inside == 0

    f, c = np.nonzero(fill)
    p, q = prev[f, c], nxt[f, c]
    a, b = xys[p, c, :2].astype(np.float64), xys[q, c, :2].astype(np.float64)
    t = ((f - p) / (q - p))[:, None]
    out = xys.copy()
    out[f, c, :2] = a + (b - a) * t
    out[f, c, 2] = np.hypot(*(b - a).T) / (q - p) * fps
    return out, fill


def interpolate(players, ball, max_gap, max_ball_gap, fps, original=None):
    """fill_gaps() on the players and the ball together. original = the raw (players, ball): cells
    tracked there but NaN here were blanked on purpose and are protected."""
    xys = np.concatenate([players, ball[:, None]], axis=1)
    protect = None
    if original is not None:
        raw = np.concatenate([original[0], original[1][:, None]], axis=1)
        protect = np.isnan(xys[..., 0]) & ~np.isnan(raw[..., 0])
    limit = np.r_[np.full(players.shape[1], max_gap), max_ball_gap]
    out, filled = fill_gaps(xys, limit, fps, protect)
    return out[:, :-1], out[:, -1], filled[:, :-1], filled[:, -1], protect


def _group(cell, pitch):
    """Metres back to the ATD's normalised 'x,y,speed'."""
    return "{:.6f},{:.6f},{:.4f}".format(cell[0] / pitch[0], cell[1] / pitch[1], cell[2])


def write_atd(raw_in, raw_out, players, ball, filled_players, filled_ball, pitch):
    """Copy the ATD text file, rewriting only the filled groups. Every other character, line
    endings included, is copied unchanged. Returns how many filled cells had no line to go on."""
    changed = filled_players.any(axis=1) | filled_ball
    written = np.zeros(len(changed), dtype=bool)
    with open(raw_in, newline="") as src, open(raw_out, "w", newline="") as dst:
        for line in src:
            body = line.rstrip("\r\n")
            parts = body.split(":")
            if len(parts) == 3 and changed[int(parts[0]) - 1]:   # ATD frame_id - 1 = video frame
                row = int(parts[0]) - 1
                written[row] = True
                groups = parts[1].split(";")
                for s in np.flatnonzero(filled_players[row]):
                    groups[s] = _group(players[row, s], pitch)
                parts[1] = ";".join(groups)
                if filled_ball[row]:
                    parts[2] = _group(ball[row], pitch)
                line = ":".join(parts) + line[len(body):]
            dst.write(line)
    return int((changed & ~written).sum())


def gap_lengths(tracked):
    """Lengths in frames of every inner gap (tracked on both sides) of a (frames,) mask."""
    edges = np.diff(np.r_[0, (~tracked).astype(np.int8), 0])
    start, stop = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    inner = (start > 0) & (stop < len(tracked))
    return stop[inner] - start[inner]


def _report(name, before, filled, max_gap, fps):
    """Counts only: gaps by length and how many frames were filled."""
    lengths = np.concatenate([gap_lengths(before[:, c]) for c in range(before.shape[1])])
    short = lengths <= max_gap
    print("{:7} {:7d} inner gaps: {:6d} <= {:.0f} s ({:8d} frames), {:5d} longer ({:8d} frames); "
          "filled {:8d} cells".format(name, len(lengths), short.sum(), max_gap / fps,
                                      lengths[short].sum(), (~short).sum(), lengths[~short].sum(),
                                      filled.sum()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--game", default=GAME)
    parser.add_argument("--in", dest="src", help="cleaned bundle to read (default <data>/clean)")
    parser.add_argument("--out", help="folder to write (default <data>/clean_interpolation)")
    parser.add_argument("--max-gap", type=float, default=MAX_GAP_S, help="seconds, players")
    parser.add_argument("--max-ball-gap", type=float, help="seconds, ball (default = --max-gap)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = parser.parse_args()

    fps, pitch = GAMES[args.game]["fps"], GAMES[args.game]["pitch"]
    src = args.src or os.path.join(DATA_DIR, "clean")
    out_dir = args.out or os.path.join(DATA_DIR, "clean_interpolation")
    max_gap = int(round(args.max_gap * fps))
    max_ball_gap = int(round((args.max_gap if args.max_ball_gap is None else args.max_ball_gap) * fps))

    target = asset_paths(args.game, out_dir)
    if os.path.exists(target["atd_raw"]) and not args.force:
        sys.exit("{} exists; pass --force to overwrite it".format(target["atd_raw"]))

    clean = read_tracking(args.game, src)
    raw = load_tracking(args.game)                   # the original ATD, to find keeper.py's blanks
    players, ball, f_players, f_ball, protect = interpolate(
        clean["players"], clean["ball"], max_gap, max_ball_gap, fps, (raw["players"], raw["ball"]))

    print("input  {}\noutput {}".format(src, out_dir))
    print("protected (blanked by keeper.py): {} cells".format(protect.sum()))
    _report("players", ~np.isnan(clean["players"][..., 0]), f_players, max_gap, fps)
    _report("ball", ~np.isnan(clean["ball"][:, :1]), f_ball, max_ball_gap, fps)

    meta_in, raw_in = require(["atd_meta", "atd_raw"], args.game, src)
    os.makedirs(os.path.dirname(target["atd_raw"]), exist_ok=True)
    shutil.copyfile(meta_in, target["atd_meta"])
    lost = write_atd(raw_in, target["atd_raw"], players, ball, f_players, f_ball, pitch)
    if lost:
        print("WARNING: {} frames had filled cells but no line in the ATD".format(lost))

    check = read_tracking(args.game, out_dir)        # never cached
    same = (np.allclose(check["players"], players, atol=1e-2, equal_nan=True)
            and np.allclose(check["ball"], ball, atol=1e-2, equal_nan=True))
    print("saved {}  (reads back the same as the fill: {})".format(target["atd_raw"], same))
