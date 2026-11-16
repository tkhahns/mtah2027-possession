"""The one command that regenerates the final submission from the provided data.

    python src/make_submission.py [--name TEAM_NAME] [--data-dir DATA_DIR]

Reads the organiser bundle as delivered (tactical and broadcast tracking, broadcast homography,
Smart Tagging, the match video's sound), runs the data pipeline (src/clean_data/pipeline.py) and
the possession pipeline (src/possession/final_pipeline.py) in memory, writes
evaluation/results/<name>.csv, prints its possession count and SHA-256, and validates it with the
organisers' checks (evaluation/validate_submission.py).

The last stage (final_pipeline.clean_up) needs the referee's whistles from the match sound: the
first run decodes it once into cache/whistle/ with ffmpeg (on PATH, or the imageio-ffmpeg package
in requirements-submission.txt), later runs read the cache. It also reads the Smart Tagging set
pieces.

Reference output (GPU: torch 2.6.0+cu124 on an RTX 4070 Ti SUPER, the pinned
requirements-submission.txt, Python 3.13.16, Windows 11), reproduced byte-for-byte on two runs:

    evaluation/results/mtah_2027_possession.csv   199319 rows, 233 possessions of mean 13.8 s
    sha256 8916418653fa775a5453663132dceed3257ad03870372e6c25a47a1dc6c87b74
"""

import argparse
import hashlib
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "starter")]

from games import DATA_DIR, GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
import load_homography  # noqa: E402
from possession import final_pipeline, rule_cleanup, whistle_cleanup  # noqa: E402
from possession.data import read_tracking, parse_atd, team_of_each_slot  # noqa: E402
from possession.labels import graded_mask, possession_summary, write_submission  # noqa: E402

TEAM_NAME = "mtah_2027_possession"


def load_tactical(data_dir):
    tracking = read_tracking(GAME, data_dir)
    return {**tracking, "pitch": GAMES[GAME]["pitch"]}


def load_broadcast(data_dir):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir, "broadcast")
    team = team_of_each_slot(meta_path)
    players, ball = parse_atd(
        raw_path, len(team), source_frames("broadcast", GAME), GAMES[GAME]["pitch"])
    return {"players": players, "ball": ball, "team": team, "pitch": GAMES[GAME]["pitch"]}


def build_labels(data_dir=DATA_DIR, whistle_cache=None, report=None):
    tracking = load_tactical(data_dir)
    _, _, broadcast_valid = load_homography.load(
        game=GAME, data_dir=data_dir, source="broadcast")
    labels = final_pipeline.predict(
        tracking, load_broadcast(data_dir), broadcast_valid, half_video_blocks(GAME),
        whistles=whistle_cleanup.load_whistles(data_dir, whistle_cache, game=GAME),
        set_pieces=rule_cleanup.load_set_pieces(GAME, data_dir), report=report)
    labels[~graded_mask(GAME)] = "X"
    return labels


def describe(report):
    """One line of counts from final_pipeline.predict's report."""
    moved = report.get("players", {})
    filled = report.get("interpolated", {})
    reckoned = report.get("extrapolated", {})
    held = filled.get("keeper_hold", {})
    given = report.get("keeper_possession", {})
    return ("data pipeline: {} player-team moves ({:.1f} s moved, {} skipped); keeper holds the "
            "ball in {} gaps ({} cells); interpolated {} player / {} ball cells; ball dead "
            "reckoning {} of {} long gaps, {} cells; broadcast shifts {}; keeper's possession {} "
            "of {} holds ({} frames changed, {} after a whistle)".format(
                               moved.get("moves", 0), moved.get("frames_moved", 0) / 25,
                               moved.get("skipped", 0), held.get("gaps", 0),
                               held.get("ball_cells", 0), filled.get("player_cells", 0),
                               filled.get("ball_cells", 0), *reckoned.get("ball_gaps", (0, 0)),
                               reckoned.get("ball_cells", 0), report.get("broadcast_shifts"),
                               given.get("given to the keeper's team", 0), given.get("holds", 0),
                               given.get("frames changed", 0),
                               given.get("whistle before: left alone", 0)))


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=TEAM_NAME)
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help="provided data bundle (default: data/ or MTAH_DATA_DIR)")
    parser.add_argument("--whistle-cache", default=None,
                        help="decoded match sound (default: cache/whistle)")
    args = parser.parse_args()

    report = {}
    labels = build_labels(args.data_dir, args.whistle_cache, report)
    print(describe(report))
    path = write_submission(os.path.join(ROOT, "evaluation", "results", args.name + ".csv"), labels)
    print("wrote", os.path.relpath(path, ROOT), possession_summary(labels))
    print("sha256", sha256(path))
    sys.exit(subprocess.call([sys.executable,
                              os.path.join(ROOT, "evaluation", "validate_submission.py"), path]))
