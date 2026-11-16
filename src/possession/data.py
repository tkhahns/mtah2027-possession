"""Load the tactical ATD once into numpy arrays and cache them, so nobody re-parses 199k lines.

    from possession.data import load_tracking
    trk = load_tracking()
    trk["players"]   # (199319, 46, 3) float32: x_m, y_m, speed -- NaN where the slot is empty
    trk["ball"]      # (199319, 3)     float32: x_m, y_m, speed -- NaN where the ball is lost
    trk["team"]      # (46,)           int8:    0 = home (Barca), 1 = away (Athletic)

Row i is video frame i (= submission row i); the ATD's frame_id - 1 is applied here, once.
Coordinates are in metres (0..105.3 along x, 0..68.0 along y), not the raw 0-1.

The cache goes to cache/ (gitignored). It is derived data: never commit it, never share it.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

CACHE_DIR = os.path.join(ROOT, "cache")
HOME, AWAY = 0, 1


def team_of_each_slot(meta_path):
    """0/1 per track slot. First <Team> in the metadata is home (FIFA/EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def _parse_group(group):
    """'x,y,speed' -> three floats ('NaN' parses as nan)."""
    return [float(v) for v in group.split(",")[:3]]


def parse_atd(raw_path, n_slots, n_frames, pitch):
    length, width = pitch
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1  # ATD frame_id counts from 1, video frames from 0
            players[row] = [_parse_group(g) for g in parts[1].split(";")[:n_slots]]
            ball[row] = _parse_group(parts[2])
    scale = np.array([length, width, 1.0], dtype=np.float32)
    return players * scale, ball * scale


def read_tracking(game=GAME, data_dir=None):
    """Same arrays as load_tracking(), straight from the files and never cached: for a cleaned
    copy of the ATD (data/clean) that must not be mixed up with the cache of the original."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    team = team_of_each_slot(meta_path)
    players, ball = parse_atd(raw_path, len(team), source_frames("tactical", game),
                              GAMES[game]["pitch"])
    return {"players": players, "ball": ball, "team": team}


def load_tracking(game=GAME, data_dir=None, refresh=False):
    cache = os.path.join(CACHE_DIR, "{}_tactical_atd.npz".format(game))
    if os.path.exists(cache) and not refresh:
        return dict(np.load(cache))

    trk = read_tracking(game, data_dir)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(cache, **trk)
    return trk
