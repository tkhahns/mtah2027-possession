"""Read the organisers' data bundle: tracking (FIFA/EPTS ATD) and Smart Tagging (SportsCode XML).

The bundle layout (organisers' DATA.md):

    <data>/tactical/BAR-ATH_tactical_atd.xml             metadata: teams and player slots
    <data>/tactical/BAR-ATH_tactical_atd.txt             raw tracking, players and ball
    <data>/tactical/BAR-ATH_tactical_smart_tagging.xml   automatic tags (phases, set pieces)
    <data>/tactical/BAR-ATH_tactical.mp4                 video (only for the optional whistles)

Nothing is cached: the tracking is parsed on every run (about a minute).
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from config import AWAY, FPS, HOME, N_FRAMES, PITCH

STEM = "BAR-ATH_tactical"


def path(data_dir, suffix):
    """Path of one bundle file, with a readable error when it is missing."""
    p = os.path.join(data_dir, "tactical", STEM + suffix)
    if not os.path.isfile(p):
        raise IOError("match data not found: {}\nUnpack the organisers' bundle into {} "
                      "(or pass --data).".format(p, data_dir))
    return p


def team_of_each_slot(meta_path):
    """0/1 per track slot. The first <Team> in the metadata is home (FIFA/EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    home_id = [team.get("id") for team in root.iter("Team")][0]
    return np.array([HOME if p.get("teamId") == home_id else AWAY for p in root.iter("Player")],
                    dtype=np.int8)


def _parse_group(group):
    """'x,y,speed' -> three floats ('NaN' parses as nan)."""
    return [float(v) for v in group.split(",")[:3]]


def parse_atd(raw_path, n_slots, n_frames=N_FRAMES, pitch=PITCH):
    """(players (frames, slots, 3), ball (frames, 3)): x_m, y_m, speed; NaN where not seen."""
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
    scale = np.array([length, width, 1.0], dtype=np.float32)   # the file is in 0-1 units
    return players * scale, ball * scale


def load_tracking(data_dir):
    """{"players", "ball", "team", "pitch"} for the whole tactical video."""
    team = team_of_each_slot(path(data_dir, "_atd.xml"))
    players, ball = parse_atd(path(data_dir, "_atd.txt"), len(team))
    return {"players": players, "ball": ball, "team": team, "pitch": PITCH}


def load_smart_tags(data_dir, fps=FPS):
    """[{code, start_frame, end_frame, tags}] in file order. <start>/<end> are seconds from the
    first frame of the tactical video, so seconds x 25 is a video frame (organisers' starter)."""
    out = []
    for instance in ET.parse(path(data_dir, "_smart_tagging.xml")).iter("instance"):
        out.append({
            "code": instance.findtext("code"),
            "start_frame": int(round(float(instance.findtext("start")) * fps)),
            "end_frame": int(round(float(instance.findtext("end")) * fps)),
            "tags": {label.findtext("group"): label.findtext("text")
                     for label in instance.findall("label")},
        })
    return out
