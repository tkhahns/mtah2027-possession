"""Metrica Sports' public sample games (other matches), a test set to compare pipeline variants.

    from possession import metrica
    trk = metrica.load_game(3)        # same dict shape as data.load_tracking(), plus "pitch"
    truth = metrica.load_truth(3)     # H / A / D / X per frame, worked out from the events

The rules forbid tuning on our own match (and any human labels of it), so pipeline variants are
compared on these games instead. Fetch them once with src/training/fetch_metrica.py. They are public,
so unlike data/ they may be looked at; still never commit them (external/ is gitignored).
"""

import csv
import json
import os
import xml.etree.ElementTree as ET

import numpy as np

from .data import AWAY, HOME

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
METRICA_DIR = os.path.join(ROOT, "external", "metrica")
GAMES = (1, 2, 3)
PITCH = (105.0, 68.0)

# Events that show a team on the ball at their start frame. Left out on purpose:
# - CHALLENGE: a duel or touch without control does not change possession (organisers' rule);
# - BALL LOST: it is the loser's record of a turnover, often logged a few frames after the
#   winner's RECOVERY, which would hand the ball back to the loser for a moment.
ON_BALL = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE", "BALL OUT"}
# A spell shorter than this with none of these events by that team is a touch without control
# (a header, a deflection, a save, a clearance): the previous owner keeps the ball. Game 3 logs a
# CARRY for every touch, so a CARRY alone is not control.
CONTROL = {"PASS", "SHOT", "SET PIECE"}
MIN_SPELL_FRAMES = 25   # 1 s


def game_dir(game, root=METRICA_DIR):
    return os.path.join(root, "Sample_Game_{}".format(game))


def available(game, root=METRICA_DIR):
    """True when src/training/fetch_metrica.py has downloaded this game."""
    names = (["Sample_Game_3_events.json", "Sample_Game_3_metadata.xml", "Sample_Game_3_tracking.txt"]
             if game == 3 else ["Sample_Game_{}_RawEventsData.csv".format(game)] +
             ["Sample_Game_{}_RawTrackingData_{}_Team.csv".format(game, side) for side in ("Home", "Away")])
    return all(os.path.isfile(os.path.join(game_dir(game, root), n)) for n in names)


# -- events -> truth ---------------------------------------------------------------------------

def read_events(game, root=METRICA_DIR):
    """[{team: "H"/"A"/None, type, subtype, period, start, end}], frames 0-based like our rows."""
    folder = game_dir(game, root)
    out = []
    if game == 3:
        with open(os.path.join(folder, "Sample_Game_3_events.json")) as handle:
            data = json.load(handle)["data"]
        home_id = _game3_home_id(root)
        for e in data:
            sub = e.get("subtypes")
            subs = [s["name"] for s in sub] if isinstance(sub, list) else [sub["name"]] if sub else []
            team = (e.get("team") or {}).get("id")
            out.append({"team": None if team is None else "H" if team == home_id else "A",
                        "type": e["type"]["name"].upper(), "subtype": "-".join(subs).upper(),
                        "period": int(e["period"]), "start": int(e["start"]["frame"]) - 1,
                        "end": int(e["end"]["frame"] or 0) - 1})
    else:
        path = os.path.join(folder, "Sample_Game_{}_RawEventsData.csv".format(game))
        with open(path, newline="") as handle:
            for row in csv.DictReader(handle):
                out.append({"team": {"Home": "H", "Away": "A"}.get(row["Team"]),
                            "type": row["Type"].upper(), "subtype": row["Subtype"].upper(),
                            "period": int(row["Period"]), "start": int(row["Start Frame"]) - 1,
                            "end": int(row["End Frame"]) - 1})
    return out


def _set_piece_next(nxt):
    return nxt is not None and nxt["type"] == "SET PIECE"


def _is_restart(e, nxt=None):
    """An event that puts a dead ball back in play: a kick by a team. A set piece (throw-in,
    corner, free kick, kick-off), a pass, or a BALL LOST, which is an unsuccessful pass: Metrica
    often logs a goal kick as RECOVERY + BALL LOST "GOAL KICK-INTERCEPTION" (the keeper's kick,
    won by the other side). A BALL LOST just before a set piece is not a kick: it is the fouled
    player's record of the stoppage. nxt = the next on-ball event that isn't a RECOVERY."""
    if e["type"] == "BALL LOST" and "GOAL KICK" not in e["subtype"] and _set_piece_next(nxt):
        return False
    return bool(e["team"]) and e["type"] in ("SET PIECE", "PASS", "BALL LOST")


def _stops_play(e, nxt):
    """Frame (or None) at which this event makes the ball dead. nxt = the next on-ball event that
    isn't a RECOVERY (someone fetching the ball during a stoppage)."""
    end = max(e["start"], e["end"])
    if e["type"] == "BALL OUT":
        return end
    if e["type"] == "SHOT" and ("OUT" in e["subtype"] or "GOAL" in e["subtype"]):
        return end
    if "OFFSIDE" in e["subtype"]:
        return end
    if "FORCED" in e["subtype"]:                # the referee halts play
        return e["start"]
    # A foul or handball stops play only if a set piece (the free kick) follows; otherwise
    # advantage was played. Metrica logs fouls as FAULT RECEIVED or as CHALLENGE with FAULT.
    foul = e["type"] == "FAULT RECEIVED" or "HAND BALL" in e["subtype"] or (
        e["type"] == "CHALLENGE" and "FAULT" in e["subtype"] and "ADVANTAGE" not in e["subtype"])
    if foul and _set_piece_next(nxt):
        return e["start"]
    return None


def _line_crossing(ball, start, end, pitch):
    """First frame in [start, end] where the tracked ball is beyond a line, else None.

    The organisers' truth is "dead where the tracking marks the ball dead"; Metrica's BALL OUT is
    logged at its end, a median 2-6 frames after the tracked ball has crossed the line."""
    if ball is None:
        return None
    length, width = pitch
    seg = ball[max(0, start):end + 1, :2]
    with np.errstate(invalid="ignore"):
        out = (seg[:, 0] < 0) | (seg[:, 0] > length) | (seg[:, 1] < 0) | (seg[:, 1] > width)
    hit = np.flatnonzero(out)
    return int(max(0, start) + hit[0]) if len(hit) else None


def truth_from_events(events, n_frames, ball=None, pitch=PITCH):
    """Possession per frame from timed events (and, if given, Metrica's ball tracking):

    - live: a team has the ball from the start of its last on-ball event (pass, carry, shot,
      recovery, set piece, ball out); challenges and BALL LOST don't change it;
    - a spell under 1 s with no pass, shot or set piece by that team is a touch without control:
      the team that had the ball before keeps it;
    - dead (D) from a ball out, a shot that goes out or in, an offside, the referee halting play
      (FORCED), or a foul or handball followed by a free kick, until the RESTART: the next set
      piece or pass. Recoveries and carries during a stoppage (a keeper fetching the ball behind
      the goal) don't restart play;
    - a set piece always follows a dead ball: if no stoppage was logged, dead from the end of the
      last event before it;
    - a ball out starts where the tracked ball first crosses a line, when the ball is given;
    - X (not graded) before the first and after the last event of each half, and between halves.
    """
    labels = np.full(n_frames, "X", dtype="<U1")
    for period in sorted({e["period"] for e in events}):
        evs = sorted((e for e in events if e["period"] == period), key=lambda e: (e["start"], e["end"]))
        on_ball = [e for e in evs if e["type"] in ON_BALL and e["team"]]
        changes, dead_until = [], None          # dead_until: frame the current stoppage started
        for e in evs:
            nxt = next((x for x in on_ball if x["start"] > e["start"] and x["type"] != "RECOVERY"), None)
            if dead_until is not None:                       # ball dead: only a restart ends it
                if _is_restart(e, nxt) and e["start"] >= dead_until:
                    changes.append((e["start"], 0, e["team"]))
                    dead_until = None
                else:
                    continue
            elif e["type"] in ON_BALL and e["team"]:
                if e["type"] == "SET PIECE":                 # no stoppage logged before it
                    before = [max(x["start"], x["end"]) for x in evs if x["start"] < e["start"]]
                    if before and max(before) < e["start"]:
                        changes.append((max(before), 1, "D"))
                changes.append((e["start"], 0, e["team"]))
            stop = _stops_play(e, nxt)
            if stop is not None:
                if e["type"] in ("BALL OUT", "SHOT"):
                    stop = _line_crossing(ball, e["start"], stop, pitch) or stop
                changes.append((stop, 1, "D"))
                dead_until = stop
        if not changes:
            continue
        changes.sort()
        lo = max(0, min(e["start"] for e in evs))
        hi = min(n_frames - 1, max(max(e["start"], e["end"]) for e in evs))
        frames = np.arange(lo, hi + 1)
        at = np.searchsorted([c[0] for c in changes], frames, side="right") - 1
        known = at >= 0
        labels[frames[known]] = np.array([c[2] for c in changes])[at[known]]
    return _drop_touches(labels, events)


def _drop_touches(labels, events):
    """Give short spells without a controlled action back to the team that had the ball before."""
    starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    ends = np.r_[starts[1:], len(labels)]
    for s, e in zip(starts, ends):
        lab, prev = labels[s], labels[s - 1] if s else "X"     # prev sees earlier merges
        if lab not in "HA" or prev not in "HA" or prev == lab or e - s >= MIN_SPELL_FRAMES:
            continue
        if not any(ev["team"] == lab and ev["type"] in CONTROL and s <= ev["start"] < e
                   for ev in events):
            labels[s:e] = prev
    return labels


def load_truth(game, root=METRICA_DIR):
    ball = load_game(game, root)["ball"]
    return truth_from_events(read_events(game, root), len(ball), ball)


def halves(events):
    """[(period, first_frame, last_frame)] from the first to the last event of each half: the
    same shape as games.half_video_blocks() for our match."""
    return [(p, min(e["start"] for e in events if e["period"] == p),
             max(max(e["start"], e["end"]) for e in events if e["period"] == p))
            for p in sorted({e["period"] for e in events})]


# -- tracking ----------------------------------------------------------------------------------

def _scale(xy):
    return xy * np.array(PITCH, dtype=np.float32)


def _with_speed(xy):
    """(..., 2) metres -> (..., 3) with a NaN speed column, the shape data.load_tracking() gives."""
    return np.concatenate([xy, np.full(xy.shape[:-1] + (1,), np.nan, dtype=np.float32)], axis=-1)


def _read_csv_tracking(path):
    """Metrica CSV (3 header rows): (players (frames, n, 2), ball (frames, 2)), 0-1 units."""
    values = np.genfromtxt(path, delimiter=",", skip_header=3, dtype=np.float32)
    frames = values[:, 1].astype(np.int64) - 1
    out_players = np.full((frames.max() + 1, (values.shape[1] - 5) // 2, 2), np.nan, np.float32)
    out_ball = np.full((frames.max() + 1, 2), np.nan, np.float32)
    out_players[frames] = values[:, 3:-2].reshape(len(values), -1, 2)
    out_ball[frames] = values[:, -2:]
    return out_players, out_ball


def _game3_meta(root):
    tree = ET.parse(os.path.join(game_dir(3, root), "Sample_Game_3_metadata.xml")).getroot()
    home_id = tree.find(".//Score").get("idLocalTeam")
    players = {p.get("id"): p.get("teamId") for p in tree.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in tree.iter("PlayerChannel")}
    specs = []
    for spec in tree.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      [channel_player[r] for r in refs[::2]]))   # x channel of each slot
    return home_id, players, specs


def _game3_home_id(root):
    return _game3_meta(root)[0]


def _read_game3_tracking(root):
    """EPTS text, one column per player (slots change with substitutions)."""
    home_id, players, specs = _game3_meta(root)
    order = sorted(players)
    column = {pid: i for i, pid in enumerate(order)}
    team = np.array([HOME if players[p] == home_id else AWAY for p in order], dtype=np.int8)
    with open(os.path.join(game_dir(3, root), "Sample_Game_3_tracking.txt")) as handle:
        lines = [line.strip() for line in handle if line.strip()]
    ids = np.array([int(line.split(":", 1)[0]) for line in lines])
    n_frames = ids.max()
    out_players = np.full((n_frames, len(order), 2), np.nan, np.float32)
    out_ball = np.full((n_frames, 2), np.nan, np.float32)
    for first, last, slots in specs:
        pick = np.flatnonzero((ids >= first) & (ids <= last))
        if not len(pick):
            continue
        text = "\n".join(lines[i].replace(":", ",").replace(";", ",") for i in pick)
        values = np.loadtxt(text.splitlines(), delimiter=",", dtype=np.float32, ndmin=2)
        rows = values[:, 0].astype(np.int64) - 1
        xy = values[:, 1:1 + 2 * len(slots)].reshape(len(rows), len(slots), 2)
        cols = [column[p] for p in slots]
        out_players[rows[:, None], np.array(cols)[None, :]] = xy
        out_ball[rows] = values[:, -2:]
    return out_players, out_ball, team


def load_game(game, root=METRICA_DIR, refresh=False):
    """Tracking of one sample game as {players, ball, team, pitch, always_dead}, metres, row =
    frame - 1. "pitch" and "always_dead" replace our match's settings in pipeline.dead_ball.
    Cached next to the files (external/metrica/cache/, gitignored)."""
    cache = os.path.join(root, "cache", "game{}.npz".format(game))
    if os.path.exists(cache) and not refresh:
        return {**np.load(cache), "pitch": PITCH, "always_dead": []}
    if game == 3:
        players, ball, team = _read_game3_tracking(root)
    else:
        stem = os.path.join(game_dir(game, root), "Sample_Game_{}_RawTrackingData_".format(game))
        home, ball = _read_csv_tracking(stem + "Home_Team.csv")
        away, _ = _read_csv_tracking(stem + "Away_Team.csv")
        n = min(len(home), len(away))
        players, ball = np.concatenate([home[:n], away[:n]], axis=1), ball[:n]
        team = np.array([HOME] * home.shape[1] + [AWAY] * away.shape[1], dtype=np.int8)
    out = {"players": _with_speed(_scale(players)), "ball": _with_speed(_scale(ball)), "team": team}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, **out)
    return {**out, "pitch": PITCH, "always_dead": []}
