"""Load the public IDSSE Bundesliga tracking matches and their provider ball state.

IDSSE publishes DFL tracking at 25 Hz.  The ball stream includes ``BallStatus`` (1 live,
0 dead) and ``BallPossession`` (1 home, 2 away), so these matches supply frame-accurate
supervision without reconstructing stoppages from sparse event timestamps.

Raw files are downloaded into ``external/idsse`` and cached there.  That directory is ignored
by git.  Coordinates are converted from the DFL centre-origin convention to the repository's
0..length / 0..width convention; speeds are converted from km/h to m/s.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from .data import AWAY, HOME

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IDSSE_DIR = os.path.join(ROOT, "external", "idsse")
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
PITCH = (105.0, 68.0)
SECTIONS = ("firstHalf", "secondHalf")
HALFTIME_GAP = 25
CACHE_VERSION = 2
MIN_CONTROL_FRAMES = 25


def paths(match, root=IDSSE_DIR):
    """Return metadata, event and tracking paths for one IDSSE match."""
    match = str(match)
    return tuple(os.path.join(root, "{}_{}.xml".format(match, suffix))
                 for suffix in ("meta", "events", "tracking"))


def available(match, root=IDSSE_DIR):
    """Whether all three public files for ``match`` have been downloaded."""
    return all(os.path.isfile(path) for path in paths(match, root))


def available_matches(root=IDSSE_DIR):
    return tuple(match for match in MATCHES if available(match, root))


def _local(tag):
    return tag.rsplit("}", 1)[-1]


def _metadata(path):
    root = ET.parse(path).getroot()
    general = next(item for item in root.iter() if _local(item.tag) == "General")
    home_id = general.attrib["HomeTeamId"]
    away_id = general.attrib["GuestTeamId"]
    people = {}
    for team in (item for item in root.iter() if _local(item.tag) == "Team"):
        team_id = team.attrib["TeamId"]
        code = HOME if team_id == home_id else AWAY if team_id == away_id else None
        if code is None:
            continue
        for player in (item for item in team.iter() if _local(item.tag) == "Player"):
            people[player.attrib["PersonId"]] = code
    return people


def _scan_ball(path, people):
    """First streaming pass: collect frame bounds, present players and ball supervision."""
    current = None
    present = set()
    ball = {section: [] for section in SECTIONS}
    for event, element in ET.iterparse(path, events=("start", "end")):
        tag = _local(element.tag)
        if event == "start" and tag == "FrameSet":
            current = (element.attrib.get("GameSection"), element.attrib.get("TeamId"),
                       element.attrib.get("PersonId"))
            if current[0] in SECTIONS and current[2] in people:
                present.add(current[2])
        elif event == "end" and tag == "Frame":
            if current and current[0] in SECTIONS and current[1] == "BALL":
                ball[current[0]].append((
                    int(element.attrib["N"]), float(element.attrib["X"]),
                    float(element.attrib["Y"]), float(element.attrib.get("S", "nan")),
                    int(element.attrib["BallStatus"]),
                    int(element.attrib["BallPossession"])))
            element.clear()
        elif event == "end" and tag == "FrameSet":
            current = None
            element.clear()
    if not present:
        raise ValueError("IDSSE tracking has no recognized player FrameSets")
    for section in SECTIONS:
        if not ball[section]:
            raise ValueError("IDSSE tracking is missing the {} ball stream".format(section))
        numbers = np.asarray([row[0] for row in ball[section]])
        expected = np.arange(numbers[0], numbers[-1] + 1)
        if not np.array_equal(numbers, expected):
            raise ValueError("IDSSE {} ball frames are not continuous".format(section))
    return sorted(present, key=lambda person: (people[person], person)), ball


def _read_tracking(path, people, player_ids, raw_ball):
    section_length = {section: len(raw_ball[section]) for section in SECTIONS}
    first_frame = {section: raw_ball[section][0][0] for section in SECTIONS}
    section_players = {
        section: np.full((section_length[section], len(player_ids), 3), np.nan, np.float32)
        for section in SECTIONS
    }
    column = {person: index for index, person in enumerate(player_ids)}
    current = None
    for event, element in ET.iterparse(path, events=("start", "end")):
        tag = _local(element.tag)
        if event == "start" and tag == "FrameSet":
            current = (element.attrib.get("GameSection"), element.attrib.get("PersonId"))
        elif event == "end" and tag == "Frame":
            if current and current[0] in SECTIONS and current[1] in column:
                row = int(element.attrib["N"]) - first_frame[current[0]]
                if 0 <= row < section_length[current[0]]:
                    section_players[current[0]][row, column[current[1]]] = (
                        float(element.attrib["X"]) + PITCH[0] / 2.0,
                        float(element.attrib["Y"]) + PITCH[1] / 2.0,
                        float(element.attrib.get("S", "nan")) / 3.6)
            element.clear()
        elif event == "end" and tag == "FrameSet":
            current = None
            element.clear()

    sections = []
    truths = []
    offsets = []
    cursor = 0
    for section in SECTIONS:
        rows = raw_ball[section]
        ball = np.asarray([[x + PITCH[0] / 2.0, y + PITCH[1] / 2.0, speed / 3.6]
                           for _, x, y, speed, _, _ in rows], dtype=np.float32)
        truth = np.asarray([
            "D" if not status else "H" if possession == 1 else "A" if possession == 2 else "X"
            for _, _, _, _, status, possession in rows], dtype="<U1")
        truth = control_truth(truth)
        sections.append((section_players[section], ball))
        truths.append(truth)
        offsets.append((cursor, cursor + len(truth)))
        cursor += len(truth) + HALFTIME_GAP

    gap_players = np.full((HALFTIME_GAP, len(player_ids), 3), np.nan, np.float32)
    gap_ball = np.full((HALFTIME_GAP, 3), np.nan, np.float32)
    players = np.concatenate([sections[0][0], gap_players, sections[1][0]])
    ball = np.concatenate([sections[0][1], gap_ball, sections[1][1]])
    truth = np.concatenate([truths[0], np.full(HALFTIME_GAP, "X", dtype="<U1"), truths[1]])
    team = np.asarray([people[person] for person in player_ids], dtype=np.int8)
    return players, ball, team, truth, np.asarray(offsets, dtype=np.int32)


def _runs(labels):
    labels = np.asarray(labels)
    edges = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1], True])
    return zip(edges[:-1], edges[1:])


def control_truth(truth, min_frames=MIN_CONTROL_FRAMES):
    """Remove sub-second touch reversals from provider possession.

    DFL possession can switch for a deflection or contested touch.  The challenge retains the
    previous controller in that situation.  Only an H/A island bracketed by the same live team
    is merged; dead-ball boundaries and sustained transfers are untouched.
    """
    labels = np.asarray(truth).copy()
    while True:
        changed = False
        for start, stop in list(_runs(labels)):
            if (stop - start < int(min_frames) and labels[start] in "HA" and start > 0 and
                    stop < len(labels) and labels[start - 1] == labels[stop] and
                    labels[stop] in "HA"):
                labels[start:stop] = labels[start - 1]
                changed = True
        if not changed:
            return labels


def load_match(match, root=IDSSE_DIR, refresh=False):
    """Return ``(tracking, truth)`` for a downloaded public IDSSE match."""
    match = str(match)
    cache = os.path.join(root, "cache", "{}_v{}.npz".format(match, CACHE_VERSION))
    if os.path.isfile(cache) and not refresh:
        saved = np.load(cache)
        trk = {key: saved[key] for key in ("players", "ball", "team")}
        trk.update({"pitch": PITCH, "always_dead": [], "halves": saved["halves"]})
        return trk, saved["truth"]
    metadata_path, _, tracking_path = paths(match, root)
    if not os.path.isfile(metadata_path) or not os.path.isfile(tracking_path):
        raise FileNotFoundError("download IDSSE match {} first".format(match))
    people = _metadata(metadata_path)
    player_ids, raw_ball = _scan_ball(tracking_path, people)
    players, ball, team, truth, halves = _read_tracking(
        tracking_path, people, player_ids, raw_ball)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, players=players, ball=ball, team=team, truth=truth, halves=halves)
    trk = {"players": players, "ball": ball, "team": team, "pitch": PITCH,
           "always_dead": [], "halves": halves}
    return trk, truth


def load(match, kind="clean", root=IDSSE_DIR, refresh=False):
    """Load clean, hidden-ball or messy tracking with the same provider truth."""
    if kind not in ("clean", "hidden", "messy"):
        raise ValueError("kind must be clean, hidden or messy")
    trk, truth = load_match(match, root, refresh=refresh)
    if kind == "clean":
        return trk, truth
    # Imported lazily to keep the raw-data loader independent of perturbation code.
    from . import testsets
    seed = 2718 + sum((index + 1) * ord(char) for index, char in enumerate(str(match)))
    out = testsets.hide_ball(trk, truth, seed=seed)
    if kind == "messy":
        out = testsets.make_messy(out, seed=seed)
    return testsets.with_speed(out), truth
