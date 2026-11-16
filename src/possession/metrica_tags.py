"""Smart-Tagging-like tags for the Metrica sample games, so models that use our Smart Tagging can
be scored on the Metrica test sets too.

    from possession import metrica_tags
    tags = metrica_tags.smart_tags(1)              # same dicts as starter/load_smart_tagging.load()
    metrica_tags.write_sportscode(tags, path)      # or as a SportsCode XML file, like ours

Metrica has no Smart Tagging, so the tags are made from its event data: SET PIECES (throw-ins,
corners, goal kicks, free kicks, kick-offs, penalties) and GOALS, which is what our file gives
reliably. Phases (BUILD UP, PROGRESSION...) aren't made. To look like ours rather than like the
truth, they get the faults our tagging has (rates in TAGS below are guesses, not fitted):

- a window of ~5 s around the restart, not the instant (see starter/load_smart_tagging.py);
- every throw-in has Team "N/A"; other tags are sometimes given the wrong team;
- many restarts aren't tagged (ours has 5 free kicks in a whole match), and a few tags are false.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from . import metrica

FPS = 25
TEAM_NAME = {"H": "Barcelona", "A": "Atletic Club"}
HALF_NAME = {1: "1st Half", 2: "2nd Half"}

TAGS = {
    # share of restarts that get a tag, per Type
    "recall": {"Throw-in": 0.6, "Corner Kick": 0.9, "Goal Kick": 0.6, "Free Kick": 0.25,
               "Kick Off (Start)": 1.0, "Kick Off (After Goal)": 1.0, "Penalty": 1.0, "Goals": 1.0},
    "before_s": (1.5, 3.5),           # window starts this long before the restart...
    "after_s": (1.0, 3.0),            # ...and ends this long after it
    "goal_before_s": (6.0, 10.0),     # GOALS windows: the build-up to the goal...
    "goal_after_s": (1.0, 3.0),
    "wrong_team_p": 0.05,
    "false_per_half": 1.5,            # tags at moments with no restart
}


def _restart_type(e, kickoff_seen):
    """Our Type for a Metrica event that restarts play, or None."""
    sub = e["subtype"]
    if e["type"] == "SET PIECE":
        if sub.startswith("THROW IN"):
            return "Throw-in"
        if sub.startswith("CORNER KICK"):
            return "Corner Kick"
        if sub.startswith("FREE KICK"):
            return "Free Kick"
        if sub.startswith("PENALTY"):
            return "Penalty"
        if sub.startswith("KICK OFF"):
            return "Kick Off (After Goal)" if kickoff_seen else "Kick Off (Start)"
    if e["type"] in ("PASS", "BALL LOST") and sub.startswith("GOAL KICK"):
        return "Goal Kick"
    return None


def _is_goal(e):
    return e["type"] == "SHOT" and ("-GOAL" in e["subtype"] or e["subtype"].endswith("GOAL"))


def _tag(code, start_frame, end_frame, half, team=None, kind=None):
    start_frame = max(0, int(start_frame))
    tags = {"Team": TEAM_NAME.get(team, "N/A"), "Half": HALF_NAME.get(half, str(half))}
    if kind:
        tags["Type"] = kind
    return {"code": code, "start_frame": start_frame, "end_frame": int(end_frame),
            "start_s": start_frame / FPS, "end_s": int(end_frame) / FPS, "tags": tags}


def tags_from_events(events, seed=1729, cfg=None):
    """Fake Smart Tagging from Metrica events (see the module docstring), sorted by start."""
    cfg = {**TAGS, **(cfg or {})}
    rng = np.random.default_rng(seed)
    out = []
    for period in sorted({e["period"] for e in events}):
        evs = sorted((e for e in events if e["period"] == period), key=lambda e: e["start"])
        kickoff_seen = False
        for e in evs:
            kind = "Goals" if _is_goal(e) else _restart_type(e, kickoff_seen)
            if kind is None or not e["team"]:
                continue
            kickoff_seen |= kind.startswith("Kick Off")
            if rng.random() >= cfg["recall"][kind]:
                continue
            team = e["team"]
            if rng.random() < cfg["wrong_team_p"]:
                team = "A" if team == "H" else "H"
            if kind == "Throw-in":
                team = None
            if kind == "Goals":
                before, after = cfg["goal_before_s"], cfg["goal_after_s"]
            else:
                before, after = cfg["before_s"], cfg["after_s"]
            out.append(_tag("GOALS" if kind == "Goals" else "SET PIECES",
                            e["start"] - rng.uniform(*before) * FPS, e["start"] + rng.uniform(*after) * FPS,
                            period, team, None if kind == "Goals" else kind))
        first, last = evs[0]["start"], evs[-1]["start"]
        for _ in range(rng.poisson(cfg["false_per_half"])):
            at = rng.integers(first, last)
            kind = str(rng.choice(["Throw-in", "Free Kick", "Corner Kick"]))
            team = None if kind == "Throw-in" else str(rng.choice(["H", "A"]))
            out.append(_tag("SET PIECES", at - 2.5 * FPS, at + 2.5 * FPS, period, team, kind))
    return sorted(out, key=lambda t: t["start_frame"])


def smart_tags(game, root=metrica.METRICA_DIR, seed=None):
    """Fake Smart Tagging for one Metrica game; deterministic per game."""
    return tags_from_events(metrica.read_events(game, root), seed=1729 + game if seed is None else seed)


def write_sportscode(tags, path):
    """Write tags as a SportsCode XML file that starter/load_smart_tagging.load() can read."""
    root = ET.Element("file")
    instances = ET.SubElement(root, "ALL_INSTANCES")
    for i, t in enumerate(tags, 1):
        inst = ET.SubElement(instances, "instance")
        ET.SubElement(inst, "ID").text = str(i)
        ET.SubElement(inst, "start").text = "{:.2f}".format(t["start_s"])
        ET.SubElement(inst, "end").text = "{:.2f}".format(t["end_s"])
        ET.SubElement(inst, "code").text = t["code"]
        for group, text in t["tags"].items():
            label = ET.SubElement(inst, "label")
            ET.SubElement(label, "group").text = group
            ET.SubElement(label, "text").text = text
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
