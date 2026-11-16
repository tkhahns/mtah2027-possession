"""Rule-based clean-up after the final pipeline
  "whistle" the referee's whistle from the match sound: a foul
           whistle makes it dead until the predicted labels change; a restart whistle keeps it dead
           until the ball moves, then live for the nearest player's team until they change. Needs
           a whistle_cleanup.
  "fill"   a D run shorter than short_dead_frames takes the label of the H/A run just before
           it: the team that had the ball keeps it through a short stoppage. A D run with nothing
           live before it in the half stays D.
  "tags"   (vD around every corner kick / goal kick / throw-in / free kick tag start in Smart
           Tagging, whatever came before: [start - pre, start + tag_after_frames), pre = corner
           15 s, goal kick 5 s, free kick 5 s, throw-in 0.
  "corner" the corner is taken: at a corner kick tag start the ball is usually missing from the
           organisers' tracking and the labels are dead. From the first frame at or after the tag
           start where the tracking has the ball again, the team attacking the end of the pitch
           the ball is in (the field half) has it, up to the next change in the labels after the
           tag start. If the labels change before the ball is back, nothing changes. Runs after
           "tags", so a ball back inside the tag window wins over it. 
  "merge"  every H or A run shorter than min_possession_frames joins its longer neighbour,
           shortest first. It never turns D into H/A, so the tag windows stay D.
"""

from collections import Counter
from dataclasses import dataclass
import importlib.util
import os

import numpy as np

from clean_data import team_fit

from . import whistle_cleanup
from .data import HOME

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RULES = ("fill", "tags", "merge")       # the default: the ones that need only the labels
ORDER = ("whistle", "fill", "tags", "corner", "merge")
LIVE = ("H", "A")
TAG_KINDS = ("corner kick", "goal kick", "free kick", "throw-in")
CORNER = "corner kick"


@dataclass(frozen=True)
class Settings:
    fps: int = 25
    short_dead_frames: int = 80       # "fill": D runs shorter than this
    tag_after_frames: int = 125       # "tags": D this many frames from the tag start on...
    corner_pre_s: float = 15.0        # ...and this long before a corner kick tag start
    goal_kick_pre_s: float = 5.0      # ... before a goal kick tag start
    free_kick_pre_s: float = 5.0      # ... before a free kick tag start
    throw_in_pre_s: float = 0.0       # ... before a throw-in tag start
    min_possession_frames: int = 45   # "merge": H/A runs shorter than this

    def pre_frames(self, kind):
        seconds = {"corner kick": self.corner_pre_s, "goal kick": self.goal_kick_pre_s,
                   "free kick": self.free_kick_pre_s, "throw-in": self.throw_in_pre_s}[kind]
        return int(round(seconds * self.fps))


def label_runs(labels):
    """[[label, start, stop], ...] maximal runs, half-open."""
    labels = np.asarray(labels)
    if not len(labels):
        return []
    cuts = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = np.concatenate(([0], cuts, [len(labels)]))
    return [[str(labels[s]), int(s), int(e)] for s, e in zip(bounds[:-1], bounds[1:])]


def fill_short_dead(labels, min_frames):
    """"fill": every D run shorter than min_frames takes the label of the H/A run just before it.
    A D run with no H/A run before it (start of the labels, or after X) stays D. 0 = off."""
    out = np.array(labels, copy=True)
    if min_frames <= 0:
        return out
    prev = None
    for lab, s, e in label_runs(out):
        if lab == "D" and e - s < min_frames and prev in LIVE:
            out[s:e] = prev                  # now part of the run before, so prev stays
            continue
        prev = lab
    return out


def merge_short_spells(labels, min_frames):
    """"merge" (R20): every H or A run shorter than min_frames joins its longer neighbour, shortest
    first. 0 = off."""
    runs = label_runs(labels)
    while len(runs) > 1 and min_frames > 0:
        short = [i for i, (lab, s, e) in enumerate(runs) if lab in LIVE and e - s < min_frames]
        if not short:
            break
        i = min(short, key=lambda k: runs[k][2] - runs[k][1])
        j = max((k for k in (i - 1, i + 1) if 0 <= k < len(runs)),
                key=lambda k: runs[k][2] - runs[k][1])
        runs[i][0] = runs[j][0]
        merged = []
        for run in runs:
            if merged and merged[-1][0] == run[0]:
                merged[-1][2] = run[2]
            else:
                merged.append(run)
        runs = merged
    out = np.array(labels, copy=True)
    for lab, s, e in runs:
        out[s:e] = lab
    return out


def tag_windows(set_pieces, lo, hi, settings=Settings()):
    """"tags": [(start, stop, kind)] half-open frame windows to make D, one per corner kick / goal
    kick / free kick / throw-in tag starting in the half [lo, hi] (inclusive), clipped to it, in
    time order. set_pieces: [(start frame, kind)] (load_set_pieces)."""
    out = []
    for start, kind in sorted(set_pieces):
        if kind not in TAG_KINDS or not lo <= start <= hi:
            continue
        a = max(lo, start - settings.pre_frames(kind))
        b = min(hi + 1, start + max(0, settings.tag_after_frames))
        if b > a:
            out.append((a, b, kind))
    return out


def force_dead(labels, windows, offset=0, report=None):
    """Every (start, stop, kind) window (frames; offset is the frame of labels[0]) becomes D."""
    out = np.array(labels, copy=True)
    for a, b, kind in windows:
        span = slice(max(0, a - offset), max(0, min(b - offset, len(out))))
        if report is not None:
            report["tags: frames made dead ({})".format(kind)] += int(np.sum(out[span] != "D"))
        out[span] = "D"
    return out


@dataclass(frozen=True)
class CornerBall:
    """What the "corner" rule reads (corner_ball()): the ball's x as the organisers' tracking has
    it, and which team attacks towards +x in each half."""
    x: np.ndarray           # (frames,) metres; NaN where the tracking has no ball
    plus_x: tuple           # ((first, last, "H" or "A"), ...) per half, inclusive frames
    length: float           # pitch length in metres

    def attacker(self, frame):
        """"H" or "A": the team attacking the end of the pitch the ball is in at frame."""
        for first, last, team in self.plus_x:
            if first <= frame <= last:
                if self.x[frame] > self.length / 2:
                    return team
                return "A" if team == "H" else "H"
        raise ValueError("frame {} is in no half".format(frame))


def corner_ball(ball, players, team, halves, pitch):
    """The CornerBall of the organisers' tracking: ball (frames, >= 1) as delivered, NaN where it
    is missing (not the cleaned ball: that has the gaps filled). Who attacks which end comes from
    where each team's players stand on average in each half (clean_data.team_fit.attack_signs)."""
    signs = team_fit.attack_signs(players, team, halves, pitch)
    plus_x = tuple((int(first), int(last), "H" if signs[(h, HOME)] > 0 else "A")
                   for h, (first, last) in enumerate(halves))
    return CornerBall(np.asarray(ball, dtype=np.float64)[:, 0], plus_x, float(pitch[0]))


def corner_team(labels, lo, hi, set_pieces, corner, report=None):
    """"corner" (see the module docstring) on one half's labels (labels[0] is frame lo), for the
    corner kick tags starting in [lo, hi]. corner: a CornerBall."""
    report = report if report is not None else Counter()
    out = np.array(labels, copy=True)
    for start, kind in sorted(set_pieces):
        if kind != CORNER or not lo <= start <= hi:
            continue
        i = start - lo
        changes = np.flatnonzero(out[i + 1:] != out[i])
        stop = i + 1 + int(changes[0]) if len(changes) else len(out)   # the next change
        seen = np.flatnonzero(np.isfinite(corner.x[start:lo + stop]))
        if not len(seen):
            report["corner: labels change before the ball is back"] += 1
            continue
        back = start + int(seen[0])
        team = corner.attacker(back)
        report["corner: kicks given to {}".format(team)] += 1
        out[back - lo:stop] = team
    return out


def clean_half(labels, lo, hi, set_pieces=(), rules=RULES, settings=Settings(), report=None,
               whistle=None, corner=None):
    """The rules in `rules`, in the order of ORDER, on one half's labels (labels[0] is frame lo).
    whistle: a whistle_cleanup.Evidence, needed by the "whistle" rule; corner: a CornerBall,
    needed by the "corner" rule."""
    unknown = set(rules) - set(ORDER)
    if unknown:
        raise ValueError("unknown rules {}; have {}".format(sorted(unknown), ORDER))
    if "whistle" in rules and whistle is None:
        raise ValueError("the whistle rule needs whistle=whistle_cleanup.evidence(...)")
    if "corner" in rules and corner is None:
        raise ValueError("the corner rule needs corner=corner_ball(...)")
    report = report if report is not None else Counter()
    out = np.asarray(labels)
    steps = (("whistle", lambda x: whistle_cleanup.apply(x, lo, hi, whistle, report)),
             ("fill", lambda x: fill_short_dead(x, settings.short_dead_frames)),
             ("tags", lambda x: force_dead(x, tag_windows(set_pieces, lo, hi, settings), lo,
                                           report)),
             ("corner", lambda x: corner_team(x, lo, hi, set_pieces, corner, report)),
             ("merge", lambda x: merge_short_spells(x, settings.min_possession_frames)))
    for name, step in steps:
        if name in rules:
            new = step(out)
            report["{}: frames changed".format(name)] += int(np.sum(new != out))
            out = new
    return out


def apply(labels, halves, set_pieces=(), rules=RULES, settings=Settings(), report=None,
          whistle=None, corner=None):
    """The labels (whole video) with the rules applied inside each half [(first, last)] (inclusive
    frames); frames outside the halves are unchanged. A copy."""
    out = np.array(labels, copy=True)
    for first, last in halves:
        out[first:last + 1] = clean_half(out[first:last + 1], first, last, set_pieces, rules,
                                         settings, report, whistle, corner)
    return out


def load_set_pieces(game=None, data_dir=None):
    """[(start frame, kind)] of the Smart Tagging SET PIECES (kind lower case, e.g. "corner kick"),
    read with the organisers' starter/load_smart_tagging.py. [] if the file is not there."""
    spec = importlib.util.spec_from_file_location(
        "load_smart_tagging", os.path.join(ROOT, "starter", "load_smart_tagging.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        events = module.load(game, data_dir) if game else module.load(data_dir=data_dir)
    except (IOError, OSError):
        return []
    return [(int(e["start_frame"]), (e["tags"].get("Type") or "").strip().lower())
            for e in events if (e["code"] or "").strip().lower() == "set pieces"]
