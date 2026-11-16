"""Player-team cleaning methods, each written to its own cleaned tracking file.

A method says which stretches of which tracks follow a player of the other team: moves,
[{"slot", "first", "last", "why"}] (frames inclusive). apply_moves puts each stretch into a slot of
the other team that is free on all its frames (slot teams are fixed in the metadata, so that is how
a file says "this is the other team's player"), and write_atd rewrites the raw ATD lines with just
those groups moved: every other byte stays as it was. Keeper slots are left to clean_data.keeper:
apply_moves never moves them or moves anything into them.

    METHODS   name -> (what it is, needs the broadcast,
                       function(tactical, broadcast, halves, ...) -> moves)

The methods in SHARED share their checks (clean_data.team_checks: broadcast, roles at 3 s to 1 min,
head count); run() keeps them in `shared` so several methods on one match work them out once.
"""

import json
import os
import shutil

import numpy as np

from utils.series_utils import runs

from . import legacy_teams, team_checks, team_evidence

FPS = 25
SIDELINE_SHARE = 0.8  # tracks this share of their life on a side line (or off the pitch) are
                      # left out as referees / ball boys


def moves_from_slot_team(slot_team, team, why):
    """Moves for every run of frames where slot_team (frames, slots) differs from team (slots,)."""
    out = []
    for s in range(slot_team.shape[1]):
        for a, z in runs(slot_team[:, s] != team[s]):
            out.append({"slot": int(s), "first": int(a), "last": int(z) - 1, "why": why})
    return out


def apply_moves(players, team, moves, keepers):
    """(players with each move's stretch put into a free slot of the other team, applied moves with
    "to": that slot, skipped moves). A slot is free when it is empty on every frame of the stretch
    after the moves before it; the moves go in order of their first frame. keepers: (frames, 2)
    keeper slot per frame (clean_data.keeper.find_slots, -1 = none); a slot that is a keeper on any
    frame of a stretch is neither moved nor moved into."""
    out = np.array(players, copy=True)
    team, keepers = np.asarray(team), np.asarray(keepers)
    applied, skipped = [], []
    for m in sorted(moves, key=lambda m: (m["first"], m["slot"])):
        s, a, z = m["slot"], m["first"], m["last"] + 1
        if (keepers[a:z] == s).any():
            skipped.append({**m, "reason": "keeper slot (clean_data.keeper fixes those)"})
            continue
        if np.isnan(out[a:z, s, 0]).all():
            skipped.append({**m, "reason": "nothing left to move"})
            continue
        free = [t for t in np.flatnonzero(team != team[s])
                if np.isnan(out[a:z, t, 0]).all() and not (keepers[a:z] == t).any()]
        if not free:
            skipped.append({**m, "reason": "no free slot on the other team"})
            continue
        t = int(free[0])
        out[a:z, t] = out[a:z, s]
        out[a:z, s] = np.nan
        applied.append({**m, "to": t, "team": int(team[s]), "to_team": int(team[t])})
    return out, applied, skipped


def write_atd(raw_in, raw_out, applied):
    """Copy the raw ATD text with each applied move's groups moved from its slot to "to" (frame id
    = video frame + 1); every other byte is left as it was."""
    by_frame = {}
    for m in applied:
        for f in range(m["first"], m["last"] + 1):
            by_frame.setdefault(f, []).append((m["slot"], m["to"]))
    with open(raw_in, newline="") as src, open(raw_out, "w", newline="") as dst:
        for line in src:
            parts = line.split(":")
            if len(parts) == 3 and parts[0].strip().isdigit() and int(parts[0]) - 1 in by_frame:
                groups = parts[1].split(";")
                for s, t in by_frame[int(parts[0]) - 1]:
                    if max(s, t) < len(groups):
                        groups[t], groups[s] = groups[s], "NaN,NaN,NaN"
                line = ":".join([parts[0], ";".join(groups), parts[2]])
            dst.write(line)


# -- the methods -------------------------------------------------------------------------------

def team_swaps(tactical, broadcast, halves):
    """Undo swaps between opponents near the ball, by physics and team shape."""
    trk = legacy_teams.FixTeamSwaps().clean(tactical)
    return moves_from_slot_team(trk["slot_team"], np.asarray(tactical["team"]), "team swap")


def handovers(tactical, broadcast, halves):
    """A track replaced next to an opponent by a new track of the other team."""
    graded = np.zeros(len(tactical["ball"]), dtype=bool)
    for a, z in halves:
        graded[a:z + 1] = True
    fixer = legacy_teams.FixHandovers(touchline_share=SIDELINE_SHARE)
    trk = fixer.clean({**tactical, "graded": graded})
    return moves_from_slot_team(trk["slot_team"], np.asarray(tactical["team"]), "hand-over")


def broadcast(tactical, broadcast, halves, shifts):
    """Stretches between contacts that the broadcast puts on the other team, strong ones only
    (2+ broadcast tracks say so, or 1 plus the head count)."""
    found, _ = team_evidence.broadcast_candidates(tactical, broadcast, halves, shifts,
                                                  touchline_share=SIDELINE_SHARE)
    return [{"slot": int(c["slot"]), "first": int(c["first"]), "last": int(c["stop"]) - 1,
             "why": "broadcast: {} track{} say the other team".format(
                 c["votes_against"], "" if c["votes_against"] == 1 else "s")}
            for c in found if c["strong"]]


def checks(shared, tactical, broadcast, halves, shifts=None):
    """team_checks.Checks for this match, made once per `shared` dict (the role grid is shared
    between the checks with and without the broadcast)."""
    shared = {} if shared is None else shared
    key = "checks" if broadcast is not None else "checks_no_broadcast"
    if key not in shared:
        shared[key] = team_checks.Checks(tactical, halves, broadcast, shifts, grid=shared.get("grid"),
                                         thresholds=shared.get("thresholds"),
                                         settings={"touchline_share": SIDELINE_SHARE})
        shared["grid"] = shared[key].grid
    return shared[key]


ROLE_CUT_EVENTS = 2.0  # roles alone confirm an events_checked event at 2x their threshold: on
                       # messy Metrica 1x gives 12 false moves per 90 min, 1.5x 3.8, 2x 0.3
                       # (methods_eval.py)


def find_handovers(shared, tactical, c):
    """The hand-overs of `handovers` (legacy_teams.FixHandovers.find), found once per `shared`
    dict."""
    shared = {} if shared is None else shared
    if "handovers" not in shared:
        shared["handovers"] = legacy_teams.FixHandovers(touchline_share=SIDELINE_SHARE).find(
            {**tactical, "graded": c.graded})
    return shared["handovers"]


def _best(c, options, min_signals, prefer=None, role_cut=1.0):
    """The option (slot, first, stop, name) whose checks back the other team best, or None: the
    broadcast says so, or the roles do at role_cut times their threshold; nothing says its own
    team; at least min_signals signals. Ties go to `prefer` (a slot)."""
    best = None
    for slot, first, stop, name in options:
        r = c.check(slot, first, stop)
        roles = r["r_other"] and r["roles"]["z"] >= role_cut
        if r["against"] or not (r["b_other"] or roles) or r["signals"] < min_signals:
            continue
        rank = (r["signals"], slot == prefer)
        if best is None or rank > best[0]:
            best = (rank, slot, first, stop, name, r)
    return None if best is None else best[1:]


def _handover_moves(c, shared, tactical, min_signals, role_cut):
    out = []
    for h in find_handovers(shared, tactical, c):
        options = []
        for slot, name in ((h["slot"], "new track"), (h["mate"], "teammate")):
            k = c.track_at(slot, h["start"])
            span = c.run(k, h["start"]) if k >= 0 else None
            if span:
                options.append((slot, span[0], span[1], name))
        pick = _best(c, options, min_signals, prefer=h["fix"], role_cut=role_cut)
        if pick:
            slot, first, stop, name, r = pick
            note = "" if slot == h["fix"] else ", FixHandovers picked the other"
            out.append({"slot": int(slot), "first": int(first), "last": int(stop) - 1,
                        "why": "hand-over, the {}{}: {}".format(name, note, r["text"])})
    return out


def _event_moves(c, shared, tactical, min_signals, role_cut):
    skip = set()
    for h in find_handovers(shared, tactical, c):
        skip.update({c.track_at(h["slot"], h["start"]), c.track_at(h["mate"], h["start"]),
                     c.track_at(h["old_slot"], h["old_stop"] - 1)})
    out = []
    for e in c.events():
        k = e["track"]
        if k in skip or e.get("other") in skip:
            continue
        s, a, z = c.table[k]
        options = []
        if e["kind"] == "vanish":
            span = c.run(k, z, forward=False)
            options.append((s, *span, "vanishes mid-play") if span else None)
        else:
            span = c.run(k, a)
            what = ("appears from nowhere" if e["kind"] == "fresh"
                    else "appears where an opponent's track ended")
            options.append((s, *span, what) if span else None)
        if e["kind"] == "after opponent":
            js, ja, jz = c.table[e["other"]]
            span = c.run(e["other"], jz, forward=False)
            options.append((js, *span, "ends where an opponent's track appears") if span else None)
        pick = _best(c, [o for o in options if o], min_signals, role_cut=role_cut)
        if pick:
            slot, first, stop, name, r = pick
            out.append({"slot": int(slot), "first": int(first), "last": int(stop) - 1,
                        "why": "track {}: {}".format(name, r["text"])})
    return out


def handovers_checked(tactical, broadcast, halves, shifts=None, shared=None, min_signals=1):
    """The hand-overs of `handovers`, each checked on both tracks that could be wrong (the new
    track, or the teammate whose dot jumped onto the other player), from the hand-over on, piece
    by piece up to the first piece the checks put on its own team. The one the broadcast or the
    roles put on the other team, with nothing saying its own, moves; neither: left alone."""
    c = checks(shared, tactical, broadcast, halves, shifts)
    return _handover_moves(c, shared, tactical, min_signals, 1.0)


def appear_vanish(tactical, broadcast, halves, shifts=None, shared=None, min_signals=2):
    """Tracks that appear or vanish mid-play, the three tracks of a hand-over (handovers_checked)
    left out. A track that starts where an opponent's track just ended: it, or the opponent's
    track before it, may be on the wrong team; one that appears from nowhere: it may; one that
    vanishes: it may, up to then. Each is checked from the event on (back from it for the track
    before), piece by piece up to the first piece the checks put on its own team; it moves when
    the broadcast or the roles say the other team, at least min_signals signals agree and nothing
    says its own team."""
    c = checks(shared, tactical, broadcast, halves, shifts)
    return merge_moves(_event_moves(c, shared, tactical, min_signals, 1.0))


def events_checked(tactical, broadcast, halves, shifts=None, shared=None, role_cut=ROLE_CUT_EVENTS):
    """Hand-overs (handovers_checked) and tracks that appear or vanish (appear_vanish's events)
    under one rule, the hand-overs': the event is the first opinion, the broadcast (as in
    `broadcast`) and the role assignment (roles at 3 s-1 min) the second and third. A track moves
    when one of them says the other team (the roles at role_cut times their threshold) and
    neither says its own team."""
    c = checks(shared, tactical, broadcast, halves, shifts)
    return merge_moves(_handover_moves(c, shared, tactical, 1, role_cut)
                       + _event_moves(c, shared, tactical, 1, role_cut))


def roles_scales(tactical, broadcast, halves, shifts=None, shared=None, cut=2.0, min_s=3.0):
    """The team-role test (clean_data.role_scales) at several scales. Each track's pieces between contacts lean to the other team or
    its own by the roles (team_checks.lean: the median at the longest scale that fits). Runs of
    pieces leaning to the other team (pieces too short to tell may sit between them) are tested
    as one stretch: longer, so a longer window can confirm what the short ones lean to. It moves
    when its median at the longest scale is at least `cut` times that scale's threshold and every
    shorter scale leans the same way. cut 2: on messy Metrica with injected stretches (src/training/
    player_cleaning/methods_eval.py) 1 gives ~37 false moves per 90 min, 2 gives ~2."""
    c = checks(shared, tactical, None, halves)
    out = []
    for k, (s, a, z) in enumerate(c.table):
        if c.touchline[k]:
            continue
        group = []
        for lo, hi in c.pieces(k) + [(None, None)]:
            lean = None if lo is None else c.lean(s, lo, hi, use_broadcast=False)
            if lo is not None and lean != "own":
                group.append((lo, hi, lean))
                continue
            while group and group[-1][2] is None:
                group.pop()
            while group and group[0][2] is None:
                group.pop(0)
            if group and group[-1][1] - group[0][0] >= min_s * FPS:
                first, stop = group[0][0], group[-1][1]
                r = c.roles(s, first, stop)
                if r["verdict"] == "other" and r["z"] >= cut:
                    out.append({"slot": int(s), "first": int(first), "last": int(stop) - 1,
                                "why": "roles: " + team_checks.describe(None, r, 0.0)})
            group = []
    return out


def combined(tactical, broadcast, halves, shifts=None, shared=None):
    """The moves of handovers_checked, appear_vanish and roles_scales in one file; moves on a slot
    that overlap are joined."""
    return merge_moves(handovers_checked(tactical, broadcast, halves, shifts, shared=shared)
                       + appear_vanish(tactical, broadcast, halves, shifts, shared=shared)
                       + roles_scales(tactical, broadcast, halves, shifts, shared=shared))


def merge_moves(moves):
    """Moves on the same slot that overlap or touch, joined into one."""
    out = []
    for m in sorted(moves, key=lambda m: (m["slot"], m["first"])):
        if out and out[-1]["slot"] == m["slot"] and m["first"] <= out[-1]["last"] + 1:
            prev = out[-1]
            why = prev["why"] if m["why"] == prev["why"] else prev["why"] + " | " + m["why"]
            out[-1] = {**prev, "last": max(prev["last"], m["last"]), "why": why}
        else:
            out.append(dict(m))
    return sorted(out, key=lambda m: (m["first"], m["slot"]))


METHODS = {
    "team_swaps": ("team swaps: physics + team shape near the ball", False, team_swaps),
    "handovers": ("hand-overs: replaced next to an opponent", False, handovers),
    "broadcast": ("broadcast: confirmed by 2+ broadcast tracks or 1 + head count", True, broadcast),
    "handovers_checked": ("hand-overs checked by broadcast + roles at 3 s-1 min", True,
                          handovers_checked),
    "appear_vanish": ("tracks that appear / vanish, checked by broadcast + roles + head count",
                      True, appear_vanish),
    "roles_scales": ("roles at 3 s-1 min, longer runs confirm shorter pieces", False,
                     roles_scales),
    "combined": ("handovers_checked + appear_vanish + roles_scales", True, combined),
    "events_checked": ("hand-overs + appear / vanish, checked by broadcast or roles", True,
                       events_checked),
}
SHARED = {"handovers_checked", "appear_vanish", "roles_scales", "combined", "events_checked"}


def run(name, tactical, broadcast, halves, shifts=None, shared=None):
    """The method's moves on this match; shifts (tactical - broadcast frame per half) for the
    broadcast methods; shared: a dict the SHARED methods keep their checks in between calls."""
    label, needs_broadcast, method = METHODS[name]
    if name in SHARED:
        return method(tactical, broadcast, halves, shifts, shared=shared)
    if needs_broadcast:
        return method(tactical, broadcast, halves, shifts)
    return method(tactical, broadcast, halves)


def save(name, paths, out_dir, applied, skipped, n_frames):
    """out_dir/tactical/: the cleaned ATD and a copy of the metadata, as a bundle reads them; and
    out_dir/moves.json: the method, its moves (frames and slots only) and counts."""
    folder = os.path.join(out_dir, "tactical")
    os.makedirs(folder, exist_ok=True)
    write_atd(paths["atd_raw"], os.path.join(folder, os.path.basename(paths["atd_raw"])), applied)
    shutil.copyfile(paths["atd_meta"], os.path.join(folder, os.path.basename(paths["atd_meta"])))
    moved = sum(m["last"] - m["first"] + 1 for m in applied)
    with open(os.path.join(out_dir, "moves.json"), "w") as handle:
        label = METHODS[name][0] if name in METHODS else name
        json.dump({"method": name, "label": label, "moves": applied, "skipped": skipped,
                   "counts": {"moves": len(applied), "skipped": len(skipped),
                              "moved_s": round(moved / FPS, 1), "n_frames": n_frames}}, handle)
