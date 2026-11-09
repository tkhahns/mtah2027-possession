"""Clean the automatic Smart Tagging with team shape, centroid and velocity.

Smart Tagging is automatic, not truth: each set piece is a ~5 s window *around* the restart,
every throw-in has Team "N/A", some tags carry
the wrong team, a few are false, and many restarts are not tagged at all. This module checks every
SET PIECES tag against what the tracking shows:

- **when** play restarts inside the window: where the motion evidence flips from dead to live
  (players start moving), snapped to the ball being struck when the ball is seen;
- **what** restart it is, from where the ball is at that moment: a throw-in on a touchline, a
  corner at a corner, a goal kick in the goal area, a kick-off on the centre spot; a tag whose
  type the geometry contradicts is re-typed when another type fits, else kept only as a generic
  free kick if the players really were stopped, else dropped as a false tag;
- **who** takes it: corners and goal kicks from the end of the pitch and the side each team
  attacks (team_shape.attack_direction); throw-ins and the rest from the player nearest the
  ball at the restart and the team-shape evidence just after, with the tag's own team as a prior.

    restarts = tag_cleaning.clean_set_pieces(tags, feats, ev)
    ll = tag_cleaning.anchor(ll, restarts)          # soft evidence for decoder.decode()
"""

import numpy as np

import team_shape
from config import AWAY, HOME
from decoder import A, D, H

FPS = 25
SET_PIECES = "set pieces"
TEAMS = {"barcelona": HOME, "atletic club": AWAY, "athletic club": AWAY, "athletic": AWAY}
TYPES = ("Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off (Start)",
         "Kick Off (After Goal)", "Penalty")

CFG = {
    "pad": 25,               # frames searched beyond each end of the tag window
    "prior_at": 0.55,        # where in the window a restart usually is (tags open ~2.5 s early)
    "prior_sigma": 25.0,     # frames: how far from there it may be
    "motion_window": 25,     # frames compared before / after a candidate restart
    "kick_speed": 4.0,       # m/s: the ball is struck
    "min_dead": -3.0,        # mean dead log-odds in the 2 s before a restart, below which a tag
                             # whose type the geometry doesn't support is judged false
    "team_prior": 2.0,       # log-odds the tag's own team is worth
    "anchor_before": 25,     # frames of "dead" evidence before a restart ...
    "anchor_after": 25,      # ... and of "live, taker's team" after it
    "anchor_dead": 1.0,      # nats per frame
    "anchor_team": 1.0,
}


def _norm(text):
    return " ".join(str(text or "").strip().lower().split())


def tag_team(tag):
    """HOME / AWAY from a tag's Team, or None ("N/A", missing, unknown)."""
    return TEAMS.get(_norm(tag.get("tags", {}).get("Team")))


def tag_type(tag):
    wanted = _norm(tag.get("tags", {}).get("Type"))
    return next((t for t in TYPES if t.lower() == wanted), None)


def is_set_piece(tag):
    return _norm(tag.get("code")) == SET_PIECES


# -- geometry ---------------------------------------------------------------------------------

def _distances(pos, pitch):
    length, width = pitch
    x, y = float(pos[0]), float(pos[1])
    d_goal = min(abs(x), abs(length - x))
    d_touch = min(abs(y), abs(width - y))
    return {"goal": d_goal, "touch": d_touch, "corner": float(np.hypot(d_goal, d_touch)),
            "off_centre": abs(y - width / 2), "centre": float(np.hypot(x - length / 2, y - width / 2))}


def geometric_types(pos, pitch):
    """Restart types a ball position clearly fits, most specific first. pos None -> ()."""
    if pos is None or not np.all(np.isfinite(pos)):
        return ()
    d = _distances(pos, pitch)
    out = []
    if d["centre"] <= 6.0:
        out += ["Kick Off (Start)", "Kick Off (After Goal)"]
    if d["corner"] <= 6.0:
        out.append("Corner Kick")
    if 8.0 <= d["goal"] <= 14.0 and d["off_centre"] <= 5.0:
        out.append("Penalty")
    if d["goal"] <= 9.0 and d["off_centre"] <= 14.0:
        out.append("Goal Kick")
    if d["touch"] <= 4.0 and d["goal"] > 2.0:
        out.append("Throw-in")
    return tuple(out)


def impossible(kind, pos, pitch):
    """True when the ball position rules this restart type out, with generous margins (the
    position is a few frames off the kick at best, and our ball is jittery)."""
    if kind is None or pos is None or not np.all(np.isfinite(pos)):
        return False
    d = _distances(pos, pitch)
    return {"Throw-in": d["touch"] > 8.0,
            "Corner Kick": d["corner"] > 12.0,
            "Goal Kick": d["goal"] > 16.0 or d["off_centre"] > 20.0,
            "Kick Off (Start)": d["centre"] > 12.0,
            "Kick Off (After Goal)": d["centre"] > 12.0,
            "Penalty": d["goal"] > 20.0 or d["off_centre"] > 12.0}.get(kind, False)


def _ball_at(feats, r, before=2 * FPS, after=5):
    """Where the restart is taken from: the ball's last sighting in [r - before, r], else its
    first one in (r, r + after]; else None. Never later: a struck ball is metres away within
    a fraction of a second, and an estimate across a gap is not trusted for geometry."""
    seen = feats["ball_seen"]
    n = len(seen)
    earlier = np.flatnonzero(seen[max(0, r - before):r + 1])
    if len(earlier):
        return feats["ball"][max(0, r - before) + earlier[-1]]
    later = np.flatnonzero(seen[r + 1:min(n, r + after + 1)])
    if len(later):
        return feats["ball"][r + 1 + later[0]]
    return None


# -- when -------------------------------------------------------------------------------------

def restart_frame(tag, feats, ev, cfg=CFG):
    """Best estimate of the frame play restarts in this tag (see the module docstring)."""
    n = len(ev["dead"])
    start, end = int(tag["start_frame"]), int(tag["end_frame"])
    lo, hi = max(1, start - cfg["pad"]), min(n - 2, end + cfg["pad"])
    if hi < lo:
        return int(np.clip((start + end) // 2, 0, n - 1))
    w = cfg["motion_window"]
    dead = np.asarray(ev["dead"], dtype=np.float64)
    before = team_shape.window_mean(dead, w, -1)
    after = team_shape.window_mean(dead, 0, w - 1)
    cand = np.arange(lo, hi + 1)
    flip = np.nan_to_num(before[cand] - after[cand])
    prior = start + cfg["prior_at"] * (end - start)
    score = np.clip(flip, -8, 8) - 0.5 * ((cand - prior) / cfg["prior_sigma"]) ** 2
    best = int(cand[np.argmax(score)])
    # snap to the ball being struck, if it is seen moving off within half a second of `best`
    speed = np.nan_to_num(feats["ball_speed"], nan=0.0)
    seen = feats["ball_seen"]
    near = np.arange(max(lo, best - 12), min(hi, best + 12) + 1)
    kick = near[seen[near] & (speed[near] >= cfg["kick_speed"]) &
                ~(seen[near - 1] & (speed[near - 1] >= cfg["kick_speed"]))]
    if len(kick):
        best = int(kick[np.argmin(np.abs(kick - best))])
    return best


# -- who --------------------------------------------------------------------------------------

def _taker(kind, pos, feats, ev, r, prior_team, cfg):
    """(team or None, how) for a restart of `kind` taken at frame r from `pos`."""
    length = feats["pitch"][0]
    attack = feats["attack"][r]
    if pos is not None and attack.any() and kind in ("Corner Kick", "Goal Kick"):
        end_sign = 1 if pos[0] > length / 2 else -1
        wanted = end_sign if kind == "Corner Kick" else -end_sign
        return (HOME if attack[HOME] == wanted else AWAY), "pitch end"
    score = 0.0
    if prior_team is not None:
        score += cfg["team_prior"] if prior_team == HOME else -cfg["team_prior"]
    near = feats["nearest"][r]
    if feats["ball_seen"][r] and np.isfinite(near).all() and abs(near[HOME] - near[AWAY]) > 1.0:
        score += 2.0 if near[HOME] < near[AWAY] else -2.0
    after = ev["team"][r:min(len(ev["team"]), r + 2 * FPS)]
    if len(after):
        score += float(np.clip(np.mean(after), -4, 4))
    if abs(score) < 0.5:
        return prior_team, "tag" if prior_team is not None else "unknown"
    return (HOME if score > 0 else AWAY), "evidence"


def clean_set_pieces(tags, feats, ev, cfg=CFG):
    """Cleaned restarts from the SET PIECES tags, sorted by frame. Each is a dict:

    restart  frame play restarts            type     checked / corrected restart type
    team     HOME / AWAY / None             keep     False for tags judged false
    tag_type, tag_team                      what the tag said
    retyped  True when the geometry changed the type     team_from  how the team was decided
    dead_before  mean dead log-odds in the 2 s before the restart
    """
    n = len(ev["dead"])
    out = []
    for tag in tags or []:
        if not is_set_piece(tag) or int(tag["end_frame"]) < 0 or int(tag["start_frame"]) >= n:
            continue
        r = restart_frame(tag, feats, ev, cfg)
        pos = _ball_at(feats, r)
        fits = geometric_types(pos, feats["pitch"])
        stated = tag_type(tag)
        dead_before = float(np.mean(ev["dead"][max(0, r - 2 * FPS):max(1, r)]))
        kind, retyped, keep = stated, False, True
        if stated is None or impossible(stated, pos, feats["pitch"]):
            specific = [t for t in fits if t != stated]
            if specific:
                kind, retyped = specific[0], True
            elif dead_before >= cfg["min_dead"]:
                kind, retyped = "Free Kick", stated != "Free Kick"
            else:
                keep = False
        # Players clearly in open play just before the "restart": a false tag, unless the ball is
        # exactly where the tagged kind of restart is taken from (a re-typed guess doesn't count).
        if keep and dead_before < cfg["min_dead"] and (retyped or kind not in fits):
            keep = False
        team, how = _taker(kind, pos, feats, ev, r, tag_team(tag), cfg)
        out.append({"restart": r, "type": kind, "team": team, "keep": keep,
                    "tag_type": stated, "tag_team": tag_team(tag), "retyped": retyped,
                    "team_from": how, "dead_before": dead_before,
                    "start_frame": int(tag["start_frame"]), "end_frame": int(tag["end_frame"])})
    return sorted(out, key=lambda item: item["restart"])


def anchor(ll, restarts, cfg=CFG):
    """Copy of decoder emissions with soft evidence at every kept restart: dead just before it,
    live (and the taker's team, if known) just after. Soft, so strong tracking evidence still
    wins; the decoder decides how far back the stoppage goes."""
    ll = np.array(ll, dtype=np.float64, copy=True)
    n = len(ll)
    for item in restarts:
        if not item["keep"]:
            continue
        r = int(item["restart"])
        ll[max(0, r - cfg["anchor_before"]):max(0, r), D] += cfg["anchor_dead"]
        post = slice(min(n, r), min(n, r + cfg["anchor_after"]))
        ll[post, D] -= cfg["anchor_dead"]
        if item["team"] in (HOME, AWAY):
            ll[post, H if item["team"] == HOME else A] += cfg["anchor_team"]
    return ll
