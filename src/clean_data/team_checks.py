"""Checks of one hypothesis, "this stretch of this slot follows a player of the other team",
shared by the player-cleaning methods in clean_data.players.SHARED.

  broadcast   pairs with broadcast players over the stretch (clean_data.broadcast_teams, replays
              skipped): "other" when at least 1 s is paired and 80% of it says the other team,
              votes = broadcast tracks saying so for 1 s or more; "own" the same the other way.
  roles       clean_data.role_scales.stretch: windows of 3 s to 1 min inside the stretch.
  head count  its team shows 12+ players and the other 10 or fewer on half the stretch.

Tracks: each slot's runs of frames, gaps up to 0.5 s bridged. Pieces: a track's graded frames cut
at its contacts with opponents (within 1.5 m) and at its gaps, where an identity can jump.
Events: where a track starts or ends mid-play, and which track ended or started near it then.
"""

import os

import numpy as np

from utils.series_utils import runs

from . import broadcast_teams as bt, role_scales
from .team_evidence import contacts

FPS = 25
SETTINGS = {
    "gap_s": 0.5,             # a slot's gaps up to this long stay one track
    "min_paired_s": 1.0,      # broadcast: paired at least this long...
    "share": 0.8,             # ...and this share of the pairs on one side
    "min_vote_s": 1.0,        # a broadcast track votes with this much pairing
    "full": 12,               # head count: 12+ on its team, 10 or fewer on the other...
    "full_share": 0.5,        # ...on this share of the stretch
    "handover_s": 2.0,        # events: a track starting this soon after another ended nearby...
    "overlap_s": 1.0,         # ...or this long before it ended, follows it
    "reach_m": 2.0,           # nearby: within reach_m + run_mps x the gap
    "run_mps": 7.0,
    "min_life_s": 1.0,        # events only on tracks living this long
    "edge_s": 1.0,            # and not this close to a half's start or end
    "touchline_share": 0.8,   # tracks this share of their life on a side line are left out
}


class Checks:
    """Everything the checks need on one match, worked out once. tactical: {players, ball, team,
    pitch}; halves: [(first, last)] frames; broadcast (optional): {players, team, ...} blanked
    where it has no homography, lined up by shifts (tactical - broadcast frame per half)."""

    def __init__(self, tactical, halves, broadcast=None, shifts=None, thresholds=None, grid=None,
                 workers=None, settings=None):
        self.p = {**SETTINGS, **(settings or {})}
        self.players = tactical["players"]
        self.team = np.asarray(tactical["team"])
        self.halves = [(int(a), int(z)) for a, z in halves]
        n = len(self.players)
        self.graded = np.zeros(n, bool)
        for a, z in self.halves:
            self.graded[a:z + 1] = True
        seen = ~np.isnan(self.players[..., 0])
        self.ids, self.table = bt.tracks(seen, int(self.p["gap_s"] * FPS))
        self.contact = contacts(self.players, self.team)
        line = bt._touchline(self.players, tactical["pitch"], bt.DEFAULTS["touchline_m"])
        share = self.p["touchline_share"]
        self.touchline = np.array([line[a:z, s][self.ids[a:z, s] == k].mean() >= share
                                   for k, (s, a, z) in enumerate(self.table)], bool)
        self.counts = np.stack([(seen & (self.team == c)).sum(1) for c in (0, 1)], 1)
        self.thresholds = thresholds or role_scales.load_thresholds()
        workers = workers or max(1, (os.cpu_count() or 2) - 2)
        self.grid = grid if grid is not None else role_scales.grid(
            self.players, self.team, self.halves, tactical["pitch"], scales=sorted(self.thresholds),
            workers=workers)
        self.pairing = None
        if broadcast is not None:
            from . import broadcast_replays
            replay = broadcast_replays.find(tactical, broadcast, self.halves, shifts)
            self.pairing = bt.pairing(tactical, broadcast, self.halves, shifts, skip=replay)
        self._by_slot = {}
        for k, (s, a, z) in enumerate(self.table):
            self._by_slot.setdefault(s, []).append((a, z, k))

    # -- one stretch -------------------------------------------------------------------------

    def broadcast(self, slot, first, stop):
        """{"verdict", "paired_s", "disagree", "against", "for"} (votes), or None without a
        broadcast."""
        if self.pairing is None:
            return None
        f = np.arange(first, stop)
        partner, agree = self.pairing["partner"][f, slot], self.pairing["agree"][f, slot]
        on = partner >= 0
        out = {"verdict": None, "paired_s": round(on.sum() / FPS, 1), "disagree": None,
               "against": _votes(partner[on & ~agree], self.p["min_vote_s"]),
               "for": _votes(partner[on & agree], self.p["min_vote_s"])}
        if on.sum() >= self.p["min_paired_s"] * FPS:
            out["disagree"] = round(float(1 - agree[on].mean()), 3)
            if out["disagree"] >= self.p["share"] and out["against"]:
                out["verdict"] = "other"
            elif out["disagree"] <= 1 - self.p["share"] and out["for"]:
                out["verdict"] = "own"
        return out

    def roles(self, slot, first, stop):
        return role_scales.stretch(self.grid, self.thresholds, slot, first, stop)

    def head(self, slot, first, stop):
        """Share of the stretch with 12+ on the slot's team and 10 or fewer on the other."""
        if stop <= first:
            return 0.0
        own = self.counts[first:stop, self.team[slot]]
        other = self.counts[first:stop, 1 - self.team[slot]]
        return float(np.mean((own >= self.p["full"]) & (other <= self.p["full"] - 2)))

    def check(self, slot, first, stop):
        """All three checks with a summary: signals for the other team (broadcast 1, or 2 with 2+
        votes; roles 1; head count 1), against (broadcast or roles say its own team) and why."""
        b, r = self.broadcast(slot, first, stop), self.roles(slot, first, stop)
        h = self.head(slot, first, stop)
        b_other = b is not None and b["verdict"] == "other"
        signals = (b_other + (b_other and b["against"] >= 2) + (r["verdict"] == "other")
                   + (h >= self.p["full_share"]))
        against = (b is not None and b["verdict"] == "own") or r["verdict"] == "own"
        return {"broadcast": b, "roles": r, "head": round(h, 2), "signals": int(signals),
                "b_other": b_other, "r_other": r["verdict"] == "other", "against": bool(against),
                "text": describe(b, r, h, self.p["full_share"])}

    def lean(self, slot, first, stop, use_broadcast=True):
        """"other" / "own" / None for a piece: the broadcast's verdict if it has one, else the sign
        of the roles' median at their longest scale."""
        if use_broadcast:
            b = self.broadcast(slot, first, stop)
            if b is not None and b["verdict"]:
                return b["verdict"]
        z = self.roles(slot, first, stop)["z"]
        return None if not np.isfinite(z) or z == 0 else ("other" if z > 0 else "own")

    def sure_own(self, slot, first, stop, use_broadcast=True):
        """The broadcast or the roles (z <= -1, shorter scales agreeing) put it on its own team."""
        b = self.broadcast(slot, first, stop) if use_broadcast else None
        if b is not None and b["verdict"] == "own":
            return True
        return self.roles(slot, first, stop)["verdict"] == "own"

    # -- tracks, pieces, runs ----------------------------------------------------------------

    def track_at(self, slot, frame):
        """The track of slot that covers frame (gaps bridged), or -1."""
        for a, z, k in self._by_slot.get(slot, []):
            if a <= frame < z:
                return k
        return -1

    def pieces(self, k):
        """[(first, stop)]: track k's graded frames, seen and out of contact, in runs."""
        s, a, z = self.table[k]
        free = (self.ids[a:z, s] == k) & self.graded[a:z] & ~self.contact[a:z, s]
        return [(a + lo, a + hi) for lo, hi in runs(free)]

    def run(self, k, frame, forward=True, use_broadcast=True):
        """(first, stop) of track k from frame on (forward) or up to it (backward), piece by piece,
        up to the first piece the checks are sure is on its own team; the track's end when none
        is. None when the first piece already is."""
        s, a, z = self.table[k]
        pieces = [p for p in self.pieces(k) if (p[1] > frame if forward else p[0] < frame)]
        if not forward:
            pieces = pieces[::-1]
        kept = []
        for lo, hi in pieces:
            lo, hi = (max(lo, frame), hi) if forward else (lo, min(hi, frame))
            if self.sure_own(s, lo, hi, use_broadcast):
                break
            kept.append((lo, hi))
        else:
            return (frame, z) if forward else (a, frame)
        if not kept:
            return None
        return (frame, kept[-1][1]) if forward else (kept[-1][0], frame)

    # -- events --------------------------------------------------------------------------------

    def events(self):
        """Tracks that start or end mid-play: [{"kind": "after opponent" / "fresh" (a start) or
        "vanish" (an end), "track", "frame", "other": the opponent's track for "after opponent"}].
        Starts after a teammate's track ended nearby (a re-track) and ends followed by a teammate's
        start are left out; an end followed by an opponent's start is that start's event."""
        p = self.p
        life, edge = int(p["min_life_s"] * FPS), int(p["edge_s"] * FPS)
        ok = [not self.touchline[k] and z - a >= life for k, (s, a, z) in enumerate(self.table)]
        starts = np.array([a for _, a, _ in self.table])
        stops = np.array([z for _, _, z in self.table])
        inner = np.zeros(len(self.graded), bool)
        for a, z in self.halves:
            inner[a + edge:z + 1 - edge] = True
        out = []
        for k, (s, a, z) in enumerate(self.table):
            if not ok[k]:
                continue
            if inner[a]:
                prev = self._nearest(k, a, stops, before=True)
                if prev is None:
                    out.append({"kind": "fresh", "track": k, "frame": a})
                elif self.team[self.table[prev][0]] != self.team[s]:
                    out.append({"kind": "after opponent", "track": k, "frame": a,
                                "other": int(prev)})
            if inner[z - 1] and self._nearest(k, z, starts, before=False) is None:
                out.append({"kind": "vanish", "track": k, "frame": z})
        return out

    def _nearest(self, k, frame, ends, before):
        """The nearest other track that ended (before=True) or started near track k's start /
        end `frame`, close enough in time and place to be the same player; None if none."""
        p = self.p
        s, a, z = self.table[k]
        back, ahead = int(p["handover_s"] * FPS), int(p["overlap_s"] * FPS)
        lo, hi = (frame - back, frame + ahead) if before else (frame - ahead, frame + back)
        here = self.players[a if before else z - 1, s, :2]
        best, best_d = None, np.inf
        for j in np.flatnonzero((ends >= lo) & (ends <= hi)):
            js, ja, jz = self.table[j]
            if j == k or (ja >= a if before else jz <= z):
                continue
            there = self.players[jz - 1 if before else ja, js, :2]
            gap = max(0, (frame - jz) if before else (ja - frame)) / FPS
            d = float(np.linalg.norm(there - here))
            if d <= p["reach_m"] + p["run_mps"] * gap and d < best_d:
                best, best_d = j, d
        return best


def describe(b, r, h, full_share=0.5):
    """One line for the viewer: what each check says."""
    parts = []
    if b is not None:
        if b["disagree"] is None:
            parts.append("broadcast: {} s paired".format(b["paired_s"]))
        else:
            parts.append("broadcast: {:.0%} of {} s say other team ({} v {} tracks)".format(
                b["disagree"], b["paired_s"], b["against"], b["for"]))
    if r["scale"] is None:
        parts.append("roles: too short")
    else:
        shorter = [L for L in r["per_scale"] if L < r["scale"]]
        parts.append("roles: {:+.1f} at {} s{}".format(
            r["z"], r["scale"], " ({} s agree)".format("/".join(map(str, shorter)))
            if r["verdict"] and shorter else ""))
    if h >= full_share:
        parts.append("head count 12 v 10 on {:.0%}".format(h))
    return "; ".join(parts)


def _votes(track_ids, min_s):
    if not len(track_ids):
        return 0
    _, n = np.unique(track_ids, return_counts=True)
    return int((n >= min_s * FPS).sum())
