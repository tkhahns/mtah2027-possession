"""The earlier player-team cleaners, kept as they were so clean_data.players can write their
tracking files (its methods team_swaps and handovers):

    FixTeamSwaps   undo identity swaps between opponents near the ball, judged by physics and
                   team shape (broke good tracks on Metrica: not a default)
    FixHandovers   a track that ends next to an opponent and is replaced by a new track of the
                   other team: the one that follows the other team's player moves over

Each clean(trk) returns trk with "slot_team" (frames, slots): every slot's team on every frame.
"""

import numpy as np

FPS = 25


def _runs(mask):
    """(start, end) of every run of True, end exclusive."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def _team_centre(xy):
    """(frames, 2): the median position of a team's tracked players on each frame."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # frames with nobody tracked
        return np.nanmedian(xy, axis=1)


def _place(path, centre, min_seen=0.8):
    """Where a player stands relative to his team's centre, on average over a window, or None
    when the track is seen on too few of its frames."""
    rel = path - centre
    ok = np.isfinite(rel).all(axis=1)
    return rel[ok].mean(axis=0) if ok.mean() >= min_seen else None


class FixTeamSwaps:
    """Undo identity swaps between opponents near the ball: a slot whose dot slid onto a player
    of the other team.

    A pass: a track of each team within pass_m of each other for at most pass_max_s, with the
    ball seen within near_ball_m of one of them during it. Two stories for what comes after:
      no swap  each dot carries on along its own path and keeps its player's place in his team;
      swapped  each dot carries on along the other dot's path and takes the other's place.
    It is a swap only when both tests prefer the swapped story by a margin:
      physics  running each dot straight on from the half second before the pass, the "no swap"
               story lands more than min_dpos metres further off than the "swapped" one;
      shape    a player's place = where he stands relative to his team's centre (the median of
               its players). Compared over the part of each track the fix would move (from the
               pass to the track's end, at most after_max_s) against up to before_s of the same
               track before it; both at least min_window_s. The "no swap" story must move those
               places more than min_dshape metres more than the "swapped" one. Judging the whole
               part that moves keeps a false alarm on a long track from costing minutes.
    Then the rest of both tracks, from the closest moment of the pass to where each track ends,
    goes to the other team, in trk["slot_team"] (frames, slots): each slot's team on each frame.
    """

    def __init__(self, pass_m=1.5, pass_max_s=2.0, near_ball_m=5.0, before_s=30.0, after_max_s=300.0,
                 min_window_s=5.0, min_dpos=1.0, min_dshape=0.5):
        self.pass_m = pass_m
        self.pass_max_s = pass_max_s
        self.near_ball_m = near_ball_m
        self.before_s = before_s
        self.after_max_s = after_max_s
        self.min_window_s = min_window_s
        self.min_dpos = min_dpos
        self.min_dshape = min_dshape

    def clean(self, trk):
        P = np.asarray(trk["players"])[:, :, :2]
        ball = np.asarray(trk["ball"])[:, :2]
        team = np.asarray(trk["team"])
        n = len(P)
        slot_team = np.repeat(team[None, :], n, axis=0).astype(np.int8)
        teams = np.unique(team)
        if n == 0 or len(teams) < 2:
            return {**trk, "slot_team": slot_team}
        a, b = teams[0], teams[1]
        present = np.isfinite(P[:, :, 0])
        with np.errstate(invalid="ignore"):
            near = np.linalg.norm(P - ball[:, None, :], axis=2) < self.near_ball_m
        centre = {a: _team_centre(P[:, team == a]), b: _team_centre(P[:, team == b])}
        step, gap = 12, FPS                                # half a second; 1 s left around the pass
        longest = int(round(self.pass_max_s * FPS))

        found = []                                         # (start, end, closest, i, j)
        for i in np.flatnonzero(team == a):
            for j in np.flatnonzero(team == b):
                both = present[:, i] & present[:, j]
                if not (both & (near[:, i] | near[:, j])).any():
                    continue
                d = np.full(n, np.inf)
                d[both] = np.linalg.norm(P[both, i] - P[both, j], axis=1)
                for s, e in _runs(d < self.pass_m):
                    if e - s > longest or not (near[s:e, i] | near[s:e, j]).any():
                        continue
                    if s - step < 0 or e + step > n or not both[s - step:e + step].all():
                        continue
                    found.append((s, e, s + int(np.argmin(d[s:e])), i, j))

        for s, e, m, i, j in sorted(found):
            x, y = slot_team[s, i], slot_team[s, j]
            if x == y:
                continue                                   # an earlier fix already moved one of them
            if self._dpos(P[:, i], P[:, j], s, e - 1, step) <= self.min_dpos:
                continue
            spans = [self._spans(present[:, k], s, e, gap) for k in (i, j)]
            if None in spans:
                continue
            if self._dshape(P[:, i], P[:, j], centre[x], centre[y], *spans) <= self.min_dshape:
                continue
            for k, other in ((i, y), (j, x)):
                end = m + (np.argmin(present[m:, k]) if not present[m:, k].all() else n - m)
                slot_team[m:end, k] = other
        return {**trk, "slot_team": slot_team}

    @staticmethod
    def _dpos(pi, pj, s, e, step):
        """How much further off the "no swap" story lands than the "swapped" one, in metres."""
        a0, a1, b1 = s - step, s - 1, e + step
        t = (b1 - a1) / (a1 - a0)
        pred_i = pi[a1] + (pi[a1] - pi[a0]) * t
        pred_j = pj[a1] + (pj[a1] - pj[a0]) * t
        same = np.linalg.norm(pred_i - pi[b1]) + np.linalg.norm(pred_j - pj[b1])
        cross = np.linalg.norm(pred_i - pj[b1]) + np.linalg.norm(pred_j - pi[b1])
        return same - cross

    def _spans(self, present, s, e, gap):
        """(before, after) slices of one track around a pass: up to before_s of it before, and the
        part the fix would move after (to the track's end, at most after_max_s); None when either
        is shorter than min_window_s."""
        n = len(present)
        lost_before = np.flatnonzero(~present[:s])
        start = max(lost_before[-1] + 1 if len(lost_before) else 0, s - gap - int(self.before_s * FPS))
        lost_after = np.flatnonzero(~present[e:])
        end = min(e + lost_after[0] if len(lost_after) else n, e + gap + int(self.after_max_s * FPS))
        shortest = int(self.min_window_s * FPS)
        if s - gap - start < shortest or end - (e + gap) < shortest:
            return None
        return slice(start, s - gap), slice(e + gap, end)

    @staticmethod
    def _dshape(pi, pj, cx, cy, span_i, span_j):
        """How much more the "no swap" story moves the two players' places than the "swapped"
        one, in metres (-inf when a track is seen too little to tell)."""
        (bi, ai), (bj, aj) = span_i, span_j
        places = [_place(pi[bi], cx[bi]), _place(pj[bj], cy[bj]),
                  _place(pi[ai], cx[ai]), _place(pj[aj], cy[aj]),
                  _place(pi[ai], cy[ai]), _place(pj[aj], cx[aj])]
        if any(p is None for p in places):
            return -np.inf
        i_x, j_y, i_x_after, j_y_after, i_y_after, j_x_after = places
        same = np.linalg.norm(i_x_after - i_x) + np.linalg.norm(j_y_after - j_y)
        swapped = np.linalg.norm(i_y_after - j_y) + np.linalg.norm(j_x_after - i_x)
        return same - swapped


def _tracks(present):
    """Every run of frames a slot is seen: [(slot, start, stop)], and id[frame, slot] = its index."""
    tracks = [(k, s, e) for k in range(present.shape[1]) for s, e in _runs(present[:, k])]
    ids = np.full(present.shape, -1, dtype=np.int32)
    for t, (k, s, e) in enumerate(tracks):
        ids[s:e, k] = t
    return tracks, ids


class FixHandovers:
    """Hand-overs: a home and an away track stand close together with nobody else around, one of
    them ends, and a new track of the OTHER team starts at the pair, so both tracks left are on
    one team (slot teams are fixed, so this is how a pair "becomes one team"). One of the two
    follows the other team's player: it goes to the other team in trk["slot_team"] (frames,
    slots), from the new track's start to the end of that track.

    A hand-over, all distances in metres and times in seconds:
      new track   starts mid-play (on a graded frame, if the tracking says which), lives at least
                  min_life_s, and is not a touchline track (a referee or ball boy: within
                  touchline_m of a side line, or off the pitch, for touchline_share of its life);
      old track   of the other team, ends at most handover_s before the new one starts (or at most
                  overlap_s after), last seen within reach_m + run_mps x gap of where the new one
                  starts. If a track of the new track's own team also ended there, it's a normal
                  re-track: left alone;
      pair        a teammate of the new track, seen without a break from the old track's end to the
                  new one's start, was within pair_m of the old track in its last contact_s;
      alone       no other track within alone_m of the pair when the old track goes and when the
                  new one comes (a same-team copy within dup_m of one of the pair doesn't count);
      two people  the new track gets more than split_m from the teammate within split_s (else it
                  is a copy of him: left alone).
    Which one is wrong: the path check runs the old pair on in a straight line from path_s before
    the old track ended. If the teammate lands where the old track was heading (closer by
    path_margin_m), the teammate's track jumped onto the other player and goes to the other team;
    otherwise (also when the check can't tell) the new track does.
    only_full_team: fix only hand-overs whose new track's team already had 11 players on the
    pitch (median over its first count_s; same-team tracks within dup_m count once)."""

    def __init__(self, pair_m=1.5, alone_m=3.0, handover_s=2.0, overlap_s=1.0, contact_s=1.0,
                 reach_m=2.0, run_mps=7.0, min_life_s=1.0, dup_m=1.0, split_m=2.0, split_s=10.0,
                 touchline_m=1.0, touchline_share=0.8, path_s=0.5, path_margin_m=1.0,
                 only_full_team=False, count_s=2.0, pitch_margin_m=2.0):
        self.pair_m = pair_m
        self.alone_m = alone_m
        self.handover_s = handover_s
        self.overlap_s = overlap_s
        self.contact_s = contact_s
        self.reach_m = reach_m
        self.run_mps = run_mps
        self.min_life_s = min_life_s
        self.dup_m = dup_m
        self.split_m = split_m
        self.split_s = split_s
        self.touchline_m = touchline_m
        self.touchline_share = touchline_share
        self.path_s = path_s
        self.path_margin_m = path_margin_m
        self.only_full_team = only_full_team
        self.count_s = count_s
        self.pitch_margin_m = pitch_margin_m

    def clean(self, trk):
        team = np.asarray(trk["team"])
        n = len(trk["players"])
        slot_team = (np.array(trk["slot_team"], dtype=np.int8) if "slot_team" in trk
                     else np.repeat(team[None, :], n, axis=0).astype(np.int8))
        teams = np.unique(team)
        if len(teams) == 2:
            other = {teams[0]: teams[1], teams[1]: teams[0]}
            for h in self.find(trk):
                k = h["fix"]
                slot_team[h["start"]:h["fix_stop"], k] = other[team[k]]
        return {**trk, "slot_team": slot_team}

    def find(self, trk):
        """The hand-overs to fix: [{slot, start, stop, mate, verdict, fix, fix_stop, old_slot,
        old_stop}]. slot/start/stop is the new track, mate the teammate, old_slot/old_stop the
        track that ended; the fix moves slot `fix` to the other team on frames start..fix_stop."""
        P = np.asarray(trk["players"])[:, :, :2].astype(np.float32)
        team = np.asarray(trk["team"])
        n = len(P)
        if n == 0 or len(np.unique(team)) != 2:
            return []
        graded = np.asarray(trk.get("graded", np.ones(n, dtype=bool)), dtype=bool)
        present = np.isfinite(P[:, :, 0])
        tracks, ids = _tracks(present)
        last = np.array([e - 1 for _, _, e in tracks])
        order = np.argsort(last, kind="stable")
        last_sorted = last[order]
        length, width = trk["pitch"]
        x, y = P[:, :, 0], P[:, :, 1]
        with np.errstate(invalid="ignore"):
            edge = (y < self.touchline_m) | (y > width - self.touchline_m) | (x < 0) | (x > length)
            m = self.pitch_margin_m
            countable = present & (x >= -m) & (x <= length + m) & (y >= -m) & (y <= width + m)
        touch = np.array([edge[s:e, k].mean() >= self.touchline_share for k, s, e in tracks], dtype=bool)
        for t in np.flatnonzero(touch):
            k, s, e = tracks[t]
            countable[s:e, k] = False
        back, ahead = int(round(self.handover_s * FPS)), int(round(self.overlap_s * FPS))
        min_life = int(round(self.min_life_s * FPS))

        found = []
        for t_new, (k, s, e) in enumerate(tracks):
            if touch[t_new] or s == 0 or not graded[s] or e - s < min_life:
                continue
            lo, hi = np.searchsorted(last_sorted, [s - back, s + ahead], side="left")
            same, opp = False, []
            for c in order[lo:hi]:
                k2, s2, e2 = tracks[c]
                if c == t_new or s2 >= s:
                    continue
                l2 = e2 - 1
                gap, ref = ((s - l2) / FPS, P[l2, k2]) if l2 < s else (0.0, P[s, k2])
                if np.linalg.norm(P[s, k] - ref) <= self.reach_m + self.run_mps * gap:
                    if team[k2] == team[k]:
                        same = True
                        break
                    opp.append(c)
            if same:
                continue
            for c in opp:
                h = self._handover(P, team, present, tracks, ids, k, s, c)
                if h is None:
                    continue
                if h != "copy":
                    if self.only_full_team and not self._full(P, team, countable, k, s, e):
                        break
                    i, i_stop, verdict = h
                    jumped = verdict == "old track jumped"
                    found.append({"slot": k, "start": s, "stop": e, "mate": i, "verdict": verdict,
                                  "fix": i if jumped else k, "fix_stop": i_stop if jumped else e,
                                  "old_slot": int(tracks[c][0]), "old_stop": int(tracks[c][2])})
                break
        return found

    def _handover(self, P, team, present, tracks, ids, k, s, j_track):
        """None if new track k (from frame s) is no hand-over from old track j_track, "copy" if k
        never leaves the teammate, else (teammate, end of the teammate's track, path verdict)."""
        j, sj, ej = tracks[j_track]
        lj = ej - 1
        c = min(lj, s - 1)                                 # the old track's last frame before s
        w0 = max(sj, lj - int(round(self.contact_s * FPS)) + 1)
        best, best_d = None, np.inf
        for i in np.flatnonzero(team == team[k]):
            if i == k or ids[c, i] < 0 or ids[c, i] != ids[s, i]:
                continue
            d = np.linalg.norm(P[w0:lj + 1, i] - P[w0:lj + 1, j], axis=1)
            if np.isfinite(d).any() and np.nanmin(d) < best_d:
                best, best_d = i, np.nanmin(d)
        if best is None or best_d > self.pair_m:
            return None
        i = best
        for f, pair, centre in ((c, (i, j), (P[c, i] + P[c, j]) / 2),
                                (s, (i, j, k), (P[s, i] + P[s, k]) / 2)):
            with np.errstate(invalid="ignore"):
                third = present[f] & (np.linalg.norm(P[f] - centre, axis=1) < self.alone_m)
                for q in pair:
                    third[q] = False
                    if present[f, q]:
                        copy = (np.linalg.norm(P[f] - P[f, q], axis=1) < self.dup_m) & (team == team[q])
                        third &= ~copy
            if third.any():
                return None
        i_stop = tracks[ids[s, i]][2]
        hi = min(s + int(round(self.split_s * FPS)), tracks[ids[s, k]][2], i_stop)
        apart = np.linalg.norm(P[s:hi, k] - P[s:hi, i], axis=1)
        if not (np.isfinite(apart).any() and np.nanmax(apart) > self.split_m):
            return "copy"
        return i, i_stop, self._path_verdict(P, i, j, k, c, s)

    def _path_verdict(self, P, i, j, k, c, s):
        """"old track jumped" if teammate i lands where old track j was heading, "new track" if new
        track k does, "can't tell" or None (not testable) otherwise."""
        w = max(1, int(round(self.path_s * FPS)))
        if c - w < 0 or s + w >= len(P):
            return None
        pts = np.array([P[c - w, j], P[c, j], P[c - w, i], P[c, i], P[s + w, k], P[s + w, i]])
        if not np.isfinite(pts).all():
            return None
        t = (s + w - c) / w
        pj = pts[1] + (pts[1] - pts[0]) * t
        pi = pts[3] + (pts[3] - pts[2]) * t
        same = np.linalg.norm(pj - pts[4]) + np.linalg.norm(pi - pts[5])
        cross = np.linalg.norm(pj - pts[5]) + np.linalg.norm(pi - pts[4])
        if same + self.path_margin_m < cross:
            return "new track"
        if cross + self.path_margin_m < same:
            return "old track jumped"
        return "can't tell"

    def _full(self, P, team, countable, k, s, e):
        """Did new track k's team already have 11 players on the pitch over its first count_s?"""
        frames = np.arange(s, s + min(e - s, int(round(self.count_s * FPS))))
        ks = np.flatnonzero((team == team[k]) & (np.arange(len(team)) != k))
        X = P[np.ix_(frames, ks)]
        ok = countable[np.ix_(frames, ks)]
        with np.errstate(invalid="ignore"):
            close = np.linalg.norm(X[:, :, None] - X[:, None, :], axis=3) < self.dup_m
        close &= ok[:, :, None] & ok[:, None, :] & np.tri(len(ks), k=-1, dtype=bool)[None]
        return np.median((ok & ~close.any(axis=2)).sum(axis=1)) >= 11
