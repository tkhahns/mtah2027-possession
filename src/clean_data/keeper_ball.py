"""The ball in a goalkeeper's hands (step 2b of clean_data.pipeline: after the keeper-slot repair,
before the interpolation and the dead reckoning).

A ball gap is a run of frames with no ball x, y. When a gap is longer than min_frames (25 = 1 s)
and, on the last frame before it, the ball is within radius_m of a goalkeeper, the keeper is taken
to hold the ball: on every frame of the gap the ball gets that keeper's x, y and speed, following
the team's keeper slot frame by frame, until the ball is tracked again. Frames where the keeper is
not tracked stay empty (step 3 interpolates them like any other gap), and the hold stops at the
end of the half.

    ball, held, gaps = hold_ball(players, ball, slots, half_blocks, radius_m=1.0)

players are the keeper-repaired players (keeper.fix_keepers), slots the (frames, 2) home / away
keeper slots of keeper.find_slots.
"""

import numpy as np

MIN_FRAMES = 25      # only gaps longer than 1 s


def gaps(tracked):
    """(starts, stops) of every run of False in a (frames,) bool mask, stop exclusive."""
    edges = np.diff(np.r_[0, (~np.asarray(tracked, bool)).astype(np.int8), 0])
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def hold_ball(players, ball, slots, half_blocks, radius_m, min_frames=MIN_FRAMES):
    """Fill each long ball gap that starts next to a keeper with that keeper's track.

    players      (frames, slots, 3) x, y, speed in metres, NaN where empty
    ball         (frames, 3)
    slots        (frames, 2) home / away keeper slot per frame, -1 where not known
    half_blocks  [(half, first, last), ...] inclusive; a hold never runs past its half's end
    radius_m     the ball's last tracked position must be at most this far from the keeper
    min_frames   only gaps longer than this are held

    Returns (a filled copy of ball, held: (frames,) bool, one dict per held gap: first, last
    frame of the gap, until = last frame of the hold (the gap's last frame or the half's), team
    (0 home, 1 away), the keeper's distance in metres, cells filled)."""
    players, ball = np.asarray(players), np.asarray(ball)
    slots = np.broadcast_to(np.asarray(slots, dtype=np.int64), (len(ball), 2))
    out = ball.copy()
    held = np.zeros(len(ball), dtype=bool)
    half_end = np.full(len(ball), -1, dtype=np.int64)
    for _, first, last in half_blocks:
        half_end[first:last + 1] = last
    done = []
    starts, stops = gaps(~np.isnan(ball[:, 0]))
    for s, e in zip(starts, stops):
        if e - s <= min_frames or s == 0 or half_end[s - 1] < s:   # also: s - 1 ends the half
            continue
        f = s - 1
        best, team = np.inf, -1
        for t in (0, 1):
            k = slots[f, t]
            if k < 0 or np.isnan(players[f, k, 0]):
                continue
            d = float(np.hypot(*(players[f, k, :2].astype(np.float64) - ball[f, :2])))
            if d < best:
                best, team = d, t
        if team < 0 or best > radius_m:
            continue
        frames = np.arange(s, min(e, half_end[f] + 1))
        until = int(frames[-1])
        k = slots[frames, team]
        ok = k >= 0
        frames, k = frames[ok], k[ok]
        keep = players[frames, k]
        ok = ~np.isnan(keep[:, 0])
        out[frames[ok]] = keep[ok]
        held[frames[ok]] = True
        done.append({"first": int(s), "last": int(e - 1), "until": until, "team": team,
                     "distance_m": round(best, 3), "cells": int(ok.sum())})
    return out, held, done
