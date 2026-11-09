"""Team shape, centroid and velocity per frame: identity-free evidence for restarts and possession.

    import team_shape
    shape = team_shape.frame_features(trk)          # dict of arrays, first axis = frame

Every feature depends only on the *set* of visible points of each team in a frame, never on which
slot a point sits in, so it survives our tracking's faults: tracklets that appear and vanish, ~20.5
of 22 players seen, a few duplicates and identity swaps between slots (docs/DATA_SCHEMA.md).

Velocities are the mean of per-slot displacements over a short window, not the derivative of the
centroid: a player appearing or vanishing jumps a team's centroid by metres but leaves the mean
velocity alone. A slot that "moves" faster than MAX_SPEED is a swap or a re-used tracklet and is
left out.

When the ball is missing (41% of our graded frames) the near-ball features use an estimate of
where it is: linear between two sightings for short gaps, the last sighting held for a while
after that, then nothing.

Team index 0 = home (H), 1 = away (A), as in config.py. Units are metres, seconds, m/s.
"""

import numpy as np

from config import AWAY, HOME, PITCH

FPS = 25

VEL_HALF_WINDOW = 5            # frames each side for displacement velocities (0.4 s in total)
MAX_SPEED = 12.0               # m/s: faster than any sprint, so a slot swap or a new tracklet
STILL_SPEED = 1.0              # m/s: below this a player is standing or walking slowly
NEAR_M = 12.0                  # radius of the "near the ball" group
INTERP_GAP = 50                # frames: interpolate the ball across gaps up to 2 s ...
HOLD_GAP = 125                 # ... and hold its last sighting up to 5 s into a longer gap
CONTEST_M = 2.0                # both teams have a player this close to the ball ...
CONTEST_GAP_M = 1.0            # ... and neither is clearly closer: a contest, nobody in control
PENALTY_DEPTH, PENALTY_WIDTH = 16.5, 40.3


def pitch_of(trk):
    return tuple(float(v) for v in trk.get("pitch", PITCH))


def _team_moments(xy, valid):
    """(count, centroid (n,2), std (n,2)) of the valid points of each frame; NaN if none."""
    count = valid.sum(axis=1)
    safe = np.where(valid[..., None], xy, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        centre = safe.sum(axis=1) / count[:, None]
        delta = np.where(valid[..., None], xy - centre[:, None], 0.0)
        var = (delta * delta).sum(axis=1) / count[:, None]
    centre[count == 0] = np.nan
    var[count == 0] = np.nan
    return count.astype(np.int16), centre.astype(np.float32), np.sqrt(var).astype(np.float32)


def slot_velocity(xy, half_window=VEL_HALF_WINDOW, fps=FPS, max_speed=MAX_SPEED):
    """(frames, slots, 2) m/s from displacement over +-half_window frames; NaN where either end
    is missing or the implied speed is not humanly possible (a swap or a new tracklet)."""
    out = np.full(xy.shape, np.nan, dtype=np.float32)
    h = int(half_window)
    if len(xy) <= 2 * h:
        return out
    v = (xy[2 * h:] - xy[:-2 * h]) * (fps / (2.0 * h))
    with np.errstate(invalid="ignore"):
        v[np.linalg.norm(v, axis=-1) > max_speed] = np.nan
    out[h:-h] = v
    return out


def attack_direction(team_centroid_x, halves=None):
    """(frames, 2) +1 where a team attacks towards x = length, -1 towards x = 0, 0 unknown.

    Per half, the team whose centroid is further back (lower x) on most frames defends the
    goal at x = 0 and so attacks +x. A median over a whole half is untouched by any one
    passage of play. halves: [(period, first, last)] with inclusive frames, like
    games.half_video_blocks(); None means one block for the whole array.
    """
    n = len(team_centroid_x)
    out = np.zeros((n, 2), dtype=np.int8)
    blocks = halves if halves else [(1, 0, n - 1)]
    for _, first, last in blocks:
        first, last = max(0, int(first)), min(n - 1, int(last))
        if last < first:
            continue
        part = team_centroid_x[first:last + 1]
        ok = np.isfinite(part).all(axis=1)
        if ok.sum() < FPS:                       # under a second of both teams: no opinion
            continue
        home_behind = np.median(part[ok, HOME] - part[ok, AWAY]) < 0
        out[first:last + 1, HOME] = 1 if home_behind else -1
        out[first:last + 1, AWAY] = -out[first:last + 1, HOME]
    return out


def ball_estimate(ball_xy, interp_gap=INTERP_GAP, hold_gap=HOLD_GAP):
    """(frames, 2) the ball where seen; across a gap of at most interp_gap frames, on the straight
    line between the two sightings; otherwise its last sighting for hold_gap frames; else NaN."""
    ball_xy = np.asarray(ball_xy, dtype=np.float32)
    n = len(ball_xy)
    seen = np.isfinite(ball_xy).all(axis=1)
    out = np.where(seen[:, None], ball_xy, np.nan).astype(np.float32)
    if not seen.any():
        return out
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(seen, idx, -1))
    nxt = np.where(seen, idx, n)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    gap = ~seen & (prev >= 0)
    both = gap & (nxt < n) & ((nxt - prev - 1) <= interp_gap)
    if both.any():
        p, q = prev[both], nxt[both]
        w = ((idx[both] - p) / (q - p).astype(np.float32))[:, None]
        out[both] = ball_xy[p] * (1 - w) + ball_xy[q] * w
    hold = gap & ~both & ((idx - prev) <= hold_gap)
    out[hold] = ball_xy[prev[hold]]
    return out


def _speed_of(xy, half_window=VEL_HALF_WINDOW, fps=FPS):
    """(frames,) speed of one point (the ball) from displacement; NaN where an end is missing."""
    out = np.full(len(xy), np.nan, dtype=np.float32)
    h = int(half_window)
    if len(xy) > 2 * h:
        out[h:-h] = np.linalg.norm(xy[2 * h:] - xy[:-2 * h], axis=1) * (fps / (2.0 * h))
    return out


def frame_features(trk, halves=None, near_m=NEAR_M):
    """Per-frame shape, centroid and velocity features of both teams, and of the players near
    the ball. Arrays (first axis = frame):

    count (n,2)            visible players per team
    centroid (n,2,2)       team centroid (x, y)
    spread (n,2,2)         team standard deviation along x and y: how stretched it is
    all_centroid (n,2), all_spread (n,2)   the same for every visible player
    velocity (n,2,2)       team mean velocity (from per-slot displacements)
    speed (n,2)            team mean player speed
    still (n,)             share of visible players slower than STILL_SPEED
    attack (n,2)           +1 / -1: the direction along x each team attacks (0 unknown)
    forward (n,2)          velocity along each team's attacking direction
    ball (n,2), ball_seen (n,), ball_speed (n,)   ball estimate (see ball_estimate) and speed
    nearest (n,2)          distance from the ball estimate to each team's nearest player
    near_count (n,2)       players of each team within near_m of the ball estimate
    near_speed (n,)        mean speed of those players: rushing (live) or easing off (dead)
    closing (n,2)          mean speed of each team's near players *towards* the ball (m/s;
                           negative = moving away): the defending side presses, a stopped
                           game has nobody closing in
    centroid_speed (n,)    speed of the centroid of every player (mean velocity of all players)
    contest (n,)           ball seen, both teams within CONTEST_M of it, neither clearly closer
    near_centroid (n,2), near_spread (n,)    centroid and spread (rms radius) of that group
    box_count (n,2)        players inside the left (x=0) and right penalty areas
    """
    players = np.asarray(trk["players"], dtype=np.float32)
    team = np.asarray(trk["team"])
    length, width = pitch_of(trk)
    halves = halves if halves is not None else trk.get("halves")
    xy = players[..., :2]
    n = len(xy)
    valid = np.isfinite(xy).all(axis=2)

    count = np.zeros((n, 2), np.int16)
    centroid = np.full((n, 2, 2), np.nan, np.float32)
    spread = np.full((n, 2, 2), np.nan, np.float32)
    velocity = np.full((n, 2, 2), np.nan, np.float32)
    speed = np.full((n, 2), np.nan, np.float32)
    vel = slot_velocity(xy)
    vel_ok = np.isfinite(vel).all(axis=2)
    slot_speed = np.linalg.norm(np.where(vel_ok[..., None], vel, 0.0), axis=2)
    for k in (HOME, AWAY):
        cols = np.flatnonzero(team == k)
        count[:, k], centroid[:, k], spread[:, k] = _team_moments(xy[:, cols], valid[:, cols])
        ok = vel_ok[:, cols]
        m = ok.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            velocity[:, k] = np.where(ok[..., None], vel[:, cols], 0.0).sum(axis=1) / m[:, None]
            speed[:, k] = np.where(ok, slot_speed[:, cols], 0.0).sum(axis=1) / m
        velocity[m == 0, k] = np.nan
        speed[m == 0, k] = np.nan
    _, all_centroid, all_spread = _team_moments(xy, valid)
    with np.errstate(invalid="ignore", divide="ignore"):
        still = ((slot_speed < STILL_SPEED) & vel_ok).sum(axis=1) / vel_ok.sum(axis=1)
    still = still.astype(np.float32)

    attack = attack_direction(centroid[:, :, 0], halves)
    forward = velocity[:, :, 0] * attack

    ball_raw = np.asarray(trk["ball"], dtype=np.float32)[:, :2]
    ball_seen = np.isfinite(ball_raw).all(axis=1)
    ball = ball_estimate(ball_raw)
    ball_speed = _speed_of(np.where(ball_seen[:, None], ball_raw, np.nan))

    dist = np.linalg.norm(xy - ball[:, None], axis=2)                       # NaN-safe below
    near = valid & np.isfinite(dist) & (dist <= near_m)
    nearest = np.full((n, 2), np.nan, np.float32)
    near_count = np.zeros((n, 2), np.int16)
    for k in (HOME, AWAY):
        cols = np.flatnonzero(team == k)
        d = np.where(valid[:, cols] & np.isfinite(dist[:, cols]), dist[:, cols], np.inf)
        best = d.min(axis=1) if len(cols) else np.full(n, np.inf)
        nearest[:, k] = np.where(np.isfinite(best), best, np.nan)
        near_count[:, k] = near[:, cols].sum(axis=1)
    # Proximity: are the players near the ball rushing at it, or easing off?
    to_ball = ball[:, None] - xy
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = to_ball / np.linalg.norm(to_ball, axis=2, keepdims=True)
        toward = (vel * unit).sum(axis=2)                                    # m/s towards the ball
    moving_near = near & vel_ok & np.isfinite(toward)
    with np.errstate(invalid="ignore", divide="ignore"):
        near_speed = (np.where(moving_near, slot_speed, 0.0).sum(axis=1)
                      / moving_near.sum(axis=1)).astype(np.float32)
    closing = np.full((n, 2), np.nan, np.float32)
    for k in (HOME, AWAY):
        cols = np.flatnonzero(team == k)
        m = moving_near[:, cols]
        with np.errstate(invalid="ignore", divide="ignore"):
            closing[:, k] = np.where(m, toward[:, cols], 0.0).sum(axis=1) / m.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        all_velocity = np.where(vel_ok[..., None], vel, 0.0).sum(axis=1) / vel_ok.sum(axis=1)[:, None]
    centroid_speed = np.linalg.norm(all_velocity, axis=1).astype(np.float32)
    with np.errstate(invalid="ignore"):
        contest = (ball_seen & (nearest[:, HOME] <= CONTEST_M) & (nearest[:, AWAY] <= CONTEST_M)
                   & (np.abs(nearest[:, HOME] - nearest[:, AWAY]) < CONTEST_GAP_M))
    _, near_centroid, near_std = _team_moments(xy, near)
    near_spread = np.sqrt(np.nansum(near_std ** 2, axis=1)).astype(np.float32)
    near_spread[~np.isfinite(near_std).all(axis=1)] = np.nan

    in_band = valid & (np.abs(xy[..., 1] - width / 2) <= PENALTY_WIDTH / 2)
    box_count = np.stack([(in_band & (xy[..., 0] <= PENALTY_DEPTH)).sum(axis=1),
                          (in_band & (xy[..., 0] >= length - PENALTY_DEPTH)).sum(axis=1)],
                         axis=1).astype(np.int16)

    return {
        "count": count, "centroid": centroid, "spread": spread,
        "all_centroid": all_centroid, "all_spread": all_spread,
        "velocity": velocity, "speed": speed, "still": still,
        "attack": attack, "forward": forward.astype(np.float32),
        "ball": ball, "ball_seen": ball_seen, "ball_speed": ball_speed,
        "nearest": nearest, "near_count": near_count, "near_speed": near_speed,
        "closing": closing, "centroid_speed": centroid_speed, "contest": contest,
        "near_centroid": near_centroid, "near_spread": near_spread,
        "box_count": box_count, "pitch": (length, width),
    }


# -- windowed summaries ----------------------------------------------------------------------

def window_mean(values, before, after):
    """Mean of values[t-before .. t+after] per frame (along axis 0), ignoring NaN; NaN when the
    window has no value. Cumulative sums, so the window size costs nothing."""
    values = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(values)
    total = np.concatenate([np.zeros((1,) + values.shape[1:]), np.cumsum(np.where(ok, values, 0.0), axis=0)])
    count = np.concatenate([np.zeros((1,) + values.shape[1:]), np.cumsum(ok, axis=0)])
    idx = np.arange(len(values))
    lo = np.clip(idx - int(before), 0, len(values))
    hi = np.clip(idx + int(after) + 1, 0, len(values))
    with np.errstate(invalid="ignore", divide="ignore"):
        return (total[hi] - total[lo]) / (count[hi] - count[lo])
