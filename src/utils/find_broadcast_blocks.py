"""The shift between the tactical and broadcast videos in each half (tactical frame - broadcast
frame; both are 25 fps, so one number per half), from the two tracking files. Each ATD has one
row per frame of its own video, so no video is decoded.

1. kick-offs, on each camera alone: both teams lined up in their own halves with a player on the
   centre spot, then play; timed by the ball leaving the spot, else by the line-up breaking. The
   first kick-off starts half 1, the first with the teams at the other ends starts half 2.
2. shift: tactical kick-off - broadcast kick-off is the guess; the shift within search_s of it
   that lays the broadcast players and ball best on the tactical ones wins.
"""

import numpy as np

from .series_utils import runs

FPS = 25
CHUNK = 2000


def find(tactical, broadcast, halves, fps=FPS, search_s=5.0, window_s=300.0):
    """One dict per tactical half (first, last): kickoff (tactical, broadcast at the shift),
    detected (the broadcast kick-off as found), shift, matched (share of broadcast players
    within 1 m at that shift); None where not found. tactical, broadcast: {players (N, K, >=2),
    ball (N, >=2), team (K,), pitch}, metres."""
    t_kicks = half_kickoffs(kickoffs(tactical, fps=fps))
    b_kicks = half_kickoffs(kickoffs(broadcast, fps=fps))
    search = int(search_s * fps)
    out = []
    for i, (first, last) in enumerate(halves):
        t_kick = t_kicks[i] if i < len(t_kicks) and first - search <= t_kicks[i] <= last else None
        b_kick = b_kicks[i] if i < len(b_kicks) else None
        row = {"half": i + 1, "kickoff": (None, None), "detected": b_kick, "shift": None,
               "matched": None}
        if t_kick is not None and b_kick is not None:
            stop = min(int(last), t_kick + int(window_s * fps))
            shift, matched = best_shift(tactical, broadcast, t_kick, stop, t_kick - b_kick, search)
            row.update(kickoff=(t_kick, t_kick - shift), shift=shift, matched=matched)
        out.append(row)
    return out


# -- kick-offs ---------------------------------------------------------------------------------

def kickoffs(trk, fps=FPS, min_players=10, min_team=3, own_half=0.9, slack_m=1.0,
             kicker_m=3.0, lined_s=1.0, flicker_s=0.5, play_s=10.0, lag_s=0.9):
    """[(frame, side)] kick-offs; side 1: home lined up at low x, -1: at high x. A kick-off ends
    a line-up of at least lined_s (min_players seen, own_half of each team in its own half,
    breaks under flicker_s ignored) with a player within kicker_m of the centre spot in its
    last second, and is followed by play_s of play (teams mixed, players on camera)."""
    side = lineup_side(trk, min_players, min_team, own_half, slack_m)
    flicker = int(flicker_s * fps)
    side = np.where(_bridge(side == 1, flicker), 1, np.where(_bridge(side == -1, flicker), -1, 0))
    xy = np.asarray(trk["players"][..., :2], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        on_spot = (np.linalg.norm(xy - np.divide(trk["pitch"], 2), axis=2) <= kicker_m).any(1)
    seen = (~np.isnan(xy[..., 0])).sum(1)
    lined, play, out = int(lined_s * fps), int(play_s * fps), []
    for start, stop in runs(side != 0):
        after = slice(stop, stop + play)
        if (stop - start < lined or stop + play > len(side)
                or on_spot[max(start, stop - fps):stop].mean() < 0.5
                or (side[after] != 0).mean() > 0.2 or (seen[after] >= 3).mean() < 0.5):
            continue
        kick = _ball_leaves_spot(trk, stop, fps)
        out.append((kick if kick is not None else stop - int(lag_s * fps), int(side[stop - 1])))
    return out


def half_kickoffs(kicks):
    """[half 1 kick-off, half 2 kick-off] from kickoffs(): the first, then the first with home
    at the other end (teams change ends at half-time); shorter if not found."""
    if not kicks:
        return []
    first, side = kicks[0]
    later = next((frame for frame, s in kicks if frame > first and s == -side), None)
    return [first] + ([later] if later is not None else [])


def lineup_side(trk, min_players=10, min_team=3, own_half=0.9, slack_m=1.0):
    """Per frame: 1 where home lines up at low x and away at high x, -1 the other way, else 0.
    Lined up: own_half of each team's seen players on its side (slack_m over the line)."""
    x = np.asarray(trk["players"][..., 0], dtype=np.float64)
    team, middle = np.asarray(trk["team"]), trk["pitch"][0] / 2
    seen = ~np.isnan(x)
    ok = seen.sum(1) >= min_players
    sides = []
    with np.errstate(invalid="ignore"):
        for code in (0, 1):
            xs, n = x[:, team == code], seen[:, team == code].sum(1)
            ok &= n >= min_team
            sides.append(((xs <= middle + slack_m).sum(1) >= own_half * n,
                          (xs >= middle - slack_m).sum(1) >= own_half * n))
    (h_low, h_high), (a_low, a_high) = sides
    return np.where(ok & h_low & a_high, 1, np.where(ok & h_high & a_low, -1, 0)).astype(np.int8)


def _ball_leaves_spot(trk, stop, fps, spot_m=2.0, still_m=0.5, moved_m=1.0):
    """The kick in the 2 s before frame stop: the ball seen within spot_m of the centre spot, then
    moved_m further out within half a second, traced back to the spot; None if unseen."""
    lo = max(0, stop - 2 * fps)
    ball = np.asarray(trk["ball"][lo:stop + fps // 2, :2], dtype=np.float64)
    r = np.linalg.norm(ball - np.divide(trk["pitch"], 2), axis=1)
    seen = np.flatnonzero(~np.isnan(r))
    for f in seen[(r[seen] > still_m) & (r[seen] <= spot_m)]:
        later = seen[(seen > f) & (seen <= f + fps // 2)]
        if not len(later) or r[later[-1]] < r[f] + moved_m:
            continue
        back = f - int(round(r[f] * (later[-1] - f) / (r[later[-1]] - r[f])))
        resting = seen[(seen >= max(0, back)) & (seen < f) & (r[seen] <= still_m)]
        return lo + (int(resting[-1]) if len(resting) else max(0, back))
    return None


# -- shift -------------------------------------------------------------------------------------

def best_shift(tactical, broadcast, start, stop, guess, search, sigma_m=1.0, ball_weight=4.0):
    """(shift, matched): the shift within guess +- search that lays the broadcast players and
    ball best on the tactical ones over tactical frames start..stop (every 10th frame, then
    every frame around the best), and the share of broadcast players within 1 m there."""
    shifts = np.arange(guess - search, guess + search + 1)
    coarse = [_score(tactical, broadcast, np.arange(start, stop + 1, 10), s, sigma_m,
                     ball_weight)[0] for s in shifts]
    best = int(shifts[int(np.argmax(coarse))])
    near = np.arange(best - 3, best + 4)
    fine = [_score(tactical, broadcast, np.arange(start, stop + 1), s, sigma_m, ball_weight)
            for s in near]
    i = int(np.argmax([f[0] for f in fine]))
    return int(near[i]), fine[i][1]


def _score(tactical, broadcast, t, shift, sigma_m, ball_weight):
    """(kernel score, share within 1 m) of the broadcast at t - shift against the tactical at t."""
    b = t - shift
    keep = (b >= 0) & (b < len(broadcast["ball"]))
    t, b = t[keep], b[keep]
    d = _nearest(tactical["players"], broadcast["players"], t, b)
    bd = np.linalg.norm(np.asarray(tactical["ball"][t, :2], dtype=np.float64)
                        - np.asarray(broadcast["ball"][b, :2], dtype=np.float64), axis=1)
    d, bd = d[~np.isnan(d)], bd[~np.isnan(bd)]
    weight = len(d) + ball_weight * len(bd)
    if not weight:
        return -np.inf, None
    kernel = lambda x: np.exp(-x ** 2 / (2 * sigma_m ** 2)).sum()  # noqa: E731
    score = (kernel(d) + ball_weight * kernel(bd)) / weight
    return float(score), round(float(np.mean(d <= 1.0)), 4) if len(d) else None


def _nearest(t_players, b_players, t, b):
    """(len(t), K) distance from each broadcast player at frame b to the nearest tactical
    player at frame t; NaN where the broadcast slot is empty or nobody is on the tactical."""
    out = np.full((len(t), b_players.shape[1]), np.nan, np.float32)
    for a in range(0, len(t), CHUNK):
        tp = np.asarray(t_players[t[a:a + CHUNK], :, :2], dtype=np.float32)
        bp = np.asarray(b_players[b[a:a + CHUNK], :, :2], dtype=np.float32)
        gap = np.linalg.norm(bp[:, :, None] - tp[:, None], axis=-1)
        gap = np.where(np.isnan(gap), np.inf, gap).min(axis=2)
        out[a:a + CHUNK] = np.where(np.isinf(gap), np.nan, gap)
    return out


# -- helpers -----------------------------------------------------------------------------------

def _bridge(mask, frames):
    """mask with False gaps of at most `frames` between two True runs set True."""
    out = np.asarray(mask, dtype=bool).copy()
    spans = runs(out)
    for (_, stop), (start, _) in zip(spans, spans[1:]):
        if start - stop <= frames:
            out[stop:start] = True
    return out
