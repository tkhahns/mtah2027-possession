"""Does a player's place in the team shape fit his own team, or the other one?

Per half, each team attacks away from where its players stand on average; positions are turned
so each team attacks towards +x, then taken relative to the team's median player.
1. formation: every step_s, a team's 11 roles over +-window_s (k-means on the relative positions
   of its tracked players, from frames where it shows 5 to 11 players, so a player of the other
   team counted in it (12) doesn't become a role; side-line and off-pitch tracks left out);
2. fit, on the given frames: each team's players go to its roles, closest pairs first. A player's
   own cost is the distance to his role (none left: lam_m). His other cost is the distance to the
   nearest role of the other team that he would fill better than its current player, or an empty
   one (none: lam_m). support = own - other: above 0, he fits the other team better.
"""

import numpy as np

FPS = 25
K = 11


def attack_signs(players, team, halves, pitch):
    """{(half index, team): +1 if that team attacks towards +x, else -1}."""
    out = {}
    for h, (first, last) in enumerate(halves):
        x = np.asarray(players[first:last + 1, :, 0], dtype=np.float64)
        for code in (0, 1):
            centre = np.nanmedian(x[:, np.asarray(team) == code])
            out[(h, code)] = 1 if centre < pitch[0] / 2 else -1
    return out


def oriented(xy, sign, pitch):
    """xy turned so the team attacks towards +x (a half turn when sign is -1)."""
    return xy if sign > 0 else np.asarray(pitch) - xy


def formations(players, team, halves, pitch, window_s, step_s=30.0, every=10, seed=0,
               touchline_m=1.0, max_points=4000, restarts=1):
    """Per half: (grid frames (G,), roles (G, 2 teams, K, 2)) in each team's attacking frame,
    relative to the team's median player; k-means keeps the best of `restarts` starts."""
    team = np.asarray(team)
    signs = attack_signs(players, team, halves, pitch)
    rng = np.random.default_rng(seed)
    out = []
    for h, (first, last) in enumerate(halves):
        grid = np.arange(first, last + 1, int(step_s * FPS))
        roles = np.full((len(grid), 2, K, 2), np.nan)
        for g, f in enumerate(grid):
            lo = max(first, f - int(window_s * FPS))
            hi = min(last, f + int(window_s * FPS))
            xy = np.asarray(players[lo:hi + 1:every, :, :2], dtype=np.float64)
            for code in (0, 1):
                pts = _relative(xy[:, team == code], signs[(h, code)], pitch, touchline_m)
                if len(pts) > max_points:
                    pts = pts[rng.choice(len(pts), max_points, replace=False)]
                if len(pts) >= 5 * K:
                    roles[g, code] = min((_kmeans(pts, K, rng) for _ in range(restarts)),
                                         key=lambda c: _inertia(pts, c))
        out.append((grid, roles))
    return out


def support(players, team, halves, pitch, forms, frames, lam_m=10.0, touchline_m=1.0):
    """(own, other) costs in metres, (len(frames), slots), NaN where the slot is unseen, on the
    pitch's side line or its team has fewer than 5 players; support = own - other."""
    team = np.asarray(team)
    signs = attack_signs(players, team, halves, pitch)
    own = np.full((len(frames), players.shape[1]), np.nan)
    other = np.full_like(own, np.nan)
    for i, f in enumerate(frames):
        h = next((h for h, (a, z) in enumerate(halves) if a <= f <= z), None)
        if h is None:
            continue
        grid, roles = forms[h]
        g = int(np.argmin(np.abs(grid - f)))
        xy = np.asarray(players[f, :, :2], dtype=np.float64)
        ok = ~np.isnan(xy[:, 0]) & ~_on_line(xy, pitch, touchline_m)
        fits = {}
        for code in (0, 1):
            slots = np.flatnonzero(ok & (team == code))
            if len(slots) < 5 or np.isnan(roles[g, code]).any():
                continue
            turned = oriented(xy[slots], signs[(h, code)], pitch)
            rel = turned - np.median(turned, axis=0)
            d = np.linalg.norm(rel[:, None] - roles[g, code][None], axis=-1)
            cost, occupied = _greedy(d, lam_m)
            fits[code] = (slots, cost, occupied, np.median(turned, axis=0))
        for code in fits:
            o = 1 - code
            if o not in fits:
                continue
            slots, cost, _, _ = fits[code]
            _, _, occupied, centre = fits[o]
            rel = oriented(xy[slots], signs[(h, o)], pitch) - centre
            d = np.linalg.norm(rel[:, None] - roles[g, o][None], axis=-1)
            better = np.where(d < occupied[None], d, np.inf).min(axis=1)
            own[i, slots] = cost
            other[i, slots] = np.minimum(better, lam_m)
    return own, other


def _greedy(d, lam_m):
    """(cost per player, cost of each role's player (inf if empty)): closest pairs first;
    players left over cost lam_m, and no pair costs more than lam_m."""
    n, k = d.shape
    cost = np.full(n, lam_m)
    occupied = np.full(k, np.inf)
    used_p, used_r = np.zeros(n, bool), np.zeros(k, bool)
    for flat in np.argsort(d, axis=None):
        p, r = divmod(int(flat), k)
        if used_p[p] or used_r[r]:
            continue
        used_p[p] = used_r[r] = True
        cost[p] = occupied[r] = min(d[p, r], lam_m)
        if used_p.all() or used_r.all():
            break
    return cost, occupied


def _relative(xy, sign, pitch, touchline_m):
    """(points, 2) positions relative to each frame's team median, in the attacking frame, from
    frames with 5 to K players (all frames with 5+ if those give too few points)."""
    seen = ~np.isnan(xy[..., 0]) & ~_on_line(xy, pitch, touchline_m)
    turned = oriented(xy, sign, pitch)
    with np.errstate(invalid="ignore"):
        masked = np.where(seen[..., None], turned, np.nan)
    n = seen.sum(1)
    enough = (n >= 5) & (n <= K)
    if n[enough].sum() < 5 * K:
        enough = n >= 5
    if not enough.any():
        return np.zeros((0, 2))
    centre = np.nanmedian(masked[enough], axis=1)
    rel = masked[enough] - centre[:, None]
    return rel[seen[enough]]


def _on_line(xy, pitch, margin):
    x, y = xy[..., 0], xy[..., 1]
    with np.errstate(invalid="ignore"):
        return (y <= margin) | (y >= pitch[1] - margin) | (x < 0) | (x > pitch[0])


def _inertia(points, centres):
    return float(((points[:, None] - centres[None]) ** 2).sum(-1).min(1).sum())


def _kmeans(points, k, rng, iters=15):
    """(k, 2) centres: k-means++ start, then Lloyd steps."""
    centres = [points[rng.integers(len(points))]]
    for _ in range(1, k):
        d2 = np.min(((points[:, None] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        centres.append(points[rng.choice(len(points), p=d2 / d2.sum())])
    centres = np.array(centres)
    for _ in range(iters):
        label = ((points[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        for c in range(k):
            if (label == c).any():
                centres[c] = points[label == c].mean(0)
    return centres
