"""Per-frame evidence -> H / A / D labels with long, stable possessions (docs/STRATEGIES.md, Level 3).

    ll = decoder.emissions(ev)                     # (n, 3) log-likelihood of H, A, D per frame
    labels = decoder.decode(ll, halves)            # "H" / "A" / "D", one Viterbi pass per half

The switching costs are the knob that sets how many possessions come out: changing team costs
`switch_team`, going between live and dead costs `switch_dead` (in nats, against per-frame
evidence that is itself a log-odds). They were chosen on the Metrica test sets with
leave-one-game-out evidence models (experiments/khanh/train_shape_evidence.py), never on our match.
"""

import numpy as np

STATES = np.array(["H", "A", "D"])
H, A, D = 0, 1, 2

DEFAULTS = {
    "scale": 1.0,          # multiplies both evidence log-odds (frames are not independent)
    "switch_team": 10.0,   # cost of H <-> A
    "switch_dead": 100.0,  # cost of live <-> dead
    "oob": 6.0,            # extra dead log-odds while the ball's last sighting is off the pitch
}


def config(overrides=None):
    return {**DEFAULTS, **(overrides or {})}


def emissions(ev, cfg=None):
    """(n, 3) float64 log P(evidence | H, A, D) up to a constant, from shape_evidence.evidence()."""
    cfg = config(cfg)
    team = np.asarray(ev["team"], dtype=np.float64) * cfg["scale"]
    dead = np.asarray(ev["dead"], dtype=np.float64) * cfg["scale"]
    if "oob" in ev:
        dead = dead + cfg["oob"] * np.asarray(ev["oob"], dtype=np.float64)
    log_live, log_dead = -np.logaddexp(0.0, dead), -np.logaddexp(0.0, -dead)
    log_home, log_away = -np.logaddexp(0.0, -team), -np.logaddexp(0.0, team)
    return np.stack([log_live + log_home, log_live + log_away, log_dead], axis=1)


def viterbi(ll, switch_team, switch_dead):
    """Most likely state path (int8: 0 H, 1 A, 2 D) under the two switching costs.

    Plain Python floats rather than numpy per frame: three states, so the per-call overhead of
    numpy would dominate. About a second per half of our match."""
    ll = np.asarray(ll, dtype=np.float64)
    n = len(ll)
    path = np.zeros(n, dtype=np.int8)
    if n == 0:
        return path
    ct, cd = float(switch_team), float(switch_dead)
    col_h, col_a, col_d = ll[:, H].tolist(), ll[:, A].tolist(), ll[:, D].tolist()
    back_h, back_a, back_d = bytearray(n), bytearray(n), bytearray(n)
    sh, sa, sd = col_h[0], col_a[0], col_d[0]
    for t in range(1, n):
        # into H: stay, from A, from D (ties keep the current state)
        bh, vh = 0, sh
        if sa - ct > vh:
            bh, vh = 1, sa - ct
        if sd - cd > vh:
            bh, vh = 2, sd - cd
        ba, va = 1, sa
        if sh - ct > va:
            ba, va = 0, sh - ct
        if sd - cd > va:
            ba, va = 2, sd - cd
        bd, vd = 2, sd
        if sh - cd > vd:
            bd, vd = 0, sh - cd
        if sa - cd > vd:
            bd, vd = 1, sa - cd
        back_h[t], back_a[t], back_d[t] = bh, ba, bd
        sh, sa, sd = vh + col_h[t], va + col_a[t], vd + col_d[t]
    state = int(np.argmax([sh, sa, sd]))
    backs = (back_h, back_a, back_d)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = backs[state][t]
    return path


def decode(ll, halves=None, cfg=None):
    """Labels for every frame: Viterbi inside each half, "D" outside them (never graded).

    halves: [(period, first, last)] inclusive, like games.half_video_blocks() and
    metrica.halves(); None decodes the whole array as one stretch."""
    cfg = config(cfg)
    n = len(ll)
    labels = np.full(n, "D", dtype="<U1")
    for _, first, last in (halves or [(1, 0, n - 1)]):
        first, last = max(0, int(first)), min(n - 1, int(last))
        if last >= first:
            path = viterbi(ll[first:last + 1], cfg["switch_team"], cfg["switch_dead"])
            labels[first:last + 1] = STATES[path]
    return labels
