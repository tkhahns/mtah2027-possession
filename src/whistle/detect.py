"""Referee whistle events from the per-frame tones of features.py. 

Steps, all frame-based (25 fps):

1. chord frames  the best pair of tones on the frame with low in [lo_min, lo_max] Hz,
                 high / low in [ratio_min, ratio_max] and both tones' p1 >= p1_min dB.
2. events        chord frames at most `gap` frames apart are one event.
3. keep          an event needs >= min_hits chord frames, at most max_len frames, a steady pitch
                 (>= steady of its chord frames within +-steady_hz of their median low tone), a
                 median chord strength (the weaker tone's p1) >= min_strength dB, and a median
                 loudness of the low tone >= min_loud dB above the match's usual level at its pitch,
                 and a median high tone (low x ratio) <= hi_max Hz. An event whose clearest chord
                 frame is <= weak_db dB (0 = rule off) is dropped when the AROUND frames on each
                 side of it hold on average >= busy_around tones as clear as p1_min (a weak
                 whistle in a busy crowd).
4. edges         for the picture only: start/end grow over neighbouring frames that still hold the
                 low tone (+-steady_hz, p1 >= p1_min), up to `extend` frames each way.
5. cool-down     a whistle starting less than cooldown_s seconds after the start of the last KEPT
                 whistle is ignored (the user's rule: one stoppage, one whistle). An ignored whistle
                 does not start a cool-down of its own. 0 = off.

When in doubt it drops a whistle rather than invent one: every setting is a minimum to clear.
"""

import numpy as np

DEFAULTS = {
    "lo_min": 3600.0, "lo_max": 4000.0,      # Hz, the chord's low tone
    "hi_max": 4300.0,                        # Hz, the event's median high tone (low x ratio)
    "ratio_min": 1.09, "ratio_max": 1.13,    # high tone / low tone
    "p1_min": 9.0,                           # dB above the local floor, each tone
    "gap": 3,                                # frames
    "min_hits": 3,                           # chord frames per event
    "max_len": 125,                          # frames (5 s)
    "steady_hz": 80.0, "steady": 0.8,
    "min_strength": 9.0,                     # dB, median of the weaker tone's p1 over chord frames
    "min_loud": 12.0,                        # dB, median loudness of the low tone above the match's usual
    "weak_db": 13.0,                         # dB: an event never clearer than this is weak (0 = rule off)
    "busy_around": 2.2,                      # clear tones per frame in the 2 s around that make the crowd busy
    "extend": 0,                             # frames
    "cooldown_s": 12.0,                      # s: whistles this soon after a kept one are ignored (0 = off)
}
INTS = ("gap", "min_hits", "max_len", "extend")
AROUND = 50                                  # frames (2 s) each side looked at for a busy crowd

def clean_params(params=None):
    """DEFAULTS overridden by `params`, cast to numbers and checked; unknown keys dropped."""
    p = dict(DEFAULTS)
    p.update({k: v for k, v in (params or {}).items() if k in DEFAULTS and v is not None})
    for k in DEFAULTS:
        p[k] = int(round(float(p[k]))) if k in INTS else float(p[k])
    if not (p["lo_min"] < p["lo_max"] and p["ratio_min"] < p["ratio_max"] and p["gap"] >= 0
            and p["min_hits"] >= 1 and p["max_len"] >= 1 and 0 <= p["steady"] <= 1 and p["extend"] >= 0
            and p["weak_db"] >= 0 and p["busy_around"] >= 0 and p["cooldown_s"] >= 0):
        raise ValueError("bad whistle settings: {}".format(p))
    return p


def chord_frames(f, p1, params=None, lvl=None):
    """Per frame: (strength, low_hz, ratio, loud) of the best chord, strength = weaker tone's p1,
    loud = the low tone's lvl (NaN when lvl is not given).

    f, p1, lvl: (n_frames, k) tones per frame (NaN = no tone). Frames without a chord: strength
    -inf, the rest NaN.
    """
    p = clean_params(params)
    f_all = np.asarray(f, dtype=np.float32)
    n, k = f_all.shape
    out = (np.full(n, -np.inf, dtype=np.float32), np.full(n, np.nan, dtype=np.float32),
           np.full(n, np.nan, dtype=np.float32), np.full(n, np.nan, dtype=np.float32))
    p1_all = np.asarray(p1, dtype=np.float32)
    with np.errstate(invalid="ignore"):
        cand = ((f_all >= p["lo_min"]) & (f_all <= p["lo_max"]) & (p1_all >= p["p1_min"])).any(axis=1)
    rows = np.flatnonzero(cand)              # only frames holding a possible low tone (a few %)
    if not len(rows):
        return out
    f = f_all[rows]
    p1 = np.nan_to_num(p1_all[rows], nan=-np.inf)
    lvl = np.full((len(rows), k), np.nan, dtype=np.float32) if lvl is None else np.asarray(lvl, dtype=np.float32)[rows]
    strength, low, ratio, loud = (np.full(len(rows), -np.inf, dtype=np.float32),
                                  *(np.full(len(rows), np.nan, dtype=np.float32) for _ in range(3)))
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in range(k):
            lo_ok = (f[:, i] >= p["lo_min"]) & (f[:, i] <= p["lo_max"]) & (p1[:, i] >= p["p1_min"])
            if not lo_ok.any():
                continue
            for j in range(k):
                if j == i:
                    continue
                r = f[:, j] / f[:, i]
                s = np.minimum(p1[:, i], p1[:, j])
                ok = lo_ok & (r >= p["ratio_min"]) & (r <= p["ratio_max"]) & (p1[:, j] >= p["p1_min"]) & (s > strength)
                strength[ok], low[ok], ratio[ok], loud[ok] = s[ok], f[ok, i], r[ok], lvl[ok, i]
    for full, part in zip(out, (strength, low, ratio, loud)):
        full[rows] = part
    return out


def _spans(on, gap):
    """[start, stop) runs of True, joined across gaps of at most `gap` False frames."""
    idx = np.flatnonzero(on)
    if not len(idx):
        return []
    breaks = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    stops = np.concatenate((idx[breaks], [idx[-1]])) + 1
    return list(zip(starts.tolist(), stops.tolist()))


def _grow(f, p1, start, stop, centre, p):
    """Move start back / stop forward while the neighbouring frame still holds the low tone."""
    a, b = max(0, start - p["extend"]), min(len(f), stop + p["extend"])
    with np.errstate(invalid="ignore"):
        near = (np.abs(f[a:b] - centre) <= p["steady_hz"]) & (p1[a:b] >= p["p1_min"])
    has = near.any(axis=1)                   # has[i] is frame a + i
    lo = start
    while lo > a and has[lo - 1 - a]:
        lo -= 1
    hi = stop
    while hi < b and has[hi - a]:
        hi += 1
    return lo, hi


def find_events(f, p1, lvl=None, params=None, fps=25):
    """Whistle events (steps 1-5), in order. See find_candidates for the fields."""
    p = clean_params(params)
    return cool_down(find_candidates(f, p1, lvl, p), p["cooldown_s"], fps)[0]


def cool_down(events, seconds, fps=25):
    """(kept, ignored): an event starting less than `seconds` after the start of the last kept one
    is ignored; ignored ones get after_frame = start of the kept whistle that hides them."""
    kept, ignored = [], []
    for e in events:
        if kept and seconds > 0 and e["start_frame"] - kept[-1]["start_frame"] < seconds * fps:
            ignored.append(dict(e, after_frame=kept[-1]["start_frame"]))
        else:
            kept.append(e)
    return kept, ignored


def find_candidates(f, p1, lvl=None, params=None):
    """Whistle events of steps 1-4 (before the cool-down), in order: dicts with start_frame,
    end_frame (inclusive), peak_frame, chord_frames, low_hz, high_hz, ratio, strength_db, loud_db,
    peak_db, busy. Without lvl the loudness test is skipped. peak_db = clearest chord frame;
    busy = tones as clear as p1_min per frame in the AROUND frames on each side."""
    p = clean_params(params)
    f, p1 = np.asarray(f, dtype=np.float32), np.asarray(p1, dtype=np.float32)
    strength, low, ratio, loud = chord_frames(f, p1, p, lvl)
    on = strength >= p["p1_min"]
    with np.errstate(invalid="ignore"):
        clear = (p1 >= p["p1_min"]).sum(axis=1)      # clear tones per frame
    events = []
    for start, stop in _spans(on, p["gap"]):
        idx = start + np.flatnonzero(on[start:stop])
        if len(idx) < p["min_hits"] or stop - start > p["max_len"]:
            continue
        centre = float(np.median(low[idx]))
        if np.mean(np.abs(low[idx] - centre) <= p["steady_hz"]) < p["steady"]:
            continue
        med_strength = float(np.median(strength[idx]))
        if med_strength < p["min_strength"]:
            continue
        high = float(np.median(low[idx] * ratio[idx]))
        if high > p["hi_max"]:
            continue
        med_loud = float(np.median(loud[idx])) if lvl is not None else float("nan")
        if lvl is not None and not med_loud >= p["min_loud"]:
            continue
        peak = float(np.max(strength[idx]))
        around = np.r_[max(0, start - AROUND):start, stop:min(len(clear), stop + AROUND)]
        busy = float(np.mean(clear[around])) if len(around) else 0.0
        if peak <= p["weak_db"] and busy >= p["busy_around"]:
            continue
        lo, hi = _grow(f, p1, start, stop, centre, p) if p["extend"] else (start, stop)
        events.append({"start_frame": int(lo), "end_frame": int(hi - 1),
                       "peak_frame": int(idx[np.argmax(strength[idx])]), "chord_frames": int(len(idx)),
                       "low_hz": round(centre), "high_hz": round(high),
                       "ratio": round(float(np.median(ratio[idx])), 3),
                       "strength_db": round(med_strength, 1),
                       "loud_db": None if lvl is None else round(med_loud, 1),
                       "peak_db": round(peak, 1), "busy": round(busy, 2)})
    return events


def frame_mask(events, n_frames):
    """Boolean per frame: inside a whistle event."""
    mask = np.zeros(n_frames, dtype=bool)
    for ev in events:
        mask[ev["start_frame"]:ev["end_frame"] + 1] = True
    return mask


def overlaps(events, ranges):
    """For each (first, last) range: True when some event overlaps it."""
    return [any(ev["start_frame"] <= b and ev["end_frame"] >= a for ev in events) for a, b in ranges]
