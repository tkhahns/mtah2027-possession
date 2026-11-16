"""Per video frame: the strongest pure tones in the sound and how whistle-like each one is.

For every 25 fps video frame i (sound centred on video time i/25 + 20 ms):

1. spectrum   80 ms of sound, 4 overlapping 32 ms Hann windows averaged (Welch), so random noise
              wobbles about 2-3 dB instead of 5-6. Bins of BIN_HZ = 31.25 Hz up to 16 kHz, in dB.
2. whiten     each bin minus the mean of its neighbours 94-375 Hz away on both sides ("local
              floor"). A pure tone stands far above its own neighbourhood; crowd roar, voices and
              wind are spread over many bins and stay near 0. Loud moments raise the floor too, so
              this does not care how loud the stadium is.
3. tones      the K (6) strongest local peaks between PEAK_BAND Hz, each with
                f     its pitch in Hz (refined between bins)
                p1    how far it stands above its local floor (dB): "how pure is it"
                p2    the same at 2 x f (+-2 bins), p3 at 3 x f (+-3 bins): the overtones. A whistle
                      is a buzzy tone with clear overtones; a person whistling is almost a pure sine.
                lvl   its loudness above the whole-match median at that pitch (dB)
"""

import json
import os

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .audio import SAMPLE_RATE

VERSION = 1                      # bump when the numbers below change: old caches are rebuilt
FPS = 25
HOP = SAMPLE_RATE // FPS         # 1280 samples per video frame
WIN = 1024                       # 32 ms windows
SUB = 512                        # 16 ms apart
N_SUB = 4                        # 4 windows = 80 ms: [centre - 40 ms, centre + 40 ms)
NFFT = WIN
BIN_HZ = SAMPLE_RATE / NFFT      # 31.25 Hz
N_BINS = NFFT // 2 + 1
NOTCH, REACH = 3, 12             # local floor: bins 3..12 away on each side (94..375 Hz)
PEAK_BAND = (1000.0, 8000.0)     # where tones are looked for
K = 6                            # tones kept per frame
SPEC_POOL = 2                    # display: 2 bins per band (62.5 Hz) ...
SPEC_BANDS = 192                 # ... 192 bands, 0-12 kHz
SPEC_LO_DB, SPEC_DB_STEP = -10.0, 0.2   # display byte b = dB above median (-10 + 0.2 b)

FEATURES = ("f", "p1", "p2", "p3", "lvl")


def frame_db(samples, start, stop):
    """dB power spectrum of video frames [start, stop): float32 (stop - start, N_BINS)."""
    n = stop - start
    lo = start * HOP - HOP // 2                      # first window of frame i starts here
    hi = (stop - 1) * HOP - HOP // 2 + (N_SUB - 1) * SUB + WIN
    buf = np.zeros(hi - lo, dtype=np.float32)
    s0, s1 = max(lo, 0), min(hi, len(samples))
    if s1 > s0:
        buf[s0 - lo:s1 - lo] = samples[s0:s1]
    starts = (np.arange(n)[:, None] * HOP + np.arange(N_SUB)[None, :] * SUB).ravel()
    windows = sliding_window_view(buf, WIN)[starts] * np.hanning(WIN).astype(np.float32)
    spec = np.fft.rfft(windows, n=NFFT)
    power = (spec.real ** 2 + spec.imag ** 2).reshape(n, N_SUB, N_BINS).mean(axis=1)
    return (10 * np.log10(power + 1e-6)).astype(np.float32)


def whiten(db):
    """Each bin minus the mean of the bins NOTCH..REACH away on both sides (fewer at the edges)."""
    n_bins = db.shape[1]
    total = np.concatenate([np.zeros((len(db), 1), np.float64), np.cumsum(db, axis=1, dtype=np.float64)], axis=1)

    def span_sum(a, b):                              # sum over bins [a, b), clipped to the edges
        a, b = np.clip(a, 0, n_bins), np.clip(b, 0, n_bins)
        return total[:, b] - total[:, a], (b - a)

    k = np.arange(n_bins)
    left, nl = span_sum(k - REACH, k - NOTCH + 1)
    right, nr = span_sum(k + NOTCH, k + REACH + 1)
    floor = (left + right) / np.maximum(nl + nr, 1)
    return (db - floor).astype(np.float32)


def band_bins(lo_hz, hi_hz):
    return int(np.ceil(lo_hz / BIN_HZ)), int(np.floor(hi_hz / BIN_HZ)) + 1


def top_tones(db, white, k=K, band=PEAK_BAND):
    """(bin, f_hz, p1) of the k highest local peaks of `white` inside band, best first.

    Frames with fewer peaks get bin -1, f NaN, p1 -inf in the spare columns.
    """
    a, b = band_bins(*band)
    w = white[:, a - 1:b + 1]
    inner = w[:, 1:-1]
    is_peak = (inner > w[:, :-2]) & (inner >= w[:, 2:])
    score = np.where(is_peak, inner, -np.inf)
    order = np.argsort(-score, axis=1, kind="stable")[:, :k]
    rows = np.arange(len(db))[:, None]
    p1 = score[rows, order]
    bins = a + order
    # parabola through the three dB values around the peak -> pitch between bins
    lft, mid, rgt = db[rows, bins - 1], db[rows, bins], db[rows, bins + 1]
    den = lft - 2 * mid + rgt
    delta = np.where(den < 0, 0.5 * (lft - rgt) / np.where(den < 0, den, -1), 0.0)
    f = (bins + np.clip(delta, -0.5, 0.5)) * BIN_HZ
    missing = ~np.isfinite(p1)
    return np.where(missing, -1, bins), np.where(missing, np.nan, f).astype(np.float32), p1.astype(np.float32)


def overtone(white, f_hz, mult, tol):
    """Highest whitened value within +-tol bins of mult x f (NaN where f is NaN or off the top)."""
    centre = np.rint(np.nan_to_num(f_hz, nan=-1e9) * mult / BIN_HZ).astype(np.int64)
    cols = centre[..., None] + np.arange(-tol, tol + 1)
    ok = (centre - tol >= 0) & (centre + tol < white.shape[1]) & np.isfinite(f_hz)
    rows = np.arange(len(white)).reshape((-1,) + (1,) * (cols.ndim - 1))
    vals = white[rows, np.clip(cols, 0, white.shape[1] - 1)].max(axis=-1)
    return np.where(ok, vals, np.nan).astype(np.float32)


def match_median_db(samples, n_frames, n_probe=4000, seed=0):
    """Median dB per bin over n_probe frames spread over the match (the 'usual' sound)."""
    picks = np.unique(np.linspace(0, n_frames - 1, n_probe).astype(int))
    rows = [frame_db(samples, i, i + 1)[0] for i in picks]
    return np.median(np.array(rows), axis=0).astype(np.float32)


def compute(samples, n_frames, chunk=2048, progress=None):
    """All features for frames [0, n_frames): (dict of (n_frames, K) float32 arrays, uint8 spectrogram)."""
    med = match_median_db(samples, n_frames)
    out = {name: np.full((n_frames, K), np.nan, dtype=np.float32) for name in FEATURES}
    spec = np.zeros((n_frames, SPEC_BANDS), dtype=np.uint8)
    pooled_med = med[:SPEC_BANDS * SPEC_POOL].reshape(SPEC_BANDS, SPEC_POOL).mean(axis=1)
    for a in range(0, n_frames, chunk):
        b = min(n_frames, a + chunk)
        db = frame_db(samples, a, b)
        white = whiten(db)
        bins, f, p1 = top_tones(db, white)
        out["f"][a:b], out["p1"][a:b] = f, p1
        out["p2"][a:b] = overtone(white, f, 2, 2)
        out["p3"][a:b] = overtone(white, f, 3, 3)
        lvl = db[np.arange(b - a)[:, None], np.maximum(bins, 0)] - med[np.maximum(bins, 0)]
        out["lvl"][a:b] = np.where(bins >= 0, lvl, np.nan)
        pooled = db[:, :SPEC_BANDS * SPEC_POOL].reshape(b - a, SPEC_BANDS, SPEC_POOL).mean(axis=2)
        spec[a:b] = np.clip(np.rint((pooled - pooled_med - SPEC_LO_DB) / SPEC_DB_STEP), 0, 255)
        if progress:
            progress(b, n_frames)
    out["p1"][~np.isfinite(out["p1"])] = np.nan
    return out, spec


# ---- cache --------------------------------------------------------------------------------------
FEATURE_FILE, SPEC_FILE, META_FILE = "features.npz", "spectrogram_u8.bin", "features.json"


def save(folder, feats, spec, source=None):
    os.makedirs(folder, exist_ok=True)
    np.savez(os.path.join(folder, FEATURE_FILE), **feats)
    spec.tofile(os.path.join(folder, SPEC_FILE))
    meta = {"version": VERSION, "n_frames": len(spec), "fps": FPS, "k": K,
            "spec_bands": SPEC_BANDS, "spec_band_hz": BIN_HZ * SPEC_POOL,
            "spec_lo_db": SPEC_LO_DB, "spec_db_step": SPEC_DB_STEP, "source": source}
    with open(os.path.join(folder, META_FILE), "w", encoding="ascii") as handle:
        json.dump(meta, handle)
    return meta


def cache_ok(folder, source=None):
    path = os.path.join(folder, META_FILE)
    if not all(os.path.isfile(os.path.join(folder, name)) for name in (FEATURE_FILE, SPEC_FILE, META_FILE)):
        return False
    with open(path, encoding="ascii") as handle:
        meta = json.load(handle)
    return meta.get("version") == VERSION and (source is None or meta.get("source") == source)


def load(folder):
    """(features dict, spectrogram memmap (n_frames, SPEC_BANDS) uint8, metadata)."""
    with open(os.path.join(folder, META_FILE), encoding="ascii") as handle:
        meta = json.load(handle)
    with np.load(os.path.join(folder, FEATURE_FILE)) as data:
        feats = {name: data[name] for name in FEATURES}
    spec = np.memmap(os.path.join(folder, SPEC_FILE), dtype=np.uint8, mode="r",
                     shape=(meta["n_frames"], meta["spec_bands"]))
    return feats, spec, meta
