"""Optional: referee whistles from the match sound, as dead-ball evidence (main.py --whistles).

Detection (Thanh's detector, plain signal processing, no model). A whistle is one loud, steady
pitch at about 2.5-5 kHz; crowd noise spreads over every pitch. Per video frame:

1. spectrum  80 ms of sound centred on the frame -> power in 96 bands of 93.75 Hz from 1.5 to
             10.5 kHz, in dB, stored as uint8 steps of 0.5 dB.
2. whiten    subtract each band's whole-match median, so the usual noise is 0 at every pitch.
3. peak      the loudest band in [fmin, fmax] and how far it stands above the others' median.
4. events    frames with prominence >= prom_db; runs at most `gap` frames apart merged; kept if
             min_len <= length <= max_len frames and the pitch holds steady.

The detector's DEFAULTS are fixed, never tuned on our match (that would be calibrating on the
video). The sound is read with ffmpeg (on PATH, or --ffmpeg).

Use (fixed by reasoning, not tuned: Metrica's games have no sound to test on). Referees whistle
to *stop* play, so each whistle adds DEAD_PER_FRAME nats of "dead" evidence on the HOLD frames
from its start: 50 in all, half of the decoder's live/dead switch (100). A whistle alone, or a
false one from the stands, cannot create a stoppage; it can only pull the start of one that the
players' movement already shows back to the moment the referee blew.
"""

import shutil
import subprocess

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from config import FPS, N_FRAMES
from decoder import D

SAMPLE_RATE = 24000                 # Hz the sound is decoded at: 960 samples per 25 fps frame
NFFT = 2048                         # FFT bins of 11.72 Hz
POOL = 8                            # FFT bins per band -> 93.75 Hz bands
FIRST_BAND = 16                     # band 0 starts at 16 * 93.75 = 1500 Hz
N_BANDS = 96                        # ... and band 95 ends at 10500 Hz
BAND_HZ = SAMPLE_RATE / NFFT * POOL
BAND_FREQS = (FIRST_BAND + np.arange(N_BANDS) + 0.5) * BAND_HZ      # band centres, Hz
DB_STEP = 0.5                       # dB per uint8 step
STEADY_BANDS = 2                    # "same pitch" = within +-2 bands (+-190 Hz)
DEFAULTS = {"fmin": 2500, "fmax": 5000, "prom_db": 12.0, "gap": 3, "min_len": 4,
            "max_len": 100, "steady": 0.7}

DEAD_PER_FRAME = 1.0      # nats of dead evidence per frame after a whistle ...
HOLD = 50                 # ... for 2 s from its start: 50 in all, half of one live/dead switch
MIN_PROM_DB = 12.0        # the detector's own default threshold


# -- sound -> whistles ---------------------------------------------------------------------------

def decode_audio(video, ffmpeg=None):
    """The first sound track as mono int16 at SAMPLE_RATE, padded so sample 0 is video time 0."""
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found: install it or pass --ffmpeg PATH")
    cmd = [ffmpeg, "-nostdin", "-v", "error", "-i", video, "-map", "0:a:0", "-ac", "1",
           "-af", "aresample={}:async=1:first_pts=0".format(SAMPLE_RATE), "-f", "s16le", "-"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError("ffmpeg failed: " + proc.stderr.decode(errors="replace")[-800:])
    return np.frombuffer(proc.stdout, dtype="<i2")


def frame_spectra(samples, n_frames=N_FRAMES, fps=FPS, chunk=4096):
    """dB power per video frame (rows) and band (columns): float32 (n_frames, N_BANDS).
    Frame i uses the samples of [i - 1/2, i + 3/2) frames, Hann-windowed."""
    hop = SAMPLE_RATE // fps
    samples = np.asarray(samples)
    window = np.hanning(2 * hop).astype(np.float32)
    first = FIRST_BAND * POOL
    out = np.empty((n_frames, N_BANDS), dtype=np.float32)
    for a in range(0, n_frames, chunk):
        b = min(n_frames, a + chunk)
        lo, hi = a * hop - hop // 2, b * hop + hop // 2 + hop % 2
        buf = np.zeros(hi - lo, dtype=np.float32)
        s0, s1 = max(lo, 0), min(hi, len(samples))
        if s1 > s0:
            buf[s0 - lo:s1 - lo] = samples[s0:s1]
        frames = sliding_window_view(buf, 2 * hop)[::hop][:b - a]
        spec = np.fft.rfft(frames * window, n=NFFT)
        power = (spec.real ** 2 + spec.imag ** 2)[:, first:first + N_BANDS * POOL]
        out[a:b] = 10 * np.log10(power.reshape(b - a, N_BANDS, POOL).mean(axis=2) + 1e-12)
    return out


def quantize(db):
    """uint8 spectra: dB = db_lo + value * DB_STEP; the loudest cell maps to 255."""
    lo = float(np.max(db)) - 255 * DB_STEP if np.size(db) else 0.0
    return np.clip(np.rint((db - lo) / DB_STEP), 0, 255).astype(np.uint8)


def band_medians(spec):
    """Each band's median over all frames (the lower one of an even count), int32 per band."""
    rank = (len(spec) - 1) // 2
    return np.array([np.searchsorted(np.cumsum(np.bincount(spec[:, b], minlength=256)), rank + 1)
                     for b in range(spec.shape[1])], dtype=np.int32)


def _lower_median(x):
    k = x.shape[1]
    return np.partition(x, (k - 1) // 2, axis=1)[:, (k - 1) // 2]


def _runs(mask):
    """Half-open (start, stop) runs where a boolean mask is true."""
    edges = np.diff(np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def detect(spec, p=DEFAULTS):
    """Whistles in uint8 spectra: [{start_frame, end_frame (inclusive), peak_frame, freq_hz,
    prom_db}], in order."""
    w = np.asarray(spec, dtype=np.int32) - band_medians(spec)
    band = np.flatnonzero((BAND_FREQS >= p["fmin"]) & (BAND_FREQS <= p["fmax"]))
    sub = w[:, band]
    peak = (band[0] + np.argmax(sub, axis=1)).astype(np.int32)
    prom = (sub.max(axis=1) - _lower_median(sub)).astype(np.int32)
    on = prom * DB_STEP >= p["prom_db"]
    spans = []
    for start, stop in _runs(on):
        if spans and start - spans[-1][1] <= p["gap"]:
            spans[-1][1] = stop
        else:
            spans.append([start, stop])
    events = []
    for start, stop in spans:
        if not p["min_len"] <= stop - start <= p["max_len"]:
            continue
        idx = start + np.flatnonzero(on[start:stop])
        mid = int(np.sort(peak[idx])[(len(idx) - 1) // 2])
        if np.sum(np.abs(peak[idx] - mid) <= STEADY_BANDS) < p["steady"] * len(idx):
            continue
        best = int(idx[np.argmax(prom[idx])])
        events.append({"start_frame": int(start), "end_frame": int(stop - 1), "peak_frame": best,
                       "freq_hz": int(round(float(BAND_FREQS[mid]))),
                       "prom_db": float(prom[best] * DB_STEP)})
    return events


def from_video(video, ffmpeg=None):
    """Whistles of the tactical video's sound track (a minute or two)."""
    return detect(quantize(frame_spectra(decode_audio(video, ffmpeg))))


# -- whistles -> decoder evidence ----------------------------------------------------------------

def anchor(ll, events, halves, dead_per_frame=DEAD_PER_FRAME, hold=HOLD, min_prom_db=MIN_PROM_DB):
    """Copy of decoder emissions with dead evidence after each whistle inside a half."""
    ll = np.array(ll, dtype=np.float64, copy=True)
    n = len(ll)
    for w in events or []:
        start = int(w["start_frame"])
        if w.get("prom_db", min_prom_db) < min_prom_db:
            continue
        if not any(first <= start <= last for _, first, last in halves):
            continue                       # before kick-off, half-time, after the final whistle
        ll[max(0, start):min(n, start + hold), D] += dead_per_frame
    return ll
