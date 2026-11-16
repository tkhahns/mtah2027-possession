"""The match sound: decoded once with ffmpeg into cache/whistle/, then read as a numpy memmap.

The cache is mono 16-bit PCM at SAMPLE_RATE, sample 0 = video time 0. It is derived match data:
git-ignored (cache/), never commit or share it. ffmpeg comes from --ffmpeg, else PATH, else the
imageio-ffmpeg package (pip install imageio-ffmpeg ships one).
"""

import json
import os
import shutil
import subprocess

import numpy as np

SAMPLE_RATE = 32000          # Hz; keeps everything up to 16 kHz (a whistle's 3rd harmonic)
SOUND_FILE = "sound_32k_mono.s16"
SOUND_META = "sound.json"


def find_ffmpeg(path=None):
    """Path of an ffmpeg executable, or RuntimeError saying how to get one."""
    if path:
        return path
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg not found: pass --ffmpeg PATH, put it on PATH, "
                           "or pip install imageio-ffmpeg") from None


def stamp(video):
    """Name, size and time of the video, to notice a cache made from another file."""
    st = os.stat(video)
    return {"name": os.path.basename(video), "size": st.st_size, "mtime": int(st.st_mtime)}


def sound_ok(folder, video=None):
    """True when folder holds a decoded sound (from this video, when it is given and exists)."""
    raw, meta_path = os.path.join(folder, SOUND_FILE), os.path.join(folder, SOUND_META)
    if not (os.path.isfile(raw) and os.path.isfile(meta_path)):
        return False
    with open(meta_path, encoding="ascii") as handle:
        meta = json.load(handle)
    if os.path.getsize(raw) != 2 * meta["n_samples"] or meta["sample_rate"] != SAMPLE_RATE:
        return False
    return not (video and os.path.isfile(video)) or meta.get("source") == stamp(video)


def decode(video, folder, ffmpeg=None):
    """Video's first sound track -> folder/SOUND_FILE (mono int16, SAMPLE_RATE). Returns metadata."""
    os.makedirs(folder, exist_ok=True)
    raw = os.path.join(folder, SOUND_FILE)
    tmp = raw + ".part"
    cmd = [find_ffmpeg(ffmpeg), "-nostdin", "-v", "error", "-y", "-i", video, "-map", "0:a:0",
           "-ac", "1", "-af", "aresample={}:async=1:first_pts=0".format(SAMPLE_RATE),
           "-f", "s16le", tmp]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode:
        raise RuntimeError("ffmpeg failed: " + proc.stderr.decode(errors="replace")[-800:])
    os.replace(tmp, raw)
    meta = {"sample_rate": SAMPLE_RATE, "n_samples": os.path.getsize(raw) // 2,
            "source": stamp(video)}
    with open(os.path.join(folder, SOUND_META), "w", encoding="ascii") as handle:
        json.dump(meta, handle)
    return meta


def load(folder):
    """(int16 memmap of the samples, metadata)."""
    with open(os.path.join(folder, SOUND_META), encoding="ascii") as handle:
        meta = json.load(handle)
    samples = np.memmap(os.path.join(folder, SOUND_FILE), dtype="<i2", mode="r",
                        shape=(meta["n_samples"],))
    return samples, meta
