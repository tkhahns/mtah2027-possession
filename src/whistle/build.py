"""Make sure cache/whistle/ holds the decoded sound and the per-frame features; build what is missing."""

import os
import time

from . import audio, features


def ensure(video, folder, n_frames, ffmpeg=None, rebuild=False, log=print):
    """(features dict, spectrogram, metadata) for `video`, reading the sound only when needed."""
    has_video = bool(video) and os.path.isfile(video)
    if rebuild or not audio.sound_ok(folder, video):
        if not has_video:
            raise FileNotFoundError("no video at {} (pass --video or --data-dir)".format(video))
        log("Reading the sound of {} (once; a minute or two)...".format(os.path.basename(video)))
        t0 = time.time()
        audio.decode(video, folder, ffmpeg)
        log("  done in {:.0f} s".format(time.time() - t0))
        rebuild = True
    samples, sound_meta = audio.load(folder)
    if rebuild or not features.cache_ok(folder, sound_meta["source"]):
        log("Measuring the tones of {} frames (once; under a minute)...".format(n_frames))
        t0 = time.time()
        last = [0]

        def progress(done, total):
            if done - last[0] >= total // 10 or done == total:
                last[0] = done
                log("  {:3.0f}%".format(100 * done / total))

        feats, spec = features.compute(samples, n_frames, progress=progress)
        features.save(folder, feats, spec, source=sound_meta["source"])
        log("  done in {:.0f} s".format(time.time() - t0))
    return features.load(folder)
