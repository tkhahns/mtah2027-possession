"""Referee whistles from the match sound (no model, just the spectrum).

    audio      decode the video's sound once with ffmpeg -> cache/whistle/ (memmap)
    features   per video frame: the strongest pure tones, their overtones and loudness
    detect     tones -> whistle events, with the settings in DEFAULTS
"""
