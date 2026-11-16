"""Reusable numpy helpers for any part of the pipeline. Plain arrays in, plain arrays out.

    series_utils   runs, rolling counts and means, last/next seen frame
    pitch_utils    lines, boxes, corners, halves (metres)
    ball_utils     gap filling, smoothing, speed, direction, acceleration, line crossings
    player_utils   player speeds, distances to the ball, team centroid / speed / shape, keepers
    whistle_utils  referee whistles from the sound: per-frame spectra, peak prominence, events
    find_broadcast_blocks  tactical - broadcast frame shift per half, from kick-offs in the two ATDs
    keeper_utils   each team's keeper slot: defended sides, deepest / in-box shares in a window

Nothing here reads the match files, so every helper works the same on small synthetic arrays.
"""
