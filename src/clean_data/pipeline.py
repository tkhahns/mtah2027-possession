"""The data pipeline: the organiser's tactical tracking -> the cleaned tracking that the possession
pipeline (possession.final_pipeline) reads. Everything runs in memory, in this order:

1. players      player-team cleaning (clean_data.players, method "combined" = handovers_checked +
                appear_vanish + roles_scales), on the tracking as delivered: a stretch of a track
                slot that follows a player of the other team is moved into a free slot of that
                team (slot teams are fixed in the metadata, so that is how the file says "the
                other team's player"). Its checks read
                the broadcast tracking on frames with a valid homography, put on the tactical
                timeline with the per-half shifts, and role thresholds fitted on the public Metrica
                games (src/models/team_role_scales.json).
                Goalkeeper slots are never moved or moved into: step 2 handles them.
2. keepers      goalkeeper-slot repair (clean_data.keeper) on the slots found per frame.
2b. keeper ball a ball gap longer than 25 frames that starts within 2 m of a goalkeeper is
                the keeper holding the ball: the ball gets that keeper's track until it is
                tracked again (clean_data.keeper_ball). The cleaned tracking lists the holds as
                "keeper_hold": [(first, until, team), ...] for possession.keeper_possession.
3. interpolate  linear interpolation (clean_data.interpolate) of inner player and ball gaps up to
                15 s, written at the precision of the ATD text (six decimals in normalised x/y,
                four in speed); a gap holding a deliberately blanked keeper ghost stays empty.
4. extrapolate  ball dead reckoning (clean_data.extrapolate): every ball gap still longer than
                15 s starts with 20 frames that carry on the ball's direction and speed over the
                5 frames before it; the rest of the gap stays empty. Players are left as they are.

    from clean_data import pipeline
    out = pipeline.clean(trk, half_blocks, broadcast, shifts)
    out["tracking"]   # {players, ball, team, pitch}: what the models read
    out["report"]     # counts only

trk is {players (frames, slots, 3), ball (frames, 3), team (slots,), pitch}, x/y in metres;
half_blocks [(half, first, last), ...]; broadcast the broadcast tracking blanked where it has no
homography (possession.final_pipeline.with_homography); shifts the tactical - broadcast frame per
half (possession.final_pipeline.broadcast_offsets). players_method=None skips step 1 (it is the
only step that needs the broadcast), keeper_hold_m=None skips step 2b, extrapolate_frames=0
skips step 4.
"""

import numpy as np

from . import extrapolate as dead_reckoning
from . import interpolate, keeper, keeper_ball
from . import players as player_cleaning

FPS = 25
MAX_PLAYER_GAP_SECONDS = 15.0       # step 3: longest player gap filled
MAX_BALL_GAP_SECONDS = 15.0         # step 3: longest ball gap filled
PLAYERS_METHOD = "combined"         # step 1: a clean_data.players.METHODS name
EXTRAPOLATE_FRAMES = dead_reckoning.FILL_FRAMES   # step 4: 20 frames at the start of a long gap
EXTRAPOLATE_MIN_GAP_SECONDS = 15.0  # step 4: only gaps longer than this
EXTRAPOLATE_BALL_ONLY = True        # step 4: the ball only
KEEPER_HOLD_M = 2.0                 # step 2b: keeper-to-ball radius in metres; None = off
KEEPER_HOLD_MIN_FRAMES = keeper_ball.MIN_FRAMES   # step 2b: only ball gaps longer than this


def move_players(trk, half_blocks, broadcast, shifts, method=PLAYERS_METHOD, fps=FPS):
    """Step 1: (tracking with the method's moves applied, applied moves, skipped moves). The moves
    are found on trk as given, and the keeper slots apply_moves leaves alone are found there too."""
    team = np.asarray(trk["team"])
    halves = [(first, last) for _, first, last in half_blocks]
    keepers = keeper.find_slots(trk["players"], team, half_blocks, trk.get("pitch", keeper.PITCH), fps)
    moves = player_cleaning.run(method, trk, broadcast, halves, shifts, shared={})
    players, applied, skipped = player_cleaning.apply_moves(trk["players"], team, moves, keepers)
    return {**trk, "players": players}, applied, skipped


def repair_and_interpolate(trk, half_blocks, fps=FPS,
                           max_player_gap_seconds=MAX_PLAYER_GAP_SECONDS,
                           max_ball_gap_seconds=MAX_BALL_GAP_SECONDS,
                           keeper_hold_m=KEEPER_HOLD_M,
                           keeper_hold_min_frames=KEEPER_HOLD_MIN_FRAMES):
    """Steps 2-3: (tracking, counts). The keeper-slot repair, on the slots keeper.find_slots()
    finds per frame, then (keeper_hold_m not None) the keeper's hold of the ball, then bounded
    interpolation. The caller's arrays are not changed."""
    players = np.asarray(trk["players"])
    ball = np.asarray(trk["ball"])
    slots = keeper.find_slots(players, trk["team"], half_blocks,
                              trk.get("pitch", keeper.PITCH), fps)
    fixed_players, _, _ = keeper.fix_keepers(players, half_blocks, slots)
    held_ball = ball
    if keeper_hold_m is not None:
        held_ball, held, held_gaps = keeper_ball.hold_ball(
            fixed_players, ball, slots, half_blocks, keeper_hold_m, keeper_hold_min_frames)
    max_player_gap = int(round(float(max_player_gap_seconds) * fps))
    max_ball_gap = int(round(float(max_ball_gap_seconds) * fps))
    filled_players, filled_ball, player_fill, ball_fill, _ = interpolate.interpolate(
        fixed_players, held_ball, max_player_gap, max_ball_gap, fps,
        original=(players, ball))
    length, width = map(float, trk.get("pitch", (105.3, 68.0)))
    for values, mask in ((filled_players, player_fill), (filled_ball, ball_fill)):
        values[..., 0][mask] = np.round(values[..., 0][mask] / length, 6) * length
        values[..., 1][mask] = np.round(values[..., 1][mask] / width, 6) * width
        values[..., 2][mask] = np.round(values[..., 2][mask], 4)
    counts = {"player_cells": int(player_fill.sum()), "ball_cells": int(ball_fill.sum())}
    out = {**trk, "players": filled_players, "ball": filled_ball}
    if keeper_hold_m is not None:
        counts["keeper_hold"] = {"gaps": len(held_gaps), "ball_cells": int(held.sum())}
        out["keeper_hold"] = [(g["first"], g["until"], g["team"]) for g in held_gaps]
    return out, counts


def extrapolate(trk, original, fps=FPS, frames=EXTRAPOLATE_FRAMES,
                min_gap_seconds=EXTRAPOLATE_MIN_GAP_SECONDS, ball_only=EXTRAPOLATE_BALL_ONLY):
    """Step 4: (tracking, counts). original is the tracking before steps 2-3: a cell tracked there
    but empty in trk was blanked on purpose (a keeper ghost), and its gap is left alone."""
    res = dead_reckoning.extrapolate(
        trk["players"], trk["ball"], int(round(min_gap_seconds * fps)), int(frames),
        dead_reckoning.LOOKBACK_FRAMES, fps,
        (np.asarray(original["players"]), np.asarray(original["ball"])), ball_only=ball_only)
    counts = {"player_cells": int(res["filled_players"].sum()),
              "ball_cells": int(res["filled_ball"].sum()),
              "player_gaps": [res["done_players"], res["long_players"]],
              "ball_gaps": [res["done_ball"], res["long_ball"]]}
    return {**trk, "players": res["players"], "ball": res["ball"]}, counts


def clean(trk, half_blocks, broadcast=None, shifts=None, fps=FPS,
          players_method=PLAYERS_METHOD,
          max_player_gap_seconds=MAX_PLAYER_GAP_SECONDS,
          max_ball_gap_seconds=MAX_BALL_GAP_SECONDS,
          extrapolate_frames=EXTRAPOLATE_FRAMES,
          extrapolate_ball_only=EXTRAPOLATE_BALL_ONLY,
          keeper_hold_m=KEEPER_HOLD_M):
    """Steps 1-4 (see the module docstring): {"tracking": cleaned tracking, "report": counts}."""
    report = {}
    if players_method:
        trk, applied, skipped = move_players(trk, half_blocks, broadcast, shifts,
                                             players_method, fps)
        report["players"] = {"method": players_method, "moves": len(applied),
                             "skipped": len(skipped),
                             "frames_moved": sum(m["last"] - m["first"] + 1 for m in applied)}
    cleaned, report["interpolated"] = repair_and_interpolate(
        trk, half_blocks, fps, max_player_gap_seconds, max_ball_gap_seconds, keeper_hold_m)
    if extrapolate_frames:
        cleaned, report["extrapolated"] = extrapolate(
            cleaned, trk, fps, extrapolate_frames, ball_only=extrapolate_ball_only)
    return {"tracking": cleaned, "report": report}
