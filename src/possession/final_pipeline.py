"""Submission pipeline: the data pipeline (clean_data.pipeline: player-team cleaning, keeper-slot
repair, the keeper's ball, bounded interpolation, ball dead reckoning), learned timing, broadcast
corroboration, the live-control veto, the keeper's possession (keeper_possession: a ball the data pipeline put in a
goalkeeper's hands is his team's, unless a whistle stopped play just before), then the rule
clean-up: foul whistles, short-dead fill, set-piece tag windows, the corner taker's team once the
ball is back after a corner kick tag, and short-spell merge."""

import numpy as np

from clean_data import pipeline as data_pipeline
from utils import find_broadcast_blocks
from . import (broadcast_deadball, keeper_possession, live_control_veto, rule_cleanup,
               set_segment_pipeline, whistle_cleanup)

FPS = data_pipeline.FPS
MAX_PLAYER_GAP_SECONDS = data_pipeline.MAX_PLAYER_GAP_SECONDS
MAX_BALL_GAP_SECONDS = data_pipeline.MAX_BALL_GAP_SECONDS
PLAYERS_METHOD = data_pipeline.PLAYERS_METHOD            # "combined"
EXTRAPOLATE_FRAMES = data_pipeline.EXTRAPOLATE_FRAMES    # 20 frames, ball only
KEEPER_POSSESSION = True                                 # keeper_possession after the veto
PITCH = (105.3, 68.0)                                    # metres, when the tracking has no "pitch"


def clean_tracking(trk, half_blocks, fps=FPS,
                   max_player_gap_seconds=MAX_PLAYER_GAP_SECONDS,
                   max_ball_gap_seconds=MAX_BALL_GAP_SECONDS,
                   players_method=None, broadcast=None, shifts=None, extrapolate_frames=0,
                   report=None):
    """The data pipeline (clean_data.pipeline.clean) in memory. With the defaults only the
    keeper-slot repair and bounded interpolation run; predict() also passes the player-team
    cleaning (players_method, with broadcast and shifts) and the ball dead reckoning
    (extrapolate_frames). report: a dict that gets the pipeline's counts."""
    out = data_pipeline.clean(
        trk, half_blocks, broadcast, shifts, fps=fps, players_method=players_method,
        max_player_gap_seconds=max_player_gap_seconds, max_ball_gap_seconds=max_ball_gap_seconds,
        extrapolate_frames=extrapolate_frames)
    if report is not None:
        report.update(out["report"])
    return out["tracking"]


def with_homography(trk, valid):
    """The tracking with its frames that have no homography blanked (a copy)."""
    keep = np.zeros(len(trk["ball"]), bool)
    n = min(len(keep), len(valid))
    keep[:n] = np.asarray(valid, bool)[:n]
    players, ball = np.array(trk["players"], dtype=float), np.array(trk["ball"], dtype=float)
    players[~keep], ball[~keep] = np.nan, np.nan
    return {**trk, "players": players, "ball": ball}


def broadcast_offsets(tactical_tracking, broadcast_tracking, broadcast_valid, half_blocks):
    """{half: tactical frame - broadcast frame}, found from the two tracking files
    (utils.find_broadcast_blocks). ValueError if a half's shift is not found."""
    halves = find_broadcast_blocks.find(
        tactical_tracking, with_homography(broadcast_tracking, broadcast_valid),
        [(first, last) for _, first, last in half_blocks])
    missing = [h["half"] for h in halves if h["shift"] is None]
    if missing:
        raise ValueError("no broadcast shift found for half {}".format(
            ", ".join(map(str, missing))))
    return {half: h["shift"] for (half, _, _), h in zip(half_blocks, halves)}


def clean_up(labels, cleaned, half_blocks, whistles=(), set_pieces=(), tracking=None):
    """The clean-up."""
    evidence = whistle_cleanup.evidence(cleaned, whistles)
    halves = [(first, last) for _, first, last in half_blocks]
    rules, corner = rule_cleanup.ORDER, None
    if tracking is None:
        rules = tuple(rule for rule in rules if rule != "corner")
    else:
        corner = rule_cleanup.corner_ball(tracking["ball"], cleaned["players"], cleaned["team"],
                                          halves, cleaned.get("pitch", PITCH))
    return rule_cleanup.apply(labels, halves, set_pieces, rules, whistle=evidence, corner=corner)


def predict(tactical_tracking, broadcast_tracking, broadcast_valid, half_blocks,
            whistles=None, set_pieces=None, players_method=PLAYERS_METHOD,
            extrapolate_frames=EXTRAPOLATE_FRAMES, keeper_possession_on=KEEPER_POSSESSION,
            report=None):
    """Return final H/A/D labels on the tactical timeline before ungraded frames become X.
    broadcast_tracking: {players, ball, team, pitch} by broadcast frame; broadcast_valid: its
    homography is valid. players_method / extrapolate_frames: the data pipeline's player-team
    cleaning (None: off) and ball dead reckoning (0: off). keeper_possession_on: the keeper's
    possession of the holds the data pipeline found (needs whistles). With whistles and/or
    set_pieces (see clean_up) the clean-up runs last; with neither, the labels stop after the live-control veto.
    report: a dict that gets the data pipeline's counts and the broadcast shifts."""
    offsets = broadcast_offsets(tactical_tracking, broadcast_tracking, broadcast_valid,
                                half_blocks)
    kw = {"extrapolate_frames": extrapolate_frames}
    if report is not None:
        report["broadcast_shifts"] = dict(offsets)
        kw["report"] = report
    if players_method:
        kw.update(players_method=players_method,
                  broadcast=with_homography(broadcast_tracking, broadcast_valid),
                  shifts=[offsets[half] for half, _, _ in half_blocks])
    cleaned = clean_tracking(tactical_tracking, half_blocks, **kw)
    labels = set_segment_pipeline.predict(cleaned)
    labels = broadcast_deadball.predict(
        labels, cleaned, broadcast_valid, half_blocks, offsets,
        broadcast_tracking_valid=broadcast_deadball.tracking_valid(broadcast_tracking["players"]))
    labels = live_control_veto.restore_controlled_live(labels, cleaned)
    if keeper_possession_on and whistles is not None:
        counts = {}
        labels = keeper_possession.apply(labels, cleaned.get("keeper_hold", ()), whistles,
                                         report=counts)
        if report is not None:
            report["keeper_possession"] = dict(counts)
    if whistles is None and set_pieces is None:
        return labels
    return clean_up(labels, cleaned, half_blocks, whistles or (), set_pieces or (),
                    tactical_tracking)
