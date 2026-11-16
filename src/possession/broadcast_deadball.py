"""Broadcast replay evidence for conservative dead-ball interval completion.

An invalid broadcast homography usually means that the programme feed has cut away from the
pitch.  That is useful evidence for a stoppage, but it is not sufficient by itself: close-ups and
replays can occur while a goalkeeper still has live possession.  Existing dead intervals may be
extended by the camera signal.  A new interval additionally needs a visible, stopped ball state
followed by a strong restart transition in the tactical tracking.
"""

import os

import numpy as np

from . import constrained_restart, hybrid_pipeline, restart_state

FPS = 25

# Existing dead intervals use the previously validated, deliberately permissive extension rule.
EXTENSION_PRE_FRAMES = 2 * FPS

# A replay may seed a new interval only with all of the stronger tactical checks below.
MIN_REPLAY_FRAMES = 2 * FPS
STANDALONE_PRE_FRAMES = 3 * FPS
MAX_RESTART_SEARCH_FRAMES = 20 * FPS
DEAD_PROBABILITY = 0.60
MAX_BALL_SPEED = 1.5
MAX_PLAYER_SPEED = 2.0
MOTION_THRESHOLD = 0.75
MIN_STOPPED_FRAMES = 2 * FPS
MIN_BALL_VISIBLE_SHARE = 0.80

# Some replays begin before the restart and remain off-pitch after play resumes.  For that shape,
# inspect the five seconds leading into the replay and allow a sustained state rise to backtrack
# the boundary.  The one-second margin accounts for the contextual model's expected onset lag.
INSIDE_REPLAY_LOOKBACK_FRAMES = 5 * FPS
ONSET_PROBABILITY = 0.30
MIN_ONSET_FRAMES = 12
ONSET_PRE_FRAMES = FPS

# A close-up can occasionally retain a plausible homography even though broadcast tracking has
# lost every complete player sample.  Treat that disagreement as a separate, weaker camera-cut
# cue.  It needs a
# longer low-confidence stop state, a restart, and a ball away from either goal line so that a
# close-up of a goalkeeper holding the live ball cannot seed a dead interval.
MIN_TRACKING_DROPOUT_FRAMES = 2 * FPS
TRACKING_DROPOUT_LOOKBACK_FRAMES = 5 * FPS
TRACKING_DROPOUT_PRE_FRAMES = 3 * FPS
TRACKING_DROPOUT_RESTART_SEARCH_FRAMES = 10 * FPS
TRACKING_DROPOUT_PROBABILITY = 0.15
MIN_TRACKING_DROPOUT_STOPPED_FRAMES = 2 * FPS
MIN_BALL_GOAL_LINE_DISTANCE = 0.15

# Delayed broadcast control: a foul-like tactical stop may be confirmed by a replay/close-up cue
# that starts after the stop but before the true restart.  The restart must lead to sustained ball
# activity; this rejects player repositioning and temporary ball handling during long stoppages.
CONTROL_MIN_CUE_FRAMES = 2 * FPS
CONTROL_LOOKBACK_FRAMES = 8 * FPS
CONTROL_RESTART_SEARCH_FRAMES = 10 * FPS
CONTROL_STATE_PROBABILITY = 0.15
CONTROL_STATE_PROBE_FRAMES = 12
CONTROL_LIVE_TRANSITION_LOOKBACK = 5 * FPS
CONTROL_POST_RESTART_FRAMES = 3 * FPS
CONTROL_ACTIVE_BALL_SPEED = 3.0
CONTROL_MIN_ACTIVE_BALL_FRAMES = 2 * FPS
CONTROL_MIN_STOPPED_FRAMES = 2 * FPS

# A long VAR can lose the tactical ball for most of the stoppage.  In that case, use a
# later broadcast interruption to backfill an already-existing dead interval to the start of a
# sustained high-confidence state.  The missing-ball requirement keeps this separate from live
# goalkeeper possession.
REVIEW_LOOKBACK_FRAMES = 60 * FPS
REVIEW_STATE_PROBABILITY = 0.60
REVIEW_MAX_PLAYER_SPEED = 2.0
REVIEW_MIN_STATE_FRAMES = 2 * FPS
REVIEW_MIN_MISSING_SHARE = 0.80
REVIEW_MIN_CONTROL_FRAMES = 2 * FPS

# Repair short dead spells created while a goalkeeper (or deep defender) is visibly carrying the
# ball.  These must be surrounded by the same team, remain close to that team's player, move by a
# meaningful distance, and stay within penalty-area depth of a goal line.
KEEPER_MAX_DEAD_FRAMES = 10 * FPS
KEEPER_MAX_GOAL_LINE_DISTANCE = 18.0
KEEPER_MAX_PLAYER_DISTANCE = 2.5
KEEPER_MIN_VISIBLE_SHARE = 0.80
KEEPER_MIN_CLOSE_SHARE = 0.80
KEEPER_MIN_SAME_TEAM_SHARE = 0.80
KEEPER_MIN_BALL_DISPLACEMENT = 1.0
KEEPER_DEEPEST_MARGIN = 3.0

# A broadcast-added D interval can be much longer than the actual keeper hold.  Veto only the
# sustained frames where the tactical-only decoder, visible ball and goalkeeper-like control all
# agree.  Recent hard out-of-pitch evidence protects goal kicks after the ball is placed back down.
KEEPER_HOLD_MIN_FRAMES = 2 * FPS
KEEPER_HOLD_HARD_LOOKBACK_FRAMES = 15 * FPS


def runs(mask):
    """Half-open ``(start, stop)`` intervals for true components of a boolean mask."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def _first_sustained(mask, minimum):
    for start, stop in runs(mask):
        if stop - start >= int(minimum):
            return int(start)
    return None


def mapped_replay_runs(invalid, half_blocks, offsets, n_frames):
    """Map broadcast invalid-homography runs onto the tactical submission timeline."""
    invalid = np.asarray(invalid, dtype=bool)
    mapped = []
    for half, first, last in half_blocks:
        offset = int(offsets[half])
        broadcast_first = max(0, int(first) - offset)
        broadcast_last = min(len(invalid) - 1, int(last) - offset)
        if broadcast_last < broadcast_first:
            continue
        for start, stop in runs(invalid[broadcast_first:broadcast_last + 1]):
            start += broadcast_first
            stop += broadcast_first
            tactical_start = max(int(first), start + offset)
            tactical_stop = min(int(last) + 1, stop + offset, int(n_frames))
            if tactical_stop > tactical_start:
                mapped.append((half, tactical_start, tactical_stop, int(first), int(last) + 1))
    return mapped


def tracking_valid(players, min_visible_players=1):
    """Frames with at least ``min_visible_players`` complete broadcast player samples."""
    players = np.asarray(players)
    if players.ndim != 3 or players.shape[2] < 2:
        raise ValueError("broadcast players must have shape (frames, slots, coordinates)")
    visible = np.isfinite(players).all(axis=2).sum(axis=1)
    return visible >= int(min_visible_players)


def extend_existing(labels, replay_runs, pre_frames=EXTENSION_PRE_FRAMES):
    """Extend only replay candidates that overlap an existing dead interval."""
    labels = np.asarray(labels)
    output = labels.copy()
    candidate = np.zeros(len(labels), dtype=bool)
    for _, start, stop, first, _ in replay_runs:
        candidate[max(first, start - int(pre_frames)):stop] = True
    for start, stop in runs(candidate):
        if np.any(labels[start:stop] == "D"):
            output[start:stop] = "D"
    return output


def seed_standalone(labels, replay_runs, features, probability, motion,
                    min_replay_frames=MIN_REPLAY_FRAMES,
                    pre_frames=STANDALONE_PRE_FRAMES,
                    max_restart_search_frames=MAX_RESTART_SEARCH_FRAMES,
                    dead_probability=DEAD_PROBABILITY,
                    max_ball_speed=MAX_BALL_SPEED,
                    max_player_speed=MAX_PLAYER_SPEED,
                    motion_threshold=MOTION_THRESHOLD,
                    min_stopped_frames=MIN_STOPPED_FRAMES,
                    min_ball_visible_share=MIN_BALL_VISIBLE_SHARE):
    """Create replay-anchored dead intervals only when tactical tracking corroborates them."""
    output = np.asarray(labels).copy()
    features = np.asarray(features)
    probability = np.asarray(probability)
    motion = np.asarray(motion)
    if not (len(output) == len(features) == len(probability) == len(motion)):
        raise ValueError("labels, features, probability and motion must have the same length")

    index = {name: i for i, name in enumerate(restart_state.FEATURE_NAMES)}
    ball_speed = features[:, index["ball_speed"]] * 15.0
    player_speed = features[:, index["player_speed"]] * 5.0
    ball_visible = features[:, index["ball_visible"]]

    for _, replay_start, replay_stop, first, last in replay_runs:
        if replay_stop - replay_start < int(min_replay_frames):
            continue
        candidate_start = max(first, replay_start - int(pre_frames))
        if np.any(output[candidate_start:replay_stop] == "D"):
            continue

        search_stop = min(last, replay_stop + int(max_restart_search_frames))
        restart_frames = np.flatnonzero(
            motion[replay_stop:search_stop] >= float(motion_threshold)) + replay_stop
        for restart in restart_frames:
            if restart <= replay_stop:
                continue
            interval = slice(replay_stop, restart)
            if np.mean(ball_visible[interval]) < float(min_ball_visible_share):
                continue
            stopped = (
                (probability[interval] >= float(dead_probability))
                & (ball_speed[interval] <= float(max_ball_speed))
                & (player_speed[interval] <= float(max_player_speed))
            )
            if np.sum(stopped) < int(min_stopped_frames):
                continue
            output[candidate_start:restart] = "D"
            break
    return output


def seed_inside_replay(labels, replay_runs, features, probability, motion,
                       lookback_frames=INSIDE_REPLAY_LOOKBACK_FRAMES,
                       onset_probability=ONSET_PROBABILITY,
                       min_onset_frames=MIN_ONSET_FRAMES,
                       onset_pre_frames=ONSET_PRE_FRAMES,
                       pre_frames=STANDALONE_PRE_FRAMES,
                       min_replay_frames=MIN_REPLAY_FRAMES,
                       dead_probability=DEAD_PROBABILITY,
                       max_ball_speed=MAX_BALL_SPEED,
                       max_player_speed=MAX_PLAYER_SPEED,
                       motion_threshold=MOTION_THRESHOLD,
                       min_stopped_frames=MIN_STOPPED_FRAMES,
                       min_ball_visible_share=MIN_BALL_VISIBLE_SHARE):
    """Seed a stoppage when the restart happens before a long replay returns to the pitch."""
    output = np.asarray(labels).copy()
    features = np.asarray(features)
    probability = np.asarray(probability)
    motion = np.asarray(motion)
    if not (len(output) == len(features) == len(probability) == len(motion)):
        raise ValueError("labels, features, probability and motion must have the same length")

    index = {name: i for i, name in enumerate(restart_state.FEATURE_NAMES)}
    ball_speed = features[:, index["ball_speed"]] * 15.0
    player_speed = features[:, index["player_speed"]] * 5.0
    ball_visible = features[:, index["ball_visible"]]

    for _, replay_start, replay_stop, first, _ in replay_runs:
        if replay_stop - replay_start < int(min_replay_frames):
            continue
        lookback = max(first, replay_start - int(lookback_frames))
        fixed_start = max(first, replay_start - int(pre_frames))
        onset = _first_sustained(
            probability[lookback:replay_start] >= float(onset_probability),
            min_onset_frames)
        if onset is not None:
            onset = max(first, lookback + onset - int(onset_pre_frames))
        candidate_start = min(fixed_start, onset if onset is not None else fixed_start)
        if np.any(output[candidate_start:replay_start] == "D"):
            continue

        restart_frames = np.flatnonzero(
            motion[replay_start:replay_stop] >= float(motion_threshold)) + replay_start
        for restart in restart_frames:
            if restart <= lookback:
                continue
            interval = slice(lookback, restart)
            if np.mean(ball_visible[interval]) < float(min_ball_visible_share):
                continue
            stopped = (
                (probability[interval] >= float(dead_probability))
                & (ball_speed[interval] <= float(max_ball_speed))
                & (player_speed[interval] <= float(max_player_speed))
            )
            if np.sum(stopped) < int(min_stopped_frames):
                continue
            output[candidate_start:restart] = "D"
            break
    return output


def seed_tracking_dropout(labels, dropout_runs, features, probability, motion,
                          min_dropout_frames=MIN_TRACKING_DROPOUT_FRAMES,
                          lookback_frames=TRACKING_DROPOUT_LOOKBACK_FRAMES,
                          pre_frames=TRACKING_DROPOUT_PRE_FRAMES,
                          max_restart_search_frames=TRACKING_DROPOUT_RESTART_SEARCH_FRAMES,
                          state_probability=TRACKING_DROPOUT_PROBABILITY,
                          min_onset_frames=MIN_ONSET_FRAMES,
                          max_ball_speed=MAX_BALL_SPEED,
                          max_player_speed=MAX_PLAYER_SPEED,
                          motion_threshold=MOTION_THRESHOLD,
                          min_stopped_frames=MIN_TRACKING_DROPOUT_STOPPED_FRAMES,
                          min_ball_visible_share=MIN_BALL_VISIBLE_SHARE,
                          min_ball_goal_line_distance=MIN_BALL_GOAL_LINE_DISTANCE):
    """Seed false-valid broadcast cuts with conservative tactical restart evidence.

    ``dropout_runs`` must contain only intervals where the homography is valid but every complete
    broadcast player sample is missing.  This complements invalid-homography replay detection
    without weakening its frozen thresholds.
    """
    output = np.asarray(labels).copy()
    features = np.asarray(features)
    probability = np.asarray(probability)
    motion = np.asarray(motion)
    if not (len(output) == len(features) == len(probability) == len(motion)):
        raise ValueError("labels, features, probability and motion must have the same length")

    index = {name: i for i, name in enumerate(restart_state.FEATURE_NAMES)}
    ball_speed = features[:, index["ball_speed"]] * 15.0
    player_speed = features[:, index["player_speed"]] * 5.0
    ball_visible = features[:, index["ball_visible"]]
    ball_x = features[:, index["ball_x"]]

    for _, dropout_start, dropout_stop, first, last in dropout_runs:
        if dropout_stop - dropout_start < int(min_dropout_frames):
            continue
        lookback = max(first, dropout_start - int(lookback_frames))
        candidate_start = max(first, dropout_start - int(pre_frames))
        onset = _first_sustained(
            probability[lookback:dropout_start] >= float(state_probability),
            min_onset_frames)
        if onset is not None:
            candidate_start = min(candidate_start, lookback + onset)
        if np.any(output[candidate_start:dropout_stop] == "D"):
            continue

        search_stop = min(last, dropout_stop + int(max_restart_search_frames))
        restart_frames = np.flatnonzero(
            motion[dropout_stop:search_stop] >= float(motion_threshold)) + dropout_stop
        for restart in restart_frames:
            if restart <= dropout_stop:
                continue
            interval = slice(candidate_start, restart)
            if np.mean(ball_visible[interval]) < float(min_ball_visible_share):
                continue
            goal_line_distance = np.minimum(ball_x[interval], 1.0 - ball_x[interval])
            if np.median(goal_line_distance) < float(min_ball_goal_line_distance):
                continue
            stopped = (
                (probability[interval] >= float(state_probability))
                & (ball_speed[interval] <= float(max_ball_speed))
                & (player_speed[interval] <= float(max_player_speed))
            )
            if np.sum(stopped) < int(min_stopped_frames):
                continue
            output[candidate_start:restart] = "D"
            break
    return output


def seed_delayed_control(labels, control_runs, features, probability, motion,
                         min_cue_frames=CONTROL_MIN_CUE_FRAMES,
                         lookback_frames=CONTROL_LOOKBACK_FRAMES,
                         max_restart_search_frames=CONTROL_RESTART_SEARCH_FRAMES,
                         state_probability=CONTROL_STATE_PROBABILITY,
                         state_probe_frames=CONTROL_STATE_PROBE_FRAMES,
                         live_transition_lookback=CONTROL_LIVE_TRANSITION_LOOKBACK,
                         post_restart_frames=CONTROL_POST_RESTART_FRAMES,
                         active_ball_speed=CONTROL_ACTIVE_BALL_SPEED,
                         min_active_ball_frames=CONTROL_MIN_ACTIVE_BALL_FRAMES,
                         min_onset_frames=MIN_ONSET_FRAMES,
                         max_ball_speed=MAX_BALL_SPEED,
                         max_player_speed=MAX_PLAYER_SPEED,
                         motion_threshold=MOTION_THRESHOLD,
                         min_stopped_frames=CONTROL_MIN_STOPPED_FRAMES,
                         min_ball_visible_share=MIN_BALL_VISIBLE_SHARE,
                         min_ball_goal_line_distance=MIN_BALL_GOAL_LINE_DISTANCE):
    """Use a delayed replay/close-up as confirmation of a tactical stop and restart.

    The required order is stop-state onset, broadcast interruption, then sustained restart.  A
    nearby H/A transition is reverted to the previous team because the possession model often
    changes team at the foul contact before its lower-confidence dead state becomes sustained.
    """
    output = np.asarray(labels).copy()
    features = np.asarray(features)
    probability = np.asarray(probability)
    motion = np.asarray(motion)
    if not (len(output) == len(features) == len(probability) == len(motion)):
        raise ValueError("labels, features, probability and motion must have the same length")

    index = {name: i for i, name in enumerate(restart_state.FEATURE_NAMES)}
    ball_speed = features[:, index["ball_speed"]] * 15.0
    player_speed = features[:, index["player_speed"]] * 5.0
    ball_visible = features[:, index["ball_visible"]]
    ball_x = features[:, index["ball_x"]]

    for _, cue_start, cue_stop, first, last in control_runs:
        if cue_stop - cue_start < int(min_cue_frames):
            continue
        probe_stop = min(cue_stop, cue_start + int(state_probe_frames))
        if (np.mean(probability[cue_start:probe_stop]) < float(state_probability)
                or np.mean(player_speed[cue_start:probe_stop]) > float(max_player_speed)):
            continue

        lookback = max(first, cue_start - int(lookback_frames))
        onset = _first_sustained(
            probability[lookback:cue_start + 1] >= float(state_probability),
            min_onset_frames)
        if onset is None:
            continue
        candidate_start = lookback + onset

        # A foul commonly creates a short, false H/A spell before the state model rises.  Remember
        # the most recent live-team transition so it can be restored to the preceding team.
        transition_start = max(first, candidate_start - int(live_transition_lookback))
        changes = (np.flatnonzero(
            output[transition_start + 1:candidate_start + 1]
            != output[transition_start:candidate_start]) + transition_start + 1)
        live_changes = [int(change) for change in changes
                        if output[change - 1] in "HA" and output[change] in "HA"]
        live_transition = live_changes[-1] if live_changes else None

        search_stop = min(last, cue_stop + int(max_restart_search_frames))
        restart_frames = np.flatnonzero(
            motion[cue_start:search_stop] >= float(motion_threshold)) + cue_start
        for restart in restart_frames:
            active_stop = min(last, restart + int(post_restart_frames))
            if np.sum(ball_speed[restart:active_stop] >= float(active_ball_speed)) \
                    < int(min_active_ball_frames):
                continue
            edit_start = candidate_start if live_transition is None else live_transition
            if np.any(output[edit_start:restart] == "D"):
                continue
            interval = slice(candidate_start, restart)
            if np.mean(ball_visible[interval]) < float(min_ball_visible_share):
                continue
            goal_line_distance = np.minimum(ball_x[interval], 1.0 - ball_x[interval])
            if np.median(goal_line_distance) < float(min_ball_goal_line_distance):
                continue
            stopped = (
                (probability[interval] >= float(state_probability))
                & (ball_speed[interval] <= float(max_ball_speed))
                & (player_speed[interval] <= float(max_player_speed))
            )
            if np.sum(stopped) < int(min_stopped_frames):
                continue
            if live_transition is not None:
                output[live_transition:candidate_start] = output[live_transition - 1]
            output[candidate_start:restart] = "D"
            break
    return output


def backfill_review_onsets(labels, control_runs, features, probability,
                           lookback_frames=REVIEW_LOOKBACK_FRAMES,
                           state_probability=REVIEW_STATE_PROBABILITY,
                           max_player_speed=REVIEW_MAX_PLAYER_SPEED,
                           min_state_frames=REVIEW_MIN_STATE_FRAMES,
                           min_missing_share=REVIEW_MIN_MISSING_SHARE,
                           min_control_frames=REVIEW_MIN_CONTROL_FRAMES):
    """Backfill long VAR-like stoppages whose ball tracking disappears before an existing D run."""
    output = np.asarray(labels).copy()
    features = np.asarray(features)
    probability = np.asarray(probability)
    if not (len(output) == len(features) == len(probability)):
        raise ValueError("labels, features and probability must have the same length")

    index = {name: i for i, name in enumerate(restart_state.FEATURE_NAMES)}
    player_speed = features[:, index["player_speed"]] * 5.0
    ball_visible = features[:, index["ball_visible"]]
    state = ((probability >= float(state_probability))
             & (player_speed <= float(max_player_speed)))
    control = np.zeros(len(output), dtype=bool)
    half_bounds = []
    for _, start, stop, first, last in control_runs:
        half_bounds.append((int(first), int(last)))
        if stop - start >= int(min_control_frames):
            control[int(start):int(stop)] = True

    for dead_start, _ in runs(output == "D"):
        containing = [(first, last) for first, last in half_bounds
                      if first <= dead_start < last]
        if not containing:
            continue
        first, _ = containing[0]
        search_start = max(first, dead_start - int(lookback_frames))
        candidates = runs(state[search_start:dead_start + 1])
        for state_start, state_stop in reversed(candidates):
            state_start += search_start
            state_stop += search_start
            if state_stop < dead_start + 1:
                continue
            interval = slice(state_start, dead_start)
            if dead_start - state_start < int(min_state_frames):
                break
            if np.mean(ball_visible[interval] < 0.5) < float(min_missing_share):
                break
            if np.sum(control[interval]) < int(min_control_frames):
                break
            output[state_start:dead_start] = "D"
            break
    return output


def restore_live_goalkeeper_carries(labels, trk,
                                     max_dead_frames=KEEPER_MAX_DEAD_FRAMES,
                                     max_goal_line_distance=KEEPER_MAX_GOAL_LINE_DISTANCE,
                                     max_player_distance=KEEPER_MAX_PLAYER_DISTANCE,
                                     min_visible_share=KEEPER_MIN_VISIBLE_SHARE,
                                     min_close_share=KEEPER_MIN_CLOSE_SHARE,
                                     min_same_team_share=KEEPER_MIN_SAME_TEAM_SHARE,
                                     min_ball_displacement=KEEPER_MIN_BALL_DISPLACEMENT):
    """Restore short same-team dead spells when a near-goal player visibly carries the ball."""
    output = np.asarray(labels).copy()
    players = np.asarray(trk["players"])
    ball = np.asarray(trk["ball"])
    team = np.asarray(trk["team"])
    if len(output) != len(players) or len(output) != len(ball):
        raise ValueError("labels and tactical tracking must have the same length")
    pitch_length = float(trk.get("pitch", (105.0, 68.0))[0])

    for start, stop in runs(output == "D"):
        if (start == 0 or stop == len(output)
                or stop - start > int(max_dead_frames)):
            continue
        live_label = output[start - 1]
        if live_label not in "HA" or output[stop] != live_label:
            continue

        ball_xy = ball[start:stop, :2]
        visible = np.isfinite(ball_xy).all(axis=1)
        if np.mean(visible) < float(min_visible_share):
            continue
        visible_xy = ball_xy[visible]
        goal_line_distance = np.minimum(visible_xy[:, 0], pitch_length - visible_xy[:, 0])
        if np.median(goal_line_distance) > float(max_goal_line_distance):
            continue

        player_xy = players[start:stop, :, :2]
        separation = np.linalg.norm(player_xy - ball_xy[:, None, :], axis=2)
        finite = np.isfinite(separation)
        safe_separation = np.where(finite, separation, np.inf)
        nearest_slot = np.argmin(safe_separation, axis=1)
        nearest_distance = safe_separation[np.arange(stop - start), nearest_slot]
        target_team = 0 if live_label == "H" else 1
        if np.mean(nearest_distance <= float(max_player_distance)) < float(min_close_share):
            continue
        if np.mean(team[nearest_slot] == target_team) < float(min_same_team_share):
            continue
        if np.linalg.norm(visible_xy[-1] - visible_xy[0]) < float(min_ball_displacement):
            continue
        output[start:stop] = live_label
    return output


def restore_broadcast_goalkeeper_holds(labels, base_labels, trk,
                                       min_hold_frames=KEEPER_HOLD_MIN_FRAMES,
                                       hard_lookback_frames=KEEPER_HOLD_HARD_LOOKBACK_FRAMES,
                                       max_goal_line_distance=KEEPER_MAX_GOAL_LINE_DISTANCE,
                                       max_player_distance=KEEPER_MAX_PLAYER_DISTANCE,
                                       deepest_margin=KEEPER_DEEPEST_MARGIN,
                                       hard_dead=None):
    """Veto sustained broadcast-only D frames under visible goalkeeper control.

    This operates on the provenance of each frame instead of the length of the final D run.  It
    therefore handles a short stationary keeper hold embedded in a longer camera interruption.
    Actual goal kicks remain protected by current or recent hard out-of-pitch evidence.
    """
    output = np.asarray(labels).copy()
    base = np.asarray(base_labels)
    players = np.asarray(trk["players"])
    ball = np.asarray(trk["ball"])
    team = np.asarray(trk["team"])
    if not (output.shape == base.shape == (len(players),) == (len(ball),)):
        raise ValueError("labels, base labels and tactical tracking must have the same length")
    hard = (hybrid_pipeline.dead_ball(trk) if hard_dead is None
            else np.asarray(hard_dead, dtype=bool))
    if hard.shape != output.shape:
        raise ValueError("hard dead evidence must have the same shape as labels")

    pitch_length = float(trk.get("pitch", (105.0, 68.0))[0])
    ball_xy = ball[:, :2]
    visible = np.isfinite(ball_xy).all(axis=1)
    separation = np.linalg.norm(players[..., :2] - ball_xy[:, None, :], axis=2)
    separation = np.where(np.isfinite(separation), separation, np.inf)
    nearest_slot = np.argmin(separation, axis=1)
    nearest_distance = separation[np.arange(len(output)), nearest_slot]
    nearest_team = team[nearest_slot]
    ball_left = ball_xy[:, 0] <= pitch_length / 2.0
    goal_distance = np.where(ball_left, ball_xy[:, 0], pitch_length - ball_xy[:, 0])

    keeper_like = np.zeros(len(output), dtype=bool)
    for label, target_team in (("H", 0), ("A", 1)):
        frames = base == label
        slots = np.flatnonzero(team == target_team)
        target_x = players[:, slots, 0]
        depth = np.where(ball_left[:, None], target_x, pitch_length - target_x)
        depth = np.where(np.isfinite(depth), depth, np.inf)
        deepest = np.min(depth, axis=1)
        nearest_x = players[np.arange(len(output)), nearest_slot, 0]
        nearest_depth = np.where(ball_left, nearest_x, pitch_length - nearest_x)
        keeper_like[frames] = (np.isfinite(nearest_depth[frames])
                               & (nearest_depth[frames]
                                  <= deepest[frames] + float(deepest_margin)))

    # Protect the placement and wait before a goal kick even after the ball is observed in bounds.
    cumulative = np.r_[0, np.cumsum(hard.astype(np.int64))]
    index = np.arange(len(output))
    lookback = np.maximum(0, index - int(hard_lookback_frames))
    recent_hard = (cumulative[index + 1] - cumulative[lookback]) > 0

    for label, target_team in (("H", 0), ("A", 1)):
        candidate = (
            (output == "D") & (base == label) & visible & ~recent_hard
            & (goal_distance <= float(max_goal_line_distance))
            & (nearest_distance <= float(max_player_distance))
            & (nearest_team == target_team) & keeper_like
        )
        for start, stop in runs(candidate):
            if stop - start >= int(min_hold_frames):
                output[start:stop] = label
    return output


def refine(labels, broadcast_valid, features, probability, motion, half_blocks, offsets,
           broadcast_tracking_valid=None):
    """Apply existing-interval extension, then strongly corroborated standalone seeding."""
    replay_runs = mapped_replay_runs(
        ~np.asarray(broadcast_valid, dtype=bool), half_blocks, offsets, len(labels))
    output = extend_existing(labels, replay_runs)
    output = seed_standalone(output, replay_runs, features, probability, motion)
    output = seed_inside_replay(output, replay_runs, features, probability, motion)
    broadcast_valid = np.asarray(broadcast_valid, dtype=bool)
    control = ~broadcast_valid
    if broadcast_tracking_valid is not None:
        broadcast_tracking_valid = np.asarray(broadcast_tracking_valid, dtype=bool)
        if broadcast_valid.shape != broadcast_tracking_valid.shape:
            raise ValueError("broadcast homography and tracking validity must have the same shape")
        dropout_runs = mapped_replay_runs(
            broadcast_valid & ~broadcast_tracking_valid, half_blocks, offsets, len(labels))
        output = seed_tracking_dropout(
            output, dropout_runs, features, probability, motion)
        control |= ~broadcast_tracking_valid
    control_runs = mapped_replay_runs(control, half_blocks, offsets, len(labels))
    output = seed_delayed_control(
        output, control_runs, features, probability, motion)
    return backfill_review_onsets(output, control_runs, features, probability)


def predict(labels, trk, broadcast_valid, half_blocks, offsets,
            restart_model=restart_state.DEFAULT_MODEL, broadcast_tracking_valid=None):
    """Convenience wrapper that derives tactical restart and motion evidence from tracking."""
    restart_model = (restart_state.load_model(restart_model)
                     if isinstance(restart_model, (str, os.PathLike)) else restart_model)
    features = restart_state.frame_features(trk)
    probability = restart_state.probabilities(features, restart_model)
    motion = constrained_restart.motion_evidence(features)
    output = refine(labels, broadcast_valid, features, probability, motion, half_blocks, offsets,
                    broadcast_tracking_valid=broadcast_tracking_valid)
    output = restore_live_goalkeeper_carries(output, trk)
    return restore_broadcast_goalkeeper_holds(output, labels, trk)
