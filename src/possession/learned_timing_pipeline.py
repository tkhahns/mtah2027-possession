"""Frozen timing pipeline with learned extension of control-aware short-spell merging."""

import os

from . import (boundary_timing, constrained_restart, dead_boundary_timing, hybrid_pipeline,
               restart_state, spell_candidate, spell_filter)


def predict(trk, restart_model=restart_state.DEFAULT_MODEL,
            candidate_model=spell_candidate.DEFAULT_MODEL):
    restart_model = (restart_state.load_model(restart_model)
                     if isinstance(restart_model, (str, os.PathLike)) else restart_model)
    candidate_model = (spell_candidate.load_model(candidate_model)
                       if isinstance(candidate_model, (str, os.PathLike)) else candidate_model)
    features = restart_state.frame_features(trk)
    probability = restart_state.probabilities(features, restart_model)
    labels = constrained_restart.predict_from_probabilities(trk, probability, features)
    control = spell_filter.control_team(trk)
    ownership = boundary_timing.ownership_evidence(trk, control=control)
    labels = spell_candidate.merge(
        labels, trk, probability, candidate_model,
        restart_features=features, control=control, ownership=ownership,
        max_frames=int(candidate_model.get("max_frames", spell_candidate.MAX_FRAMES)))
    labels = boundary_timing.refine(labels, trk, evidence=ownership)
    dead = dead_boundary_timing.dead_evidence(probability)
    return dead_boundary_timing.refine(
        labels, dead, hybrid_pipeline.dead_ball(trk), radius=dead_boundary_timing.RADIUS,
        min_gain=dead_boundary_timing.MIN_GAIN)
