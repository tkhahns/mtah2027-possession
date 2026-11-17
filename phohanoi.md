# Ball Possession from Automatic Tracking: Technical Document

**More than a Hack 2027, Phase 1** · Team: **phohanoi** ·
Repository: <https://github.com/tkhahns/mtah2027-possession> ·
Submission file: `phohanoi.csv`

---

## 1. Summary

- **The pipeline is fully algorithmic.** It cleans the tracking, builds learned dead/live and team
  evidence, decodes whole segments with a neural semi-Markov decoder, checks stoppages against the
  broadcast camera, the referee's whistle and Smart Tagging, and finishes with a rule clean-up.
- **No challenge-match labels.** Every learned component was trained and selected only on
  **10 public external matches** (3 Metrica Sports, 7 IDSSE/DFL). 
- **Submission file.** 199,319 rows, 233 possessions with a mean of 13.8 s. There are no
  abstentions on graded frames; `X` appears only on the 40,462 ungraded frames.
- **Public 5-minute window, used only as a pipeline check:** spellF1
  **0.648**, macroF1 **0.920**, alive/dead F1 0.980. The organisers' `smooth50` baseline scores 0.621 / 0.326 on the
  same window.
  
- **For the strict leave-one-match-out evaluation across all 10 external Metrica/IDSSE matches**, the current frozen production pipeline scored: spellF1 **0.64341**, macro F1 **0.91702**
  
---

## 2. Approach

### 2.1 Design rationale

- **The metric scores possessions, not frames.** spellF1 rewards whole possessions, so the
  decisions are made at segment level. After the base labels, each stage either moves existing
  boundaries without changing the number of spells, or needs independent corroboration before it
  creates a new segment.
- **Most errors come from noisy tracking.** Cross-team identity switches, fragmented slots and a
  ball that is missing on 41% of graded frames cause most mistakes, so the tracking is repaired.
- **External labelled data is small and cleaner than ours.** We therefore use slot-invariant
  features and compact models, and train on synthetically degraded copies of the external
  matches that imitate the challenge tracking.
- **Strong cues that cannot be learned externally are used as conservative rules.** Broadcast
  camera cuts, whistle audio and set-piece tags are strong stoppage cues, but the external
  tracking datasets do not contain them. The whistle detector was designed instead on the audio
  of two external match videos (§4).

### 2.2 Pipeline

```
tactical + broadcast tracking, broadcast homography, Smart Tagging, tactical video audio
  │
  ├─ 1. Data cleaning ............ repaired tactical tracking
  ├─ 2. Base labels .............. learned dead/live + nearest-team + learned short-spell merge
  ├─ 3. Segment decoding ......... set-temporal network + semi-Markov decoder
  ├─ 4. Corroboration ............ broadcast camera cuts, live-control veto, keeper possession
  └─ 5. Rule clean-up ............ whistles, short stoppages, set-piece tags, corners, short spells
  → H / A / D on graded frames, X outside the two halves
```

**Stage 1: Data cleaning** (`src/clean_data/pipeline.py`)

- **Camera alignment.** One tactical-to-broadcast frame shift per half, found from the two
  tracking files: kick-off detected on each camera, then the shift that best overlays players and
  ball. 
- **Player-team cleaning.** Tracking sometimes swaps player identities, so we find stretches where
a track follows a player from the other team and move them to a free slot on the correct team.
- **Goalkeeper repair.** Keeper slots that jump into the opposite penalty box are blanked, assign the correct coordinate back to the correct slot.
- **Keeper hold, ball misising.** If a ball gap of more than 1 s starts within 2 m of a goalkeeper, the ball
  follows that keeper until it is tracked again. These holds are recorded for Stage 4.
- **Gap filling.** Player and ball gaps of up to 15 s are filled by linear interpolation. Longer
  ball gaps get dead reckoning (constant velocity), and the rest stays empty.

**Stage 2: Base labels** (`src/possession/learned_timing_pipeline.py`). Refer 4. External data for data used

- **Hard dead ball.** If the last observed ball position is outside the pitch lines, the frame is
  `D`. No later stage can override this.
- **Restart-state MLP** (47→32→16→1). It reads 47 slot-invariant context features: team
  centroids, spreads and linearity, shape near the ball, ball visibility, speed and distance to
  the line, and ±0.5 s before/after trends. The output is P(dead). A constrained decoder uses it
  in two ways:
  - P ≥ 0.80 may extend an existing hard-dead interval;
  - a learned-only interval needs P ≥ 0.85, at least 4 s, and a stopped-to-moving restart
    signature.
- **Team.** The team of the tracked player nearest the ball. It is carried through missing-ball
  frames while play is live (never across `D`), then smoothed with a ±2 s majority vote.
- **Learned short-spell filter** (29→16→1 MLP). An `H`/`A` run of 6 s or less, sandwiched by the
  other team, is kept or merged. The 29 spell-level features cover control frames, the
  ownership-evidence profile, distance margins, ball speed and visibility, and P(dead). The
  decision threshold is 0.40.
- **Boundary timing.** `H`↔`A` boundaries move up to ±1.4 s to the best split of signed
  ownership evidence. `D` interval edges move up to ±2 s on the smoothed P(dead) and always keep
  their hard-dead frames. Neither step changes the number or order of spells.

**Stage 3: Segment decoding** (`set_temporal.py`, `segmental_decoder.py`, `set_segment_pipeline.py`). Refer 4. External data for data used

- **Set-temporal network** (64k parameters, PyTorch, runs at 5 Hz):
  - a per-player encoder over 8 features (position, speed, offset to ball, team flag), with
    mean and max pooling per team, so it is invariant to slot order and slot count;
  - the 47 context features from Stage 2;
  - 4 dilated residual temporal convolutions (dilation 1/2/4/8);
  - two heads: the `H`/`A`/`D` state, and three boundary types (transfer, stop, restart).
- **Semi-Markov decoder.** Viterbi over a sparse boundary lattice. It combines:
  - the state emissions;
  - transition-specific boundary probabilities;
  - log-normal segment-duration priors learned from the external matches;
  - the Stage 2 labels as a soft prior;
  - hard out-of-pitch frames as anchors (they cannot be live).
- **Why semi-Markov.** It scores whole segments, which matches what spellF1 counts and suppresses
  fragmentation directly.

**Stage 4: Corroboration** (`broadcast_deadball.py`, `live_control_veto.py`, `keeper_possession.py`)

- **Broadcast camera cuts.** Frames where the broadcast homography is invalid (a replay,
  close-up or crowd shot) are mapped onto the tactical timeline:
  - a cut that touches an existing `D` interval extends it;
  - a cut can create a new `D` interval only if the tactical tracking also shows a stopped,
    visible ball, P(dead) ≥ 0.6, and a restart within 20 s.

  The same corroboration applies to broadcast tracking dropouts, foul-like delayed cues and long
  reviews. A guard restores live play where a goalkeeper is visibly carrying the ball.
- **Live-control veto.** A short `D` island (≤ 8 s) between two spells of the same team becomes
  live again only if all of these hold:
  - that team shows sustained moving control (≥ 50% of frames);
  - the opponent controls ≤ 5% of frames;
  - the ball moves at least 1.5 m;
  - the ball was not outside the pitch in the previous 10 s.
- **Keeper possession.** Every keeper ball hold from Stage 1 is assigned with the keeper's team, unless
  a referee whistle occurred in the 20 s before it.

**Stage 5: Rule clean-up** (`rule_cleanup.py`, `whistle_cleanup.py`, `src/whistle/`), applied per
half in this order:

1. **Whistle.** Referee whistles are detected in the tactical video's audio track with no learned
   model: a steady two-tone chord with a low tone of 3.6–4.0 kHz and a tone ratio of 1.09–1.13,
   purity and loudness thresholds, and a 12 s cool-down. If the ball was moving (≥ 2 m/s) in the
   second before a whistle, the whistle is treated as a foul and play is `D` until the predicted
   run ends. Refer 4. External data for data used
2. **Short stoppage fill.** A `D` run shorter than 3.2 s that follows live play takes the
   preceding team's label.
3. **Set-piece tags.** Smart Tagging corner, goal kick, free kick and throw-in tags force a `D`
   window from *start − pre* to *start + 5 s*. The *pre* value is 15 s for a corner, 5 s for a
   goal kick or free kick, and 0 s for a throw-in.
4. **Corner taker.** After a corner tag, from the first frame where the ball is tracked again, the
   team attacking that end has possession.
5. **Short-spell merge.** An `H`/`A` run shorter than 1.8 s joins its longer neighbour. This step
   never turns `D` into live play.

Frames outside the two halves (as defined in the organisers' `games.py`) are written as `X`.

---

## 3. Assets

**Used**

| Asset | How it is used |
|---|---|
| Tactical tracking (ATD) | Primary input to every stage |
| Broadcast tracking (ATD) | Per-half time alignment; independent team evidence for player-team cleaning; dropout cue for camera cuts |
| Broadcast homography | Only its per-frame validity flag, as a camera-cut / stoppage cue. Also used to blank broadcast tracking that has no homography |
| Smart Tagging | Set-piece tags only (corner, goal kick, free kick, throw-in), plus the 2nd-half kick-off frame |
| Tactical video | **Audio track only**, for referee whistles. No image frames are decoded |

**Tried and dropped, or not used**

- **Dropped decoders and settings (after validate on Metrica and IDSSE):**
  - Plain-threshold restart decoder: replaced by the constrained decoder.
  - Boundary-only refinement of the set-temporal model: the semi-Markov (`segmental`) mode was
    chosen instead.
  - "Restart" whistles: detected but disabled; only foul whistles act.
- **Not used:**
  - Tactical homography, the broadcast video image, and Smart Tagging phase tags (build-up,
    progression and so on).
  - The organisers' 5-minute ground truth was used only to check frame alignment and file format.


---

## 4. External data


| Dataset | Source | Owner | Amount | Target used |
|---|---|---|---|---|
| Metrica Sports sample data, Games 1–3 | [github.com/metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) | Metrica Sports (provider event data) | 3 matches | Timed events converted algorithmically into frame-level `H`/`A`/`D` (`possession/metrica.py`) |
| IDSSE | [huggingface.co/datasets/pysport/idsse-data](https://huggingface.co/datasets/pysport/idsse-data), CC BY 4.0 | DFL official tracking feed | 7 matches | Frame-level live/dead and possession. Sub-second touch reversals between spells of the same team are merged (`possession/idsse.py`) |

**Whistle detection (audio only).** We used these two public full-match tactical-camera videos
of other games to design the referee-whistle detector (`src/whistle/`). None of them is the
challenge match, and none of the video is stored in the repository.

| Video | Source | Amount | Used for |
|---|---|---|---|
| Croatia 1–1 Czechia, UEFA EURO 2020, tactical camera | [youtube.com/watch?v=KgQquW68E-Q](https://www.youtube.com/watch?v=KgQquW68E-Q&t=433s) | 1 full match | Whistle sound profile: two-tone chord band, tone ratio, purity and loudness thresholds |
| Manchester City vs Chelsea, 2021 UEFA Champions League final, tactical camera | [youtube.com/watch?v=JA0p0Bg9N1w](https://www.youtube.com/watch?v=JA0p0Bg9N1w) | 1 full match | Same as above |


- **Metrica event rules.**
  - Possession starts at a team's on-ball event (pass, carry, shot, recovery or set piece).
    Challenges and "ball lost" events do not change possession.
  - A spell shorter than 1 s with no pass, shot or set piece counts as an uncontrolled touch.
  - Dead time runs from a ball out, goal, offside or foul to the next restart.
- **Synthetic views** (`possession/testsets.py`). Generated in code with fixed seeds:
  - *hidden*: the ball is blanked on 40% of live play, more often in crowds and on fast balls;
  - *messy*: *hidden*, plus 46 tracklet slots, about 20.5 of 22 players visible, duplicate
    tracks, jitter, and opponent ID swaps in tackles.

  Models train on the *hidden* and *messy* views. The noise rates come from the published
  description of the challenge data, and some are guesses. 
- **Download and storage.** The raw files are downloaded by `src/training/fetch_*.py` into
  `external/` and are not committed. The frozen models live in `src/models/`.


---

## 5. Training and model selection

- **Validation splits.** All validation holds out complete matches (leave-one-match-out). The
  base predictions fed to the temporal model are cross-fitted, so it never sees labels from
  checkpoints that were trained on the held-out match.
- **Selection criterion.** The official spellF1 code, run on the *hidden* and *messy* views. A
  candidate must predict at least 90% of the true possession count, and macroF1 breaks ties.
- **Set-temporal network.** AdamW (lr 1e-3), 6 epochs, 25.6 s windows. The loss is state
  cross-entropy plus boundary BCE on both the clean and the noisy view, plus a 0.25-weighted
  clean/noisy consistency term, so slot count and occlusion cannot become shortcuts.
- **Restart-state model.** Trained only on frames that the hard out-of-pitch rule does not
  already cover, so its capacity goes to fouls, offsides and central restarts.
- **Short-spell classifier.** Trained on 2,471 candidate spells (740 "keep"). The target is "keep"
  if the spell matches a true spell of the same team at IoU ≥ 0.3.
- **The challenge match is never opened by any training script.** The public 5 minutes were not
  used to choose between variants.

---

## 6. Possession rules implemented

- **Ball in play:**
  - The ball is dead whenever it was last seen outside the pitch lines, until it is seen inside
    again. This rule is never overridden.
  - Fouls, offsides and central free kicks are recognised from the team shape and from players
    and ball coming to a stop. They need tracking corroboration, and are reinforced by foul
    whistles, broadcast camera cuts and set-piece tags.
  - A stoppage shorter than 3.2 s inside live play is treated as continuous play by the same
    team.
- **Controlled possession.** A frame counts as controlled when all of these hold:
  - the nearest player is within 2.5 m of the ball;
  - that player is at least 0.45 m closer than the nearest opponent;
  - the ball is moving slower than 10 m/s.
- **Loose balls and contested phases.** Possession stays with the last team in control. The owner
  is carried through missing-ball frames while play is live, never across a stoppage.
- **Challenges, deflections and aerial duels.** An opposing spell of 6 s or less between two
  spells of the same team is merged back unless the learned classifier finds real control by the
  opponent. Any spell shorter than 1.8 s is absorbed.
- **Restarts:**
  - After a stoppage, the team nearest the ball when play resumes has possession.
  - After a corner, the attacking team of that end has possession once the ball is tracked again.
  - Restart tag windows are always `D`.
- **Goalkeepers.** A ball held by a keeper is live possession of the keeper's team, unless a
  whistle in the preceding 20 s means play was stopped. A goal-kick tag still makes the frame
  `D`.
- **Dead-ball ownership is never predicted.** A stoppage is always `D`.
- **Abstention.** We never abstain (`X`) on graded frames. `X` is written only outside the two
  halves.

---

## 7. Limitations and future work

- **Long ball occlusions.** Gaps over 15 s are not filled. Hand-overs during an occlusion are timed
  from carried ownership and the learned boundary model, and can be late or missed.
- **Domain shift.**
  - The models learn from 10 external matches, mostly cleaner than ours.
  - The synthetic noise rates are partly guessed.
  - The 64k-parameter network is deliberately small, so its capacity is limited.
- **Conservative fragmentation.** The pipeline tends to merge short genuine possessions: on the
  public window it predicts 12 possessions against 17 true ones. A short-spell decision with
  better calibration is the biggest expected gain.
- **Pipeline complexity.** Many sequential stages interact, so it is hard to attribute an error to
  one of them.
- **With more time:**
  - Add more provider datasets with realistic tracking noise.
  - Detect the ball on the broadcast to fill tactical gaps.


---

## 8. Reproduction

```bash
python -m pip install -r requirements-submission.txt   # pinned; Python 3.13, torch 2.6.0+cu124
# unpack the organisers' bundle into ./data  (or set MTAH_DATA_DIR)
python src/make_submission.py                          # writes evaluation/results/phohanoi.csv
python evaluation/validate_submission.py evaluation/results/phohanoi.csv
```

- **Inference uses only the frozen models in `src/models/`**: `restart_state.json`,
  `spell_candidate.json`, `set_temporal.pt` and `team_role_scales.json`. No external data is
  needed.
- **Whistle audio cache.** The first run decodes the match audio once, using ffmpeg from PATH or
  the pinned `imageio-ffmpeg`, into `cache/whistle/`. This folder is git-ignored derived data.
  Later runs read the cache.
- **Reference output.** Two runs reproduced the file byte for byte with SHA-256
  `8916418653fa775a5453663132dceed3257ad03870372e6c25a47a1dc6c87b74`: 199,319 rows and
  233 possessions. The hash does not depend on the output name (`--name`, default `phohanoi`).
  Environment: Windows 11,
  Python 3.13.16, torch 2.6.0+cu124, RTX 4070 Ti SUPER.
- **Retraining** from the public external data:

  ```bash
  python src/training/fetch_metrica.py
  python src/training/fetch_idsse.py
  python src/training/possession_models/train_restart_state.py --include-idsse
  python src/training/possession_models/train_spell_candidate.py
  python src/training/possession_models/train_set_temporal.py --include-idsse
  python src/training/player_cleaning/role_scales_thresholds.py
  ```

- **Runtime.** About 7-10 minutes (depend on machine capability) end to end on the reference machine, with the whistle cache
  already built.
