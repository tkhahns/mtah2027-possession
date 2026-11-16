"""The referee's whistle as a clean-up rule on predicted labels (rule_cleanup's "whistle").
    foul whistle     the ball was moving just before it (live play): play stops. Dead from the
                     whistle until the predicted labels change (the predicted run the whistle falls
                     in ends: a dead ball, or the other team). Already dead: nothing changes.
    restart whistle  the ball was still or slow just before it: the referee lets play go on.
                     Dead from the whistle until the ball moves again; from then live, given to
                     the nearest player's team, until the predicted labels change (the end of the
                     predicted run the move falls in). restart_team "predicted" keeps the predicted
                     team where that run was already live.
"""

from collections import Counter
from dataclasses import dataclass
import os
import warnings

import numpy as np

from utils import series_utils

from .data import HOME

FOUL, RESTART, MISSING = "foul", "restart", "missing"
BOTH = (FOUL, RESTART)
LIVE = ("H", "A")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass(frozen=True)
class Settings:
    fps: int = 25
    kinds: tuple = (FOUL,)            # which whistles act 
    p1_min: float = 9.0
    min_strength: float = 9.0
    min_loud: float = 12.0
    cooldown_s: float = 12.0          # a whistle this soon after a kept one is ignored
    ball_max_fill_frames: int = 25
    ball_smooth_frames: int = 7
    ball_smooth_order: int = 2
    ball_max_jump_speed: float = 40.0
    interp_max_accel: float = 0.005   # filled gap: velocity changes less than this per frame...
    interp_min_s: float = 0.5         # ...for at least this long...
    interp_min_speed: float = 0.02    # ...while moving
    # foul or restart
    speed_window_s: float = 1.0       # ball speed read over this long before the whistle
    ball_speed: float = 2.0           # mean ball speed at least this = foul, else restart
    players_speed: float = 1.5        # ball unseen: median player speed below this = restart
    # restart whistle
    move_speed: float = 2.0           # the ball moves again faster than this...
    move_frames: int = 3              # ...for this many frames in a row...
    restart_wait_s: float = 15.0      # ...within this long, or the whistle is left alone
    restart_min_dead_s: float = 1.5   # in live play, a ball moving sooner than this is ignored
    restart_team: str = "kicker"      # "kicker" or "predicted" (see the module docstring)

    def frames(self, seconds):
        return max(1, int(round(seconds * self.fps))) if seconds > 0 else 0

    def detect_params(self):
        return {"p1_min": self.p1_min, "min_strength": self.min_strength,
                "min_loud": self.min_loud, "cooldown_s": self.cooldown_s}


# -- the whistles -------------------------------------------------------------------------------

def load_whistles(data_dir=None, cache=None, settings=Settings(), game=None):
    """Start frames of the referee's whistles in the whole video, in order. Reads the sound from
    the video once into cache/whistle/ (ffmpeg), then only the cached per-frame tones."""
    from games import GAME, GAMES, asset_paths
    from whistle import build, detect

    game = game or GAME
    folder = cache or os.path.join(ROOT, "cache", "whistle")
    video = asset_paths(game, data_dir=data_dir)["video"]
    feats, _, _ = build.ensure(video, folder, GAMES[game]["n_video_frames"], log=lambda *a: None)
    events = detect.find_events(feats["f"], feats["p1"], feats["lvl"], settings.detect_params(),
                                settings.fps)
    return [int(e["start_frame"]) for e in events]


# -- the ball and the players (only the parts this rule needs) ----------------------------------

def _savgol_weights(window, order):
    t = np.arange(-(window // 2), window // 2 + 1, dtype=np.float64)
    return np.linalg.pinv(np.vander(t, order + 1, increasing=True))[0]


def ball_motion(ball_xy, s=Settings()):
    """{xy, speed, missing} of the ball: gaps up to ball_max_fill_frames filled linearly, split
    where it jumps faster than ball_max_jump_speed, Savitzky-Golay smoothed, velocity by central
    differences inside each piece. missing: not known, or on a straight-line filled stretch."""
    xy = np.array(ball_xy, dtype=np.float64)[:, :2]
    known = ~np.isnan(xy[:, 0])
    for a, b in series_utils.runs(~known):
        if a == 0 or b == len(xy) or b - a > s.ball_max_fill_frames:
            continue
        weight = (np.arange(a, b) - (a - 1)) / float(b - (a - 1))
        xy[a:b] = xy[a - 1] + weight[:, None] * (xy[b] - xy[a - 1])
    known = ~np.isnan(xy[:, 0])
    step = np.full(len(xy), np.inf)
    step[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) * s.fps
    with np.errstate(invalid="ignore"):
        jump = step > s.ball_max_jump_speed
    pieces = []
    for a, b in series_utils.runs(known):
        cuts = [a] + [int(f) for f in np.flatnonzero(jump[a + 1:b]) + a + 1] + [b]
        pieces += list(zip(cuts[:-1], cuts[1:]))
    weights, half = _savgol_weights(s.ball_smooth_frames, s.ball_smooth_order)[::-1], \
        s.ball_smooth_frames // 2
    smooth = xy.copy()
    for a, b in pieces:
        if b - a >= s.ball_smooth_frames:
            for axis in (0, 1):
                smooth[a + half:b - half, axis] = np.convolve(xy[a:b, axis], weights, "valid")
    vel = np.full_like(smooth, np.nan)
    for a, b in pieces:
        if b - a >= 2:
            vel[a:b] = np.gradient(smooth[a:b], axis=0) * s.fps
    speed = np.linalg.norm(vel, axis=1)
    acc = np.full(len(xy), np.inf)
    acc[1:] = np.linalg.norm(np.diff(vel, axis=0), axis=1)
    with np.errstate(invalid="ignore"):
        flat = (acc < s.interp_max_accel) & (speed > s.interp_min_speed)
    filled = np.zeros(len(xy), dtype=bool)
    for a, b in series_utils.runs(flat):
        if b - a >= s.frames(s.interp_min_s):
            filled[max(0, a - 1):b] = True
    return {"xy": smooth, "speed": speed, "missing": ~known | filled}


def median_player_speed(players, fps=25):
    """(frames,) median speed of the tracked players: the tracking's speed, or the speed from
    positions (central difference) where it is missing."""
    players = np.asarray(players, dtype=np.float32)
    given = players[..., 2] if players.shape[-1] > 2 else np.full(players.shape[:2], np.nan)
    from_xy = np.full(players.shape[:2], np.nan, dtype=np.float32)
    from_xy[1:-1] = np.linalg.norm(players[2:, :, :2] - players[:-2, :, :2], axis=-1) * (fps / 2.0)
    speeds = np.where(np.isnan(given), from_xy, given)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(speeds, axis=1)


@dataclass
class Evidence:
    whistles: list              # start frames, in order
    ball_xy: np.ndarray         # (frames, 2) smoothed
    ball_speed: np.ndarray      # (frames,)
    missing: np.ndarray         # (frames,) bool
    median_speed: np.ndarray    # (frames,)
    players_xy: np.ndarray      # (frames, slots, 2)
    team: np.ndarray            # (slots,) HOME / AWAY
    settings: Settings = Settings()


def evidence(trk, whistles, settings=Settings()):
    """Evidence for the rule from a tracking dict (players, ball, team) and the whistle frames."""
    ball = ball_motion(trk["ball"], settings)
    players = np.asarray(trk["players"])
    return Evidence(sorted(int(w) for w in whistles), ball["xy"], ball["speed"], ball["missing"],
                    median_player_speed(players, settings.fps), players[..., :2],
                    np.asarray(trk["team"]), settings)


# -- reading one whistle --------------------------------------------------------------------------

def whistle_kind(ev, w):
    """FOUL, RESTART or MISSING (ball unseen and no players either) for a whistle at frame w."""
    s = ev.settings
    a = max(0, w - s.frames(s.speed_window_s))
    seen = ~ev.missing[a:w]
    if w > a and seen.sum() * 2 >= w - a:
        speed = float(np.mean(ev.ball_speed[a:w][seen]))
        return FOUL if speed >= s.ball_speed else RESTART
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        players = np.nanmean(ev.median_speed[a:w + 1]) if w >= a else np.nan
    if not np.isfinite(players):
        return MISSING
    return FOUL if players >= s.players_speed else RESTART


def moves_again(ev, w, hi):
    """First frame after w where the ball is seen moving faster than move_speed for move_frames
    frames in a row, within restart_wait_s (and not after hi); None if it does not."""
    s = ev.settings
    b = min(hi, w + s.frames(s.restart_wait_s))
    if b <= w:
        return None
    with np.errstate(invalid="ignore"):
        fast = ~ev.missing[w + 1:b + 1] & (ev.ball_speed[w + 1:b + 1] > s.move_speed)
    need, run = max(1, s.move_frames), 0
    for i, ok in enumerate(fast):
        run = run + 1 if ok else 0
        if run == need:
            return w + 1 + i - need + 1
    return None


def kicker_team(ev, frame):
    """"H" / "A": the team of the player nearest the ball at that frame; None if nobody or no ball."""
    d = np.linalg.norm(ev.players_xy[frame] - ev.ball_xy[frame], axis=-1)
    if not np.isfinite(d).any():
        return None
    return "H" if ev.team[int(np.nanargmin(d))] == HOME else "A"


# -- the rule -------------------------------------------------------------------------------------

def run_ends(labels):
    """(frames,) the end (exclusive) of the run each frame is in."""
    labels = np.asarray(labels)
    cuts = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    ends = np.append(cuts, len(labels))
    return ends[np.searchsorted(cuts, np.arange(len(labels)), side="right")]


def apply(labels, lo, hi, ev, report=None):
    """The whistles in [lo, hi] on one half's predicted labels (labels[0] is frame lo). How far a
    whistle reaches is read on the predicted labels, not on what earlier whistles changed. A copy."""
    s = ev.settings
    if s.restart_team not in ("kicker", "predicted"):
        raise ValueError("restart_team must be 'kicker' or 'predicted', not {!r}".format(s.restart_team))
    report = report if report is not None else Counter()
    base = np.asarray(labels)
    out = base.copy()
    ends = run_ends(base)
    for w in ev.whistles:
        if not lo <= w <= hi:
            continue
        kind = whistle_kind(ev, w)
        report["whistle heard: " + kind] += 1
        i = w - lo
        if kind not in s.kinds or base[i] not in ("D",) + LIVE:
            continue
        if kind == FOUL:
            if base[i] == "D":
                report["whistle foul: already dead"] += 1
                continue
            e = ends[i]
            out[i:e] = "D"
            report["whistle foul: made dead to the next predicted change"] += 1
            report["whistle foul: frames made dead"] += int(e - i)
            report["whistle foul: dead 10 s or longer"] += int(e - i >= 10 * s.fps)
            continue
        go = moves_again(ev, w, hi)
        if go is None:
            report["whistle restart: ball does not move"] += 1
            continue
        g = go - lo
        if base[i] in LIVE and go - w < s.frames(s.restart_min_dead_s):
            report["whistle restart: dropped short"] += 1
            continue
        report["whistle restart: frames made dead"] += int(np.sum(out[i:g] != "D"))
        out[i:g] = "D"
        e = ends[g]
        team = kicker_team(ev, go)
        if base[g] == "D":
            if team is None:
                report["whistle restart: no kicker"] += 1
                continue
            new = team
            report["whistle restart: frames made live"] += int(e - g)
        elif base[g] in LIVE:
            new = team if (team and s.restart_team == "kicker") else base[g]
            report["whistle restart: frames given to the other team"] += int(e - g) * (new != base[g])
        else:
            continue
        out[g:e] = new
        report["whistle restart: live from the ball moving"] += 1
    return out
