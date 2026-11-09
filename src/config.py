"""Match constants for BAR-ATH (from the organisers' games.py, DATA.md and FAQ.md).

Every array in this code is indexed by video frame of the tactical video: row i of the submission
is frame i. Team index 0 = home (Barca, H), 1 = away (Athletic, A).
"""

GAME = "BAR-ATH"
N_FRAMES = 199319              # frames of the tactical video = rows of the submission
FPS = 25
PITCH = (105.3, 68.0)          # metres
HOME, AWAY = 0, 1

# Graded video frames of each half, inclusive: the tracking IdFrame span of each half minus that
# half's offset (games.py: halves 1713242..1792708 and 1812866..1892255, offsets 1702057 and
# 1702058). Everything outside them is never graded.
HALVES = [(1, 11185, 90651), (2, 110808, 190197)]

# The 2nd half is graded from 110808 but kicks off at 113742: the frames in between are all dead.
KICKOFF_2ND = 113742
ALWAYS_DEAD = [(HALVES[1][1], KICKOFF_2ND)]   # [first, stop)

LABELS = ("H", "A", "D", "X")
