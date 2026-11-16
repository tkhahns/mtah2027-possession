"""Download Metrica Sports' public sample games (other matches, not ours) into external/metrica/.

    python src/training/fetch_metrica.py

Three anonymised matches with tracking and timed events, from
https://github.com/metrica-sports/sample-data (credit: Metrica Sports). About 180 MB. We use them
to choose between pipeline variants, because the rules forbid tuning on our own match. The
folder is gitignored: the data is fetched, never committed. Files already there are skipped.
"""

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "external", "metrica")
BASE = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
FILES = [
    "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
    "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
    "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    "Sample_Game_3/Sample_Game_3_events.json",
    "Sample_Game_3/Sample_Game_3_metadata.xml",
    "Sample_Game_3/Sample_Game_3_tracking.txt",
]


def main(out_dir=OUT_DIR):
    for name in FILES:
        path = os.path.join(out_dir, *name.split("/"))
        if os.path.exists(path):
            print("  have   ", name)
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("  fetch  ", name, flush=True)
        urllib.request.urlretrieve(BASE + name, path + ".part")
        os.replace(path + ".part", path)
    print("done:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
