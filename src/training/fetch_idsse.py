#!/usr/bin/env python3
"""Download the seven CC BY 4.0 IDSSE tracking/event matches from Hugging Face."""

import argparse
import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))

from possession import idsse  # noqa: E402

BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main"
COMPETITION = {
    "J03WMX": "000001", "J03WN1": "000001",
    "J03WOH": "000002", "J03WOY": "000002", "J03WPY": "000002",
    "J03WQQ": "000002", "J03WR9": "000002",
}
REMOTE = {
    "meta": "DFL_02_01_matchinformation_DFL-COM-{competition}_DFL-MAT-{match}.xml",
    "events": "DFL_03_02_events_raw_DFL-COM-{competition}_DFL-MAT-{match}.xml",
    "tracking": "DFL_04_03_positions_raw_observed_DFL-COM-{competition}_DFL-MAT-{match}.xml",
}


def download(url, destination):
    if os.path.isfile(destination) and os.path.getsize(destination):
        print("exists", destination, flush=True)
        return
    temporary = destination + ".part"
    print("download", url, flush=True)
    urllib.request.urlretrieve(url, temporary)
    os.replace(temporary, destination)
    print("saved", destination, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matches", nargs="*", choices=idsse.MATCHES,
                        default=list(idsse.MATCHES))
    parser.add_argument("--root", default=idsse.IDSSE_DIR)
    args = parser.parse_args(argv)
    os.makedirs(args.root, exist_ok=True)
    for match in args.matches:
        competition = COMPETITION[match]
        for suffix, template in REMOTE.items():
            name = template.format(competition=competition, match=match)
            download("{}/{}".format(BASE, name),
                     os.path.join(args.root, "{}_{}.xml".format(match, suffix)))


if __name__ == "__main__":
    main()
