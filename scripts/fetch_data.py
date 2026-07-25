#!/usr/bin/env python3
"""Download the evaluation datasets.

Kept out of git: SciFact is 8MB and SQuAD is 4MB, and vendoring benchmark data
into a repo is how people end up quietly evaluating on a stale copy.
"""

import os
import sys
import urllib.request
import zipfile

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

SOURCES = {
    "scifact": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
    "squad-dev-v2.json": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}


def fetch(name: str, url: str) -> None:
    os.makedirs(RAW, exist_ok=True)
    target = os.path.join(RAW, name)
    if os.path.exists(target):
        print(f"  {name} already present")
        return
    print(f"  downloading {name} ...", flush=True)
    if url.endswith(".zip"):
        tmp = target + ".zip"
        urllib.request.urlretrieve(url, tmp)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(RAW)
        os.remove(tmp)
    else:
        urllib.request.urlretrieve(url, target)
    print(f"  got {name}")


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(SOURCES)
    for name in wanted:
        if name not in SOURCES:
            sys.exit(f"unknown dataset {name}, options: {', '.join(SOURCES)}")
        fetch(name, SOURCES[name])
    print("done")
