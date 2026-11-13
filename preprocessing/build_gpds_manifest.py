"""Build a signer-disjoint GPDS-Synthetic manifest.

The GPDS folder splits by sample (01-16 train, 17-24 test) for the same 150
signers, leaking signers across splits. This pools samples and re-splits by
signer. Usage: python -m preprocessing.build_gpds_manifest
"""

from __future__ import annotations

import csv
import random
import re
from collections import Counter
from pathlib import Path

DATA = Path("data")
GPDS = DATA / "New folder (10)"
FIELDS = ["image_path", "signer_id", "is_forgery", "split"]
GENUINE_RE = re.compile(r"^c-(\d+)-\d+.*\.jpg$", re.IGNORECASE)
FORGERY_RE = re.compile(r"^cf-(\d+)-\d+.*\.jpg$", re.IGNORECASE)
def split_signers(signers, seed=42):
    shuffled = sorted(signers)
    random.Random(seed).shuffle(shuffled)
    n_val = n_test = round(len(shuffled) * 0.15)
    labels = ["test"] * n_test + ["val"] * n_val
    labels += ["train"] * (len(shuffled) - len(labels))
    return dict(zip(shuffled, labels))

def collect_rows():
    rows = []
    for subfolder, pattern, is_forgery in (
        ("genuine", GENUINE_RE, False),
        ("forge", FORGERY_RE, True),
    ):
        for path in sorted(GPDS.rglob("*" + subfolder + "/*.jpg")):
                match = pattern.match(path.name)
                if not match:
                    raise ValueError("unexpected name: %s" % path.name)
                rows.append((path, int(match.group(1)), is_forgery))
    splits = split_signers(sorted({s for _, s, _ in rows}))
    out = []
    for path, signer, is_forgery in rows:
        out.append({
            "image_path": path.relative_to(DATA).as_posix(),
            "signer_id": "gpds_%03d" % signer,
            "is_forgery": str(is_forgery).lower(),
            "split": splits[signer],
        })
    out.sort(key=lambda row: row["image_path"])
    return out

def main():
    rows = collect_rows()
    out_dir = DATA / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gpds.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %d rows (%d signers) to %s" % (len(rows), len({r["signer_id"] for r in rows}), out_path))
    print("splits:", dict(Counter(r["split"] for r in rows)))
    print("genuine per split:", dict(Counter(r["split"] for r in rows if r["is_forgery"] == "false")))

if __name__ == "__main__":
    main()
