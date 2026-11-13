"""Prune manifest rows whose image files are missing from disk.

Usage:
    python -m preprocessing.prune_manifest --manifest data/manifest_kaggle.csv --image-root data --output data/manifest_kaggle_clean.csv

Without --output, prints the counts and the missing paths only.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
FIELDS = ["image_path", "signer_id", "is_forgery", "split"]

def main():
    parser = argparse.ArgumentParser(description="Drop manifest rows with missing files.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = (args.image_root or args.manifest.parent).resolve()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or list(FIELDS)
        rows = list(reader)
    kept = []
    dropped = []
    for row in rows:
        path = Path(row["image_path"])
        full = path if path.is_absolute() else root / path
        (kept if full.exists() else dropped).append(row)
    print("%d kept, %d dropped (missing files)" % (len(kept), len(dropped)))
    for row in dropped[:10]:
        print("  missing:", row["image_path"])
    if len(dropped) > 10:
        print("  ... and %d more" % (len(dropped) - 10))
    if args.output and kept:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        print("wrote", args.output)
    elif args.output and not kept:
        print("nothing to write: all rows point to missing files")

if __name__ == "__main__":
    main()
