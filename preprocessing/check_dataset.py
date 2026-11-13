"""Audit a manifest for preprocessing/dataset consistency before training.

Checks each image referenced by a manifest for:
  * readability (missing / unreadable files),
  * correct model-input size (default 192 x 384),
  * image mode: binary (0/255) vs grayscale - a trained model should see one
    consistent mode, and binary data discards the stroke-pressure information
    the preprocessing pipeline is designed to preserve,
  * ink density (share of dark pixels) per dataset, to surface domain imbalance
    and near-empty images.

Results are grouped by dataset (the first path component of the image path).

Usage (from the repository root):
    python -m preprocessing.check_dataset --manifest data/manifest.csv
    python -m preprocessing.check_dataset --manifest data/processed/manifest.csv --sample 200

Exit code is 0 unless a manifest row is missing/unreadable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from preprocessing.signature_preprocessing import DEFAULT_TARGET_SIZE

# The model input is fixed-size; images at any other size mean the manifest does
# not point at already-preprocessed data (train with --already-preprocessed only).
EXPECTED = (DEFAULT_TARGET_SIZE[1], DEFAULT_TARGET_SIZE[0])  # (height, width)


def _read_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _dataset(row: dict[str, str]) -> str:
    """Group rows by dataset: path first component, except BHSig260 which is
    split by language (e.g. ``BHSig260/BHSig100_Bengali``)."""
    parts = Path(row["image_path"]).parts
    if len(parts) >= 2 and parts[0] == "BHSig260":
        return f"{parts[0]}/{parts[1]}"
    return parts[0] or "(root)"


def _audit_one(path: Path) -> dict:
    """Return a dict with shape/mode/density, or an error key."""
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return {"error": "unreadable"}
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    density = float(np.mean(gray < 128))
    return {
        "shape": gray.shape,
        "unique": int(len(np.unique(gray))),
        "mode": "binary" if int(len(np.unique(gray))) <= 2 else "grayscale",
        "density": density,
    }


def audit_manifest(manifest: Path, image_root: Path | None = None) -> dict:
    """Run the checks over a manifest and return a nested summary."""
    rows = _read_rows(manifest)
    per_dataset: dict[str, dict] = {}
    totals = {
        "rows": len(rows),
        "missing": 0,
        "unreadable": 0,
        "bad_size": 0,
        "binary": 0,
        "grayscale": 0,
    }
    densities: list[float] = []

    for row in rows:
        root = image_root or manifest.parent
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = root / path
        dataset = _dataset(row)
        bucket = per_dataset.setdefault(
            dataset,
            {
                "count": 0,
                "missing": 0,
                "unreadable": 0,
                "bad_size": 0,
                "binary": 0,
                "grayscale": 0,
                "densities": [],
            },
        )
        bucket["count"] += 1
        if not path.exists():
            bucket["missing"] += 1
            totals["missing"] += 1
            continue
        info = _audit_one(path)
        if "error" in info:
            bucket["unreadable"] += 1
            totals["unreadable"] += 1
            continue
        if tuple(info["shape"]) != EXPECTED:
            bucket["bad_size"] += 1
            totals["bad_size"] += 1
        if info["mode"] == "binary":
            bucket["binary"] += 1
            totals["binary"] += 1
        else:
            bucket["grayscale"] += 1
            totals["grayscale"] += 1
        bucket["densities"].append(info["density"])
        densities.append(info["density"])

    summaries = {}
    for name, b in per_dataset.items():
        dens = b["densities"]
        summaries[name] = {
            "rows": b["count"],
            "missing": b["missing"],
            "unreadable": b["unreadable"],
            "bad_size": b["bad_size"],
            "binary": b["binary"],
            "grayscale": b["grayscale"],
            "ink_density_mean_pct": round(100 * float(np.mean(dens)), 2) if dens else None,
            "ink_density_min_pct": round(100 * float(np.min(dens)), 2) if dens else None,
            "ink_density_max_pct": round(100 * float(np.max(dens)), 2) if dens else None,
        }
    totals["ink_density_mean_pct"] = round(100 * float(np.mean(densities)), 2) if densities else None
    return {"expected_size": EXPECTED, "totals": totals, "datasets": summaries}


def _fmt_pct(x) -> str:
    return f"{x:6.2f}" if x is not None else "   -- "


def _fmt_ink(s: dict) -> str:
    parts = [_fmt_pct(s.get(k)).strip() for k in ("ink_density_min_pct", "ink_density_mean_pct", "ink_density_max_pct")]
    return "/".join(parts)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.csv"))
    parser.add_argument("--image-root", type=Path, help="Base dir for relative image paths (default: manifest dir).")
    args = parser.parse_args()

    result = audit_manifest(args.manifest, args.image_root)
    totals = result["totals"]
    print(f"Auditing {args.manifest}  (expected {EXPECTED[1]}w x {EXPECTED[0]}h)\n")
    header = (
        f"{'dataset':<14} {'rows':>5} {'miss':>4} {'unread':>6} {'badsize':>7} "
        f"{'binary':>6} {'gray':>5} {'ink% min/mean/max':>20}"
    )
    print(header)
    print("-" * len(header))
    for name, s in result["datasets"].items():
        print(
            f"{name:<14} {s['rows']:>5} {s['missing']:>4} {s['unreadable']:>6} "
            f"{s['bad_size']:>7} {s['binary']:>6} {s['grayscale']:>5} {_fmt_ink(s):>20}"
        )
    print("-" * len(header))
    t = totals
    print(
        f"{'TOTAL':<14} {t['rows']:>5} {t['missing']:>4} {t['unreadable']:>6} "
        f"{t['bad_size']:>7} {t['binary']:>6} {t['grayscale']:>5}"
    )

    problems = []
    if t["missing"] or t["unreadable"]:
        problems.append("missing/unreadable files (fix paths or typos)")
    if t["binary"] and t["grayscale"]:
        problems.append("MIXED binary and grayscale images: a model should see one consistent mode")
    if t["binary"] and t["binary"] == t["rows"] and t["rows"]:
        problems.append("ALL images are binary (0/255): stroke-pressure information is discarded")
    for name, s in result["datasets"].items():
        mean = s.get("ink_density_mean_pct")
        if mean is not None and mean < 1.5:
            problems.append(f"{name}: near-empty signatures (mean ink {mean}%)")
    print("\n" + "Issues found:" if problems else "\nNo obvious consistency issues found.")
    for issue in problems:
        print("  -", issue)
    if t["missing"] or t["unreadable"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()