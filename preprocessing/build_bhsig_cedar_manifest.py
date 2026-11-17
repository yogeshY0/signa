"""Build 4-way signer-disjoint manifest for BHSig + CEDAR (no GPDS).

Discovers all images under ``data/BHSig260/``, parses filenames, assigns
signer IDs, and creates a 4-way stratified split: train 60%, fusion 15%,
val 10%, test 15% (seed 42).  Saves to ``data/manifests/bhsig_cedar.csv``.

Forgeries use the signer ID of the signer they imitate.

Usage::

    python -m preprocessing.build_bhsig_cedar_manifest

"""
from __future__ import annotations
import csv, random, re
from pathlib import Path

DATA = Path("data")
ROOT = DATA / "BHSig260"
MANIFEST_DIR = DATA / "manifests"
FIELDS = ["image_path", "signer_id", "is_forgery", "split"]
SEED = 42

BHSIG_RE = re.compile(r"^([BH])-S-(\d+)-([GF])-\d+\.tif$", re.IGNORECASE)
LANGUAGE = {"B": "bn", "H": "hi"}
CEDAR_ORG_RE = re.compile(r"^original_(\d+)_(\d+)\.png$")
CEDAR_FORG_RE = re.compile(r"^forgeries_(\d+)_(\d+)\.png$")


def _classify_bhsig(path: Path) -> dict | None:
    m = BHSIG_RE.match(path.name)
    if not m:
        return None
    lang, num, kind = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
    ds = "bhsig_bn" if lang == "B" else "bhsig_hi"
    sid = f"{ds}_{num:03d}"
    isf = kind == "F"
    if path.parent.name.lower() != ("forged" if isf else "genuine"):
        raise ValueError(f"Folder mismatch: {path}")
    return {"sid": sid, "isf": isf, "img": str(path.relative_to(DATA).as_posix())}


def _classify_cedar(path: Path) -> dict | None:
    m = CEDAR_ORG_RE.match(path.name) or CEDAR_FORG_RE.match(path.name)
    if not m:
        return None
    sid = f"cedar_{int(m.group(1)):03d}"
    isf = CEDAR_FORG_RE.match(path.name) is not None
    return {"sid": sid, "isf": isf, "img": str(path.relative_to(DATA).as_posix())}


def _dataset_of(signer_id: str) -> str:
    return signer_id.rsplit("_", 1)[0]


def _stratified_split(signers: list[str], seed: int) -> dict[str, str]:
    """Assign each signer to train/fusion/val/test (60/15/10/15)."""
    shuffled = sorted(signers)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = round(n * 0.15)
    n_val = round(n * 0.10)
    n_fusion = round(n * 0.15)
    labels = ["test"] * n_test + ["val"] * n_val + ["fusion"] * n_fusion
    labels += ["train"] * (n - len(labels))
    return dict(zip(shuffled, labels))


def main() -> None:
    if not ROOT.is_dir():
        raise FileNotFoundError(f"Root not found: {ROOT.resolve()}")

    # Collect all rows, grouped by dataset for stratified splitting.
    by_dataset: dict[str, dict[str, list[dict]]] = {}
    # Mapping from signer -> dataset
    dataset_of: dict[str, str] = {}

    for p in sorted(ROOT.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in (".tif", ".tiff", ".png"):
            continue
        meta = _classify_bhsig(p) or _classify_cedar(p)
        if meta is None:
            continue
        ds = _dataset_of(meta["sid"])
        by_dataset.setdefault(ds, {})
        by_dataset[ds].setdefault(meta["sid"], [])
        by_dataset[ds][meta["sid"]].append(meta)
        dataset_of[meta["sid"]] = ds

    # Assign splits per dataset (stratified), then merge.
    split_of: dict[str, str] = {}
    for ds, signers_dict in by_dataset.items():
        signers = sorted(signers_dict)
        splits = _stratified_split(signers, SEED + hash(ds) % 10000)
        split_of.update(splits)

    # Build rows.
    rows: list[dict[str, str]] = []
    for ds in sorted(by_dataset):
        for sid in sorted(by_dataset[ds]):
            for rec in by_dataset[ds][sid]:
                rows.append({
                    "image_path": rec["img"],
                    "signer_id": rec["sid"],
                    "is_forgery": str(rec["isf"]).lower(),
                    "split": split_of[rec["sid"]],
                })

    # Verify disjointness.
    splits_actual: dict[str, set] = {}
    for r in rows:
        splits_actual.setdefault(r["split"], set()).add(r["signer_id"])
    for s1 in splits_actual:
        for s2 in splits_actual:
            if s1 < s2:
                overlap = splits_actual[s1] & splits_actual[s2]
                if overlap:
                    raise ValueError(f"Signer overlap between {s1} and {s2}: {sorted(overlap)[:5]}")

    # Write manifest.
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = MANIFEST_DIR / "bhsig_cedar.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {manifest_path}")

    # Summary.
    counts: dict[str, dict[str, int]] = {}
    for r in rows:
        ds = _dataset_of(r["signer_id"])
        sp = r["split"]
        counts.setdefault(ds, {})
        counts[ds].setdefault(sp, {"signers": set(), "rows": 0})
        counts[ds][sp]["signers"].add(r["signer_id"])
        counts[ds][sp]["rows"] += 1

    print(f"\nSigners: {len(split_of)} | Rows: {len(rows)}")
    print(f"{'Dataset':<12} {'Split':<8} {'Signers':<8} {'Rows':<8}")
    print("-" * 36)
    for ds in sorted(counts):
        for sp in ("train", "fusion", "val", "test"):
            if sp in counts[ds]:
                c = counts[ds][sp]
                print(f"{ds:<12} {sp:<8} {len(c['signers']):<8} {c['rows']:<8}")


if __name__ == "__main__":
    main()