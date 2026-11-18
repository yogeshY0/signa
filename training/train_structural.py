"""Train a structural-only classifier (logistic regression on pairwise features).

Pairwise features via ``structural_features.structural_pair_features``.
First genuine = reference. Fit on **train** writers only.
Thresholds (EER, target FAR) on **val** writers.

Usage::

    python -m training.train_structural --manifest data/manifests/bhsig_cedar.csv \\
        --image-root data --output-dir artifacts/structural

"""
from __future__ import annotations
import argparse, csv, json, pickle
from pathlib import Path
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from structural_features import extract_structural_features, structural_pair_features


def _read_records(manifest: Path, image_root: Path, split: str) -> list[dict]:
    root = image_root.resolve()
    records = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split"].strip().lower() != split.lower(): continue
            p = Path(row["image_path"])
            records.append({"image_path": (p if p.is_absolute() else root / p).resolve(),
                            "signer_id": row["signer_id"].strip(),
                            "is_forgery": row["is_forgery"].strip().lower() in {"1","true","yes"}})
    return records


def _build_pairs(records: list[dict]) -> list[dict]:
    by_signer: dict = {}
    for r in records:
        by_signer.setdefault(r["signer_id"], {"gen": [], "forg": []})
        by_signer[r["signer_id"]]["gen" if not r["is_forgery"] else "forg"].append(r["image_path"])
    pairs = []
    for sid, grp in by_signer.items():
        g = sorted(grp["gen"])
        if len(g) < 2: continue
        ref = g[0]
        for q in g[1:]: pairs.append({"signer_id": sid, "ref": ref, "q": q, "label": 1})
        for q in sorted(grp["forg"]): pairs.append({"signer_id": sid, "ref": ref, "q": q, "label": 0})
    return pairs
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/structural"))
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()
    out_dir = args.output_dir.resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    train_recs = _read_records(args.manifest, args.image_root, "train")
    val_recs = _read_records(args.manifest, args.image_root, "val")
    train_pairs = _build_pairs(train_recs)
    val_pairs = _build_pairs(val_recs)
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")

    all_paths = sorted(set(str(p["ref"]) for p in train_pairs + val_pairs) |
                       set(str(p["q"]) for p in train_pairs + val_pairs))
    print(f"Unique images: {len(all_paths)}")
    cache = {}
    for i, p in enumerate(all_paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError(f"Unreadable: {p}")
        try: cache[p] = extract_structural_features(img)
        except ValueError as e: raise ValueError(f"Extraction failed: {p}: {e}")
        if (i+1) % 500 == 0: print(f"  {i+1}/{len(all_paths)}")

    def _feats(pairs):
        X, y = [], []
        for p in pairs:
            f1, f2 = cache[str(p["ref"])], cache[str(p["q"])]
            X.append(structural_pair_features(f1, f2))
            y.append(p["label"])
        return np.array(X, dtype=np.float64), np.array(y, dtype=np.int32)

    X_tr, y_tr = _feats(train_pairs)
    X_val, y_val = _feats(val_pairs)
    scaler = StandardScaler(); X_tr_s = scaler.fit_transform(X_tr); X_val_s = scaler.transform(X_val)

    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_tr_s, y_tr)

    val_s = clf.predict_proba(X_val_s)[:, 1]
    vg, vf = val_s[y_val == 1], val_s[y_val == 0]
    cand = np.unique(np.concatenate([vg, vf]))
    gs, fs = np.sort(vg), np.sort(vf)
    frr = np.searchsorted(gs, cand, side="left") / max(len(vg), 1)
    far = 1 - np.searchsorted(fs, cand, side="left") / max(len(vf), 1)
    best = int(np.argmin(np.abs(far - frr)))
    eer_th, eer = float(cand[best]), float((far[best] + frr[best]) / 2)
    print(f"Val EER: {eer:.4f} (thresh {eer_th:.4f})")

    thresh = {"eer": eer_th}
    for tgt in [0.01, 0.05]:
        valid = np.where(far <= tgt)[0]
        t = float(cand[valid[0]]) if len(valid) else float(cand[-1])
        thresh[f"far_{tgt:.4f}"] = t
        print(f"  At FAR={tgt}: thresh={t:.4f}, FRR={(gs < t).mean():.4f}")

    for name, obj in [("scaler.pkl", scaler), ("model.pkl", clf)]:
        p = out_dir / name
        with p.open("wb") as f: pickle.dump(obj, f)
        print(f"Wrote {p}")
    (out_dir / "thresholds.json").write_text(json.dumps(thresh, indent=2))
    print(f"Wrote thresholds.json")
    summary = {"train_pairs": len(train_pairs), "val_pairs": len(val_pairs),
               "val_eer": eer, "val_eer_threshold": eer_th, "thresholds": thresh,
               "coef_norm": float(np.linalg.norm(clf.coef_)),
               "intercept": float(clf.intercept_[0])}
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("Done.")

if __name__ == "__main__":
    main()