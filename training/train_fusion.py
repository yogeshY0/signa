"""Train fusion: logistic regression on [neural_cos, structural_proba] (2-D).

Fit on **fusion** writers (neural model frozen; unseen during neural training).
Threshold on **val** writers.

Usage::

    python -m training.train_fusion --manifest data/manifests/bhsig_cedar.csv \\
        --image-root data --neural-ckpt runs/A_grayscale_s42/best_model.pt \\
        --structural-artifacts artifacts/structural --output-dir artifacts/fusion

"""
from __future__ import annotations
import argparse, csv, json, pickle
from pathlib import Path
import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from structural_features import extract_structural_features, structural_pair_features
from training.stage3_metric_learning import SiameseSignatureModel, SignatureDataset, SignatureRecord
from torch.utils.data import DataLoader


def _read_records(manifest, image_root, split):
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


def _build_pairs(records):
    by_signer = {}
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


def _neural_sims(model, device, ref_paths, query_paths, batch_size=64):
    all_p = list(set(str(p) for p in ref_paths) | set(str(p) for p in query_paths))
    idx = {p: i for i, p in enumerate(all_p)}
    recs = [SignatureRecord(image_path=Path(p), signer_id=f"d_{i}", is_forgery=False)
            for i, p in enumerate(all_p)]
    ds = SignatureDataset(recs, {r.signer_id: 0 for r in recs}, already_preprocessed=True)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    embs = []
    with torch.no_grad():
        for images, _, _ in loader: embs.append(model(images.to(device)).cpu().numpy())
    emb = np.concatenate(embs)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-8)
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--neural-ckpt", type=Path, required=True)
    parser.add_argument("--structural-artifacts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/fusion"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--C", type=float, default=1.0)
    args = parser.parse_args()
    out_dir = args.output_dir.resolve(); out_dir.mkdir(parents=True, exist_ok=True)

    with (args.structural_artifacts / "scaler.pkl").open("rb") as f: struct_scaler = pickle.load(f)
    with (args.structural_artifacts / "model.pkl").open("rb") as f: struct_model = pickle.load(f)
    print("Loaded structural artifacts")
    ckpt = torch.load(args.neural_ckpt, map_location="cpu")
    model = SiameseSignatureModel(pretrained=False, embedding_dim=ckpt["embedding_dim"])
    model.load_state_dict(ckpt["model_state_dict"]); model.to(args.device)
    print("Loaded neural checkpoint")

    fusion_recs = _read_records(args.manifest, args.image_root, "fusion")
    val_recs = _read_records(args.manifest, args.image_root, "val")
    fusion_pairs = _build_pairs(fusion_recs)
    val_pairs = _build_pairs(val_recs)
    print(f"Fusion pairs: {len(fusion_pairs)}, Val pairs: {len(val_pairs)}")

    def _compute(pairs):
        refs = [p["ref"] for p in pairs]; qs = [p["q"] for p in pairs]
        y = np.array([p["label"] for p in pairs], dtype=np.int32)
        neural = _neural_sims(model, args.device, refs, qs, args.batch_size)
        all_p = sorted(set(str(r) for r in refs) | set(str(q) for q in qs))
        cache = {}
        for i, p in enumerate(all_p):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None: raise ValueError(f"Unreadable: {p}")
            cache[p] = extract_structural_features(img)
            if (i+1) % 500 == 0: print(f"  struct {i+1}/{len(all_p)}")
        struct = []
        for p in pairs:
            pf = structural_pair_features(cache[str(p["ref"])], cache[str(p["q"])])
            struct.append(struct_model.predict_proba(struct_scaler.transform(pf.reshape(1,-1)))[0,1])
        return np.column_stack([neural, np.array(struct)]), y

    X_f, y_f = _compute(fusion_pairs)
    X_v, y_v = _compute(val_pairs)
    scaler = StandardScaler(); X_fs = scaler.fit_transform(X_f); X_vs = scaler.transform(X_v)
    clf = LogisticRegression(C=args.C, class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_fs, y_f)
    vs = clf.predict_proba(X_vs)[:, 1]
    vg, vf = vs[y_v == 1], vs[y_v == 0]
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
    for name, obj in [("fusion_scaler.pkl", scaler), ("fusion_model.pkl", clf)]:
        p = out_dir / name
        with p.open("wb") as f: pickle.dump(obj, f)
        print(f"Wrote {p}")
    (out_dir / "thresholds.json").write_text(json.dumps(thresh, indent=2))
    summary = {"fusion_pairs": len(fusion_pairs), "val_pairs": len(val_pairs),
               "val_eer": eer, "val_eer_threshold": eer_th, "thresholds": thresh,
               "coef": clf.coef_.tolist(), "intercept": float(clf.intercept_[0])}
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("Done.")

if __name__ == "__main__":
    main()
    return np.array([float(np.dot(emb[idx[str(r)]], emb[idx[str(q)]]))
                     for r, q in zip(ref_paths, query_paths)], dtype=np.float64)