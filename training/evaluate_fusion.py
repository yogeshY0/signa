"""Evaluate fusion model on test signers.
Usage:
    python -m training.evaluate_fusion --neural-ckpt <path> \\
        --structural-dir <path> --fusion-dir <path> \\
        --manifest <path> --image-root <path> [--device cuda]
"""
from __future__ import annotations
import argparse, csv, json, pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from structural_features import extract_structural_features, structural_pair_features
from training.stage3_metric_learning import (
    SignatureDataset, SignatureRecord, SiameseSignatureModel,
)


def _read_records(manifest, root, splits):
    records = []
    with manifest.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split"].strip().lower() not in splits:
                continue
            p = Path(row["image_path"].strip())
            records.append(SignatureRecord(
                image_path=(p if p.is_absolute() else root / p).resolve(),
                signer_id=row["signer_id"].strip(),
                is_forgery=row["is_forgery"].strip().lower() in {"1", "true", "yes"},
            ))
    return records


@torch.no_grad()
def _embed(model, records, device):
    ds = SignatureDataset(records, {r.signer_id: 0 for r in records},
                          already_preprocessed=False)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    model.eval()
    return np.concatenate([model(imgs.to(device)).cpu().numpy()
                           for imgs, _, _ in loader])


def _pool_scores(records, embeddings, struct_cache, struct_scaler,
                 struct_model, fuse_scaler, fuse_model):
    """Return (genuine_scores, skilled_forgery_scores) arrays."""
    by = defaultdict(lambda: {"gen": [], "forg": []})
    for i, r in enumerate(records):
        by[r.signer_id]["forg" if r.is_forgery else "gen"].append(i)

    gen_scores, skilled_scores = [], []
    for sid, grp in by.items():
        gs = sorted(grp["gen"])
        if len(gs) < 2:
            continue
        ref_i = gs[0]
        ref_e = embeddings[ref_i]
        ref_p = str(records[ref_i].image_path)
        ref_sf = struct_cache[ref_p]

        def _score(q_i):
            qe = embeddings[q_i]
            qp = str(records[q_i].image_path)
            cos = float(np.dot(ref_e, qe) / (
                np.linalg.norm(ref_e) * np.linalg.norm(qe)))
            pf = structural_pair_features(ref_sf, struct_cache[qp])
            sp = struct_model.predict_proba(
                struct_scaler.transform(pf.reshape(1, -1)))[0, 1]
            fv = fuse_scaler.transform([[cos, sp]])[0]
            return float(fuse_model.predict_proba(fv.reshape(1, -1))[0, 1])

        for qi in gs[1:]:
            gen_scores.append(_score(qi))
        for qi in sorted(grp["forg"]):
            skilled_scores.append(_score(qi))

    return np.array(gen_scores), np.array(skilled_scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--neural-ckpt", type=Path, required=True)
    parser.add_argument("--structural-dir", type=Path, required=True)
    parser.add_argument("--fusion-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load artifacts
    print("Loading artifacts ...")
    with (args.structural_dir / "scaler.pkl").open("rb") as f:
        struct_scaler = pickle.load(f)
    with (args.structural_dir / "model.pkl").open("rb") as f:
        struct_model = pickle.load(f)
    with (args.fusion_dir / "fusion_scaler.pkl").open("rb") as f:
        fuse_scaler = pickle.load(f)
    with (args.fusion_dir / "fusion_model.pkl").open("rb") as f:
        fuse_model = pickle.load(f)
    with (args.fusion_dir / "thresholds.json").open() as f:
        thresholds = json.load(f)
    print(f"  Thresholds: {thresholds}")

    ckpt = torch.load(args.neural_ckpt, map_location="cpu")
    model = SiameseSignatureModel(pretrained=False,
                                   embedding_dim=ckpt["embedding_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)
    print("  Neural model loaded.")

    # Read test records
    test_recs = _read_records(args.manifest, args.image_root, {"test"})
    print(f"Test signer records: {len(test_recs)}")

    # Embeddings
    embs = _embed(model, test_recs, args.device)
    print(f"Embeddings: {embs.shape}")

    # Structural feature cache
    all_paths = sorted({str(r.image_path) for r in test_recs})
    struct_cache = {}
    for i, p in enumerate(all_paths):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            struct_cache[p] = extract_structural_features(img)
        if (i + 1) % 500 == 0:
            print(f"  structural {i+1}/{len(all_paths)}")
    print("  Structural cache ready.")

    # Score pools
    gen, skilled = _pool_scores(test_recs, embs, struct_cache,
                                struct_scaler, struct_model,
                                fuse_scaler, fuse_model)

    # Report for each threshold
    print()
    for tname, thr in thresholds.items():
        far = float((skilled >= thr).mean())
        frr = float((gen < thr).mean())
        print(f"  {tname}: threshold={thr:.4f}  FRR={frr:.4f}  FAR={far:.4f}  "
              f"N_gen={len(gen)}  N_skilled={len(skilled)}")


if __name__ == "__main__":
    main()