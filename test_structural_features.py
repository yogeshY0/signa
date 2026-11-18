"""Test structural features on preprocessed BHSig+CEDAR data."""
from __future__ import annotations
from pathlib import Path
import csv
import cv2
import numpy as np
from structural_features import (
    extract_structural_features,
    structural_pair_features,
    structural_similarity,
    fit_scaler,
    apply_scaler,
)


def test_extract_85d():
    d = np.full((192, 384), 255, dtype=np.uint8)
    d[80:120, 150:250] = 50
    f = extract_structural_features(d)
    assert f["feature_dim"] == 85, f"Expected 85, got {f['feature_dim']}"
    assert len(f["hu"]) == 7
    assert len(f["h_profile"]) == 32
    assert len(f["v_profile"]) == 32
    assert len(f["geometry"]) == 6
    assert len(f["contour"]) == 4
    assert len(f["skeleton"]) == 4
    assert 0 < f["mask_fraction"] < 1
    print("PASS test_extract_85d")


def test_self_similarity():
    d = np.full((192, 384), 255, dtype=np.uint8)
    d[80:120, 150:250] = 50
    f = extract_structural_features(d)
    assert abs(structural_similarity(f, f) - 1.0) < 1e-6
    print("PASS test_self_similarity")


def test_pair_features_zero():
    d = np.full((192, 384), 255, dtype=np.uint8)
    d[80:120, 150:250] = 50
    f = extract_structural_features(d)
    pf = structural_pair_features(f, f)
    assert pf.shape == (14,)
    assert np.all(pf < 1e-5)
    print("PASS test_pair_features_zero")


def test_scaler():
    feats = []
    for i in range(10):
        d = np.full((192, 384), 255, dtype=np.uint8)
        d[80 + i:120 + i, 150:250] = 50
        feats.append(extract_structural_features(d))
    s = fit_scaler(feats)
    assert set(s.keys()) == {"hu", "h_profile", "v_profile", "geometry", "contour", "skeleton"}
    for bn in s:
        m, std = s[bn]
        assert len(m) == len(feats[0][bn]) and np.all(std > 0)
    out = apply_scaler(feats[0], s)
    assert abs(np.mean(out["hu"])) < 1.0
    print("PASS test_scaler")


def test_empty_raises():
    d = np.full((192, 384), 255, dtype=np.uint8)
    try:
        extract_structural_features(d)
        assert False
    except ValueError:
        pass
    print("PASS test_empty_raises")


def test_real_preprocessed():
    mp = Path("data/preprocessed_bhsig_cedar/grayscale/manifest.csv")
    if not mp.exists():
        print("SKIP test_real_preprocessed")
        return
    with mp.open() as f:
        rows = list(csv.DictReader(f))[:10]
    root = mp.parent
    for row in rows:
        img = cv2.imread(str(root / row["image_path"]), cv2.IMREAD_GRAYSCALE)
        assert img is not None
        f = extract_structural_features(img)
        assert f["feature_dim"] == 85
    print(f"PASS test_real_preprocessed ({len(rows)} images)")


if __name__ == "__main__":
    test_extract_85d()
    test_self_similarity()
    test_pair_features_zero()
    test_scaler()
    test_empty_raises()
    test_real_preprocessed()
    print("\nAll structural-features tests passed.")