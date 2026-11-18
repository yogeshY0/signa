"""Structural feature extraction for signature verification.

- ``extract_structural_features(image)``: per-image feature dict.
- ``structural_pair_features(f1, f2)``: ~14-D deltas for a classifier.
- ``structural_similarity(f1, f2)``: raw cosine (not clipped).
- ``fit_scaler(feature_list)`` / ``apply_scaler(features, scaler)``: normalise.

Empty-mask policy: Otsu; if ink < 0.5%, 5th-percentile; if still empty, raise.
"""
from __future__ import annotations
import cv2
import numpy as np

# ── mask extraction ────────────────────────────────────────────────────────

def _extract_mask(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3: gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else: gray = image
    if gray.mean() < 128: gray = cv2.bitwise_not(gray)
    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = bin_ < 128
    if mask.mean() < 0.005:
        mask = gray < np.percentile(gray, 5)
    if mask.mean() < 0.002:
        raise ValueError(f"too little ink ({mask.mean() * 100:.3f}%)")
    return mask

def _skeletonize(mask: np.ndarray) -> np.ndarray:
    binary = (mask * 255).astype(np.uint8)
    try:
        skel = cv2.ximgproc.thinning(binary)
    except AttributeError:
        skel = np.zeros_like(binary)
        temp = binary.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(temp, kernel)
            if cv2.countNonZero(eroded) == 0: break
            opening = cv2.morphologyEx(temp, cv2.MORPH_OPEN, kernel)
            skel = cv2.bitwise_or(skel, cv2.subtract(temp, opening))
            temp = eroded.copy()
    return skel > 0

# ── block extractors ───────────────────────────────────────────────────────

def _hu_moments(mask: np.ndarray) -> np.ndarray:
    moments = cv2.moments(mask.astype(np.uint8) * 255)
    if moments["m00"] == 0: return np.zeros(7, dtype=np.float64)
    hu = cv2.HuMoments(moments)
    return (np.sign(hu) * np.log10(np.abs(hu) + 1e-10)).flatten()

def _projection_profiles(mask: np.ndarray, bins: int = 32) -> tuple[np.ndarray, np.ndarray]:
    h, w = mask.shape
    hp = np.sum(mask, axis=1).astype(np.float64) / w
    vp = np.sum(mask, axis=0).astype(np.float64) / h
    hb = np.mean(hp.reshape(bins, -1), axis=1) if h >= bins else np.pad(hp, (0, bins - h))
    vb = np.mean(vp.reshape(bins, -1), axis=1) if w >= bins else np.pad(vp, (0, bins - w))
    return hb, vb

def _geometry(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    ink = float(mask.mean())
    ys, xs = np.where(mask)
    if len(ys) == 0: return np.zeros(6, dtype=np.float64)
    cy, cx = float(ys.mean()) / h, float(xs.mean()) / w
    y0, x0, y1, x1 = int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1
    bbox_area = (y1 - y0) * (x1 - x0)
    return np.array([ink, cy, cx, bbox_area / (h * w), (y1 - y0) / max(x1 - x0, 1),
                     len(ys) / max(bbox_area, 1)], dtype=np.float64)

def _contour(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return np.zeros(4, dtype=np.float64)
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
    x, y, bw, bh = cv2.boundingRect(cnt)
    perim = float(cv2.arcLength(cnt, True))
    comp = (perim ** 2) / max(area, 1)
    if len(cnt) >= 5:
        (_, _), (ma, ma2), _ = cv2.fitEllipse(cnt)
        ecc = np.sqrt(1 - (min(ma, ma2) / max(ma, ma2)) ** 2) if max(ma, ma2) > 0 else 0.0
    else: ecc = 0.0
    return np.array([area / max(hull_area, 1), area / max(bw * bh, 1), comp, ecc], dtype=np.float64)

def _skeleton(mask: np.ndarray) -> np.ndarray:
    skel = _skeletonize(mask)
    if skel.sum() == 0: return np.zeros(4, dtype=np.float64)
    kernel = np.ones((3, 3), dtype=np.uint8)
    nbrs = cv2.filter2D(skel.astype(np.uint8), -1, kernel)
    nbrs[skel == 0] = 0
    eps = int(((nbrs - 1) == 1).sum())
    juncs = int(((nbrs - 1) >= 3).sum())
    npix = int(skel.sum())
    return np.array([eps / max(npix, 1), juncs / max(npix, 1),
                     npix / max(eps + juncs, 1),
                     juncs / max(eps, 1) if eps > 0 else 0.0], dtype=np.float64)

BLOCK_NAMES = ["hu", "h_profile", "v_profile", "geometry", "contour", "skeleton"]
BLOCK_SLICES = [(0, 7), (7, 39), (39, 71), (71, 77), (77, 81), (81, 85)]

def extract_structural_features(image: np.ndarray) -> dict:
    mask = _extract_mask(image)
    hu = _hu_moments(mask)
    hp, vp = _projection_profiles(mask)
    geom = _geometry(mask)
    cont = _contour(mask)
    skel = _skeleton(mask)
    allf = np.concatenate([hu, hp, vp, geom, cont, skel])
    return {"hu": hu, "h_profile": hp, "v_profile": vp, "geometry": geom,
            "contour": cont, "skeleton": skel, "all_features": allf,
            "feature_dim": len(allf), "mask_fraction": float(mask.mean())}

def structural_pair_features(f1: dict, f2: dict) -> np.ndarray:
    return np.array([
        np.linalg.norm(f1["hu"] - f2["hu"]),
        1 - np.corrcoef(f1["h_profile"], f2["h_profile"])[0, 1],
        1 - np.corrcoef(f1["v_profile"], f2["v_profile"])[0, 1],
        abs(f1["geometry"][0] - f2["geometry"][0]),
        np.linalg.norm(f1["geometry"][1:3] - f2["geometry"][1:3]),
        abs(f1["geometry"][3] - f2["geometry"][3]),
        abs(f1["geometry"][4] - f2["geometry"][4]),
        abs(f1["contour"][0] - f2["contour"][0]),
        abs(f1["contour"][1] - f2["contour"][1]),
        abs(f1["contour"][3] - f2["contour"][3]),
        abs(f1["skeleton"][0] - f2["skeleton"][0]),
        abs(f1["skeleton"][1] - f2["skeleton"][1]),
        abs(f1["skeleton"][2] - f2["skeleton"][2]),
        abs(f1["skeleton"][3] - f2["skeleton"][3]),
    ], dtype=np.float64)

def structural_similarity(f1: dict, f2: dict) -> float:
    a, b = f1["all_features"], f2["all_features"]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

def fit_scaler(feature_list: list[dict]) -> dict:
    arrays = {bn: [] for bn in BLOCK_NAMES}
    for fd in feature_list:
        for bn in BLOCK_NAMES: arrays[bn].append(fd[bn])
    return {bn: (np.mean(np.stack(arrays[bn]), 0), np.std(np.stack(arrays[bn]), 0).clip(min=1e-8))
            for bn in BLOCK_NAMES}

def apply_scaler(features: dict, scaler: dict) -> dict:
    out = dict(features)
    for bn in BLOCK_NAMES:
        m, s = scaler[bn]
        out[bn] = (features[bn] - m) / s
    out["all_features"] = np.concatenate([out[bn] for bn in BLOCK_NAMES])
    return out

if __name__ == "__main__":
    d = np.full((192, 384), 255, dtype=np.uint8); d[80:120, 150:250] = 50
    f = extract_structural_features(d)
    print(f"dim {f['feature_dim']}, hu={len(f['hu'])}, hp={len(f['h_profile'])}, "
          f"vp={len(f['v_profile'])}, geo={len(f['geometry'])}, cont={len(f['contour'])}, "
          f"skel={len(f['skeleton'])}")
    print(f"self-sim {structural_similarity(f, f):.4f}, pair-deltas {structural_pair_features(f, f)}")
    print("OK")