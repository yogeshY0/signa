"""Structural feature extraction for signature verification."""

import cv2
import numpy as np


def extract_structural_features(image: np.ndarray) -> dict:
    """Extract structural features from a signature image."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Simple binarization
    if image.mean() < 128:
        image = cv2.bitwise_not(image)
    
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Hu moments
    contours, _ = cv2.findContours(255 - binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hu_moments = np.zeros(7, dtype=np.float32)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        moments = cv2.moments(largest)
        if moments["m00"] != 0:
            hu = cv2.HuMoments(moments)
            hu_moments = np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
            hu_moments = hu_moments.flatten()
    
    # Projection profiles (simplified)
    h, w = binary.shape
    horizontal = np.sum(binary == 0, axis=1) / w
    vertical = np.sum(binary == 0, axis=0) / h
    
    # Simple features
    features = np.concatenate([
        hu_moments,
        horizontal[:16],  # first 16 values
        vertical[:16]    # first 16 values
    ])
    
    return {
        "all_features": features,
        "feature_dim": len(features)
    }


def structural_similarity(features1: dict, features2: dict) -> float:
    """Compute similarity between two feature sets."""
    f1 = features1["all_features"]
    f2 = features2["all_features"]
    
    # Simple cosine similarity
    dot = np.dot(f1, f2)
    norm1 = np.linalg.norm(f1)
    norm2 = np.linalg.norm(f2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    sim = dot / (norm1 * norm2)
    return float(max(0, sim))


if __name__ == "__main__":
    dummy = np.full((192, 384), 255, dtype=np.uint8)
    dummy[80:120, 150:250] = 50
    
    feats = extract_structural_features(dummy)
    print(f"Feature dimension: {feats['feature_dim']}")
    
    sim = structural_similarity(feats, feats)
    print(f"Self-similarity: {sim:.4f}")