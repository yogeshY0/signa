"""Test structural features on a small sample."""

import sys
sys.path.insert(0, '.')

import csv
from pathlib import Path
import cv2
import numpy as np
from structural_features import extract_structural_features, structural_similarity
from preprocessing.signature_preprocessing import preprocess_signature


def test_on_sample():
    """Test structural features on a few sample images."""
    # Use test images from the repository
    test_dir = Path("test_signs")
    if not test_dir.exists():
        print("No test_signs directory found")
        return
    
    test_images = list(test_dir.glob("*.png"))
    if len(test_images) < 2:
        print("Need at least 2 test images")
        return
    
    print(f"Found {len(test_images)} test images")
    
    # Extract features for all images
    features = {}
    for img_path in test_images[:5]:  # Just first 5
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
                
            # Preprocess
            preprocessed = preprocess_signature(img)
            
            # Extract features
            feats = extract_structural_features(preprocessed)
            features[img_path.name] = feats
            
            print(f"{img_path.name}: feature dim={feats['feature_dim']}")
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
    
    # Compute similarities
    print("\nSimilarity matrix:")
    names = list(features.keys())
    for i, name1 in enumerate(names):
        row = []
        for j, name2 in enumerate(names):
            sim = structural_similarity(features[name1], features[name2])
            row.append(f"{sim:.3f}")
        print(f"{name1:40} {' '.join(row)}")


def test_with_manifest_sample():
    """Test with a few rows from the manifest."""
    manifest_path = Path("data/manifest_kaggle_clean.csv")
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        return
    
    # Read first few rows
    records = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 10:  # Just first 10
                break
            records.append({
                "path": Path(row["image_path"].strip()),
                "signer": row["signer_id"].strip(),
                "is_forgery": row["is_forgery"].strip().lower() in {"1", "true", "yes"},
                "split": row["split"].strip().lower()
            })
    
    print(f"Read {len(records)} records from manifest")
    
    # Group by signer
    by_signer = {}
    for record in records:
        by_signer.setdefault(record["signer"], []).append(record)
    
    # For each signer with both genuine and forged
    for signer, samples in by_signer.items():
        genuine = [r for r in samples if not r["is_forgery"]]
        forged = [r for r in samples if r["is_forgery"]]
        
        if len(genuine) >= 1 and len(forged) >= 1:
            print(f"\nSigner {signer}: {len(genuine)} genuine, {len(forged)} forged")
            
            # Try to load and process one of each
            try:
                # Use first genuine as reference
                ref_path = Path("data") / genuine[0]["path"]
                if ref_path.exists():
                    ref_img = cv2.imread(str(ref_path), cv2.IMREAD_UNCHANGED)
                    if ref_img is not None:
                        ref_preprocessed = preprocess_signature(ref_img)
                        ref_feats = extract_structural_features(ref_preprocessed)
                        
                        # Compare with another genuine
                        if len(genuine) > 1:
                            gen_path = Path("data") / genuine[1]["path"]
                            if gen_path.exists():
                                gen_img = cv2.imread(str(gen_path), cv2.IMREAD_UNCHANGED)
                                if gen_img is not None:
                                    gen_preprocessed = preprocess_signature(gen_img)
                                    gen_feats = extract_structural_features(gen_preprocessed)
                                    sim = structural_similarity(ref_feats, gen_feats)
                                    print(f"  Genuine-Genuine similarity: {sim:.4f}")
                        
                        # Compare with a forgery
                        if forged:
                            forg_path = Path("data") / forged[0]["path"]
                            if forg_path.exists():
                                forg_img = cv2.imread(str(forg_path), cv2.IMREAD_UNCHANGED)
                                if forg_img is not None:
                                    forg_preprocessed = preprocess_signature(forg_img)
                                    forg_feats = extract_structural_features(forg_preprocessed)
                                    sim = structural_similarity(ref_feats, forg_feats)
                                    print(f"  Genuine-Forgery similarity: {sim:.4f}")
            except Exception as e:
                print(f"  Error: {e}")
            break  # Just do first signer with both


if __name__ == "__main__":
    print("Testing structural features...")
    print("-" * 50)
    
    test_on_sample()
    print("\n" + "=" * 50)
    test_with_manifest_sample()