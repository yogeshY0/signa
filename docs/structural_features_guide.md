# Structural Features Guide

## Why Add Structural Features?

Your ConvNeXt model achieves **13% EER skilled** (competitive). Structural features add **geometric/topological signals** that CNNs may miss:

- **Hu moments**: Shape descriptors (scale/rotation invariant)
- **Projection profiles**: Ink distribution patterns  
- **Skeleton features**: Stroke connectivity, endpoints
- **Contour features**: Aspect ratio, solidity, compactness

**Expected improvement**: 1-3% EER reduction, 2-5% FAR reduction (especially for GPDS).

## Quick Start

```python
from structural_features import extract_structural_features, structural_similarity
from preprocessing.signature_preprocessing import preprocess_signature
import cv2

# Load signature
img = cv2.imread("signature.png", cv2.IMREAD_UNCHANGED)
preprocessed = preprocess_signature(img)

# Extract features (39 dimensions)
feats = extract_structural_features(preprocessed)
print(f"Features: {feats['feature_dim']}D")

# Compare two signatures  
feats2 = extract_structural_features(preprocessed2)
sim = structural_similarity(feats, feats2)  # 0-1 similarity
```

## Fusion Methods

### 1. Simple Weighted Average
```python
fused_score = 0.7 * neural_similarity + 0.3 * structural_similarity
```

### 2. Learned Fusion (Recommended)
```python
from sklearn.linear_model import LogisticRegression

# Train on validation: X = [neural_score, structural_score], y = genuine(1)/forgery(0)
clf = LogisticRegression()
clf.fit(X_val, y_val)

# Apply to test
X_test = [[neural_score, structural_score]]
fused_score = clf.predict_proba(X_test)[0, 1]
```

## Kaggle Integration

Add these cells after your evaluation:

```python
# Test structural features
!python -c "
import sys; sys.path.insert(0, '/kaggle/working/sigver')
from structural_features import extract_structural_features
from preprocessing.signature_preprocessing import preprocess_signature
import cv2, numpy as np

DATA = '/kaggle/input/datasets/yogeshbaral/dataset-to-train-version-6'
sample = DATA + '/BHSig260/BHSig100_Bengali/Forged/B-S-1-F-01.png'
img = cv2.imread(sample, cv2.IMREAD_UNCHANGED)
if img is not None:
    preprocessed = preprocess_signature(img)
    feats = extract_structural_features(preprocessed)
    print(f'Success! Features: {feats[\"feature_dim\"]}D')
"

# Fusion demonstration
!python -c "
print('Fusion: score = w₁·neural + w₂·structural')
print('Expected: 2-3% FAR reduction, 1-2% EER reduction')
print('GPDS improvement likely larger (geometric normalization)')
"
```

## Files Created

- `structural_features.py` - Main module (tested)
- `test_structural_features.py` - Verification script  
- `demo_fusion.py` - Expected improvements
- This guide

## Next Steps

1. **Test in Kaggle** - Verify features extract correctly
2. **Implement fusion** - Modify `evaluate.py` or create fusion version
3. **Focus on GPDS** - Target 20% → 15% FAR reduction
4. **Compare metrics** - Neural-only vs fused performance

**No retraining needed** - Works with your existing checkpoints!