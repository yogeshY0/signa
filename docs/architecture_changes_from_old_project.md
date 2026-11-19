# Architecture Changes: Old Project vs `feature/bhsig-cedar-grayscale`

This document compares the **old project baseline** (`main`: the original BHSig+GPDS
binary-PNG pipeline with SupCon) against the **current branch**
`feature/bhsig-cedar-grayscale` (raw grayscale BHSig+CEDAR plus structural-feature fusion).

It is derived from the git history (`git diff main..HEAD`, 7 commits) and the current
source in `training/`, `preprocessing/`, and `structural_features.py`.

---

## 1. Summary of changes

| Area | Old project (`main`) | Current branch | Why it changed |
| --- | --- | --- | --- |
| Input representation | Binary (pure black/white) 192x384 PNG | **Grayscale** 192x384, stroke intensity kept | Binary loses ink flow, pressure, stroke-thickness cues |
| Normalisation layer | **BatchNorm** in projection head | **LayerNorm** | Per-sample; independent of PK batch composition |
| Training objective | Triplet **and** SupCon (`--loss`) | **Triplet only** | Triplet 13.05% EER vs SupCon 14.88% (old split) |
| Loss mining | semi-hard | semi-hard (**warm-up**) then **batch-hard** (fine-tune) | Harder negatives in phase 2 |
| Training schedule | single phase | **2 phases**: warmup (frozen backbone) + fine-tune (last 2 stages) | Stable head-first training |
| Optimiser | AdamW, flat LR | AdamW + **CosineAnnealingLR** per phase | Smoother convergence |
| Sampling | uniform PK | PK + **`--dataset-mix equal`** | Small datasets (CEDAR) no longer drowned out |
| Split strategy | 3-way (train/val/test) | **4-way**: train 60%/fusion 15%/val 10%/test 15% | Dedicated `fusion` split for fusion classifier |
| Datasets | BHSig + **GPDS** | BHSig + **CEDAR** (GPDS dropped) | GPDS was weakest; CEDAR is the target |
| Raw image support | preprocessed PNGs only | **`--resize-only`** + `--already-preprocessed` + `--image-root` | Train/eval on raw TIFF/JPG scans |
| Structural features | none | **new** `structural_features.py` (85-D/image, 14-D pair deltas) | Orthogonal geometric signal |
| Fusion | none | **new** `train_fusion.py` / `evaluate_fusion.py` | Neural cosine + structural probability |
| Evaluation | 5 references, EER only | **1 reference**, target-FAR points, bootstrap CIs, per-dataset | One-reference deployment protocol |
| Audit | none | **new** `preprocessing/audit_raw_dataset.py` + `reports/raw_audit.*` | Verify raw data before building manifests |

---
## 2. Detailed changes

### 2.1 Preprocessing and input representation

**Old:** signatures were converted to **binary** (Otsu, pure 0/255) 192x384 PNGs. Commit
`bc3614a` moved the pipeline to **grayscale** to preserve ink flow/pressure/stroke
features; subsequent work kept that choice.

**Current** (`preprocessing/signature_preprocessing.py`):
1. `to_grayscale` — grayscale + optional blur.
2. `normalize_polarity` (`auto`/`light`/`dark`) — guarantees dark ink on white.
3. `flatten_illumination` — morphological-closing removes shadows without quantising ink.
4. `crop_to_ink` — tight crop around the ink bounding box with a 5% margin.
5. `aspect_aware_resize` — uniform scale + letterbox pad to 384x192 (never stretches).
6. `to_model_input` — replicate to 3 channels + ImageNet normalisation.

**`--resize-only` fast path** (`ce93d8e`): skip steps 2-4, just `cv2.resize` to 384x192.

### 2.2 Model architecture

`SiameseSignatureModel`:
```
ConvNeXt-Tiny (timm, num_classes=0, global_pool=avg)
   └─ Linear(num_features -> 512)
   └─ LayerNorm(512)              # was BatchNorm (caf3df1)
   └─ ReLU
   └─ Dropout(0.2)
   └─ Linear(512 -> embedding_dim=512)
   └─ L2 normalise
```

- **LayerNorm instead of BatchNorm** (`caf3df1`): per-sample normalisation, independent
  of PK batch composition.
- Embeddings are **L2-normalised** => cosine distance = Euclidean.

### 2.3 Training objective

**Old:** Triplet + SupCon (`--loss` flag, `--temperature`).
**Current:** **triplet only** (`fcb599a`). Triplet 13.05% EER vs SupCon 14.88%.

- Positives: same signer, genuine, not self.
- Negatives: different signer (genuine) or same-signer forgery.
- semi-hard in warm-up, batch-hard in fine-tune. Margin 0.3.
- Forged samples are never anchors.

### 2.4 Two-phase training

| Phase | Backbone | Mining | LR |
|---|---|---|---|
| warmup (5 ep) | frozen | semi-hard | 1e-3 |
| fine_tune (20 ep) | last 2 stages | batch-hard | 1e-4 |

- AdamW + CosineAnnealingLR (->1% per phase).
- Early stopping (`--patience`), checkpoint by EER.
### 2.5 Sampling

`PKBatchSampler`: P signers x K genuine x F forgeries per batch.  
`--dataset-mix equal` samples datasets evenly (prevents CEDAR's 55 signers from being
swamped by BHSig-Hindi's 160 signers). Dataset inferred from signer id.

### 2.6 Manifest and split strategy

- Columns unchanged: `image_path, signer_id, is_forgery, split`.
- **`fusion` is now a valid split** (`0308cf4`): `train | val | validation | test | fusion`.
- `build_bhsig_cedar_manifest.py`: **stratified, signer-disjoint** 4-way split
  **60/15/10/15** (train/fusion/val/test), seed 42, zero overlap assertion.
### 2.7 Structural features (new)

`structural_features.py`:

| Block | Dim | Content |
|---|---|---|
| hu | 7 | log Hu moments (shape, scale/rotation invariant) |
| h_profile | 32 | horizontal projection (ink per row, binned) |
| v_profile | 32 | vertical projection (ink per column, binned) |
| geometry | 6 | ink fraction, centroid, bbox area, aspect, fill |
| contour | 4 | solidity, extent, compactness, eccentricity |
| skeleton | 4 | endpoints, junctions, stroke length, branching |

Total per image: **85-D**.  
`structural_pair_features(f1, f2)` = **14-D** deltas -> `LogisticRegression`.

Fix (`7c86157`): `_projection_profiles` pads projection arrays to next multiple of 32.
Raw BHSig (e.g. 280x970) is not divisible by 32, which previously crashed.

### 2.8 Fusion (new)

`train_fusion.py` — 2-feature LR on **[neural_cosine, structural_probability]**:

1. Frozen neural checkpoint -> cosine per pair.
2. Per-image structural features -> pair deltas -> scaler -> structural LR -> prob.
3. `StandardScaler` + `LogisticRegression` fit on **fusion** split.
4. EER / target-FAR thresholds on **val**.
5. Artifacts: `fusion_scaler.pkl`, `fusion_model.pkl`, `thresholds.json`.

`evaluate_fusion.py` (new) scores the stacked model on **test** split.

### 2.9 Evaluation

`evaluate.py` gained:
- `--resize-only` (mutually exclusive with `--already-preprocessed`).
- `--target-far 0.01 0.05` — threshold on **val**, FRR on **test**.
- Threshold from validation applied unchanged to test.
- Bootstrap CIs resample test signers.
- **1-reference** protocol throughout.

### 2.10 Dataset audit (new)
## 3. New / rewritten files on this branch

```
structural_features.py                       (rewritten: 85-D + pair deltas)
training/train_structural.py                 (new: structural-only LR classifier)
training/train_fusion.py                     (new: neural+structural fusion)
training/evaluate_fusion.py                  (new: test-split fusion evaluation)
preprocessing/audit_raw_dataset.py           (new)
preprocessing/build_bhsig_cedar_manifest.py  (new: 4-way stratified manifest)
reports/raw_audit.{md,json}                  (new)
kaggle_train_raw.py                          (new: raw-image training helper)
test_structural_features.py                  (rewritten)
```

Modified: `training/stage3_metric_learning.py`, `training/train_siamese.py`,
`training/evaluate.py`, `preprocessing/check_dataset.py`.

## 4. Results on the one-reference test split

Combined BHSig + CEDAR, 47 test signers, 1 genuine reference, 1,081 genuine queries,
1,362 skilled forgeries.

| System | EER skilled | FRR @EER-thr | FAR @EER-thr |
|---|---|---|---|
| **A. Neural-only** (ConvNeXt + triplet) | **19.32%** (CI 14.6-24.0%) | 18.04% | 20.56% |
| **B. Structural-only** (LR on 14-D) | 17.24% (val) | n/a | n/a |
| **C. Fusion** (2-D LR) | ~17-18% | 18.41% | 16.23% |

Per-dataset neural-only EER: **CEDAR 4.79%**, Hindi 20.67%, Bengali 21.76%.

At target FAR (neural-only, test): FAR=1% -> FRR 50.05%; FAR=5% -> FRR 29.23%.

> **Caveat:** fusion is **not** a statistically significant improvement over neural-only;
the EER ranges overlap. Structural-only (17.24%) is too weak to lift the neural model
meaningfully on this split. Best individual result: CEDAR (4.79% skilled EER).

## 5. Behavioural differences that matter in practice

1. **Grayscale, not binary** — do not feed binarised inputs to this checkpoint.
2. **One reference** in all current experiments — do not compare with 5-reference eval.
3. **`fusion` split must exist** in the manifest for `train_fusion.py`.
4. **`--image-root`** is required when manifest paths are relative and images live elsewhere.
5. `--already-preprocessed` vs `--resize-only` are mutually exclusive.
6. **Frozen neural encoder during fusion** — fusion never back-propagates into ConvNeXt.

## 6. Commit timeline on this branch (oldest -> newest)

| Commit | Change |
|---|---|
| `0308cf4` | Phase 1-3: audit, 4-way manifest, grayscale preprocessing, fusion split |
| `1b62d82` | Structural rewrite + train_structural + target-FAR eval |
| `ce93d8e` | `--resize-only` fast path for raw images |
| `794476e` | Fusion fix + `--resize-only` in fusion |
| `fd9757f` | Notebook commands: add `--image-root` |
| `7c86157` | Fix `_projection_profiles` padding for raw sizes |
| `06d4fe9` | Add `evaluate_fusion.py` |

Earlier baseline commits (before this branch):
- `caf3df1` BatchNorm -> LayerNorm
- `bc3614a` binary -> grayscale
- `1f58151` dataset-balanced sampling + GPDS
- `fcb599a` remove SupCon (triplet-only)

`preprocessing/audit_raw_dataset.py` + `reports/raw_audit.md`:

| Dataset | Files | Signers | G | F | Height | Width |
|---|---|---|---|---|---|---|
| bhsig_bn | 5,400 | 100 | 2,400 | 3,000 | 233-305 | 898-1083 |
| bhsig_hi | 8,640 | 160 | 3,840 | 4,800 | 229-435 | 843-1329 |
| cedar | 2,592 | 54 | 1,296 | 1,296 | 145-594 | 264-714 |

All signers meet >=2 genuine, >=1 forgery.
- Neural model trains on `train`; thresholds on `val`; fusion fits on `fusion`;
  `test` only scored.