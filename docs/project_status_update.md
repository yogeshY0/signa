# Offline Signature Verification Project 

**Project:** Offline handwritten signature verification  
**Current objective:** Verify one questioned signature against one known genuine reference and distinguish genuine signatures, skilled forgeries, and random signatures.

## 1. Executive summary

The project currently uses a deep metric-learning system rather than a conventional
three-class classifier. The model converts each signature into a normalized embedding;
verification is then performed by comparing the cosine similarity between a reference
signature and a query signature.

The main implementation is complete for:

- signer-disjoint dataset manifests;
- grayscale signature preprocessing;
- ConvNeXt-Tiny feature extraction;
- a 512-dimensional normalized embedding space;
- signer-balanced batches;
- triplet-loss training with semi-hard and batch-hard mining;
- validation threshold selection and FAR/FRR/EER evaluation;
- one-reference evaluation support;
- CPU-compatible Streamlit inference.


## 2. Dataset status

### Previously validated dataset

The earlier cleaned manifest, [`data/manifest.csv`](../data/manifest.csv), contains:

| Property | Value |
|---|---:|
| Total images | 20,616 |
| Signers | 438 |
| Training images | 14,436 |
| Validation images | 3,066 |
| Test images | 3,114 |
| Datasets | CEDAR, Bengali BHSig260, Hindi BHSig260, SOHI |
| Missing/unreadable files | 0 in the completed audit |
| Split leakage | No signer overlap between train, validation, and test |

### New Kaggle experiment dataset

The newer [`data/manifest_kaggle.csv`](../data/manifest_kaggle.csv) contains:

| Property | Value |
|---|---:|
| Total manifest rows | 24,780 |
| Signers | 465 |
| Training rows | 17,376 |
| Validation rows | 3,678 |
| Test rows | 3,726 |
| Datasets | CEDAR, Bengali BHSig260, Hindi BHSig260, GPDS |
| GPDS images | 8,100 |

The GPDS addition increases dataset diversity and introduces a larger range of raw
image dimensions. This is useful for testing robustness, but it also makes consistent
preprocessing more important.

### Dataset source

The working dataset collection is available through the project Drive folder:

[CEDAR, BHSig260, and GPDS dataset folder](https://drive.google.com/drive/folders/1vXKuH4UyRKTYB6LAAvVdEBWSTgddLK4k?usp=sharing)

The manifest remains the source of truth for image paths, signer IDs, forgery labels,
and splits. Before training, manifest paths must be checked against the Kaggle dataset
mount because a Drive link does not guarantee identical directory names or complete
file availability.


## 3. Preprocessing pipeline

The implementation is in
[`preprocessing/signature_preprocessing.py`](../preprocessing/signature_preprocessing.py).
The current pipeline:

1. Converts the input to grayscale.
2. Resolves ink polarity to dark ink on a light background.
3. Flattens uneven illumination and paper background.
4. Detects the ink region and crops it with a margin.
5. Resizes while preserving the original aspect ratio.
6. Pads to a fixed size of **384 × 192 pixels**.
7. Preserves grayscale stroke intensity rather than forcing binary black/white values.
8. Repeats the grayscale channel for the ImageNet-pretrained backbone.
9. Applies ImageNet normalization before model inference.

The preprocessing is intended to remove irrelevant background variation while retaining
stroke pressure and ink-flow information. It must be applied consistently during
training, validation, testing, and deployment.

### Model input contract

| Property | Specification |
|---|---|
| Stored image | 192 pixels high × 384 pixels wide |
| Channels before model | 1 grayscale channel |
| Channels sent to ConvNeXt | 3 channels, grayscale repeated three times |
| Tensor layout | `3 × 192 × 384` per image |
| Normalization | ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)` |
| Geometric distortion | None from stretching; aspect ratio is preserved and remaining space is padded |

This contract must be identical for training and inference. Raw, binary, or differently
sized inputs can change the score distribution and invalidate the selected threshold.

The main preprocessing risk is over-cleaning. If illumination correction or cropping
removes faint stroke details, genuine and forged signatures can become artificially
similar. Therefore, raw-versus-preprocessed comparisons should use the same dataset,
same signer split, same reference count, same threshold-selection procedure, and same
loss before drawing conclusions.

## 4. Current architecture

The current learned verification architecture is:

```text
Input signature image
        |
Grayscale and geometric preprocessing
        |
Shared ConvNeXt-Tiny encoder
        |
Projection head
        |
512-dimensional L2-normalized embedding
        |
Cosine similarity with reference embedding
        |
Threshold-based decision
```

The architecture is implemented in
[`training/stage3_metric_learning.py`](../training/stage3_metric_learning.py).

Important properties:

- **Shared weights:** reference and query signatures use the same encoder.
- **Metric learning:** the model learns distances between signatures instead of
  memorizing signer class IDs.
- **Normalized embeddings:** cosine similarity is stable and easy to calibrate.
- **Signer-aware negatives:** skilled forgeries associated with a signer are treated
  as negatives for that signer, but forged images are never anchors or positives.
- **Transfer learning:** ConvNeXt-Tiny starts from ImageNet weights for normal runs.
- **Staged fine-tuning:** the backbone is first frozen, then the final two stages are
  unfrozen for task-specific adaptation.

### Layer-level model details

1. **Input:** normalized `3 × 192 × 384` tensor.
2. **Backbone:** ImageNet-pretrained ConvNeXt-Tiny with the classifier removed and
   global average pooling, producing a 768-dimensional visual feature vector.
3. **Projection layer 1:** `Linear(768, 512)`.
4. **Normalization:** `LayerNorm(512)`.
5. **Activation:** ReLU.
6. **Regularization:** dropout with probability `0.2`.
7. **Projection layer 2:** `Linear(512, 512)`.
8. **Output:** L2-normalized 512-dimensional embedding.

The projection head is not a classification head. It maps the backbone representation
into a comparison space where same-signer signatures should be close and different or
forged signatures should be farther apart.

### Verification calculation

For unit-length reference embedding `r` and query embedding `q`:

```text
cosine_similarity = r · q
score >= threshold  -> MATCH
score < threshold   -> NO MATCH
```

The score is not a probability. A score of `0.90` does not mean a 90% probability of
genuineness. The threshold must be selected for the same reference count used in
deployment.

The current design is appropriate for deployment with one reference and one query.
Adding a structural-analysis branch remains a possible controlled experiment, but it
should not replace the current baseline until the data and evaluation protocol are
stable.

## 5. Training procedure

Training is orchestrated by
[`training/train_siamese.py`](../training/train_siamese.py).

### Batch construction

The default batch is signer-balanced:

- 12 signers;
- 4 genuine signatures per signer;
- up to 2 skilled forgeries per signer;
- 72 images per batch when all requested samples are available.

An alternative F4 experiment uses four forgeries per signer. More forgeries can
increase exposure to difficult negative examples, but it does not guarantee better
generalization. It changes the training distribution and can increase the influence
of dataset-specific forgery artifacts.

### Loss and mining

The established baseline uses triplet loss:

- semi-hard negative mining during warm-up;
- batch-hard mining during fine-tuning;
- genuine signatures from the same signer form positive pairs;
- genuine signatures from other signers and associated skilled forgeries form negatives.

The newer notebook also proposes supervised contrastive loss (SupCon), equal dataset
mixing, one-reference validation, and early stopping. SupCon is currently blocked by a
forged-sample masking/dimension issue and should not be used until the focused tests
pass.

### Training phases and optimization

| Phase | Default duration | Backbone | Mining | Learning rate |
|---|---:|---|---|---:|
| Warm-up | 5 epochs | Frozen | Semi-hard triplet mining | `1e-3` |
| Fine-tuning | 20 epochs | Final 2 stages unfrozen | Batch-hard mining | `1e-4` |

Both phases use AdamW, weight decay `1e-4`, gradient clipping at norm `5`, and cosine
learning-rate decay by default. The best checkpoint is selected using validation
skilled-forgery EER unless loss selection is explicitly requested.

## 6. Evaluation protocol

Evaluation is implemented in
[`training/evaluate.py`](../training/evaluate.py).

For each signer:

1. Select the configured number of genuine reference signatures.
2. Compare each remaining genuine signature against the references.
3. Compare skilled forgeries associated with that signer against the references.
4. Compare genuine signatures from other signers as random negatives.
5. Select the threshold on validation signers only.
6. Apply the unchanged threshold to held-out test signers.

The evaluator supports both the previous five-reference protocol and the intended
deployment protocol:

```bash
python -m training.evaluate \
  --checkpoint checkpoints/<run>/best_model.pt \
  --manifest data/processed/manifest.csv \
  --already-preprocessed \
  --references 1
```

The one-reference setting is the relevant production measurement. A threshold
selected with five references must not be used as evidence for one-reference
performance.

## 7. Results to date

Earlier controlled experiments showed that skilled forgeries are substantially harder
than random signatures. Reported F2/F4 results were approximately:

| Experiment | Skilled EER | Skilled FAR | Skilled FRR |
|---|---:|---:|---:|
| F2 | 12.97% | 13.44% | 11.71% |
| F4 | 12.07% | 14.87% | 8.56% |

An independent raw-image run reported the following skilled-forgery trade-off:

| Threshold | Skilled FAR | FRR | Balanced accuracy |
|---:|---:|---:|---:|
| 0.80 | 11.67% | 6.38% | 90.97% |
| 0.85 | 5.36% | 14.45% | 90.10% |
| 0.90 | 1.03% | 28.33% | 85.32% |

These results should be treated as directional, not as a final comparison, because
the runs did not all use identical datasets, preprocessing, reference counts,
threshold selection, or loss settings. The next valid comparison is Triplet versus
SupCon, and raw versus preprocessed input, under one fixed protocol.

## 8. Metric definitions

### False Accept Rate (FAR)

FAR is the percentage of forged queries incorrectly accepted as genuine:

```text
FAR = forged queries with score >= threshold / all forged queries
```

Skilled-forgery FAR is the most important security measure because skilled forgeries
are intentionally made to resemble the target signer.

### False Reject Rate (FRR)

FRR is the percentage of genuine queries incorrectly rejected:

```text
FRR = genuine queries with score < threshold / all genuine queries
```

A higher threshold generally reduces FAR but increases FRR. This is the security
versus usability trade-off.

### Equal Error Rate (EER)

EER is the operating point where FAR and FRR are closest. It summarizes overlap
between genuine and forgery score distributions:

- lower EER generally indicates better separation;
- EER is not necessarily the production threshold;
- a deployment may choose a lower-FAR threshold while accepting higher FRR.

### Threshold, review zone, and confidence intervals

A practical deployment can define a review zone around the threshold:

```text
score well above threshold -> likely match
score well below threshold -> likely no match
score near threshold       -> manual review
```

The evaluator can bootstrap test signers to produce 95% confidence intervals. These
intervals show uncertainty caused by the finite number of signers; small differences
between models are not meaningful when their intervals overlap substantially.

## 9. Deployment status

[`streamlit_app.py`](../streamlit_app.py) provides a CPU-compatible verification
interface:

- upload one known genuine reference;
- upload one questioned signature;
- preprocess both images;
- generate two embeddings;
- calculate cosine similarity;
- compare the score with a configurable threshold;
- display the decision and diagnostic metrics.

GPU is recommended for training and large-scale evaluation. GPU is not required for
one-reference/one-query inference. A CPU benchmark should report model loading,
preprocessing, embedding inference, end-to-end latency, P50/P95 latency, and
verifications per second on the intended deployment hardware.

The displayed similarity is a model score, not a calibrated probability or proof of
authenticity. Borderline scores should be routed for manual review rather than treated
as definitive decisions.

## 10. Current risks and limitations

- Skilled forgeries can be visually close to genuine signatures, causing score
  overlap and false accepts.
- CEDAR contains artifacts and border/ink-pattern variation that may create shortcuts
  if not audited.
- GPDS has highly variable raw dimensions and requires consistent normalization.
- A single global threshold may not be optimal for every dataset or acquisition
  condition.
- Validation metrics can be optimistic because the threshold is selected on the
  validation signers.
- The current system is a learned visual verifier; it does not yet include a
  production-ready structural or document-forensics branch.
- SupCon and the new Kaggle manifest are not yet fully validated.

## 11. Recommended next steps


### Controlled experiments

1. Train Triplet and SupCon with the same seed, data, splits, batch budget, and
   one-reference validation protocol.
2. Evaluate both checkpoints on the same held-out test signers.
3. Compare FAR, FRR, EER, confidence intervals, and per-dataset results.
4. Compare raw and preprocessed inputs under the same conditions.
5. Repeat the best configuration with multiple seeds before selecting a production
   checkpoint.

### Architecture extension, only after baseline validation

If skilled-forgery false accepts remain high after the controlled baseline, add a
secondary structural branch using contour, skeleton, projection, Hu-moment, and
shape descriptors. Fuse its score with the neural similarity using a validation-only
calibration layer. Compare neural-only, structural-only, and fused results rather than
assuming that a more complex architecture is automatically better.

## 12. Conclusion

The project has a sound baseline architecture and a reproducible evaluation design.
The earlier dataset is structurally validated, and the newer GPDS-inclusive dataset
is a promising expansion. The most important current work is not immediately
replacing ConvNeXt-Tiny; it is completing the new data pipeline, fixing SupCon,
standardizing one-reference evaluation, and measuring performance with matched
experiments. This will make the next model comparison scientifically reliable and
will identify whether the remaining skilled-forgery errors come primarily from
preprocessing, data coverage, loss design, or the model itself.
