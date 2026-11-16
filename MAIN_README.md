# Offline Signature Verification

This guide describes the project pipeline and the Stage 3 implementation added
for deep metric learning. The project is intended to compare cheque signatures
using learned image embeddings; it is not, by itself, a production fraud
decision system.

## Pipeline status

| Stage | Responsibility | Status |
| --- | --- | --- |
| 1. Signature extraction | Detect and crop signatures from cheque images, currently upstream (for example, YOLO). | Upstream |
| 2.1 Artifact cleaning | Remove stamps, lines, and noise from the crop. | Upstream |
| 2.2 Image preprocessing | Preserve grayscale stroke intensity, dark ink on white, crop, resize without stretch, normalize. | preprocessing/signature_preprocessing.py |
| 3. Deep metric learning | Train a ConvNeXt-Tiny encoder to map signatures into an embedding space. | training/stage3_metric_learning.py, training/train_siamese.py |
| 4. Scoring and verification | Compare embeddings with cosine similarity and evaluate a policy. | Similarity helper implemented; thresholds and policy need project data |

Stage 3 expects a 384 x 192 grayscale image with dark ink on white, grayscale intensity retained for stroke pressure and ink flow. The model input is ImageNet-normalized C x H x W (3 channels, 192 high, 384 wide). Keep the same preprocessing for training and inference.

## Environment setup
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Dependencies are listed in requirements.txt: PyTorch, torchvision, timm, OpenCV, scikit-learn, Albumentations. Use --device auto to pick CUDA or CPU. On Apple Silicon, --device mps may crash with the installed PyTorch build; use CPU unless MPS is confirmed working. The first pretrained run downloads ConvNeXt weights.

## Datasets

Signature images are not committed; the data folders live in data/ (Git-ignored) and are described by CSV manifests with these columns:

| Column | Meaning |
| --- | --- |
| image_path | Path to an image; absolute or relative to the manifest directory, unless --image-root is given. |
| signer_id | Signer identifier. For a skilled forgery, use the identifier of the signer being imitated. |
| is_forgery | true/false or 1/0. |
| split | train, val (or validation), or test. |

Keep signers disjoint across train, validation, and test. Test rows are not used for training or validation. Forgeries carry the id of the signer they imitate.

### Dataset inventory

| Dataset | Signers | Images | Genuine | Forged | Format |
| --- | --- | --- | --- | --- | --- |
| BHSig-Bengali | 100 | 5,400 | 2,400 | 3,000 | 192 x 384 binary PNG |
| BHSig-Hindi | 160 | 8,640 | 3,840 | 4,800 | 192 x 384 binary PNG |
| CEDAR | 55 | 2,640 | 1,320 | 1,320 | 192 x 384 binary PNG |
| GPDS-Synthetic | 150 | 8,100 | 3,600 | 4,500 | raw JPG scans |
| Total | 465 | 24,780 | 11,160 | 13,620 | |

A combined, signer-disjoint manifest data/manifest_kaggle.csv merges BHSig + CEDAR + GPDS for Kaggle.
NOTE: 24 CEDAR forgery rows of signer 7 are referenced but missing from disk; restore them or drop those
rows before evaluation. Training is unaffected because they belong to the test split.

### GPDS notes
The downloaded GPDS folder splits train and test by sample (01-16 train, 17-24 test) for the same
150 signers, which leaks signers across splits. Run preprocessing/build_gpds_manifest.py to pool all
samples per signer and re-split by signer (about 70/15/15), writing data/manifests/gpds.csv. GPDS images
are raw large grayscale JPGs, and about 40 percent have very faint (light) ink; run check_dataset before training.

### Data audit
preprocessing/check_dataset.py audits a manifest for missing or unreadable files, wrong size, binary
versus grayscale mode, and per-dataset ink density:

```bash
python -m preprocessing.check_dataset --manifest data/manifests/gpds.csv --image-root data
```

Run it on every new manifest to catch near-empty images, mixed modes, and path problems before training.

BHSig and CEDAR are already preprocessed 192 x 384 images while GPDS is raw. Training without
--already-preprocessed runs the pipeline on both, re-processing the already-processed ones. This is an
acceptable experiment, but for a clean baseline unify the representation first (for example preprocess GPDS
with --grayscale into data/processed/).

## Stage 3 implementation
### Model and embeddings

- Backbone: convnext_tiny from timm, ImageNet weights by default.
- Projection head: Linear, LayerNorm, ReLU, Dropout, Linear. LayerNorm is used because signer-structured
  P-K batches make batch statistics less reliable, and it behaves identically in training and inference.
- Embedding: 512 dimensions, L2-normalized.
- SiameseSignatureModel.compare(reference, query) returns pairwise cosine similarities.
### Sampling and loss

Training batches are signer-balanced: sample P signers, K genuine samples per signer, and up to F skilled
forgeries that imitate each signer. Genuine signatures are anchors and positives; genuine signatures of other
signers and skilled forgeries of the same signer are negatives; forgeries are never anchors or positives.

Dataset-balanced batches (--dataset-mix equal): by default signers are sampled uniformly, so a large dataset
such as BHSig-Hindi (160 signers) dominates and a small one (CEDAR, 55) is starved. With --dataset-mix equal
the batch is split evenly across datasets, so every dataset is represented in every batch.
The only training objective is triplet loss over the L2-normalized embeddings; warm-up uses semi-hard
mining and fine-tuning uses batch-hard mining. SupCon was removed from the pipeline because the
triplet objective performs better on this dataset mix.

### Two training phases
1. Warm-up: freeze ConvNeXt, train the projection head for 5 epochs with semi-hard mining and lr 1e-3.
2. Fine-tune: unfreeze the last two ConvNeXt stages for 20 epochs with batch-hard mining and lr 1e-4.

The default batch uses 12 signers, 4 genuine signatures per signer, and up to 2 skilled forgeries per signer.
The learning rate follows a cosine decay to 1 percent of its start within each phase. The best checkpoint is
selected by the skilled-forgery EER on the signer-disjoint validation set (--select-by loss uses validation
loss with batch-hard mining).

Early stopping (--patience N): stops when the validation score has not improved for N epochs. The validation
EER uses --references (default 5); set --references 1 to match a one-reference deployment.
## Train

From the activated environment:

```bash
python -m training.train_siamese --manifest data/manifest.csv --output-dir checkpoints/stage3
```

The script saves the best checkpoint to checkpoints/stage3/best_model.pt. The checkpoint contains model
weights, embedding dimension, validation loss, validation EER and threshold, epoch, phase, seed, and
preprocessing settings (including binarize and the loss).

Useful options:

```text
--image-root PATH                 Base directory for relative image paths
--device auto|cpu|cuda|mps        Compute device (auto: CUDA, otherwise CPU)
--ink auto|light|dark             Ink-polarity resolution (auto detects it)
--already-preprocessed            Require 192x384 grayscale input files
--phase1-epochs N                 Warm-up epochs (default 5)
--phase2-epochs N                 Fine-tuning epochs (default 20)
--signers-per-batch N             P (default 12)
--genuine-per-signer N            K (default 4)
--forgeries-per-signer N          F (default 2)
--margin FLOAT                    Triplet margin (default 0.3)
--dataset-mix none|equal          Equal signers per dataset per batch (default none)
--references N                    Genuine references for validation EER (default 5)
--patience N                      Early-stop after N epochs without improvement (0 disables)
--workers N                       DataLoader workers (default 0)
--seed N                          Random seed (default 42)
```

Example with balanced batches, triplet loss, one-reference protocol, and early stopping:

```bash
python -m training.train_siamese --manifest data/manifest_kaggle.csv --image-root data --output-dir checkpoints/v5_triplet --phase1-epochs 5 --phase2-epochs 30 --dataset-mix equal --references 1 --patience 6
```

For a quick CPU trial without pretrained weights, pass --no-pretrained and small epoch counts.

## Tests and checks

```bash
python -m unittest discover -s tests -t .
```

The tests cover manifest signer-split validation, P-K batch composition, triplet loss (including batches
that contain forgeries), and ink-polarity normalization. These are implementation checks only; they do not
establish signature-recognition quality or select operating thresholds.

## Before using real data or deployment
1. Verify forged signer_id values identify the imitated signer.
2. Split by signer before training; never tune on the test set.
3. Run preprocessing/check_dataset.py on every manifest and confirm no missing files and one consistent mode.
4. Select thresholds on validation data, ideally for a target skilled-FAR, and report FAR/FRR with uncertainty on an untouched test set.
5. Treat cosine similarity as a score, not a probability; use human review and approved decision policies.

See docs/offline_signature_verification_architecture.md for the full pipeline and evaluation design.
