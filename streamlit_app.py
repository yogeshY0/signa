"""Streamlit UI for comparing two signatures with a trained checkpoint."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import streamlit as st
import torch

from preprocessing.signature_preprocessing import preprocess_signature, to_model_input
from training.stage3_metric_learning import SiameseSignatureModel


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "best_model.pt"
DEFAULT_THRESHOLD = 0.88


def _read_image(uploaded_file: Any, label: str) -> np.ndarray:
    data = uploaded_file.getvalue()
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"{label} could not be decoded as an image.")
    return image


def _checkpoint_bytes() -> bytes:
    if not DEFAULT_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"No checkpoint found at {DEFAULT_CHECKPOINT}."
        )
    return DEFAULT_CHECKPOINT.read_bytes()


@st.cache_resource(show_spinner="Loading trained model...")
def _load_model(checkpoint_data: bytes) -> tuple[SiameseSignatureModel, dict[str, Any]]:
    checkpoint = torch.load(io.BytesIO(checkpoint_data), map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("The checkpoint must contain a model_state_dict.")
    embedding_dim = int(checkpoint.get("embedding_dim", 512))
    model = SiameseSignatureModel(pretrained=False, embedding_dim=embedding_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _embed(
    model: SiameseSignatureModel,
    image: np.ndarray,
    *,
    binarize: bool,
) -> tuple[torch.Tensor, np.ndarray]:
    processed = preprocess_signature(image, binarize=binarize)
    tensor = torch.from_numpy(to_model_input(processed)).unsqueeze(0)
    with torch.inference_mode():
        embedding = model(tensor).squeeze(0)
    return embedding, processed


def _metric_rows(
    similarity: float,
    threshold: float,
    checkpoint: dict[str, Any],
    input_shape: tuple[int, ...],
) -> list[tuple[str, str]]:
    distance = float(np.sqrt(max(0.0, 2.0 - 2.0 * similarity)))
    score_percent = (similarity + 1.0) * 50.0
    margin = similarity - threshold
    decision = "MATCH" if similarity >= threshold else "NO MATCH"
    return [
        ("Decision", decision),
        ("Cosine similarity", f"{similarity:.4f}"),
        ("Similarity score", f"{score_percent:.2f}%"),
        ("Threshold", f"{threshold:.4f}"),
        ("Margin above threshold", f"{margin:+.4f}"),
        ("Embedding Euclidean distance", f"{distance:.4f}"),
        ("Input image shape", str(input_shape)),
        ("Embedding dimension", str(checkpoint.get("embedding_dim", 512))),
        ("Checkpoint phase", str(checkpoint.get("phase", "unknown"))),
        ("Checkpoint epoch", str(checkpoint.get("epoch", "unknown"))),
        (
            "Best validation loss",
            f"{float(checkpoint['best_validation_loss']):.6f}"
            if "best_validation_loss" in checkpoint
            else "unknown",
        ),
    ]


st.set_page_config(page_title="Signature Verification", page_icon="✍️", layout="wide")
st.title("Offline Signature Verification")
st.caption(
    "Upload a genuine reference signature and a test signature. "
    "The model compares their learned embeddings."
)

with st.sidebar:
    st.header("Model settings")
    threshold = st.number_input(
        "Cosine threshold",
        min_value=-1.0,
        max_value=1.0,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        format="%.4f",
        help="Adjust the cosine similarity threshold. A value of 0.88 corresponds to 94% similarity.",
    )
    binarize = st.checkbox(
        "Use binary black/white preprocessing",
        value=False,
        help="Training used grayscale preprocessing. Leave this unchecked for matching train/inference preprocessing.",
    )
    if binarize:
        st.warning("Binary preprocessing differs from the grayscale training path.")

uploaded_reference = st.file_uploader(
    "Reference image (known genuine signature)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    key="reference",
)
uploaded_test = st.file_uploader(
    "Test image (signature to verify)",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    key="test",
)

if uploaded_reference is None or uploaded_test is None:
    st.info("Upload both images to run verification.")
    st.stop()

try:
    checkpoint_data = _checkpoint_bytes()
    model, checkpoint = _load_model(checkpoint_data)
    reference_image = _read_image(uploaded_reference, "Reference image")
    test_image = _read_image(uploaded_test, "Test image")
    reference_embedding, reference_processed = _embed(
        model, reference_image, binarize=binarize
    )
    test_embedding, test_processed = _embed(model, test_image, binarize=binarize)
except (FileNotFoundError, OSError, RuntimeError, ValueError) as error:
    st.error(str(error))
    st.stop()

similarity = float(torch.dot(reference_embedding, test_embedding).item())
metrics = _metric_rows(
    similarity,
    float(threshold),
    checkpoint,
    tuple(reference_image.shape),
)

st.subheader("Verification result")
decision = metrics[0][1]
if decision == "MATCH":
    st.success(decision)
else:
    st.error(decision)

columns = st.columns(2)
with columns[0]:
    st.image(reference_image, caption="Reference image", channels="BGR")
    st.image(reference_processed, caption="Reference after preprocessing", clamp=True)
with columns[1]:
    st.image(test_image, caption="Test image", channels="BGR")
    st.image(test_processed, caption="Test after preprocessing", clamp=True)

st.subheader("Metrics")
metric_columns = st.columns(3)
for index, (name, value) in enumerate(metrics):
    with metric_columns[index % 3]:
        st.metric(name, value)

st.info(
    "The similarity score is a model score, not a calibrated probability or proof "
    "of authenticity. The threshold must be validated for your deployment data; "
    "manual review and institution-approved controls are still required."
)

with st.expander("Model details"):
    checkpoint_hash = hashlib.sha256(checkpoint_data).hexdigest()
    st.write(
        {
            "checkpoint_sha256": checkpoint_hash,
            "preprocessing": "binary black/white" if binarize else "grayscale",
            "target_size": "384 x 192 (width x height)",
            "checkpoint_metadata": checkpoint.get("preprocessing", {}),
        }
    )
