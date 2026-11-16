"""Measure CPU resource usage and verification throughput.

One request is one reference image plus one query image. The benchmark reports
model-only latency, end-to-end latency, parameter/checkpoint size, and peak RSS.
"""

from __future__ import annotations

import argparse
import os
import resource
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from preprocessing.signature_preprocessing import preprocess_signature, to_model_input
from training.stage3_metric_learning import SiameseSignatureModel


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "test_signs/reference_genuine__cedar__cedar_015__original_15_1.png"
)
DEFAULT_QUERY = (
    PROJECT_ROOT
    / "test_signs/same_signer_genuine__cedar__cedar_015__original_15_10.png"
)


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    return value / (1024 * 1024) if os.uname().sysname == "Darwin" else value / 1024


def _prepare(path: Path) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read signature image: {path}")
    return torch.from_numpy(to_model_input(preprocess_signature(image)))


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "best_model.pt")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--query", type=Path, default=DEFAULT_QUERY)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=0, help="Torch CPU threads; 0 keeps the default.")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        raise ValueError("runs must be positive and warmup cannot be negative")

    if args.threads:
        torch.set_num_threads(args.threads)

    checkpoint_size_mb = args.checkpoint.stat().st_size / (1024 * 1024)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = SiameseSignatureModel(
        pretrained=False, embedding_dim=int(checkpoint["embedding_dim"])
    ).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_mb = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / (
        1024 * 1024
    )

    preprocess_times: list[float] = []
    start = time.perf_counter()
    reference = _prepare(args.reference)
    query = _prepare(args.query)
    preprocess_times.append((time.perf_counter() - start) * 1000)
    pair = torch.stack((reference, query))

    def verify() -> float:
        with torch.inference_mode():
            # Match streamlit_app.py: one forward for the reference and one for
            # the query, followed by the cosine comparison.
            reference_embedding = model(reference.unsqueeze(0))
            query_embedding = model(query.unsqueeze(0))
            return float(torch.dot(reference_embedding[0], query_embedding[0]))

    with torch.inference_mode():
        for _ in range(args.warmup):
            verify()

    model_times: list[float] = []
    scores: list[float] = []
    for _ in range(args.runs):
        start = time.perf_counter()
        scores.append(verify())
        model_times.append((time.perf_counter() - start) * 1000)

    print("CPU verification benchmark")
    print(f"checkpoint: {args.checkpoint}")
    print(f"torch threads: {torch.get_num_threads()}")
    print(f"checkpoint size: {checkpoint_size_mb:.2f} MB")
    print(f"model parameters: {parameter_count:,}")
    print(f"model parameter memory: {parameter_mb:.2f} MB (float32)")
    print(f"input tensor per image: {tuple(reference.unsqueeze(0).shape)}")
    print(f"preprocessing: {preprocess_times[0]:.2f} ms (one reference + one query)")
    print(f"model-only mean: {np.mean(model_times):.2f} ms")
    print(f"model-only median: {_percentile(model_times, 50):.2f} ms")
    print(f"model-only p95: {_percentile(model_times, 95):.2f} ms")
    print(f"end-to-end estimate: {np.mean(model_times) + preprocess_times[0]:.2f} ms")
    print(f"model-only throughput: {1000 / np.mean(model_times):.2f} requests/sec")
    print(
        f"end-to-end throughput estimate: "
        f"{1000 / (np.mean(model_times) + preprocess_times[0]):.2f} requests/sec"
    )
    print(f"example cosine similarity: {scores[-1]:.4f}")
    print(f"peak process RSS: {_peak_rss_mb():.2f} MB")


if __name__ == "__main__":
    main()
