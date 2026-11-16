# Model Performance and Real-Time Usage FAQ

## 1. What does one request mean?

One verification request contains:

1. One known genuine reference signature.
2. One questioned signature.
3. Preprocessing of both images.
4. Two embedding inferences.
5. One cosine-similarity comparison.
6. One threshold-based decision.

The model is not classifying a single image independently. It is comparing a pair of
signatures.

## 2. How fast is the current model on CPU?

The current local benchmark used [`best_model.pt`](../best_model.pt), the Streamlit
inference path, and four PyTorch CPU threads:

| Measurement | Result |
|---|---:|
| Checkpoint size | 108.71 MB |
| Parameters | 28,477,536 |
| Input per image | `1 × 3 × 192 × 384` |
| Preprocessing for two images | approximately 8–10 ms |
| Model-only latency | approximately 1,035 ms |
| End-to-end estimated latency | approximately 1,045 ms |
| Sequential throughput | approximately 0.95 requests/second |
| Approximate sequential capacity | approximately 57 verifications/minute |
| Peak process RSS | approximately 579–609 MB |

These are measurements on one local CPU and are not universal hardware guarantees.
The benchmark is available at [`benchmark_inference.py`](../benchmark_inference.py).

## 3. How many requests per second can it handle?

On the tested CPU, the current sequential capacity is approximately:

```text
0.95 verification requests/second
```

This is approximately:

```text
57 verification requests/minute
3,420 verification requests/hour
```

This result assumes one process handling one request at a time. It is not a measured
concurrent-server capacity.

## 4. Can we claim that it supports multiple users?

Not from the current benchmark alone. The current result measures sequential inference.
Concurrent capacity must be measured with a load test using multiple simultaneous
requests.

Concurrent requests may:

- increase latency;
- increase memory usage;
- cause CPU contention;
- reduce requests per second;
- require a request queue.

The correct production statement at present is: **the prototype supports CPU
inference, with approximately 0.95 sequential verifications per second on the tested
machine. Concurrent capacity is not yet certified.**

## 5. Is this real-time?

For an interactive single-user application, approximately one second per verification
is generally acceptable as near-real-time behavior.

For high-volume or strict real-time service requirements, the system needs additional
load testing and likely hardware or deployment optimization. “Real-time” should be
defined using a latency target, for example:

```text
P95 end-to-end latency <= 1.5 seconds
```

The P95 target must be measured under the expected number of simultaneous users.

## 6. What is the expected response time for one user?

The current estimate is approximately one second for:

- decoding two images;
- preprocessing both images;
- generating two embeddings;
- calculating similarity;
- returning the decision.

The exact result depends on image upload time, disk/network access, CPU load, and
Streamlit overhead. The benchmark measures the model path, not internet upload time.

## 7. What is the P95 latency?

In the local model-only benchmark, the P95 latency was approximately 1.12 seconds in
the ten-run measurement. P95 means that 95% of measured requests completed at or below
that time, while approximately 5% were slower.

For production, P95 should be measured with:

- the actual deployment server;
- realistic image sizes;
- realistic concurrent users;
- the complete API or Streamlit path;
- image upload and response time included.

## 8. How much memory does the model require?

The checkpoint and float32 parameters require approximately 109 MB. The measured
process peak was approximately 579–609 MB because memory also includes:

- Python;
- PyTorch;
- ConvNeXt intermediate tensors;
- OpenCV;
- Streamlit;
- allocator overhead;
- loaded images and runtime objects.

A practical single-process deployment should reserve at least 1 GB of RAM. More memory
is recommended for concurrent requests or multiple worker processes.

## 9. How much GPU memory is required?

Training requires substantially more memory than inference because it stores
activations, gradients, optimizer state, and batches. GPU training memory depends on:

- batch size;
- number of signers per batch;
- image size;
- number of workers;
- frozen versus unfrozen backbone;
- optimizer state.

Inference requires much less memory than training, but the exact GPU requirement must be
measured on the selected hardware. The current model does not require a GPU for one
reference plus one query.

## 10. What is the main performance bottleneck?

The main bottleneck is the ConvNeXt-Tiny encoder. Each request performs two model
forward passes because the reference and query are embedded separately.

Preprocessing contributes only a small portion of the measured latency. Potential
optimizations include:

- cache the reference embedding when the same reference is reused;
- use a GPU;
- use TorchScript or another inference runtime;
- use reduced precision where accuracy remains acceptable;
- batch independent requests;
- keep the model loaded instead of loading it per request.

Caching a reference embedding can approximately remove one of the two forward passes
for repeated comparisons against the same reference.

## 11. Can the model process more requests by batching?

Yes, independent requests can be batched. Batching may improve hardware utilization,
especially on a GPU, but it changes latency:

- throughput may increase;
- individual requests may wait for a batch to fill;
- memory usage increases;
- response time may become less predictable.

For interactive use, small batches or immediate execution are usually preferable. For
offline bulk verification, larger batches can improve throughput.

## 12. How does a GPU change capacity?

A GPU will usually reduce embedding latency and increase throughput, especially for
batched requests. However, the improvement cannot be stated without measuring the
specific GPU, batch size, precision, and serving setup.

The correct comparison is:

| Environment | Required measurements |
|---|---|
| CPU | P50, P95, memory, sequential and concurrent requests/sec |
| GPU | P50, P95, VRAM, batch throughput, sequential and concurrent requests/sec |

The CPU benchmark should not be used to predict GPU capacity.

## 13. What accuracy metrics should be reported?

Performance is not only speed. The model must also report:

- **Skilled-forgery FAR:** forged signatures incorrectly accepted as genuine.
- **Random-signature FAR:** unrelated genuine signatures incorrectly accepted.
- **FRR:** genuine signatures incorrectly rejected.
- **EER:** operating point where FAR and FRR are closest.
- **Per-dataset metrics:** CEDAR, Bengali BHSig260, Hindi BHSig260, and GPDS.
- **Confidence intervals:** uncertainty from the finite test signer population.

Skilled-forgery FAR is the most important security metric because skilled forgeries
are intentionally similar to the target signature.

## 14. What happens when the threshold changes?

Increasing the cosine threshold generally:

- reduces false accepts;
- increases false rejects;
- improves security at the cost of usability.

Decreasing the threshold generally:

- accepts more genuine signatures;
- increases the risk of accepting forgeries.

The threshold must be selected on validation signers and frozen before test
evaluation. A threshold selected with five references must not automatically be used
for a one-reference deployment.

## 15. Does a high similarity score prove authenticity?

No. Cosine similarity is a model score, not a calibrated probability and not proof of
authenticity. A high score means the two images are close in the learned embedding
space.

Borderline cases should be routed to manual review. The application should not present
the score as a legal or forensic guarantee.

## 16. What happens if the input image is poor?

Poor scans, excessive cropping, shadows, background artifacts, low resolution, and
broken ink can reduce reliability. The preprocessing pipeline attempts to normalize
these conditions, but it cannot recover information that is absent from the image.

The deployment should reject or flag:

- unreadable files;
- images with no detected ink;
- extremely low-resolution images;
- unsupported formats;
- scores near the review threshold.

## 17. Is the current model suitable for production?

It is suitable for a controlled prototype or pilot deployment, but production
readiness requires:

1. One-reference test evaluation.
2. Dataset and path validation.
3. Multiple training seeds.
4. Per-dataset FAR/FRR/EER.
5. Confidence intervals.
6. CPU and/or GPU load testing.
7. Error logging and input validation.
8. A documented threshold and review policy.
9. Monitoring for data drift.
10. Security and privacy controls for uploaded signatures.

The architecture is operationally usable, but its final security performance against
skilled forgeries must be established on the cleaned GPDS-inclusive dataset.

## 18. What should be answered if asked for a single capacity number?

Use this answer:

> On the tested CPU, one verification request (one reference plus one questioned
> signature) takes approximately 1.05 seconds end-to-end and supports approximately
> 0.95 sequential requests per second, or about 57 per minute. This is a
> single-process sequential benchmark; concurrent capacity has not yet been measured.
> The model uses approximately 109 MB for parameters and approximately 579–609 MB
> peak process memory in the tested environment. A production capacity number
> requires a load test on the target hardware.

## 19. What is the next performance test?

Run a controlled load test on the target deployment machine with:

- 1, 2, 4, and 8 concurrent clients;
- at least 100 requests per concurrency level;
- warm-up requests discarded;
- P50, P95, and P99 latency;
- successful requests per second;
- peak RAM and CPU utilization;
- error and timeout counts.

The result should be reported as a table rather than a single optimistic throughput
number.
