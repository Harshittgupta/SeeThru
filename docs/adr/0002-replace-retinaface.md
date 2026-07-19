# ADR 0002: Replace RetinaFace/TensorFlow with SCRFD on onnxruntime

- **Status:** Proposed — deferred until after the first successful training run
- **Date:** 2026-07-16
- **Task:** BUILD_PLAN T45
- **Affects:** T41/T43 (extraction throughput), T54 (backend VRAM), T82 (dependency tree), T86 (deployment)

## Context

`retina-face` is a TensorFlow package. SEETHRU is a PyTorch project. It is the
only reason TensorFlow is in the dependency tree at all, and it has surfaced as
the root cause of a distinct problem in almost every milestone so far:

| Where | What it cost | Measured |
|---|---|---|
| T39 | Importing `ml.preprocessing.video_processor` — a module that lists directories — loaded all of TensorFlow | **17.9 s → 4.45 s** after making the import lazy; **11.0 s of it was TF**. Test suite 110 s → 12.9 s. |
| T43 | Every extraction worker builds its own detector, so every worker loads TF | **~600 MB RSS per worker**; ~2.4 GB at `--workers 4`, for a process pool whose real work is decoding video |
| T41/T43 | The API is single-image: `detect_faces(one_image)`. No batching. | The 1.5–3 h bottleneck of the whole extraction pass |
| T54 | The backend would hold TF *and* PyTorch on one GPU | TF preallocates VRAM by default and starves torch; needs `TF_FORCE_GPU_ALLOW_GROWTH=true` as a workaround |
| T82 | TF is the numpy-2 laggard | Forces `numpy<2` across the entire project (see requirements.txt) |
| T86 | CPU inference | RetinaFace-ResNet50 is **0.5–2 s per 1080p frame** on CPU and *dominates* end-to-end latency; EfficientNet-B3 is only 40–80 ms |

Each of these was worked around individually. The workarounds are all correct,
and none of them address the cause.

## Decision

**Replace RetinaFace with SCRFD via `onnxruntime`, but not yet.** Do it after the
first end-to-end training run has produced a real number.

## Rationale for the swap

- **Deletes TensorFlow entirely.** One framework, one CUDA context, one memory
  allocator. The TF/torch VRAM contention class disappears rather than being
  configured around.
- **Batching.** SCRFD via onnxruntime takes a batch; RetinaFace takes one image.
  The extraction bottleneck is per-image detector calls, so this is where the
  1.5–3 h goes.
- **~10× faster on CPU**, which is what makes a CPU-only demo deployment viable
  at all (T86).
- **Unblocks numpy 2.x**, removing a pin that currently constrains every other
  package.
- **Smaller images.** ~600 MB of TF leaves both Docker images.

## Rationale for the delay

This is the reasoning that matters, because the swap looks like an obvious win:

**Changing the face detector changes the data.** Different detectors produce
different crops — different boxes, different alignment, different margins. A
model trained on RetinaFace crops and evaluated on SCRFD crops is being evaluated
on a distribution it never saw, and the resulting drop would be indistinguishable
from a modelling failure. So the swap means **re-running the full extraction**
(1.5–3 h) and **retraining**, or else carefully measuring the crop-distribution
shift.

Doing that *before* we have any baseline means changing two things at once and
having nothing to compare against. The correct order is: get one real
cross-dataset number with the current detector, then swap, then re-measure. If
the number moves, we know why.

There is also a scheduling argument. The GPU server is the scarce resource, and
the EULA data has landed. Extraction throughput is a *wall-clock* problem we can
absorb once (a one-time 1.5–3 h), whereas the detector swap is a correctness risk
to the headline result. Absorb the throughput cost; do not risk the number.

## Consequences of deferring

- `--workers 4` stays the practical ceiling for extraction (~2.4 GB of TF).
- `numpy<2` stays pinned. `requirements.txt` records the exit criterion.
- The backend must set `TF_FORCE_GPU_ALLOW_GROWTH=true` (already in
  `docker-compose.yml`) and warm up RetinaFace at startup (T54) — it builds and
  globally caches its TF model on first call, which is a multi-second stall and
  is not thread-safe.
- CPU-only deployment stays impractical for video (20–60 s/clip). T86's
  recommendation (local GPU + Cloudflare Tunnel) is unaffected.

## Do it when

Any of:
- The first FF++ → Celeb-DF number exists and is recorded (the baseline to
  compare against).
- Extraction has to be re-run for another reason anyway — swap in the same pass,
  since the cost is the extraction, not the code.
- A CPU-only deployment becomes a real requirement (T86).

## Implementation sketch

- `insightface`'s SCRFD, or the standalone `scrfd` ONNX weights, via
  `onnxruntime-gpu`.
- Keep the `FaceDetector` interface exactly (`detect_and_align(image) -> list[np.ndarray]`)
  and add `detect_and_align_batch(images)`. The current interface is already the
  right shape; only the implementation changes.
- Drop `retina-face` and `tf-keras` from `requirements.txt`; unpin numpy.
- **Measure the crop-distribution shift before retraining**: run both detectors
  over the same 100 videos and compare box IoU and alignment offsets. If they
  agree closely, a full retrain may not be needed. If they do not, that
  disagreement is precisely the thing that would otherwise have been misread as a
  modelling regression.
