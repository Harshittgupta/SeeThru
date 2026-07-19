"""Extract face crops from the raw video datasets (BUILD_PLAN T41/T43).

    python ml/preprocessing/prepare_datasets.py \
        --ff_root data/raw/FaceForensics++ \
        --celebdf_root data/raw/Celeb-DF-v2 \
        --out data/processed --workers 4

Writes one ``.npy`` per video (16 aligned 224x224 BGR crops, uint8) plus a
``manifest.jsonl`` describing every one. Both the frame view (stage 1) and the
clip view (stage 2) read that manifest -- see data/processed_dataset.py.

WHAT THIS REPLACES, AND WHY:

The previous version accumulated every decoded clip in one Python list and
pickled it whole. Computed: **3.4 GB for ff_train, 15.7 GB for celebdf_test,
resident in RAM before the first byte was written**, plus a full copy during
``pickle.dump``. It then had to be re-loaded in *every* DataLoader worker
(Windows spawns, so each gets its own copy). It was also all-or-nothing: a crash
at video 4,900 of 5,000 lost everything. And pickle is arbitrary-code-execution
on load.

Now: stream to disk per video, append to the manifest per video, skip what is
already there on re-run. A crash costs one video.

**Failures are recorded, not counted.** The old code incremented a `skipped`
integer. That hides the question that actually matters -- were the failures
disproportionately real or fake? A face detector that fails more often on one
class silently shifts the label prior, and every metric downstream inherits it,
while the run reports a cheerful "skipped: 400". ``failures.csv`` names each one.

RUNTIME: with `--workers 4` and the fixes in T39/T42, roughly 1.5-3 h for FF++
and 1-2 h for Celeb-DF. The old code (serial, per-frame keyframe seeks, TF
RetinaFace) measured out at 20-40 h.

⚠️ Each worker builds its own FaceDetector, and RetinaFace drags in TensorFlow
(~600 MB RSS per worker). At --workers 4 that is ~2.4 GB for TF alone. This is
the strongest argument for T45 (SCRFD on onnxruntime), which deletes TF outright
and adds batching.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.manifest import (  # noqa: E402
    MANIFEST_NAME,
    ManifestRow,
    append_row,
    done_videos,
    read_manifest,
    summarize,
    validate,
)
from ml.preprocessing.video_processor import (  # noqa: E402
    CelebDFLoader,
    DatasetConfig,
    FFPlusPlusLoader,
    VideoProcessor,
)

logger = logging.getLogger(__name__)

FAILURES_NAME = "failures.csv"

# Set once per worker process. Building a FaceDetector costs a TensorFlow import
# (~600 MB), so it must happen once per process, never once per video.
_PROCESSOR: VideoProcessor | None = None


def _init_worker(n_frames: int, face_size: int) -> None:
    global _PROCESSOR
    _PROCESSOR = VideoProcessor(n_frames=n_frames, face_size=face_size)


def _npy_path_for(video: dict) -> Path:
    """Unique relative path for a video's crops.

    Includes the manipulation because FF++ **reuses stems across methods**:
    `033_097` exists under Deepfakes, Face2Face, FaceSwap AND NeuralTextures.
    Keying on the stem alone would have four videos overwrite one file -- losing
    3/4 of the fakes silently, and leaving four manifest rows pointing at one
    array.
    """
    stem = Path(video["path"]).stem
    bucket = video.get("manipulation") or "original"
    return Path(video["dataset"]) / bucket / f"{stem}.npy"


def _process_one(video: dict, out_root: str, n_frames: int, face_size: int) -> dict:
    """Extract one video → a result dict. Runs in a worker process.

    Returns ``{"ok": bool, ...}`` rather than raising: one unreadable video must
    not take down a 3-hour extraction.
    """
    global _PROCESSOR
    if _PROCESSOR is None:  # direct call (tests, --workers 1)
        _init_worker(n_frames, face_size)

    path = video["path"]
    try:
        seq = _PROCESSOR.build_face_sequence(path)
    except (OSError, ValueError) as exc:
        return {"ok": False, "video": video, "reason": f"{type(exc).__name__}: {exc}"}

    if not seq.usable:
        return {
            "ok": False,
            "video": video,
            "reason": (
                f"insufficient_faces: {seq.n_missing}/{len(seq.sample.frames)} "
                f"frames had no detectable face"
            ),
        }

    npy_rel = _npy_path_for(video)
    npy_abs = Path(out_root) / npy_rel
    npy_abs.parent.mkdir(parents=True, exist_ok=True)

    # uint8, NOT re-encoded to JPEG. Re-encoding would overwrite the compression
    # artifacts the frequency branch is trained to read -- that is the signal.
    np.save(npy_abs, np.asarray(seq.faces, dtype=np.uint8))

    return {
        "ok": True,
        "row": ManifestRow(
            video_path=path,
            npy_path=str(npy_rel).replace("\\", "/"),  # POSIX in the manifest
            dataset=video["dataset"],
            split=video["split"],
            label=video["label"],
            identity=video["identity"],
            identities=list(video.get("identities", ())),
            manipulation=video.get("manipulation") or "none",
            n_frames=len(seq.faces),
            fps=seq.sample.fps,
            duration_s=seq.sample.duration_s,
            total_frames=seq.sample.total_frames,
            source_indices=seq.sample.source_indices,
            n_missing=seq.n_missing,
            face_rate=seq.face_rate,
            interpolated=seq.interpolated,
        ),
    }


# --------------------------------------------------------------------------- #
def collect_videos(config: DatasetConfig, include_celebdf: bool = True) -> list[dict]:
    """Enumerate every video to process, tagged with its dataset and split."""
    videos: list[dict] = []

    ff = FFPlusPlusLoader(config.ff_root)
    for split in ("train", "val", "test"):
        # balance=False: balancing by downsampling throws away 75% of FF++'s
        # fakes. Extract everything; weight the loss at training time (T16).
        for video in ff.get_split(split, balance=False):
            videos.append({**video, "dataset": "ffpp", "split": split})

    if include_celebdf:
        cdf = CelebDFLoader(config.celebdf_root)
        # Celeb-DF is test-only: it is the held-out cross-dataset benchmark, and
        # one frame of it in training destroys the only number that predicts
        # real-world performance.
        for video in cdf.get_video_paths(official_only=True):
            videos.append({**video, "dataset": "celebdf", "split": "test"})

    return videos


def prepare(
    config: DatasetConfig,
    out_root: Path,
    workers: int = 4,
    include_celebdf: bool = True,
    resume: bool = True,
) -> list[ManifestRow]:
    """Extract all videos → manifest rows. Resumable, parallel, fail-soft."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / MANIFEST_NAME
    failures_path = out_root / FAILURES_NAME

    videos = collect_videos(config, include_celebdf=include_celebdf)
    logger.info("Found %d videos to process", len(videos))

    if resume:
        already = done_videos(manifest_path)
        if already:
            before = len(videos)
            videos = [v for v in videos if v["path"] not in already]
            logger.info(
                "Resuming: %d already in the manifest, %d to go",
                before - len(videos), len(videos),
            )
    if not videos:
        logger.info("Nothing to do.")
        return read_manifest(manifest_path)

    failures: list[dict] = []
    n_ok = 0

    def _record(result: dict) -> None:
        nonlocal n_ok
        if result["ok"]:
            # Append per video: this file IS the resume state, so a crash at
            # video 4,900 costs one video, not 4,900.
            append_row(result["row"], manifest_path)
            n_ok += 1
        else:
            failures.append(
                {
                    "path": result["video"]["path"],
                    "dataset": result["video"]["dataset"],
                    "split": result["video"]["split"],
                    "label": result["video"]["label"],
                    "reason": result["reason"],
                }
            )

    if workers <= 1:
        _init_worker(config.n_frames, config.face_size)
        for i, video in enumerate(videos, 1):
            _record(_process_one(video, str(out_root), config.n_frames, config.face_size))
            if i % 50 == 0:
                logger.info("  %d/%d (%d ok, %d failed)", i, len(videos), n_ok, len(failures))
    else:
        logger.info(
            "Using %d worker processes. NOTE: each builds its own RetinaFace, "
            "which imports TensorFlow (~600 MB RSS per worker, ~%.1f GB total). "
            "T45 (SCRFD on onnxruntime) would remove this entirely.",
            workers, workers * 0.6,
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(config.n_frames, config.face_size),
        ) as pool:
            futures = {
                pool.submit(
                    _process_one, v, str(out_root), config.n_frames, config.face_size
                ): v
                for v in videos
            }
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    _record(future.result())
                except Exception as exc:  # noqa: BLE001 - a worker died; keep going
                    video = futures[future]
                    logger.warning("Worker died on %s: %s", video["path"], exc)
                    failures.append(
                        {
                            "path": video["path"], "dataset": video["dataset"],
                            "split": video["split"], "label": video["label"],
                            "reason": f"worker_crash: {exc}",
                        }
                    )
                if i % 50 == 0:
                    logger.info("  %d/%d (%d ok, %d failed)", i, len(videos), n_ok, len(failures))

    if failures:
        _write_failures(failures, failures_path)

    rows = read_manifest(manifest_path)
    logger.info("\n%s", summarize(rows))

    problems = validate(rows, out_root)
    if problems:
        logger.warning("Manifest validation found %d problem(s):", len(problems))
        for p in problems[:10]:
            logger.warning("  %s", p)
    return rows


def _write_failures(failures: list[dict], path: Path) -> None:
    """Record which videos were lost and why.

    The per-class breakdown is the point. A detector that fails more on one class
    silently shifts the label prior, and every metric downstream inherits it.
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "dataset", "split", "label", "reason"])
        writer.writeheader()
        writer.writerows(failures)

    n_real = sum(f["label"] == 0 for f in failures)
    n_fake = len(failures) - n_real
    logger.warning(
        "%d videos failed (%d real / %d fake) -> %s", len(failures), n_real, n_fake, path
    )
    if n_real and n_fake:
        ratio = n_real / n_fake
        if ratio > 2 or ratio < 0.5:
            logger.warning(
                "  Failures are SKEWED by class (%.1f:1 real:fake). This shifts the "
                "label prior of every split -- inspect %s before trusting any metric.",
                ratio, path,
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Extract face crops -> .npy + manifest.jsonl (T41/T43).",
    )
    parser.add_argument("--ff_root", required=True, type=str)
    parser.add_argument("--celebdf_root", type=str, default="")
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--n_frames", type=int, default=16)
    parser.add_argument("--face_size", type=int, default=224)
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Process pool size. Each worker loads TensorFlow (~600 MB).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-process videos already in the manifest.",
    )
    args = parser.parse_args()

    config = DatasetConfig(
        ff_root=args.ff_root,
        celebdf_root=args.celebdf_root or args.ff_root,
        n_frames=args.n_frames,
        face_size=args.face_size,
    )
    prepare(
        config,
        out_root=args.out,
        workers=args.workers,
        include_celebdf=bool(args.celebdf_root),
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    # Required: ProcessPoolExecutor spawns on Windows by re-importing __main__.
    main()
