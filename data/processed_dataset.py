"""Datasets over the processed manifest (BUILD_PLAN T41/T41b).

**One extraction, two views.** Both datasets read the *same* manifest and the
same `.npy` files:

* :class:`DeepfakeFrameDataset` -- each **frame** is a sample. Stage 1 (image).
* :class:`DeepfakeClipDataset`  -- each **clip** is a sample. Stage 2 (video).

This closes the T41b gap. Stage 1 previously had no data path from the real
dataset at all: ``DeepfakeDataset`` wants ``root/real/*.jpg`` image folders, and
``prepare_datasets.py`` produces 16-frame sequences. Nothing bridged them, and it
worked only because ``create_dummy_dataset.py`` hand-writes the image-folder
layout. It would have surfaced the moment real data landed -- the worst possible
moment.

Sharing one extraction matters for three reasons, in increasing order:

1. Face detection is the 1.5-3 h bottleneck. Doing it once beats twice.
2. Dumping JPEGs for stage 1 would **re-encode** the crops, overwriting the exact
   compression artifacts the frequency branch is trained to read (T41).
3. ``identity`` and ``manipulation`` come from the manifest rather than being
   re-parsed from filenames -- so the T15 leak fix and the per-method breakdown
   apply to both stages for free, and cannot drift apart.

``.npy`` files are opened with ``mmap_mode="r"``: a clip is 16x224x224x3 = 2.4 MB,
and the frame view touches one frame of it. Memory-mapping reads the ~150 KB it
needs instead of the whole array, and lets the OS page cache do the work across
DataLoader workers.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.manifest import MANIFEST_NAME, ManifestRow, filter_rows, read_manifest

logger = logging.getLogger(__name__)

REAL, FAKE = 0, 1


class _ManifestDataset(Dataset):
    """Shared manifest loading/filtering for the frame and clip views."""

    def __init__(
        self,
        root: str | Path,
        split: str | None = "train",
        dataset: str | None = None,
        transform: Callable | None = None,
        manifest_name: str = MANIFEST_NAME,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform

        manifest_path = self.root / manifest_name
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No manifest at {manifest_path}. Run:\n"
                f"  python ml/preprocessing/prepare_datasets.py "
                f"--ff_root <FaceForensics++> --celebdf_root <Celeb-DF-v2>"
            )

        self.rows: list[ManifestRow] = filter_rows(
            read_manifest(manifest_path), split=split, dataset=dataset
        )
        if not self.rows:
            # Same lesson as T11: an empty split must be loud. A DataLoader over
            # an empty dataset yields no batches, so training "succeeds" having
            # never seen the data.
            raise ValueError(
                f"No manifest rows for split={split!r} dataset={dataset!r} in "
                f"{manifest_path}. Available: "
                f"{sorted({(r.dataset, r.split) for r in read_manifest(manifest_path)})}"
            )

    # ------------------------------------------------------------------ #
    def _load(self, row: ManifestRow) -> np.ndarray:
        """Memory-map a clip's crops → ``(T, H, W, 3)`` uint8 BGR."""
        return np.load(row.resolve(self.root), mmap_mode="r")

    def _to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        # Crops are stored BGR (OpenCV's order, straight from the detector);
        # the transforms and ImageNet stats expect RGB.
        rgb = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2RGB)
        out = self.transform(rgb)
        if isinstance(out, dict):  # a raw albumentations Compose
            out = out["image"]
        return out

    def identities(self) -> list[str]:
        return sorted({i for r in self.rows for i in r.identities})

    def labels(self) -> list[int]:
        raise NotImplementedError


class DeepfakeFrameDataset(_ManifestDataset):
    """Each FRAME is a sample → ``(tensor, label)``. Stage 1 (T41b).

    Returns a plain tuple, matching ``DeepfakeDataset`` and the torchvision
    convention every training loop unpacks. Per-sample metadata is available via
    :meth:`metadata` without decoding the image.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Flatten (row, frame) -> a single index space. Built once; it is just
        # (int, int) pairs, so ~16 x 5000 = 80k tuples is nothing.
        self.index: list[tuple[int, int]] = [
            (r, f) for r, row in enumerate(self.rows) for f in range(row.n_frames)
        ]
        logger.info(
            "Frame view: %d frames from %d videos (split=%s)",
            len(self.index), len(self.rows), self.split,
        )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        row_idx, frame_idx = self.index[i]
        row = self.rows[row_idx]
        return self._to_tensor(self._load(row)[frame_idx]), row.label

    def metadata(self, i: int) -> dict:
        """Per-frame metadata -- no pixels decoded.

        ``interpolated`` is here because it matters: a copied frame is not an
        independent observation, and an eval that treats it as one is counting
        the same evidence twice.
        """
        row_idx, frame_idx = self.index[i]
        row = self.rows[row_idx]
        return {
            "video_path": row.video_path,
            "identity": row.identity,
            "identities": row.identities,
            "manipulation": row.manipulation,
            "label": row.label,
            "frame_index": frame_idx,
            "source_index": (
                row.source_indices[frame_idx] if row.source_indices else frame_idx
            ),
            "t_seconds": (
                row.source_indices[frame_idx] / row.fps
                if row.fps > 0 and row.source_indices
                else None
            ),
            "interpolated": (
                row.interpolated[frame_idx] if row.interpolated else False
            ),
        }

    def labels(self) -> list[int]:
        """Per-sample labels, for class weights / a sampler (T16)."""
        return [self.rows[r].label for r, _ in self.index]

    def video_ids(self) -> list[str]:
        """Per-sample video id, so frame scores can be pooled to video level (T28)."""
        return [self.rows[r].video_path for r, _ in self.index]


class DeepfakeClipDataset(_ManifestDataset):
    """Each CLIP is a sample → dict with ``frames (T,3,H,W)``. Stage 2.

    Applies ONE transform draw across the whole clip (T38) -- see
    :meth:`_apply_clip`. Re-sampling per frame injects flip/crop jitter that
    dwarfs the temporal signal the BiLSTM exists to find.
    """

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        clip = self._load(row)  # (T, H, W, 3) BGR
        rgb = [
            cv2.cvtColor(np.ascontiguousarray(clip[t]), cv2.COLOR_BGR2RGB)
            for t in range(clip.shape[0])
        ]
        frames = torch.stack(self._apply_clip(rgb), dim=0)  # (T, 3, H, W)
        return {
            "frames": frames,
            "label": row.label,
            "identity": row.identity,
            "manipulation": row.manipulation or "none",
            "video_path": row.video_path,
        }

    def _apply_clip(self, rgb_frames: list[np.ndarray]) -> list[torch.Tensor]:
        """One parameter draw for the whole clip (T38)."""
        import albumentations as A

        pipeline = getattr(self.transform, "transform", None)
        if pipeline is None and isinstance(self.transform, A.BaseCompose):
            pipeline = self.transform
        if not isinstance(pipeline, A.BaseCompose):
            return [self._to_tensor_rgb(f) for f in rgb_frames]

        if not hasattr(self, "_replay"):
            self._replay = A.ReplayCompose(list(pipeline.transforms))

        first = self._replay(image=rgb_frames[0])
        params = first["replay"]
        out = [first["image"]]
        out.extend(
            A.ReplayCompose.replay(params, image=f)["image"] for f in rgb_frames[1:]
        )
        return out

    def _to_tensor_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        out = self.transform(rgb)
        return out["image"] if isinstance(out, dict) else out

    def labels(self) -> list[int]:
        return [row.label for row in self.rows]
