"""Processed-dataset manifest for SEETHRU (BUILD_PLAN T41).

One row per processed video. The row is the *only* place split membership,
identity, and provenance live -- the `.npy` files beside it are just pixels.

**Why JSONL and not Parquet.** The audit suggested Parquet; measured, the
manifest tops out at ~8 MB (613 bytes/row x 13k videos). Parquet's wins --
columnar reads, compression, predicate pushdown -- all arrive somewhere north of
a million rows, and it would cost pandas + pyarrow (~100 MB) to get them. JSONL
needs no new dependency, is greppable and diffable, and is **append-only**, which
makes resumable extraction (T43) nearly free: crash, re-run, skip what is already
in the manifest.

**Why a manifest at all**, rather than an image-folder tree:

* ``prepare_datasets.py`` used to accumulate every decoded clip in one Python
  list and pickle it whole -- computed at **3.4 GB for ff_train and 15.7 GB for
  celebdf_test, resident in RAM** before the first byte was written, plus a copy
  during ``pickle.dump``. Then ``DeepfakeVideoDataset`` re-loaded the whole thing
  in **every DataLoader worker** (Windows spawns, so each gets a full copy).
* Pickle is also arbitrary-code-execution on load.
* And the image model had no data path at all (T41b): ``DeepfakeDataset`` wants
  ``root/real/*.jpg``, which nothing produced.

One extraction writes ``.npy`` + this manifest; the frame view (stage 1) and the
clip view (stage 2) then read the *same* manifest. That means one face-detection
pass instead of two -- and that pass is the 1.5-3 h bottleneck -- with identity
and manipulation coming from the manifest rather than being re-parsed out of
filenames, so the T15 leak fix and the per-method breakdown apply to both stages
for free.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.jsonl"
MANIFEST_VERSION = 1

REAL, FAKE = 0, 1
VALID_SPLITS = ("train", "val", "test")


@dataclass
class ManifestRow:
    """One processed video."""

    # Identity of the artifact
    video_path: str          # source video, for provenance
    npy_path: str            # relative to the manifest's directory
    dataset: str             # "ffpp" | "celebdf"
    split: str               # train | val | test

    # Labels and grouping
    label: int               # 0 real, 1 fake
    identity: str            # canonical GROUP key (union-find, T15)
    identities: list[str]    # every identity in the video -- BOTH ids of a swap
    manipulation: str        # "Deepfakes" ... | "none" for real

    # Timing / provenance (T40)
    n_frames: int
    fps: float = 0.0
    duration_s: float = 0.0
    total_frames: int = 0
    source_indices: list[int] = field(default_factory=list)

    # Quality
    n_missing: int = 0            # frames where no face was detected
    face_rate: float = 1.0
    interpolated: list[bool] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> ManifestRow:
        return cls(**json.loads(line))

    def resolve(self, root: Path) -> Path:
        """Absolute path to this row's .npy."""
        return root / self.npy_path


def write_manifest(rows: list[ManifestRow], path: Path) -> Path:
    """Write a manifest atomically (tmp + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(row.to_json() + "\n")
    tmp.replace(path)
    logger.info("Wrote %d manifest rows -> %s", len(rows), path)
    return path


def append_row(row: ManifestRow, path: Path) -> None:
    """Append one row, flushing immediately.

    Flushed per row on purpose: this file IS the resume state (T43). A 3-hour
    extraction that crashes at hour 2 must not lose hour 2's work to a buffer.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(row.to_json() + "\n")
        fh.flush()


def read_manifest(path: Path) -> list[ManifestRow]:
    """Read a manifest, skipping blank lines. Returns [] if absent."""
    path = Path(path)
    if not path.is_file():
        return []
    rows: list[ManifestRow] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(ManifestRow.from_json(line))
            except (json.JSONDecodeError, TypeError) as exc:
                # A torn last line is expected after a crash mid-append. Warn and
                # carry on: the video is simply re-processed, which is exactly
                # what resumability is for.
                logger.warning("%s:%d is unreadable (%s); skipping", path, lineno, exc)
    return rows


def done_videos(path: Path) -> set[str]:
    """Source videos already in the manifest -- the skip set for a resumed run."""
    return {row.video_path for row in read_manifest(path)}


def filter_rows(
    rows: list[ManifestRow],
    split: str | None = None,
    dataset: str | None = None,
) -> list[ManifestRow]:
    out = rows
    if split is not None:
        out = [r for r in out if r.split == split]
    if dataset is not None:
        out = [r for r in out if r.dataset == dataset]
    return out


def summarize(rows: list[ManifestRow]) -> str:
    """A human-readable table. Reports the label prior, which is the thing to
    look at: a skew here means downstream metrics are measuring the prior."""
    if not rows:
        return "  (empty manifest)"

    by: dict[tuple[str, str], list[ManifestRow]] = {}
    for row in rows:
        by.setdefault((row.dataset, row.split), []).append(row)

    lines = [f"  {'dataset/split':<20}{'total':>7}{'real':>7}{'fake':>7}{'ids':>6}{'face_rate':>11}"]
    lines.append("  " + "-" * 58)
    for (dataset, split), group in sorted(by.items()):
        n_real = sum(r.label == REAL for r in group)
        ids = {i for r in group for i in r.identities}
        rate = sum(r.face_rate for r in group) / len(group)
        lines.append(
            f"  {dataset + '/' + split:<20}{len(group):>7}{n_real:>7}"
            f"{len(group) - n_real:>7}{len(ids):>6}{rate:>10.1%}"
        )
    return "\n".join(lines)


def validate(rows: list[ManifestRow], root: Path) -> list[str]:
    """Structural checks → list of problems (empty == fine).

    Deliberately NOT a leakage audit -- ``data/audit_splits.py`` (T17) owns that
    and is the thing CI runs. This catches the mundane failures that make a
    manifest unusable: missing files, bad splits, a lost identity.
    """
    problems: list[str] = []
    if not rows:
        return ["manifest is empty"]

    seen_npy: set[str] = set()
    for row in rows:
        if row.split not in VALID_SPLITS:
            problems.append(f"{row.video_path}: bad split {row.split!r}")
        if row.label not in (REAL, FAKE):
            problems.append(f"{row.video_path}: bad label {row.label!r}")
        if not row.identities:
            problems.append(
                f"{row.video_path}: no identities -- splitting cannot be verified"
            )
        if row.npy_path in seen_npy:
            problems.append(f"duplicate npy_path {row.npy_path!r}")
        seen_npy.add(row.npy_path)
        if not row.resolve(root).is_file():
            problems.append(f"{row.npy_path}: file missing on disk")
        if len(row.interpolated) not in (0, row.n_frames):
            problems.append(
                f"{row.video_path}: interpolated has {len(row.interpolated)} "
                f"entries for {row.n_frames} frames"
            )
    return problems
