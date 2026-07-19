"""Split-leakage auditor for SEETHRU (BUILD_PLAN T17).

Run this in CI, and before every training run. It exits non-zero on any finding,
so a leak breaks the build instead of quietly inflating a number in a report.

WHY THIS EXISTS AS A SEPARATE, RUNNABLE THING:

The loaders already fix the known leaks (T15). This is the independent check that
they *stayed* fixed. Leakage is the failure mode with the worst signal-to-noise
in this whole project -- it makes results look BETTER, so nothing about your
training curves will ever hint that it happened. By the time you notice, the
number is in a slide deck.

The audits, and what each would have caught:

  1. Non-empty splits          -- the collapse of T10/T13: val/test silently
                                  empty, and every other check trivially passing
                                  over them.
  2. Identity disjointness     -- counting BOTH ids of every fake. The T15 leak:
                                  033_097 in train while 097's real video is in
                                  test.
  3. Path disjointness         -- the same file in two splits.
  4. Group integrity           -- an identity group straddling a split boundary,
                                  i.e. a swap chain that got cut.
  5. Class balance             -- train should be ~50:50; eval splits should NOT
                                  be (balancing them discards data for nothing).
  6. Manifest fingerprint      -- a hash of the split assignment, so "did the
                                  split change?" is answerable after the fact.

Usage::

    python data/audit_splits.py --images data/dummy/images
    python data/audit_splits.py --ffpp /path/to/FaceForensics++
    python data/audit_splits.py --celebdf /path/to/Celeb-DF-v2
    python data/audit_splits.py --ffpp ... --celebdf ... --images ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

SPLITS = ("train", "val", "test")
REAL, FAKE = 0, 1


# --------------------------------------------------------------------------- #
# Findings
# --------------------------------------------------------------------------- #
@dataclass
class Audit:
    """Accumulates findings. Any FAIL means a non-zero exit."""

    name: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)

    @property
    def ok(self) -> bool:
        return not self.failures

    def report(self) -> None:
        status = "PASS" if self.ok else "FAIL"
        print(f"\n{'=' * 70}\n{self.name}: {status}\n{'=' * 70}")
        for note in self.notes:
            print(f"  .  {note}")
        for warning in self.warnings:
            print(f"  !  WARN {warning}")
        for failure in self.failures:
            print(f"  X  FAIL {failure}")


# --------------------------------------------------------------------------- #
# Generic checks -- operate on {split: [video dicts]}
# --------------------------------------------------------------------------- #
def _ids_of(videos: Sequence[dict]) -> set[str]:
    """Every identity in every video, counting BOTH ids of a swap.

    Using video["identity"] (the group key) here instead would be exactly the
    bug this file exists to catch.
    """
    out: set[str] = set()
    for video in videos:
        out.update(video["identities"])
    return out


def audit_split_mapping(name: str, per_split: dict[str, list[dict]]) -> Audit:
    audit = Audit(name)

    # 1. Non-empty. Must come FIRST: every check below is vacuously true over
    #    empty sets, which is precisely how the original bug hid.
    empty = [s for s, v in per_split.items() if not v]
    if empty:
        audit.fail(
            f"split(s) {empty} are EMPTY -- every check below would pass "
            f"vacuously, so nothing here is evidence of anything."
        )
        return audit

    for split, videos in per_split.items():
        n_real = sum(v["label"] == REAL for v in videos)
        audit.note(
            f"{split:<5} {len(videos):>5} videos  "
            f"{n_real:>4} real / {len(videos) - n_real:<4} fake  "
            f"{len(_ids_of(videos)):>3} identities  "
            f"{len({v['identity'] for v in videos}):>3} groups"
        )

    # 2. Identity disjointness -- the headline check.
    ids = {split: _ids_of(videos) for split, videos in per_split.items()}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            shared = ids[a] & ids[b]
            if shared:
                audit.fail(
                    f"IDENTITY LEAK: {len(shared)} identit(ies) in both "
                    f"{a} and {b}: {sorted(shared)[:8]}"
                )

    # 3. Path disjointness.
    paths = {split: {v["path"] for v in videos} for split, videos in per_split.items()}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1 :]:
            shared = paths[a] & paths[b]
            if shared:
                audit.fail(
                    f"SAMPLE LEAK: {len(shared)} video(s) in both {a} and {b}: "
                    f"{sorted(shared)[:3]}"
                )

    # 4. Group integrity: a group must not straddle splits. This catches a cut
    #    swap chain, which the raw identity check can miss if the chain's
    #    members happen not to co-occur pairwise.
    group_to_splits: dict[str, set[str]] = {}
    for split, videos in per_split.items():
        for video in videos:
            group_to_splits.setdefault(video["identity"], set()).add(split)
    straddling = {g: s for g, s in group_to_splits.items() if len(s) > 1}
    if straddling:
        audit.fail(
            f"GROUP STRADDLE: {len(straddling)} identity group(s) span multiple "
            f"splits (a swap chain was cut): "
            f"{[(g, sorted(s)) for g, s in list(straddling.items())[:3]]}"
        )

    # 5. Balance policy: train ~50:50, eval splits left at their natural prior.
    train = per_split["train"]
    n_real = sum(v["label"] == REAL for v in train)
    n_fake = len(train) - n_real
    if n_real and n_fake:
        skew = abs(n_real - n_fake) / max(n_real, n_fake)
        if skew > 0.01:
            audit.warn(
                f"train is {n_real} real / {n_fake} fake ({skew:.1%} skew). "
                f"Expected ~50:50, or a class-weighted loss to compensate."
            )
    for split in ("val", "test"):
        videos = per_split[split]
        n_r = sum(v["label"] == REAL for v in videos)
        if n_r and n_r == len(videos) - n_r:
            audit.warn(
                f"{split} is exactly 50:50, which suggests it was balanced. "
                f"Eval splits should keep their natural prior: balancing "
                f"discards data and skews the per-manipulation breakdown for "
                f"no benefit (AUC is prevalence-insensitive)."
            )

    return audit


def split_fingerprint(per_split: dict[str, list[dict]]) -> str:
    """Stable hash of the split assignment.

    Log this with every run. When a metric moves and you cannot explain why,
    the first question is "did the split change?" -- and without this, that
    question has no answer after the fact.
    """
    payload = {
        split: sorted(Path(v["path"]).name for v in videos)
        for split, videos in sorted(per_split.items())
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Dataset-specific entry points
# --------------------------------------------------------------------------- #
def audit_ffpp(ff_root: Path, seed: int = 42) -> Audit:
    from ml.preprocessing.video_processor import FFPlusPlusLoader

    loader = FFPlusPlusLoader(ff_root)
    per_split = {
        split: loader.get_split(split, seed=seed, balance=(split == "train"))
        for split in SPLITS
    }
    audit = audit_split_mapping(f"FaceForensics++ ({ff_root})", per_split)

    if loader.load_official_splits() is None:
        audit.warn(
            "official splits/*.json not found -- using the grouped fallback. "
            "Results will not be comparable to published FF++ baselines."
        )
    else:
        audit.note("using FF++ official splits/*.json")

    audit.note(f"split fingerprint: {split_fingerprint(per_split)}")
    return audit


def audit_celebdf(celebdf_root: Path) -> Audit:
    from ml.preprocessing.video_processor import CelebDFLoader

    loader = CelebDFLoader(celebdf_root)
    audit = Audit(f"Celeb-DF v2 ({celebdf_root})")

    videos = loader.get_video_paths(official_only=True)
    if not videos:
        audit.fail("no videos found")
        return audit

    n_real = sum(v["label"] == REAL for v in videos)
    n_fake = len(videos) - n_real
    audit.note(f"test set: {len(videos)} videos ({n_real} real / {n_fake} fake)")

    if loader.load_testing_list() is None:
        audit.fail(
            "List_of_testing_videos.txt is missing, so the FULL corpus is in "
            "use. That set is ~86% fake -- an always-'fake' model scores ~86% "
            "on it -- and the number is incomparable to every published result."
        )
    else:
        audit.note("using the official List_of_testing_videos.txt benchmark subset")

    missing = [v["path"] for v in videos if not Path(v["path"]).is_file()]
    if missing:
        audit.fail(
            f"{len(missing)}/{len(videos)} listed videos are absent from disk "
            f"(first: {missing[0]})"
        )

    # Celeb-DF is held out entirely, so there is no train/test boundary within it
    # to leak across. The thing that CAN go wrong is label polarity: their list
    # uses real=1/fake=0, the inverse of ours.
    if n_fake and n_real and n_fake < n_real:
        audit.warn(
            f"more real ({n_real}) than fake ({n_fake}) -- the official subset "
            f"is 178 real / 340 fake, so this smells like inverted label "
            f"polarity (Celeb-DF encodes real=1, we use real=0)."
        )
    return audit


def audit_images(images_root: Path, seed: int = 42) -> Audit:
    from data.dataset_manager import build_splits

    audit = Audit(f"Image dataset ({images_root})")
    try:
        splits = build_splits(images_root, seed=seed, balance=False)
    except ValueError as exc:
        audit.fail(f"could not build splits: {str(exc).splitlines()[0]}")
        return audit

    per_split = {
        name: [
            {
                "path": str(p),
                "label": label,
                "identity": identity,
                "identities": (identity,),
            }
            for p, label, identity in ds.samples
        ]
        for name, ds in splits.items()
    }
    result = audit_split_mapping(f"Image dataset ({images_root})", per_split)
    result.note(f"split fingerprint: {split_fingerprint(per_split)}")
    return result


# --------------------------------------------------------------------------- #
def run_audits(
    images: Path | None = None,
    ffpp: Path | None = None,
    celebdf: Path | None = None,
    seed: int = 42,
) -> int:
    """Run the requested audits. Returns a process exit code."""
    audits: list[Audit] = []
    if images:
        audits.append(audit_images(images, seed=seed))
    if ffpp:
        audits.append(audit_ffpp(ffpp, seed=seed))
    if celebdf:
        audits.append(audit_celebdf(celebdf))

    if not audits:
        print("Nothing to audit. Pass --images, --ffpp and/or --celebdf.")
        return 2

    for audit in audits:
        audit.report()

    failed = [a.name for a in audits if not a.ok]
    print(f"\n{'=' * 70}")
    if failed:
        print(f"AUDIT FAILED: {len(failed)}/{len(audits)} -> {failed}")
        print("Do not train on this. Any metric produced would be fiction.")
        return 1
    print(f"AUDIT PASSED: {len(audits)}/{len(audits)}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit dataset splits for identity leakage (BUILD_PLAN T17).",
    )
    parser.add_argument("--images", type=Path, help="Image root with real/ and fake/")
    parser.add_argument("--ffpp", type=Path, help="FaceForensics++ root")
    parser.add_argument("--celebdf", type=Path, help="Celeb-DF v2 root")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raise SystemExit(
        run_audits(
            images=args.images, ffpp=args.ffpp, celebdf=args.celebdf, seed=args.seed
        )
    )


if __name__ == "__main__":
    main()
