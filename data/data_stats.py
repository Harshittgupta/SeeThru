"""Dataset statistics reporter for SEETHRU.

Prints class distribution, total image count, and per-split sizes for a
deepfake dataset laid out as ``root/real`` and ``root/fake``.

Usage::

    python data/data_stats.py /path/to/dataset
    python data/data_stats.py /path/to/dataset --no-balance --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from dataset_manager import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    DeepfakeDataset,
    default_identity_fn,
)


def _pct(part: int, whole: int) -> str:
    return f"{(100.0 * part / whole):.1f}%" if whole else "0.0%"


def raw_class_distribution(root: Path) -> Dict[str, int]:
    """Count images on disk per class, before any balancing or splitting."""
    from dataset_manager import IMAGE_EXTENSIONS

    counts: Dict[str, int] = {}
    for class_name in CLASS_TO_LABEL:
        class_dir = root / class_name
        if not class_dir.is_dir():
            counts[class_name] = 0
            continue
        counts[class_name] = sum(
            1
            for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return counts


def print_report(
    root: Path,
    split_ratios=(0.7, 0.15, 0.15),
    balance: bool = True,
    seed: int = 42,
) -> None:
    print("=" * 60)
    print(f"SEETHRU dataset statistics: {root}")
    print("=" * 60)

    # --- Raw class distribution (on disk) ---------------------------------
    raw = raw_class_distribution(root)
    raw_total = sum(raw.values())
    print("\nClass distribution (on disk):")
    for class_name, count in raw.items():
        print(f"  {class_name:<6} {count:>8}  ({_pct(count, raw_total)})")
    print(f"  {'total':<6} {raw_total:>8}")

    if raw_total == 0:
        print("\nNo images found — nothing else to report.")
        return

    # --- Identity counts --------------------------------------------------
    from dataset_manager import IMAGE_EXTENSIONS

    identities = set()
    for class_name in CLASS_TO_LABEL:
        class_dir = root / class_name
        if class_dir.is_dir():
            for p in class_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                    identities.add(default_identity_fn(p))
    print(f"\nUnique identities (subjects): {len(identities)}")

    # --- Per-split sizes --------------------------------------------------
    print(
        f"\nSplit sizes  (ratios={split_ratios}, "
        f"balance={balance}, seed={seed}):"
    )
    header = f"  {'split':<7}{'total':>8}{'real':>8}{'fake':>8}{'ids':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    grand_total = 0
    for split in ("train", "val", "test"):
        ds = DeepfakeDataset(
            root,
            split=split,
            split_ratios=split_ratios,
            balance=balance,
            seed=seed,
        )
        counts = ds.class_counts()
        n = len(ds)
        grand_total += n
        print(
            f"  {split:<7}{n:>8}"
            f"{counts['real']:>8}{counts['fake']:>8}"
            f"{len(ds.identities()):>7}"
        )
    print("  " + "-" * (len(header) - 2))
    print(f"  {'total':<7}{grand_total:>8}")

    # --- Identity-separation sanity check ---------------------------------
    split_ids = {}
    for split in ("train", "val", "test"):
        ds = DeepfakeDataset(
            root,
            split=split,
            split_ratios=split_ratios,
            balance=balance,
            seed=seed,
        )
        split_ids[split] = set(ds.identities())

    overlaps = []
    splits = ("train", "val", "test")
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            shared = split_ids[splits[i]] & split_ids[splits[j]]
            if shared:
                overlaps.append((splits[i], splits[j], len(shared)))

    print("\nIdentity separation check:")
    if overlaps:
        for a, b, n in overlaps:
            print(f"  WARNING: {n} identities shared between {a} and {b}")
    else:
        print("  OK — no identity appears in more than one split.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Report deepfake dataset stats.")
    parser.add_argument("root", help="Dataset root containing real/ and fake/")
    parser.add_argument(
        "--split-ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Report raw split sizes without 50:50 balancing.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print_report(
        Path(args.root),
        split_ratios=tuple(args.split_ratios),
        balance=not args.no_balance,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
