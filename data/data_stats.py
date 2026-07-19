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

# Support `python data/data_stats.py`, `python -m data.data_stats`, and import
# from tests. The bare `from dataset_manager import ...` only resolved because
# sys.path[0] happens to be data/ when run as a script.
try:
    from .dataset_manager import (
        CLASS_TO_LABEL,
        IMAGE_EXTENSIONS,
        DeepfakeDataset,
        default_identity_fn,
    )
except ImportError:  # pragma: no cover - direct-script execution
    from dataset_manager import (
        CLASS_TO_LABEL,
        IMAGE_EXTENSIONS,
        DeepfakeDataset,
        default_identity_fn,
    )


def _pct(part: int, whole: int) -> str:
    return f"{(100.0 * part / whole):.1f}%" if whole else "0.0%"


def raw_class_distribution(root: Path) -> dict[str, int]:
    """Count images on disk per class, before any balancing or splitting."""
    counts: dict[str, int] = {}
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

    # Build each split once and reuse -- the previous version constructed every
    # split twice (once for sizes, once for the separation check), doubling a
    # full disk scan for no reason.
    datasets: dict[str, DeepfakeDataset] = {}
    failures: dict[str, str] = {}
    for split in ("train", "val", "test"):
        try:
            datasets[split] = DeepfakeDataset(
                root,
                split=split,
                split_ratios=split_ratios,
                balance=balance,
                seed=seed,
            )
        except ValueError as exc:
            # DeepfakeDataset now raises on a degenerate/empty split (T11).
            # This is a *reporter*: surface the problem legibly rather than
            # dying with a traceback, which is the whole point of running it.
            failures[split] = str(exc).splitlines()[0]

    grand_total = 0
    for split in ("train", "val", "test"):
        if split in failures:
            print(f"  {split:<7}{'FAILED':>8}{'-':>8}{'-':>8}{'-':>7}")
            continue
        ds = datasets[split]
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
    #
    # This check used to be VACUOUS, and it mattered. With a broken identity_fn
    # every image collapsed to one identity, val/test came back EMPTY, and the
    # pairwise intersections were all empty-vs-empty -- so it cheerfully printed
    # "OK - no identity appears in more than one split" over a completely broken
    # split. A leak check without a non-empty check is a false green, and a false
    # green is worse than no check: it actively tells you to stop looking.
    print("\nIdentity separation check:")

    if failures:
        for split, message in failures.items():
            print(f"  FAIL: split '{split}' could not be built -- {message}")
        print("  Separation is UNVERIFIED: fix the above before trusting any metric.")
        print()
        return

    split_ids = {name: set(ds.identities()) for name, ds in datasets.items()}

    empty = [name for name, ids in split_ids.items() if not ids]
    if empty:
        print(f"  FAIL: split(s) {empty} contain ZERO identities.")
        print("  Separation is UNVERIFIED (empty sets trivially do not overlap).")
        print()
        return

    overlaps = []
    splits = ("train", "val", "test")
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            shared = split_ids[splits[i]] & split_ids[splits[j]]
            if shared:
                overlaps.append((splits[i], splits[j], sorted(shared)))

    if overlaps:
        for a, b, shared in overlaps:
            print(f"  WARNING: {len(shared)} identities shared between {a} and {b}")
            print(f"           e.g. {shared[:5]}")
    else:
        total_ids = sum(len(ids) for ids in split_ids.values())
        sizes = ", ".join(f"{name}={len(ids)}" for name, ids in split_ids.items())
        # State the evidence, not just the verdict, so "OK" is falsifiable.
        print(f"  OK — {total_ids} identities ({sizes}), none shared across splits.")
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
