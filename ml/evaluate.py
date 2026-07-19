"""Evaluation for SEETHRU (BUILD_PLAN T35).

    python ml/evaluate.py --checkpoint runs/image/best.pt
    python ml/evaluate.py --checkpoint runs/image/best.pt --cross-dataset data/processed/celebdf

**The threshold protocol, which is the whole point of this file.**

The operating point is chosen ONCE, on val, and then applied *frozen* to test and
to the cross-dataset set. It is never re-derived downstream. That flow is
enforced structurally rather than by comment: ``select_threshold`` raises on any
split but val, and every downstream ``compute_metrics`` call is passed the frozen
value explicitly.

The reason for the paranoia is that this mistake is **invisible**. Re-tuning the
threshold on test does not error, does not warn, and does not look wrong -- the
number simply comes out better than the model deserves, and stays wrong through
the write-up.

**What to expect**, so a correct result is not mistaken for a failure:

    in-dataset   (FF++ -> FF++)         video-level AUC 0.97-0.995
    cross-dataset(FF++ -> Celeb-DF)     AUC 0.65-0.75   <- THE headline
    per-method                          Deepfakes/Face2Face/FaceSwap ~0.98-0.99
                                        NeuralTextures ~0.90-0.95 (mouth only)

The ~30-point in-domain-to-cross-dataset drop is expected physics, not a bug.
Published baselines on Celeb-DF: Xception 0.653, EfficientNet-B4 0.64-0.69,
RECCE 0.687. Only blending-artifact methods (SBI 0.93, Face X-ray 0.74) break
0.80. **If you see cross-dataset AUC above ~0.95, you have a leak, not a
result** -- and :func:`sanity_commentary` says so in the report rather than
leaving you to notice.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml import checkpoint  # noqa: E402
from ml.config import Config  # noqa: E402
from ml.engine import build_criterion, evaluate, resolve_device  # noqa: E402
from ml.utils.logging import setup_logging  # noqa: E402
from ml.utils.metrics import select_threshold  # noqa: E402

logger = logging.getLogger("seethru.evaluate")

# Published Celeb-DF cross-dataset AUCs, for context in the report.
CROSS_DATASET_EXPECTED = (0.65, 0.75)
CROSS_DATASET_SUSPICIOUS = 0.95
IN_DATASET_EXPECTED = (0.97, 0.995)


def _build_loader(root: str, split: str, cfg: Config, meta: dict) -> DataLoader:
    """A loader for one split, using the checkpoint's own preprocessing.

    image_size and normalization come from ``meta`` (recorded at train time), not
    from a config file that may since have changed. Evaluating with different
    preprocessing than training used is a silent, and very effective, way to
    destroy a model's numbers.
    """
    from data.dataset_manager import DeepfakeDataset
    from ml.preprocessing.augmentation import build_val_transform

    dataset = DeepfakeDataset(
        root=root,
        split=split,
        split_ratios=cfg.data.split_ratios,
        # NEVER balance an evaluation split (T16). AUC is prevalence-insensitive,
        # so balancing buys nothing and costs 75% of FF++'s fakes plus the
        # per-method breakdown.
        balance=False,
        image_size=meta.get("image_size", cfg.data.image_size),
        transform=build_val_transform(meta.get("image_size", cfg.data.image_size)),
    )
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_root: str | None = None,
    cross_dataset_root: str | None = None,
    device_spec: str = "auto",
) -> dict[str, Any]:
    """Evaluate a checkpoint on val, test, and (optionally) a cross-dataset set.

    Returns a results dict. The threshold is selected on **val** and reused
    unchanged everywhere else.
    """
    model, meta = checkpoint.load_for_inference(checkpoint_path, map_location="cpu")
    device = resolve_device(device_spec)
    model = model.to(device)

    # Rebuild the run's config from the checkpoint so evaluation matches training.
    full = meta.get("full") or checkpoint.load_raw(checkpoint_path)["config"].get("full")
    cfg = Config.from_dict(full) if full else Config()
    root = data_root or cfg.data.root

    logger.info("=" * 72)
    logger.info("Evaluating %s", checkpoint_path)
    logger.info("  arch=%s fusion=%s epoch=%s", meta["arch"], meta["fusion"], meta["epoch"])
    logger.info("  git=%s%s", meta["git"].get("sha", "?")[:12],
                " (DIRTY)" if meta["git"].get("dirty") else "")
    logger.info("  data=%s device=%s", root, device)
    logger.info("=" * 72)

    criterion = build_criterion(cfg)
    results: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "arch": meta["arch"],
        "git": meta["git"],
        "calibrated": meta["calibrated"],
    }

    # ---------------------------------------------------------------- #
    # 1. VAL -- and this is the ONLY place a threshold is ever chosen.
    # ---------------------------------------------------------------- #
    val_loader = _build_loader(root, "val", cfg, meta)
    val = evaluate(model, val_loader, criterion, device, cfg)
    logger.info("val:  %s", val.summary())

    # Routed through select_threshold(split="val") on purpose, rather than just
    # reading val.eer_threshold. Every split's Metrics carries an eer_threshold
    # computed from its own data, so reading one off `test` would be trivially
    # easy, entirely silent, and wrong. select_threshold raises on any split but
    # val, which makes the rule enforceable instead of merely documented.
    threshold = select_threshold(val.y_true, val.y_score, split="val")
    logger.info("FROZEN operating point (EER on val): %.4f", threshold)
    results["val"] = val.to_dict()
    results["threshold"] = float(threshold)

    # ---------------------------------------------------------------- #
    # 2. TEST -- frozen threshold, passed in explicitly.
    # ---------------------------------------------------------------- #
    test_loader = _build_loader(root, "test", cfg, meta)
    test = evaluate(model, test_loader, criterion, device, cfg, threshold=threshold)
    logger.info("test: %s", test.summary())
    results["test"] = test.to_dict()

    # ---------------------------------------------------------------- #
    # 3. CROSS-DATASET -- the headline. Same frozen threshold.
    # ---------------------------------------------------------------- #
    if cross_dataset_root:
        cross_loader = _build_loader(cross_dataset_root, "all", cfg, meta)
        cross = evaluate(model, cross_loader, criterion, device, cfg, threshold=threshold)
        logger.info("cross-dataset: %s", cross.summary())
        results["cross_dataset"] = cross.to_dict()
    else:
        logger.warning(
            "No --cross-dataset given. In-dataset AUC is the easy number and "
            "predicts very little; the FF++ -> Celeb-DF result is the one that "
            "says whether this model works on media it has never seen."
        )

    _report(results)
    for line in sanity_commentary(results):
        logger.warning("SANITY: %s", line)

    out = Path(checkpoint_path).parent / "results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("wrote %s", out)
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def _report(results: dict) -> None:
    """Print the metric table. AUC first, because AUC is the headline."""
    logger.info("")
    logger.info("=" * 72)
    logger.info("RESULTS  (threshold %.4f, chosen on val and frozen)", results["threshold"])
    logger.info("=" * 72)
    header = f"  {'split':<16}{'AUC':>9}{'AP':>9}{'EER':>9}{'acc@thr':>10}{'n':>8}"
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))

    for name in ("val", "test", "cross_dataset"):
        m = results.get(name)
        if not m:
            continue
        logger.info(
            "  %-16s%9.4f%9.4f%9.4f%10.4f%8d",
            name, m["auc"], m["ap"], m["eer"], m["accuracy_at_eer"], m["n"],
        )

    # Per-method breakdown: the thing a single averaged AUC hides.
    for name in ("test", "cross_dataset"):
        per = (results.get(name) or {}).get("per_manipulation") or {}
        if not per:
            continue
        logger.info("")
        logger.info("  %s -- per manipulation method:", name)
        for method, entry in sorted(per.items(), key=lambda kv: kv[1].get("auc", 1.0)):
            logger.info(
                "    %-18s auc=%.4f  recall=%.4f  (n=%d)",
                method, entry.get("auc", float("nan")),
                entry.get("recall", float("nan")), int(entry.get("n", 0)),
            )
    logger.info("=" * 72)


def sanity_commentary(results: dict) -> list[str]:
    """Flag results that are too good, or suspiciously shaped.

    Exists because the failure modes of this project all make numbers look
    BETTER. A leak does not throw; it just quietly hands you 0.98 cross-dataset
    AUC, and by the time anyone is suspicious it is in a report. So the report
    itself has to be suspicious on your behalf.
    """
    notes: list[str] = []

    cross = results.get("cross_dataset")
    if cross and not _isnan(cross.get("auc")):
        auc = cross["auc"]
        if auc > CROSS_DATASET_SUSPICIOUS:
            notes.append(
                f"cross-dataset AUC {auc:.4f} is ABOVE {CROSS_DATASET_SUSPICIOUS}. "
                f"Published results on Celeb-DF sit at 0.65-0.75 and the best "
                f"specialised methods reach ~0.93. A number this high almost "
                f"certainly means a leak, not a breakthrough. Run "
                f"`python data/audit_splits.py` and check that Celeb-DF never "
                f"touched training."
            )
        elif auc < 0.55:
            notes.append(
                f"cross-dataset AUC {auc:.4f} is near chance (0.5). Check label "
                f"polarity -- Celeb-DF's own list encodes real=1/fake=0, the "
                f"INVERSE of this project's real=0/fake=1. An inverted label "
                f"gives ~1-AUC, so {1 - auc:.4f} would be the real score."
            )
        elif CROSS_DATASET_EXPECTED[0] <= auc <= CROSS_DATASET_EXPECTED[1]:
            notes.append(
                f"cross-dataset AUC {auc:.4f} is squarely in the expected "
                f"0.65-0.75 band. This is a NORMAL, publishable result -- the "
                f"~30-point drop from in-domain is the known generalization gap, "
                f"not a failure."
            )

    test = results.get("test")
    if test and cross and not _isnan(test.get("auc")) and not _isnan(cross.get("auc")):
        gap = test["auc"] - cross["auc"]
        if gap < 0.05:
            notes.append(
                f"in-dataset and cross-dataset AUC differ by only {gap:.4f}. That "
                f"gap is ~0.25-0.30 for every published FF++ detector. Suspect "
                f"the two sets share identities or preprocessing."
            )

    per = (results.get("test") or {}).get("per_manipulation") or {}
    if per and "NeuralTextures" in per:
        worst = min(per.items(), key=lambda kv: kv[1].get("auc", 1.0))[0]
        if worst != "NeuralTextures":
            notes.append(
                f"NeuralTextures is usually the WEAKEST method (it edits only the "
                f"mouth region), but {worst!r} scored lower here. Worth a look "
                f"before reporting."
            )

    if not results.get("calibrated"):
        notes.append(
            "This model is NOT calibrated (T78), so its softmax outputs are not "
            "probabilities. Do not present 'confidence' from this checkpoint -- "
            "see BUILD_PLAN T58/T63."
        )
    return notes


def _isnan(x) -> bool:
    return x is None or (isinstance(x, float) and x != x)


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SEETHRU checkpoint (T35).")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", type=str, help="Override the data root.")
    parser.add_argument(
        "--cross-dataset", type=str,
        help="Celeb-DF root. THE headline metric -- omit it and you only measure "
             "the easy in-domain number.",
    )
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    setup_logging(args.checkpoint.parent)
    evaluate_checkpoint(
        args.checkpoint,
        data_root=args.data,
        cross_dataset_root=args.cross_dataset,
        device_spec=args.device,
    )


if __name__ == "__main__":
    main()
