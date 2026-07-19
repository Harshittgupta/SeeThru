"""Run logging and provenance (BUILD_PLAN T27).

Every run writes ``runs/<name>/train.log`` alongside its checkpoints, and dumps
the **resolved** config plus the git SHA at startup.

That last part is the point. Six weeks from now the only question that matters
about a checkpoint is "what produced this?", and the answers have to live next to
the weights -- not in a shell history, not in a terminal that has been closed.
`--dirty` is recorded too: a run from an uncommitted tree is not reproducible
from its SHA alone, and it is better to know that than to assume otherwise.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
DATE_FORMAT = "%H:%M:%S"


def setup_logging(run_dir: Path | str | None = None, level: int = logging.INFO) -> Path | None:
    """Configure root logging to stdout and (optionally) ``<run_dir>/train.log``.

    Returns the log file path, or None if no run_dir was given.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):  # idempotent across repeated calls
        root.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    log_path = None
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "train.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # TensorFlow (via retina-face) is loud on import and we cannot fix it here.
    # T45 removes it entirely.
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    return log_path


def log_run_header(config, run_dir: Path | str) -> None:
    """Log the resolved config + git provenance, and write config.yaml.

    Writing the *resolved* config -- not the file that was passed in -- is what
    makes this useful: it captures CLI overrides and defaults that never appeared
    in any YAML, which are exactly the values you will fail to remember.
    """
    from ml.checkpoint import git_revision

    logger = logging.getLogger("seethru")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    git = git_revision()
    dirty = " (DIRTY -- uncommitted changes)" if git.get("dirty") else ""

    logger.info("=" * 70)
    logger.info("SEETHRU run: %s", run_dir)
    logger.info("git: %s%s", git.get("sha", "unknown")[:12], dirty)
    logger.info("=" * 70)
    for line in config.to_yaml().splitlines():
        logger.info("  %s", line)
    logger.info("=" * 70)

    (run_dir / "config.yaml").write_text(config.to_yaml(), encoding="utf-8")

    if git.get("dirty"):
        logger.warning(
            "Working tree is dirty: this run is NOT reproducible from its git "
            "SHA alone. Commit before a run whose numbers you intend to report."
        )
