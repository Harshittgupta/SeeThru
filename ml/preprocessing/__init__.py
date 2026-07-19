"""Preprocessing utilities for SEETHRU (face detection, alignment, cropping).

``FaceDetector`` is exported lazily via PEP 562 module ``__getattr__``.

Importing it eagerly pulled in ``retinaface`` -> **all of TensorFlow**, measured
at 11.0 s and ~600 MB RSS (`python -X importtime`). That cost was paid by anyone
who touched this package for any reason -- ``from ml.preprocessing import
build_train_transform`` loaded TensorFlow -- including every DataLoader worker
(Windows spawns, so each re-imports) and the backend at startup, where TF would
then also sit on the GPU competing with PyTorch for VRAM.

``from ml.preprocessing import FaceDetector`` still works exactly as before; the
import just happens on first *access* rather than at module load. See
BUILD_PLAN T39, and T45 for removing TensorFlow entirely.
"""

from typing import TYPE_CHECKING, Any

from .augmentation import (
    build_train_transform,
    build_val_transform,
    train_transform,
    val_transform,
    visualize_augmentations,
)

if TYPE_CHECKING:  # so type checkers and IDEs still resolve the name
    from .face_detector import FaceDetector

__all__ = [
    "FaceDetector",
    "build_train_transform",
    "build_val_transform",
    "train_transform",
    "val_transform",
    "visualize_augmentations",
]

_LAZY = {"FaceDetector": ".face_detector"}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute access -- defers the TensorFlow import."""
    if name in _LAZY:
        from importlib import import_module

        module = import_module(_LAZY[name], __name__)
        value = getattr(module, name)
        globals()[name] = value  # cache: only the first access pays
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(__all__)
