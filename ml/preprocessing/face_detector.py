"""Face detection and alignment for SEETHRU.

Wraps the ``retina-face`` library to detect every face in an image, align it
using the detected eye landmarks, and crop to a fixed 224×224 size suitable for
downstream deepfake-detection backbones (ImageNet-pretrained CNNs / ViTs).

Example::

    from face_detector import FaceDetector

    detector = FaceDetector()
    faces = detector.detect_and_align("path/to/image.jpg")
    print(len(faces), "face(s)")          # one (H, W, 3) array per face
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from retinaface import RetinaFace

# Output crop size expected by the detection models.
DEFAULT_OUTPUT_SIZE = 224

# Where the left eye should land in the aligned, normalized output. Placing the
# eyes on a consistent line/scale removes in-plane rotation and scale variation.
DEFAULT_LEFT_EYE = (0.35, 0.35)


class FaceDetector:
    """Detect and align faces from an image using RetinaFace landmarks.

    Args:
        output_size: Side length (pixels) of the square aligned crops.
        confidence_threshold: Minimum RetinaFace detection score to keep a face.
        desired_left_eye: ``(x, y)`` position of the (image-)left eye in the
            output crop, as fractions of ``output_size``. Controls how zoomed-in
            and how vertically-centered the aligned face is.
    """

    def __init__(
        self,
        output_size: int = DEFAULT_OUTPUT_SIZE,
        confidence_threshold: float = 0.9,
        desired_left_eye: Tuple[float, float] = DEFAULT_LEFT_EYE,
    ) -> None:
        self.output_size = output_size
        self.confidence_threshold = confidence_threshold
        self.desired_left_eye = desired_left_eye

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def detect_and_align(
        self, image: str | Path | np.ndarray
    ) -> List[np.ndarray]:
        """Detect, align and crop every face in an image.

        ``image`` may be a path to an image file or an already-loaded HWC BGR
        numpy array (e.g. a video frame). Returns a list of
        ``(output_size, output_size, 3)`` uint8 BGR arrays — one per detected
        face. Returns an empty list when no face is found.
        """
        # Accept either a file path or an in-memory BGR array. RetinaFace's
        # detect_faces takes both, so we pass the array straight through.
        if isinstance(image, np.ndarray):
            frame = image
            detect_input = image
        else:
            path = str(image)
            frame = cv2.imread(path)
            if frame is None:
                raise FileNotFoundError(f"Could not read image: {path}")
            detect_input = path

        # RetinaFace returns a dict {face_1: {...}, ...}; an empty tuple/dict
        # (or no detections) means no faces were found.
        detections = RetinaFace.detect_faces(detect_input)
        if not isinstance(detections, dict) or not detections:
            return []

        faces: List[np.ndarray] = []
        for info in detections.values():
            if info.get("score", 1.0) < self.confidence_threshold:
                continue
            aligned = self._align_face(frame, info["landmarks"])
            if aligned is not None:
                faces.append(aligned)
        return faces

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _align_face(
        self, image: np.ndarray, landmarks: dict
    ) -> Optional[np.ndarray]:
        """Affine-align a single face so the eyes are level and a fixed size.

        Uses the two eye landmarks to compute the in-plane rotation and the
        scale, then warps the original image so the eyes map to canonical
        positions in an ``output_size`` square crop.
        """
        try:
            # RetinaFace landmark keys name eyes from the *subject's* view, so
            # "left_eye" is on the right of the image. Order by x instead so the
            # geometry is unambiguous regardless of pose.
            eyes = [landmarks["left_eye"], landmarks["right_eye"]]
        except (KeyError, TypeError):
            return None

        eyes.sort(key=lambda p: p[0])
        left_eye = np.asarray(eyes[0], dtype=np.float32)   # image-left
        right_eye = np.asarray(eyes[1], dtype=np.float32)  # image-right

        # Angle between the eyes; rotating by this with getRotationMatrix2D
        # (counter-clockwise positive) levels the eye line.
        dx = float(right_eye[0] - left_eye[0])
        dy = float(right_eye[1] - left_eye[1])
        angle = np.degrees(np.arctan2(dy, dx))

        # Scale so the inter-eye distance matches the desired output geometry.
        dist = float(np.hypot(dx, dy))
        if dist < 1e-3:
            return None
        desired_right_eye_x = 1.0 - self.desired_left_eye[0]
        desired_dist = (desired_right_eye_x - self.desired_left_eye[0]) * self.output_size
        scale = desired_dist / dist

        eyes_center = (
            float((left_eye[0] + right_eye[0]) / 2.0),
            float((left_eye[1] + right_eye[1]) / 2.0),
        )

        rot = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # Translate so the eyes land at the desired output location.
        tx = self.output_size * 0.5
        ty = self.output_size * self.desired_left_eye[1]
        rot[0, 2] += tx - eyes_center[0]
        rot[1, 2] += ty - eyes_center[1]

        aligned = cv2.warpAffine(
            image,
            rot,
            (self.output_size, self.output_size),
            flags=cv2.INTER_CUBIC,
        )
        return aligned


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Detect and align faces in an image."
    )
    parser.add_argument("image", help="Path to a sample image.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="If set, write each aligned face crop here as face_<i>.png.",
    )
    parser.add_argument("--size", type=int, default=DEFAULT_OUTPUT_SIZE)
    args = parser.parse_args()

    detector = FaceDetector(output_size=args.size)
    faces = detector.detect_and_align(args.image)

    print(f"Detected {len(faces)} face(s) in {args.image}")
    for i, face in enumerate(faces):
        print(f"  face {i}: shape={face.shape}, dtype={face.dtype}")

    if args.output_dir and faces:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, face in enumerate(faces):
            out_path = out_dir / f"face_{i}.png"
            cv2.imwrite(str(out_path), face)
            print(f"  saved {out_path}")


if __name__ == "__main__":
    _main()
