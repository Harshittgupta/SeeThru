"""Upload validation and safe temp handling (BUILD_PLAN T55).

This service exists because the API accepts untrusted media and hands it to
ffmpeg/OpenCV/Pillow -- parsers with a real RCE history (cf. the libwebp
CVE-2023-4863 class). Every check here is a control, not a nicety.

The order matters, cheapest-and-most-decisive first:

1. **Size, enforced while streaming.** ``Content-Length`` is checked first
   (cheap) but is spoofable and absent under chunked encoding, so it is not
   authoritative. The real guard is aborting the read at a cumulative byte cap --
   Starlette spools an ``UploadFile`` to disk past 1 MB, so a 10 GB body would
   otherwise quietly fill the container's disk before any handler runs.
2. **Magic bytes**, not the extension and not the client ``Content-Type``. A
   ``.jpg`` with an MP4 header is an MP4, whatever it is called.
3. **Decompression-bomb guard** for images: ``Image.MAX_IMAGE_PIXELS`` (PIL only
   *warns* by default), verify-then-reopen, and a pixel-count check before cv2,
   which has no bomb guard at all.
4. **A UUID temp path**, never the client filename -- path traversal, and a
   newline in a name would otherwise forge log lines. The original name is kept
   only as a sanitised display string.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import filetype

from backend.core.errors import PayloadTooLarge, UnreadableMedia, UnsupportedMedia

logger = logging.getLogger("seethru.uploads")

# Magic-byte allowlists. Keys are the extension we will give the temp file.
IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
VIDEO_TYPES = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
}

_READ_CHUNK = 1 << 20  # 1 MB


@dataclass
class SavedUpload:
    path: Path            # the UUID temp path we control
    display_name: str     # sanitised original name, for messages only
    content_type: str     # the SNIFFED type, not the client's claim
    size: int

    def cleanup(self) -> None:
        self.path.unlink(missing_ok=True)


def _sanitize_name(name: str | None) -> str:
    """A safe display string. Never used as a path component."""
    if not name:
        return "upload"
    base = Path(name).name  # strip any directory part
    cleaned = "".join(c for c in base if c.isalnum() or c in "._- ")[:80]
    return cleaned or "upload"


async def save_upload(
    upload,
    *,
    max_bytes: int,
    allowed: dict[str, str],
    tmp_dir: Path,
    declared_length: int | None,
) -> SavedUpload:
    """Stream an UploadFile to a UUID temp path, enforcing size and type.

    Raises PayloadTooLarge / UnsupportedMedia / UnreadableMedia (all mapped to the
    error envelope) -- never a bare exception into the request path.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Cheap early reject: an honest oversized client saves everyone the transfer.
    # NOT authoritative -- see the streaming cap below.
    if declared_length is not None and declared_length > max_bytes:
        raise PayloadTooLarge(
            f"Upload is {declared_length} bytes; the limit is {max_bytes}.",
            {"max_bytes": max_bytes},
        )

    tmp_path = tmp_dir / f"{uuid.uuid4().hex}.part"
    size = 0
    header = b""
    try:
        with open(tmp_path, "wb") as fh:
            while chunk := await upload.read(_READ_CHUNK):
                size += len(chunk)
                # The authoritative cap. Abort mid-stream so a hostile client
                # cannot fill the disk regardless of a spoofed Content-Length.
                if size > max_bytes:
                    raise PayloadTooLarge(
                        f"Upload exceeds the {max_bytes}-byte limit.",
                        {"max_bytes": max_bytes},
                    )
                if len(header) < 4096:
                    header += chunk[: 4096 - len(header)]
                fh.write(chunk)

        if size == 0:
            raise UnreadableMedia("The uploaded file is empty.")

        # Magic bytes decide the type. The extension and the client's declared
        # Content-Type are both attacker-controlled and carry no weight.
        kind = filetype.guess(header)
        mime = kind.mime if kind else None
        if mime not in allowed:
            raise UnsupportedMedia(
                f"Unsupported or unrecognised media type "
                f"({mime or 'unknown'}). Allowed: {sorted(allowed)}.",
                {"detected": mime, "allowed": sorted(allowed)},
            )

        final = tmp_path.with_suffix(allowed[mime])
        tmp_path.rename(final)
        return SavedUpload(
            path=final,
            display_name=_sanitize_name(getattr(upload, "filename", None)),
            content_type=mime,
            size=size,
        )
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def load_validated_image(path: Path, max_pixels: int):
    """Decode an image with the decompression-bomb guards in place → HWC BGR.

    cv2.imread has NO pixel-count guard, so a 50000x50000 PNG that is a few KB on
    disk becomes gigabytes in RAM the moment it is decoded. PIL's default only
    *warns* near its limit. So: set an explicit hard limit, verify the header,
    check the dimensions, and only then hand off to cv2.
    """
    from PIL import Image, ImageFile

    Image.MAX_IMAGE_PIXELS = max_pixels
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    try:
        with Image.open(path) as probe:
            probe.verify()  # header integrity, without decoding pixels
        with Image.open(path) as img:
            w, h = img.size
            if w * h > max_pixels:
                raise PayloadTooLarge(
                    f"Image is {w}x{h} = {w * h} pixels; the limit is {max_pixels}.",
                    {"max_pixels": max_pixels},
                )
            rgb = img.convert("RGB")
    except PayloadTooLarge:
        raise
    except Exception as exc:  # noqa: BLE001 - any decode failure is client's fault
        raise UnreadableMedia(f"Could not decode the image: {exc}") from exc

    import numpy as np

    # PIL gives RGB; the detector and the model pipeline speak BGR (OpenCV order).
    return np.asarray(rgb)[:, :, ::-1].copy()


def probe_video(path: Path, max_seconds: float):
    """Cheaply gate a video before any heavy decode → ``{fps, frames, duration, w, h}``.

    Uses OpenCV's metadata read (no ffprobe binary dependency, which keeps the
    image slim). The point is to reject a too-long or malformed video **before**
    ``VideoProcessor`` touches it -- its fallback ``_read_all`` path decodes an
    entire file into RAM when the frame count is unreliable (that path is capped
    at MAX_DECODE_FRAMES in T42, but rejecting up front is cheaper and clearer).
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise UnreadableMedia("Could not open the video (unsupported codec?).")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()

    duration = frames / fps if fps > 0 else 0.0
    if fps > 0 and duration > max_seconds:
        raise PayloadTooLarge(
            f"Video is {duration:.0f}s; the limit is {max_seconds:.0f}s.",
            {"max_seconds": max_seconds},
        )
    return {"fps": fps, "frames": frames, "duration_s": duration, "width": w, "height": h}
