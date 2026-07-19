// Bounding-box coordinate mapping (BUILD_PLAN T62).
//
// THE bug this file exists to prevent, and it bites every scaled-<img> UI:
// bboxes come back in ORIGINAL image pixels (e.g. a 4032x3024 photo), but the
// <img> on screen is CSS-scaled to fit its container (e.g. 600px wide). Drawing
// the raw bbox over the scaled image puts the box in the wrong place, and it
// moves as the window resizes.
//
// The robust fix is to NOT do this math in a component at all: render an SVG
// overlay with `viewBox="0 0 naturalW naturalH"` and `preserveAspectRatio`
// matching the img's object-fit, and give the <rect> the ORIGINAL coordinates.
// The browser then maps them for free, and it stays correct through every
// resize. See BBoxLayer.tsx. This module is the fallback for cases (canvas,
// tests) that need the numbers explicitly.

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/**
 * Map a bbox from original-image space to rendered (displayed) space, for an
 * image shown with `object-fit: contain`.
 *
 * `contain` letterboxes: the image is scaled by a single factor to fit inside
 * the box, and centred, so there is padding on one axis. Getting that padding
 * wrong is the usual second bug after the scale itself.
 */
export function mapBboxContain(
  bbox: [number, number, number, number],
  natural: { width: number; height: number },
  rendered: { width: number; height: number },
): Rect {
  const [x1, y1, x2, y2] = bbox;

  // One scale for both axes (that is what `contain` means), = the smaller ratio.
  const scale = Math.min(
    rendered.width / natural.width,
    rendered.height / natural.height,
  );
  const displayedW = natural.width * scale;
  const displayedH = natural.height * scale;

  // Letterbox padding, centred.
  const padX = (rendered.width - displayedW) / 2;
  const padY = (rendered.height - displayedH) / 2;

  return {
    x: padX + x1 * scale,
    y: padY + y1 * scale,
    width: (x2 - x1) * scale,
    height: (y2 - y1) * scale,
  };
}

/**
 * The `preserveAspectRatio` value for an SVG overlay whose viewBox is the
 * natural image size, matching a given CSS `object-fit`. Using this on the SVG
 * makes coordinate math unnecessary in the common case (T62).
 */
export function preserveAspectRatioFor(objectFit: "contain" | "cover"): string {
  // "meet" = contain (fit inside, letterbox); "slice" = cover (fill, crop).
  return objectFit === "cover" ? "xMidYMid slice" : "xMidYMid meet";
}
