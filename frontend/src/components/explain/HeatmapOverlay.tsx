import { useState } from "react";
import type { FaceResult } from "../../api/types";
import { preserveAspectRatioFor } from "../../lib/coords";

/**
 * Original image + GradCAM heatmap + face boxes (BUILD_PLAN T62).
 *
 * The bbox coordinate problem is solved STRUCTURALLY, not with math: the SVG
 * overlay uses `viewBox="0 0 naturalW naturalH"` and the same object-fit as the
 * <img>, and the <rect>s carry ORIGINAL image coordinates. The browser maps them
 * for free and keeps them correct through every resize -- no ResizeObserver, no
 * scale factor to get wrong.
 *
 * The heatmap is a separate <img> stacked over the original with an opacity the
 * user controls (0.4-0.6, the spec's band). Note: the heatmap is a 224x224
 * aligned CROP, not the full image, so it is shown side-by-side or as its own
 * panel rather than overlaid on the original at full frame -- overlaying a crop
 * on the original would misregister (the alignment can rotate). Here we show the
 * original with boxes, and the per-face crop+heatmap beside it.
 */
export function HeatmapOverlay({
  imageUrl,
  faces,
  naturalSize,
}: {
  imageUrl: string;
  faces: FaceResult[];
  naturalSize: { width: number; height: number } | null;
}) {
  return (
    <div className="relative inline-block max-w-full">
      <img src={imageUrl} alt="Uploaded image under analysis" className="block max-w-full rounded-lg" />
      {naturalSize && (
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox={`0 0 ${naturalSize.width} ${naturalSize.height}`}
          preserveAspectRatio={preserveAspectRatioFor("contain")}
          aria-hidden
        >
          {faces.map((f) => {
            const [x1, y1, x2, y2] = f.bbox;
            const stroke =
              f.verdict === "fake" ? "#e76f51" : f.verdict === "real" ? "#2a9d8f" : "#64748b";
            return (
              <g key={f.face_id}>
                <rect
                  x={x1}
                  y={y1}
                  width={x2 - x1}
                  height={y2 - y1}
                  fill="none"
                  stroke={stroke}
                  strokeWidth={Math.max(naturalSize.width, naturalSize.height) / 300}
                />
                <text
                  x={x1}
                  y={y1 - 4}
                  fill={stroke}
                  fontSize={Math.max(naturalSize.width, naturalSize.height) / 40}
                >
                  face {f.face_id}
                </text>
              </g>
            );
          })}
        </svg>
      )}
    </div>
  );
}

/** Per-face crop with its heatmap overlaid, opacity-controlled (T62). */
export function FaceHeatmap({ face }: { face: FaceResult }) {
  const [alpha, setAlpha] = useState(0.5);
  const original = face.artifacts["original.png"];
  const heatmap = face.artifacts["heatmap.png"];

  if (!heatmap) {
    // A degenerate CAM is omitted by the backend (T53); say so, don't fake it.
    return (
      <p className="text-sm text-slate-500">
        No heatmap for this face — the model’s attention map carried no clear
        signal, so it has been omitted rather than shown as noise.
      </p>
    );
  }

  return (
    <div>
      <div className="relative inline-block">
        {original && <img src={original} alt={`Face ${face.face_id} crop`} className="block w-48 rounded" />}
        <img
          src={heatmap}
          alt={`Attention heatmap for face ${face.face_id}`}
          className="absolute inset-0 w-48 rounded"
          style={{ opacity: alpha }}
        />
      </div>
      <label className="mt-2 flex items-center gap-2 text-xs text-slate-500">
        Heatmap opacity
        <input
          type="range"
          min={0.4}
          max={0.6}
          step={0.02}
          value={alpha}
          onChange={(e) => setAlpha(Number(e.target.value))}
          aria-valuetext={`${Math.round(alpha * 100)} percent`}
        />
      </label>
      <p className="mt-1 text-xs text-slate-400">
        Attention is coarse (a 7×7 map upscaled), so it shows a region, not a pixel.
      </p>
    </div>
  );
}
