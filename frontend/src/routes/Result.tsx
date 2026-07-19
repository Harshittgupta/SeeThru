import { useEffect, useState } from "react";
import type { ImagePrediction, VideoResult } from "../api/types";
import { VerdictBanner } from "../components/verdict/VerdictBanner";
import { DisclaimerBlock } from "../components/verdict/DisclaimerBlock";
import { AttributionBar } from "../components/explain/AttributionBar";
import { HeatmapOverlay, FaceHeatmap } from "../components/explain/HeatmapOverlay";
import { Timeline } from "../components/explain/Timeline";

function Warnings({ warnings }: { warnings: string[] }) {
  if (!warnings.length) return null;
  return (
    <ul className="space-y-1 rounded-lg bg-slate-50 p-3 text-sm text-slate-600">
      {warnings.map((w, i) => (
        <li key={i}>• {w}</li>
      ))}
    </ul>
  );
}

export function ImageResult({ result, imageUrl }: { result: ImagePrediction; imageUrl: string }) {
  const [natural, setNatural] = useState<{ width: number; height: number } | null>(null);

  // Read the natural image size once, for the SVG viewBox (T62).
  useEffect(() => {
    if (!imageUrl) return;
    const img = new Image();
    img.onload = () => setNatural({ width: img.naturalWidth, height: img.naturalHeight });
    img.src = imageUrl;
  }, [imageUrl]);

  const primary = result.faces[0];

  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div className="space-y-4">
        <VerdictBanner
          verdict={result.summary.verdict}
          pFake={primary ? primary.scores.fake : 0.5}
          calibrated={result.model.calibrated}
        />
        {primary && <AttributionBar attribution={primary.attribution} />}
        <Warnings warnings={result.warnings} />
        <DisclaimerBlock disclaimer={result.disclaimer} />
      </div>
      <div className="space-y-4">
        <HeatmapOverlay imageUrl={imageUrl} faces={result.faces} naturalSize={natural} />
        {result.faces.map((f) => (
          <div key={f.face_id} className="rounded-lg border border-slate-200 p-3">
            <div className="mb-2 text-sm font-medium text-slate-600">Face {f.face_id}</div>
            <FaceHeatmap face={f} />
          </div>
        ))}
      </div>
    </div>
  );
}

export function VideoResultView({ result, posterUrl }: { result: VideoResult; posterUrl: string | null }) {
  return (
    <div className="space-y-6">
      <div className="grid gap-6 md:grid-cols-2">
        <div className="space-y-4">
          <VerdictBanner
            verdict={result.verdict}
            pFake={result.scores.fake}
            calibrated={result.model.calibrated}
          />
          <AttributionBar attribution={result.attribution} />
          <Warnings warnings={result.warnings} />
        </div>
        <div>{posterUrl && <img src={posterUrl} alt="" className="w-full rounded-lg" />}</div>
      </div>
      <div className="rounded-xl border border-slate-200 p-4">
        <Timeline frames={result.timeline} spans={result.spans} />
      </div>
      <DisclaimerBlock disclaimer={result.disclaimer} />
    </div>
  );
}
