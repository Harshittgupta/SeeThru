import { signalBand } from "../../lib/format";

/**
 * The confidence display (BUILD_PLAN T63).
 *
 * When the model is UNCALIBRATED -- which it is today, and will be until backend
 * T78 -- this suppresses the percentage ENTIRELY and shows a 3-band qualitative
 * scale instead. An uncalibrated softmax is not a probability, so "FAKE 97%"
 * would be a fabricated number dressed as certainty. The raw score is available
 * only in a collapsed technical detail, never as the headline.
 */
export function ConfidenceBand({
  pFake,
  calibrated,
}: {
  pFake: number;
  calibrated: boolean;
}) {
  if (calibrated) {
    // Only once a calibration step exists may we show a real probability.
    return (
      <div className="text-sm text-gray-600">
        Calibrated confidence:{" "}
        <span className="font-semibold">{Math.round(pFake * 100)}%</span> likely fake
      </div>
    );
  }

  const band = signalBand(pFake);
  const bars = { weak: 1, moderate: 2, strong: 3 }[band];

  return (
    <div>
      <div className="flex items-center gap-1.5" aria-label={`${band} signal`}>
        {[0, 1, 2].map((i) => (
          <span
            key={i}
            className={`h-2.5 w-8 rounded-sm ${i < bars ? "bg-slate-600" : "bg-slate-200"}`}
          />
        ))}
        <span className="ml-2 text-sm font-medium capitalize text-slate-600">
          {band} signal
        </span>
      </div>
      <p className="mt-1 text-xs text-slate-500">
        This score is uncalibrated — it is not a probability.
      </p>
      <details className="mt-1 text-xs text-slate-400">
        <summary className="cursor-pointer">Technical detail</summary>
        <span className="font-mono">raw p(fake) = {pFake.toFixed(3)}</span>
      </details>
    </div>
  );
}
