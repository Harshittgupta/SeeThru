import type { Verdict } from "../../api/types";
import { verdictGlyph, verdictLabel } from "../../lib/format";
import { ConfidenceBand } from "./ConfidenceBand";

/**
 * The verdict headline (BUILD_PLAN T63).
 *
 * Rules that are non-negotiable here:
 * - `uncertain` is a FIRST-CLASS, visually equal state -- slate, same size, not
 *   a grey footnote under a red one.
 * - NO giant red "FAKE 97%" hero. The word plus a qualitative band, never a big
 *   number (the number lives in ConfidenceBand, suppressed when uncalibrated).
 * - Meaning is never colour alone: a glyph and the word carry it too (a11y, T64).
 */
export function VerdictBanner({
  verdict,
  pFake,
  calibrated,
}: {
  verdict: Verdict;
  pFake: number;
  calibrated: boolean;
}) {
  const tone = {
    real: "border-real/40 bg-real/5",
    fake: "border-fake/40 bg-fake/5",
    uncertain: "border-uncertain/40 bg-uncertain/5",
  }[verdict];
  const text = {
    real: "text-real",
    fake: "text-fake",
    uncertain: "text-uncertain",
  }[verdict];

  return (
    <div className={`rounded-xl border-2 p-5 ${tone}`} role="status" aria-live="polite">
      <div className="flex items-center gap-3">
        <span className={`text-3xl ${text}`} aria-hidden>
          {verdictGlyph(verdict)}
        </span>
        <div>
          <div className={`text-2xl font-bold ${text}`}>{verdictLabel(verdict)}</div>
          {verdict === "uncertain" && (
            <div className="text-sm text-slate-500">
              The signal is too weak to call this either way.
            </div>
          )}
        </div>
      </div>
      <div className="mt-4">
        <ConfidenceBand pFake={pFake} calibrated={calibrated} />
      </div>
    </div>
  );
}
