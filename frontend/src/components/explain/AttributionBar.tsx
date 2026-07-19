import type { BranchAttribution } from "../../api/types";

/**
 * Branch attribution (BUILD_PLAN T62, docs/adr/0001-fusion-mode.md).
 *
 * These are CAUSAL deltas from ablation: "removing the frequency branch moves
 * the fake logit by -0.53". They are NOT shares and do NOT sum to 1, so this is
 * a signed-axis bar chart, deliberately NOT a pie chart or a normalised stacked
 * bar. Three branches can all be large (they agree) or all ~0 (nothing is
 * driving it) -- both informative, and both impossible to draw as a pie.
 *
 * The label is "what the model leaned on", phrased causally, never "70% of the
 * evidence" -- that would imply a share these numbers are not.
 */
export function AttributionBar({ attribution }: { attribution: BranchAttribution[] }) {
  if (!attribution.length) return null;

  const maxAbs = Math.max(...attribution.map((a) => Math.abs(a.delta)), 1e-6);

  return (
    <div>
      <h3 className="text-sm font-semibold text-slate-700">What the model leaned on</h3>
      <p className="mb-2 text-xs text-slate-500">
        How much each branch changed the score when removed. Longer = more
        influence. Not a share of evidence.
      </p>
      <div className="space-y-2">
        {attribution.map((a) => {
          const frac = (Math.abs(a.delta) / maxAbs) * 50; // half-width max
          const positive = a.delta >= 0; // pushed toward "fake"
          return (
            <div key={a.branch} className="flex items-center gap-2 text-sm">
              <span className="w-20 shrink-0 capitalize text-slate-600">{a.branch}</span>
              {/* Centre line = 0; bars grow left (toward real) or right (toward fake). */}
              <div className="relative h-5 flex-1 rounded bg-slate-100">
                <div className="absolute left-1/2 top-0 h-full w-px bg-slate-300" />
                <div
                  className={`absolute top-0 h-full ${positive ? "bg-fake/70" : "bg-real/70"}`}
                  style={{
                    left: positive ? "50%" : `${50 - frac}%`,
                    width: `${frac}%`,
                  }}
                  title={`${a.branch}: ${a.delta >= 0 ? "+" : ""}${a.delta.toFixed(3)} logits`}
                />
              </div>
              <span className="w-16 shrink-0 text-right font-mono text-xs text-slate-500">
                {a.delta >= 0 ? "+" : ""}
                {a.delta.toFixed(2)}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
