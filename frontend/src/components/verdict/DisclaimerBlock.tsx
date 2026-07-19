import type { Disclaimer } from "../../api/types";

/**
 * The disclaimer (BUILD_PLAN T63).
 *
 * Rendered inline and NON-collapsible, next to the verdict -- not tucked in a
 * footer nobody scrolls to. FF++-trained detectors generalise poorly, and the
 * user has to see that where they see the answer, not where they don't.
 */
export function DisclaimerBlock({ disclaimer }: { disclaimer: Disclaimer }) {
  return (
    <aside className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-900">
      <p className="font-semibold">This is a research tool, not forensic evidence.</p>
      <ul className="mt-2 list-disc space-y-1 pl-5 text-amber-800">
        {disclaimer.known_limitations.map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
    </aside>
  );
}
