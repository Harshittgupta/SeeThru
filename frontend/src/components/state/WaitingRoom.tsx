import { useEffect, useState } from "react";
import type { JobStatus } from "../../api/types";

/**
 * What the user sees during a 30 s - 5 min video analysis (BUILD_PLAN T61).
 *
 * The hard part of this screen is honesty about time. The determinate bar is
 * shown ONLY if the backend sends a real `progress`; otherwise the bar is
 * indeterminate and we never invent an ETA. A monotonic elapsed clock and the
 * current stage give a truthful sense of movement without a fake countdown.
 *
 * `reconnecting` keeps the last known state on screen during a network blip --
 * the job is still running server-side, so the screen must not wipe to an error.
 */
export function WaitingRoom({
  status,
  elapsedMs,
  reconnecting,
  posterUrl,
  onCancel,
}: {
  status: JobStatus | undefined;
  elapsedMs: number;
  reconnecting: boolean;
  posterUrl: string | null;
  onCancel: () => void;
}) {
  const stages = ["queued", "extracting faces", "analysing", "rendering"];
  const currentStage = status?.stage ?? status?.state ?? "queued";
  const hasProgress = typeof status?.progress === "number" && status.progress > 0;

  return (
    <div className="rounded-xl border border-slate-200 p-8 text-center">
      {posterUrl && (
        <img src={posterUrl} alt="" className="mx-auto mb-4 max-h-40 rounded opacity-60" />
      )}
      <h2 className="text-lg font-semibold text-slate-800">Analysing video…</h2>

      <div className="mx-auto mt-4 h-2 max-w-sm overflow-hidden rounded-full bg-slate-200">
        {hasProgress ? (
          <div
            className="h-full bg-slate-700 transition-all"
            style={{ width: `${Math.round((status!.progress) * 100)}%` }}
          />
        ) : (
          // Indeterminate: honest when there is no real progress to show.
          <div className="h-full w-1/3 animate-pulse bg-slate-400" />
        )}
      </div>

      <ol className="mx-auto mt-4 flex max-w-sm justify-between text-xs">
        {stages.map((s) => {
          const active = currentStage.includes(s) || (s === "queued" && currentStage === "queued");
          return (
            <li key={s} className={active ? "font-semibold text-slate-800" : "text-slate-400"}>
              {s}
            </li>
          );
        })}
      </ol>

      <p className="mt-3 text-sm text-slate-500" aria-live="polite">
        {reconnecting ? "Reconnecting…" : `${Math.floor(elapsedMs / 1000)}s elapsed`}
      </p>

      <button
        onClick={onCancel}
        className="mt-4 rounded-lg border border-slate-300 px-4 py-2 text-sm text-slate-600 hover:bg-slate-50"
      >
        Cancel
      </button>
    </div>
  );
}

/** Update the browser tab title so a backgrounded tab shows progress (T61). */
export function useTabTitle(title: string) {
  useEffect(() => {
    const prev = document.title;
    document.title = title;
    return () => {
      document.title = prev;
    };
  }, [title]);
}

/** A monotonic elapsed-time ticker. */
export function useElapsed(startMs: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(id);
  }, []);
  return now - startMs;
}
