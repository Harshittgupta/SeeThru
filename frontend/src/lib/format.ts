// Presentation helpers, with the honesty rules baked in (BUILD_PLAN T63).

import type { Verdict } from "../api/types";

/**
 * Map a fake-probability to a qualitative signal band.
 *
 * This exists because the model is UNCALIBRATED (backend T78): its softmax is
 * not a probability, so a "97%" would be a fabricated precision. When the
 * backend reports `calibrated: false`, the UI shows one of these words instead
 * of a number (T63). Three bands, not a percentage.
 */
export function signalBand(pFake: number): "weak" | "moderate" | "strong" {
  const margin = Math.abs(pFake - 0.5) * 2; // 0 at the boundary, 1 at the extremes
  if (margin < 0.34) return "weak";
  if (margin < 0.67) return "moderate";
  return "strong";
}

export function verdictLabel(v: Verdict): string {
  return { real: "Likely Real", fake: "Likely Fake", uncertain: "Uncertain" }[v];
}

export function verdictColorClass(v: Verdict): string {
  return { real: "text-real", fake: "text-fake", uncertain: "text-uncertain" }[v];
}

/** A shape/icon per verdict, so meaning is never carried by colour alone (a11y, T64). */
export function verdictGlyph(v: Verdict): string {
  return { real: "○", fake: "△", uncertain: "◇" }[v]; // ○ △ ◇
}

export function formatSeconds(s: number | null): string {
  if (s == null) return "—";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return m > 0 ? `${m}:${String(sec).padStart(2, "0")}` : `${s.toFixed(1)}s`;
}

export function formatBytes(n: number | undefined): string {
  if (!n) return "";
  const units = ["B", "KB", "MB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(1)} ${units[i]}`;
}
