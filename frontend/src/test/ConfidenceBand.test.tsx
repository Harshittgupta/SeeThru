import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { ConfidenceBand } from "../components/verdict/ConfidenceBand";
import { pollInterval } from "../hooks/useVideoJob";

// The single most important UI-honesty guarantee (T63): an uncalibrated model
// must NEVER show a percentage. A "97%" from an uncalibrated softmax is a
// fabricated number.
describe("ConfidenceBand — uncalibrated", () => {
  it("suppresses the percentage entirely", () => {
    render(<ConfidenceBand pFake={0.97} calibrated={false} />);
    // No "97%" anywhere in the headline.
    expect(screen.queryByText(/97\s*%/)).toBeNull();
    expect(screen.getByText(/not a probability/i)).toBeInTheDocument();
    expect(screen.getByText(/signal/i)).toBeInTheDocument();
  });

  it("only exposes the raw score inside a collapsed technical detail", () => {
    render(<ConfidenceBand pFake={0.97} calibrated={false} />);
    // The raw value exists but behind <details>, not as the headline number.
    expect(screen.getByText(/raw p\(fake\) = 0\.970/)).toBeInTheDocument();
  });

  it("shows a percentage ONLY when calibrated", () => {
    render(<ConfidenceBand pFake={0.9} calibrated={true} />);
    expect(screen.getByText(/90%/)).toBeInTheDocument();
  });
});

// The polling backoff schedule (T61): fast early, easing off.
describe("pollInterval backoff", () => {
  it("polls fast in the first 10s, then eases off", () => {
    expect(pollInterval(0)).toBe(1000);
    expect(pollInterval(5_000)).toBe(1000);
    expect(pollInterval(20_000)).toBe(2000);
    expect(pollInterval(90_000)).toBe(5000);
  });

  it("is monotonically non-decreasing", () => {
    let prev = 0;
    for (const t of [0, 9_000, 10_000, 30_000, 60_000, 120_000]) {
      const v = pollInterval(t);
      expect(v).toBeGreaterThanOrEqual(prev);
      prev = v;
    }
  });
});
