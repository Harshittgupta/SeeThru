import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ErrorState } from "../components/state/ErrorState";
import { AttributionBar } from "../components/explain/AttributionBar";
import { SeethruError } from "../api/client";

describe("ErrorState — keyed on error_code (T64)", () => {
  it("treats no_face_detected as a gentle empty state, not an alarm", () => {
    const err = new SeethruError("no_face_detected", "no face", "req-123", null);
    render(<ErrorState error={err} />);
    expect(screen.getByText(/no face found/i)).toBeInTheDocument();
    // The request id is shown for support.
    expect(screen.getByText(/req-123/)).toBeInTheDocument();
  });

  it("offers retry for recoverable errors", async () => {
    const onRetry = vi.fn();
    const err = new SeethruError("gpu_busy", "busy", null, 5);
    render(<ErrorState error={err} onRetry={onRetry} />);
    await userEvent.click(screen.getByRole("button", { name: /try again/i }));
    expect(onRetry).toHaveBeenCalled();
  });

  it("falls back gracefully on an unknown code", () => {
    const err = new SeethruError("internal_error", "boom", null, null);
    render(<ErrorState error={err} />);
    expect(screen.getByText(/something went wrong/i)).toBeInTheDocument();
  });
});

describe("AttributionBar — not a pie chart (ADR 0001 / T62)", () => {
  it("renders one bar per branch with signed deltas", () => {
    render(
      <AttributionBar
        attribution={[
          { branch: "spatial", delta: -0.23, baseline: "mean" },
          { branch: "frequency", delta: 0.16, baseline: "mean" },
        ]}
      />,
    );
    // Causal framing, not "% of evidence".
    expect(screen.getByText(/what the model leaned on/i)).toBeInTheDocument();
    expect(screen.getByText(/not a share/i)).toBeInTheDocument();
    // Signed values present.
    expect(screen.getByText("-0.23")).toBeInTheDocument();
    expect(screen.getByText("+0.16")).toBeInTheDocument();
  });

  it("renders nothing for empty attribution (concat with no branch data)", () => {
    const { container } = render(<AttributionBar attribution={[]} />);
    expect(container.firstChild).toBeNull();
  });
});
