import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";
import { VerdictBanner } from "../components/verdict/VerdictBanner";
import { signalBand, verdictGlyph } from "../lib/format";

describe("VerdictBanner", () => {
  it("renders 'uncertain' as a first-class state, not a hidden footnote", () => {
    render(<VerdictBanner verdict="uncertain" pFake={0.5} calibrated={false} />);
    expect(screen.getByText(/uncertain/i)).toBeInTheDocument();
    expect(screen.getByText(/too weak to call/i)).toBeInTheDocument();
  });

  it("never shows a giant percentage for a fake verdict (uncalibrated)", () => {
    render(<VerdictBanner verdict="fake" pFake={0.97} calibrated={false} />);
    expect(screen.getByText(/likely fake/i)).toBeInTheDocument();
    expect(screen.queryByText(/97\s*%/)).toBeNull(); // no "FAKE 97%" hero
  });

  it("carries meaning with a glyph, not colour alone (a11y)", () => {
    // Distinct glyph per verdict, so colourblind users get the signal too.
    expect(verdictGlyph("real")).not.toBe(verdictGlyph("fake"));
    expect(verdictGlyph("fake")).not.toBe(verdictGlyph("uncertain"));
  });
});

describe("signalBand", () => {
  it("maps distance-from-boundary to weak/moderate/strong", () => {
    expect(signalBand(0.52)).toBe("weak"); // near 0.5
    expect(signalBand(0.75)).toBe("moderate");
    expect(signalBand(0.98)).toBe("strong");
    expect(signalBand(0.02)).toBe("strong"); // strongly REAL is also a strong signal
  });
});
