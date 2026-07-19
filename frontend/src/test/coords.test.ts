import { describe, expect, it } from "vitest";
import { mapBboxContain, preserveAspectRatioFor } from "../lib/coords";

// The bbox-mapping bug is the most common defect in any scaled-image UI (T62),
// so it is tested at several container sizes, not just one.
describe("mapBboxContain", () => {
  it("maps identity when rendered == natural", () => {
    const r = mapBboxContain([100, 200, 300, 400], { width: 800, height: 600 }, { width: 800, height: 600 });
    expect(r).toEqual({ x: 100, y: 200, width: 200, height: 200 });
  });

  it("scales down uniformly when the image is shrunk to fit", () => {
    // 800x600 shown in a 400x300 box => scale 0.5, no letterbox (same aspect).
    const r = mapBboxContain([100, 200, 300, 400], { width: 800, height: 600 }, { width: 400, height: 300 });
    expect(r.x).toBeCloseTo(50);
    expect(r.y).toBeCloseTo(100);
    expect(r.width).toBeCloseTo(100);
    expect(r.height).toBeCloseTo(100);
  });

  it("accounts for letterbox padding when aspect ratios differ", () => {
    // A tall 400x800 image in a wide 800x400 box: contain scale = 0.5 (limited
    // by height), displayed width = 200, so padX = (800-200)/2 = 300.
    const r = mapBboxContain([0, 0, 400, 800], { width: 400, height: 800 }, { width: 800, height: 400 });
    expect(r.x).toBeCloseTo(300); // the padding the naive version forgets
    expect(r.y).toBeCloseTo(0);
    expect(r.width).toBeCloseTo(200);
    expect(r.height).toBeCloseTo(400);
  });

  it("is stable across a resize (same relative box)", () => {
    const bbox: [number, number, number, number] = [50, 50, 150, 150];
    const small = mapBboxContain(bbox, { width: 1000, height: 1000 }, { width: 200, height: 200 });
    const large = mapBboxContain(bbox, { width: 1000, height: 1000 }, { width: 600, height: 600 });
    // The box should occupy the same FRACTION of the container at both sizes.
    expect(small.x / 200).toBeCloseTo(large.x / 600);
    expect(small.width / 200).toBeCloseTo(large.width / 600);
  });
});

describe("preserveAspectRatioFor", () => {
  it("maps object-fit to the SVG equivalent", () => {
    expect(preserveAspectRatioFor("contain")).toBe("xMidYMid meet");
    expect(preserveAspectRatioFor("cover")).toBe("xMidYMid slice");
  });
});
