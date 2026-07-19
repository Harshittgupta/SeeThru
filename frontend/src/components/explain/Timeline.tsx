import type { FrameScore, TimelineSpan } from "../../api/types";
import { formatSeconds } from "../../lib/format";

/**
 * The manipulation timeline (BUILD_PLAN T62).
 *
 * DISCRETE markers on a seconds axis -- NEVER a continuous line. 16 frames
 * sampled across a whole video can be ~19 s apart on a 5-minute clip, so joining
 * them would assert a continuity that was never measured. Suspicious spans are
 * shaded only where >=2 consecutive samples cleared threshold, and labelled a
 * "sampled region", not a detection.
 *
 * Interpolated frames (a copied face, not an observation) are drawn HOLLOW and
 * are never suspicious -- rendering a duplicate as evidence would be a lie.
 */
export function Timeline({
  frames,
  spans,
}: {
  frames: FrameScore[];
  spans: TimelineSpan[];
}) {
  if (!frames.length) return null;

  const times = frames.map((f) => f.t_seconds ?? f.index);
  const maxT = Math.max(...times, 1);
  const W = 640;
  const H = 120;
  const padL = 36;
  const padB = 24;
  const plotW = W - padL - 8;
  const plotH = H - padB - 8;

  const xOf = (t: number) => padL + (t / maxT) * plotW;
  const yOf = (p: number) => 8 + (1 - p) * plotH;

  return (
    <div>
      <h3 className="text-sm font-semibold text-slate-700">Per-frame score over time</h3>
      <p className="mb-2 text-xs text-slate-500">
        Sampled points, not a continuous analysis. Hollow markers are frames with
        no detected face (filled from a neighbour — not evidence).
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Per-frame fake score timeline">
        {/* threshold line at 0.5 */}
        <line x1={padL} x2={W - 8} y1={yOf(0.5)} y2={yOf(0.5)} stroke="#cbd5e1" strokeDasharray="3 3" />
        {/* suspicious spans, shaded */}
        {spans.map((s, i) => (
          <rect
            key={i}
            x={xOf(s.start_s)}
            y={8}
            width={Math.max(2, xOf(s.end_s) - xOf(s.start_s))}
            height={plotH}
            fill="#e76f51"
            opacity={0.12}
          />
        ))}
        {/* axis labels */}
        <text x={padL} y={H - 6} fontSize="10" fill="#94a3b8">0s</text>
        <text x={W - 28} y={H - 6} fontSize="10" fill="#94a3b8">{formatSeconds(maxT)}</text>
        <text x={4} y={yOf(1) + 3} fontSize="10" fill="#94a3b8">1</text>
        <text x={4} y={yOf(0) + 3} fontSize="10" fill="#94a3b8">0</text>
        {/* markers */}
        {frames.map((f) => {
          const cx = xOf(f.t_seconds ?? f.index);
          const cy = yOf(f.p_fake);
          if (f.interpolated) {
            return <circle key={f.index} cx={cx} cy={cy} r={4} fill="white" stroke="#94a3b8" strokeWidth={1.5} />;
          }
          return <circle key={f.index} cx={cx} cy={cy} r={4} fill={f.suspicious ? "#e76f51" : "#2a9d8f"} />;
        })}
      </svg>
      {spans.length > 0 && (
        <ul className="mt-2 space-y-1 text-sm text-slate-600">
          {spans.map((s, i) => (
            <li key={i}>
              <span className="font-medium text-fake">
                {formatSeconds(s.start_s)}–{formatSeconds(s.end_s)}
              </span>{" "}
              suspicious across {s.n_frames} sampled frames (mean {s.mean_p_fake.toFixed(2)})
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
