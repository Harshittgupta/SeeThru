/**
 * Methodology + limits (BUILD_PLAN T63). The honest-context page: what the model
 * was trained on, where it degrades, and what it must not be used for.
 */
export function About() {
  return (
    <div className="prose prose-slate max-w-none text-slate-700">
      <h1 className="text-2xl font-bold">About SEETHRU</h1>
      <p>
        SEETHRU is a research and educational tool for detecting manipulated
        faces in images and video. It combines a spatial (appearance), frequency
        (spectral-artifact) and temporal (cross-frame) analysis, and explains its
        reasoning with attention heatmaps, a frequency profile, and a per-branch
        attribution.
      </p>

      <h2 className="mt-6 text-lg font-semibold">What the score means</h2>
      <p>
        The model reports a signal strength, not a probability. Until it is
        calibrated, the interface deliberately shows a qualitative band (weak /
        moderate / strong) rather than a percentage — a percentage from an
        uncalibrated model would imply a precision it does not have.
      </p>

      <h2 className="mt-6 text-lg font-semibold">Where it is weak</h2>
      <ul className="list-disc pl-5">
        <li>
          It degrades on manipulation methods, compression levels and camera
          types it did not see in training — often substantially.
        </li>
        <li>Video analysis assumes a single subject and samples ~16 frames.</li>
        <li>It is not validated for legal, journalistic, or evidentiary use.</li>
      </ul>

      <p className="mt-6 rounded-lg bg-amber-50 p-4 text-sm text-amber-900">
        A result here is not proof that media is authentic or manipulated. Treat
        it as one signal among many.
      </p>
    </div>
  );
}
