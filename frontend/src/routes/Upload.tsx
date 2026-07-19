import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { predictImage, submitVideo } from "../api/predict";
import { SeethruError } from "../api/client";
import type { ImagePrediction } from "../api/types";
import { Dropzone, type PickedFile } from "../components/upload/Dropzone";
import { ErrorState } from "../components/state/ErrorState";
import { ImageResult } from "./Result";

/**
 * The landing/upload screen (BUILD_PLAN T61).
 *
 * Image -> synchronous, shown inline. Video -> submit, then navigate to
 * /analyze/:jobId BEFORE polling, so the job id lives in the URL and a refresh
 * (or a shared link) resumes the same analysis.
 */
export function Upload() {
  const navigate = useNavigate();
  const [picked, setPicked] = useState<PickedFile | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadFrac, setUploadFrac] = useState(0);
  const [error, setError] = useState<SeethruError | null>(null);
  const [imageResult, setImageResult] = useState<ImagePrediction | null>(null);

  async function run(p: PickedFile) {
    setError(null);
    setImageResult(null);
    setUploading(true);
    setUploadFrac(0);
    try {
      if (p.kind === "image") {
        const result = await predictImage(p.file, setUploadFrac);
        setImageResult(result);
      } else {
        const job = await submitVideo(p.file, setUploadFrac);
        // Navigate BEFORE polling: the URL now owns the job (refresh-survivable).
        navigate(`/analyze/${job.job_id}`, {
          state: { posterUrl: p.previewUrl },
        });
      }
    } catch (err) {
      setError(err instanceof SeethruError ? err : new SeethruError("internal_error", String(err), null, null));
    } finally {
      setUploading(false);
    }
  }

  function reset() {
    setPicked(null);
    setImageResult(null);
    setError(null);
  }

  if (imageResult) {
    return (
      <div>
        <ImageResult result={imageResult} imageUrl={picked?.previewUrl ?? ""} />
        <button onClick={reset} className="mt-6 text-sm text-slate-500 underline">
          Analyse another
        </button>
      </div>
    );
  }

  if (error) return <ErrorState error={error} onRetry={reset} />;

  return (
    <div className="space-y-4">
      {!picked && <Dropzone onPick={(p) => setPicked(p)} />}
      {picked && !uploading && (
        <div className="rounded-xl border border-slate-200 p-6">
          {picked.kind === "image" ? (
            <img src={picked.previewUrl} alt="Preview" className="mx-auto max-h-64 rounded" />
          ) : (
            <video src={picked.previewUrl} className="mx-auto max-h-64 rounded" controls />
          )}
          <div className="mt-4 flex justify-center gap-3">
            <button
              onClick={() => run(picked)}
              className="rounded-lg bg-slate-800 px-5 py-2 font-medium text-white hover:bg-slate-700"
            >
              Analyse
            </button>
            <button onClick={reset} className="rounded-lg border border-slate-300 px-5 py-2 text-slate-600">
              Choose another
            </button>
          </div>
        </div>
      )}
      {uploading && (
        <div className="rounded-xl border border-slate-200 p-8 text-center">
          <p className="text-slate-700">Uploading… {Math.round(uploadFrac * 100)}%</p>
          <div className="mx-auto mt-3 h-2 max-w-sm overflow-hidden rounded-full bg-slate-200">
            <div className="h-full bg-slate-700 transition-all" style={{ width: `${uploadFrac * 100}%` }} />
          </div>
        </div>
      )}
    </div>
  );
}
