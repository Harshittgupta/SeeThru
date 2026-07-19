import type { SeethruError } from "../../api/client";
import type { ErrorCode } from "../../api/types";

/**
 * Error and empty states, keyed on `error_code` (BUILD_PLAN T64).
 *
 * The important distinctions:
 * - `no_face_detected` is a first-class EMPTY state, not a scary error toast --
 *   it is the common case, and the model simply has nothing to analyse.
 * - `gpu_busy` / `model_not_ready` are RECOVERABLE (503): "still queued", worth
 *   retrying, not a failure.
 * - The request_id is always shown (copyable), so support is tractable.
 */
const COPY: Record<ErrorCode, { icon: string; title: string; body: string; recoverable: boolean }> = {
  no_face_detected: {
    icon: "🔍",
    title: "No face found",
    body: "SEETHRU only analyses faces, and none were detected in this image. Try a clearer, front-facing photo.",
    recoverable: false,
  },
  insufficient_faces: {
    icon: "🎞️",
    title: "Too few faces in the video",
    body: "Most sampled frames had no detectable face, so the video can’t be analysed reliably.",
    recoverable: false,
  },
  unreadable_media: {
    icon: "📄",
    title: "Couldn’t read that file",
    body: "The file may be corrupt or use an unsupported codec. For video, H.264 MP4 works best.",
    recoverable: false,
  },
  unsupported_media: {
    icon: "🚫",
    title: "Unsupported file type",
    body: "Please upload a JPEG, PNG or WebP image, or an MP4/MOV/MKV video.",
    recoverable: false,
  },
  payload_too_large: {
    icon: "📦",
    title: "File too large",
    body: "Images are limited to 15 MB and videos to 200 MB.",
    recoverable: false,
  },
  model_not_ready: {
    icon: "⏳",
    title: "The model is still starting up",
    body: "The server is loading its weights. This usually clears in a moment.",
    recoverable: true,
  },
  gpu_busy: {
    icon: "⏳",
    title: "The GPU is busy",
    body: "Another analysis is running. Your request will be retried automatically.",
    recoverable: true,
  },
  queue_full: {
    icon: "⏳",
    title: "Queue is full",
    body: "Too many videos are waiting. Try again shortly.",
    recoverable: true,
  },
  job_not_found: {
    icon: "❓",
    title: "That analysis wasn’t found",
    body: "The link may be wrong, or the job was cleaned up.",
    recoverable: false,
  },
  job_expired: {
    icon: "🗑️",
    title: "This result has expired",
    body: "Results are kept for 24 hours. Please run the analysis again.",
    recoverable: false,
  },
  network_error: {
    icon: "📡",
    title: "Can’t reach the server",
    body: "Check your connection. This will keep retrying.",
    recoverable: true,
  },
  internal_error: {
    icon: "⚠️",
    title: "Something went wrong",
    body: "An unexpected error occurred. Please try again.",
    recoverable: false,
  },
};

export function ErrorState({ error, onRetry }: { error: SeethruError; onRetry?: () => void }) {
  const copy = COPY[error.code] ?? COPY.internal_error;
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 p-8 text-center" role="alert">
      <div className="text-4xl" aria-hidden>{copy.icon}</div>
      <h2 className="mt-3 text-lg font-semibold text-slate-800">{copy.title}</h2>
      <p className="mx-auto mt-1 max-w-md text-sm text-slate-600">{copy.body}</p>
      {(copy.recoverable || onRetry) && onRetry && (
        <button
          onClick={onRetry}
          className="mt-4 rounded-lg bg-slate-800 px-4 py-2 text-sm font-medium text-white hover:bg-slate-700"
        >
          Try again
        </button>
      )}
      {error.requestId && (
        <p className="mt-4 select-all font-mono text-xs text-slate-400">
          Reference: {error.requestId}
        </p>
      )}
    </div>
  );
}
