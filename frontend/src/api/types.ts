// Mirrors the backend response contract (backend/schemas/responses.py, T58).
// Kept in one place so a contract change surfaces as a compile error, not a
// runtime surprise in a component.

export type Verdict = "real" | "fake" | "uncertain";

export interface ModelBlock {
  arch: string;
  version: string | null;
  device: string;
  // If false, `scores` are NOT probabilities and the UI must not render a
  // percentage (T63). False today -- no calibration exists yet (backend T78).
  calibrated: boolean;
}

export interface BranchAttribution {
  branch: string;
  // Causal change in the fake logit when this branch is ablated. NOT a share:
  // the branches do not sum to 1, so this must never be a pie chart (T62).
  delta: number;
  baseline: string;
}

export interface FaceResult {
  face_id: number;
  // [x1, y1, x2, y2] in ORIGINAL image pixels. The displayed <img> is scaled,
  // so these must be mapped -- see lib/coords.ts (T62).
  bbox: [number, number, number, number];
  verdict: Verdict;
  scores: { real: number; fake: number };
  attribution: BranchAttribution[];
  // name -> URL under /v1/artifacts. Never base64 (T58).
  artifacts: Record<string, string>;
  hf_energy_ratio: number | null;
}

export interface Disclaimer {
  not_forensic_evidence: boolean;
  known_limitations: string[];
}

export interface ImagePrediction {
  request_id: string | null;
  model: ModelBlock;
  media: { kind: string; sha256: string; filename?: string; bytes?: number };
  faces: FaceResult[];
  summary: {
    verdict: Verdict;
    confidence: number;
    reasoning: string[];
    any_fake: boolean;
  };
  warnings: string[];
  disclaimer: Disclaimer;
}

export interface FrameScore {
  index: number;
  source_index: number;
  t_seconds: number | null;
  p_fake: number;
  attention_norm: number | null;
  suspicious: boolean;
  // A copied face (detection failed) or padding -- NOT an observation. The
  // timeline draws these hollow and never as evidence (T62).
  interpolated: boolean;
}

export interface TimelineSpan {
  start_s: number;
  end_s: number;
  mean_p_fake: number;
  n_frames: number;
}

export interface VideoResult {
  request_id: string | null;
  model: ModelBlock;
  media: { kind: string; sha256: string; duration_s?: number };
  verdict: Verdict;
  scores: { real: number; fake: number };
  attribution: BranchAttribution[];
  timeline: FrameScore[];
  spans: TimelineSpan[];
  artifacts: Record<string, string>;
  warnings: string[];
  disclaimer: Disclaimer;
}

export type JobState =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "expired";

export interface JobStatus {
  job_id: string;
  state: JobState;
  progress: number;
  stage: string | null;
  error_code: string | null;
  created_at: number;
  poll_url: string;
}

export interface JobAccepted {
  job_id: string;
  state: string;
  poll_url: string;
}

// The error envelope every failure shares (backend/core/errors.py, T56). The UI
// switches on `error_code`, never on the prose message (T64).
export interface ApiErrorBody {
  error_code: string;
  message: string;
  request_id: string | null;
  details?: Record<string, unknown>;
}

export type ErrorCode =
  | "no_face_detected"
  | "insufficient_faces"
  | "unreadable_media"
  | "unsupported_media"
  | "payload_too_large"
  | "model_not_ready"
  | "gpu_busy"
  | "queue_full"
  | "job_not_found"
  | "job_expired"
  | "internal_error"
  | "network_error";
