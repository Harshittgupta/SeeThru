import axios, { AxiosError } from "axios";
import type { ApiErrorBody, ErrorCode } from "./types";

// Relative baseURL, always. In dev, vite.config.ts proxies /v1 to the backend;
// in prod, nginx proxy_pass does (T65). Same-origin either way, so no CORS and
// no build-time env var that would bake one environment into the image.
export const api = axios.create({
  baseURL: "/v1",
  // Not too long: a hung request should surface as a network error the UI can
  // retry, not a spinner that never resolves. Video is async (jobs), so no
  // single request here is long-running.
  timeout: 30_000,
});

/** A normalized error the UI can switch on by `code` (T64). */
export class SeethruError extends Error {
  code: ErrorCode;
  requestId: string | null;
  retryAfter: number | null;

  constructor(code: ErrorCode, message: string, requestId: string | null, retryAfter: number | null) {
    super(message);
    this.code = code;
    this.requestId = requestId;
    this.retryAfter = retryAfter;
  }
}

/** Turn any axios failure into a SeethruError with a stable `code`. */
export function toSeethruError(err: unknown): SeethruError {
  if (err instanceof AxiosError) {
    const body = err.response?.data as ApiErrorBody | undefined;
    if (body?.error_code) {
      const retry = err.response?.headers?.["retry-after"];
      return new SeethruError(
        body.error_code as ErrorCode,
        body.message,
        body.request_id ?? null,
        retry ? Number(retry) : null,
      );
    }
    // No structured body -> the request never reached the API (offline, DNS,
    // proxy down). This is recoverable and the UI treats it as such.
    return new SeethruError(
      "network_error",
      "Could not reach the server.",
      null,
      null,
    );
  }
  return new SeethruError("internal_error", String(err), null, null);
}
