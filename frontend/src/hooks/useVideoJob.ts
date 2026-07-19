import { useQuery } from "@tanstack/react-query";
import { getJobResult, getJobStatus } from "../api/jobs";
import type { JobStatus, VideoResult } from "../api/types";

/**
 * Poll a video job to completion (BUILD_PLAN T61).
 *
 * The backoff schedule is measured to the problem: a video takes 30 s - 5 min,
 * so hammering at 1 s the whole time is wasteful, and polling at 5 s from the
 * start makes a fast job feel slow. So: fast at first, then ease off.
 *
 *   0-10 s : every 1 s   (a short video may already be done)
 *   10-60 s: every 2 s
 *   60 s+  : every 5 s
 *
 * On a network blip mid-poll the query RETRIES rather than erroring: the job is
 * still running server-side, so wiping the screen to an error page would be
 * wrong (T64). The last known status stays visible with a "reconnecting" hint.
 */
export function pollInterval(elapsedMs: number): number {
  if (elapsedMs < 10_000) return 1_000;
  if (elapsedMs < 60_000) return 2_000;
  return 5_000;
}

const TERMINAL: JobStatus["state"][] = ["succeeded", "failed", "expired"];

export function useJobStatus(jobId: string | undefined, startedAt: number) {
  return useQuery<JobStatus>({
    queryKey: ["job", jobId],
    queryFn: () => getJobStatus(jobId!),
    enabled: Boolean(jobId),
    // Stop once terminal; otherwise back off on a schedule keyed to elapsed time.
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      if (state && TERMINAL.includes(state)) return false;
      return pollInterval(Date.now() - startedAt);
    },
    // A poll failure is almost always transient (the job outlives one request),
    // so keep retrying and keep the last good status on screen (T64).
    retry: true,
    retryDelay: (attempt) => Math.min(1000 * 2 ** attempt, 10_000),
  });
}

export function useJobResult(jobId: string | undefined, ready: boolean) {
  return useQuery<VideoResult>({
    queryKey: ["job-result", jobId],
    queryFn: () => getJobResult(jobId!),
    enabled: Boolean(jobId) && ready,
    staleTime: Infinity, // a finished result never changes
  });
}
