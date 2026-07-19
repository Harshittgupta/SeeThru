import { useMemo, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { SeethruError } from "../api/client";
import { cancelJob } from "../api/jobs";
import { useJobResult, useJobStatus } from "../hooks/useVideoJob";
import { ErrorState } from "../components/state/ErrorState";
import { WaitingRoom, useElapsed, useTabTitle } from "../components/state/WaitingRoom";
import { VideoResultView } from "./Result";

/**
 * The video-analysis screen, keyed on the URL job id (BUILD_PLAN T61).
 *
 * Because the job id is in the path, a refresh or a shared link lands here and
 * resumes polling cold -- no state is lost. The poster image (if we have it from
 * the upload nav state) is shown while waiting; on a cold resume it is simply
 * absent, which is fine.
 */
export function Analyze() {
  const { jobId } = useParams<{ jobId: string }>();
  const location = useLocation();
  const navigate = useNavigate();
  const posterUrl = (location.state as { posterUrl?: string } | null)?.posterUrl ?? null;

  // A stable start time for the backoff schedule + elapsed clock.
  const [startedAt] = useState(() => Date.now());
  const elapsed = useElapsed(startedAt);

  const status = useJobStatus(jobId, startedAt);
  const state = status.data?.state;
  const done = state === "succeeded";
  const failed = state === "failed" || state === "expired";
  const result = useJobResult(jobId, done);

  const stage = status.data?.stage ?? state ?? "queued";
  useTabTitle(done ? "SEETHRU — done" : `SEETHRU — ${stage}`);

  const failure = useMemo<SeethruError | null>(() => {
    if (failed) {
      const code = status.data?.error_code ?? "internal_error";
      return new SeethruError(code as SeethruError["code"], "The analysis could not be completed.", jobId ?? null, null);
    }
    // A hard status error (not just a transient poll miss) after retries.
    if (status.isError && status.error instanceof SeethruError && status.error.code === "job_not_found") {
      return status.error;
    }
    return null;
  }, [failed, status.data?.error_code, status.isError, status.error, jobId]);

  async function onCancel() {
    if (jobId) {
      try {
        await cancelJob(jobId);
      } catch {
        /* best effort — navigating away is the user's intent regardless */
      }
    }
    navigate("/");
  }

  if (failure) return <ErrorState error={failure} onRetry={() => navigate("/")} />;

  if (done && result.data) {
    return <VideoResultView result={result.data} posterUrl={posterUrl} />;
  }

  // A network blip mid-poll: keep the last known status, show "reconnecting",
  // never wipe to an error page (T64).
  const reconnecting = status.isError && !failure;

  return (
    <WaitingRoom
      status={status.data}
      elapsedMs={elapsed}
      reconnecting={reconnecting}
      posterUrl={posterUrl}
      onCancel={onCancel}
    />
  );
}
