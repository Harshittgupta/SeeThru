import { api, toSeethruError } from "./client";
import type { JobStatus, VideoResult } from "./types";

export async function getJobStatus(jobId: string): Promise<JobStatus> {
  try {
    const res = await api.get<JobStatus>(`/jobs/${jobId}`);
    return res.data;
  } catch (err) {
    throw toSeethruError(err);
  }
}

export async function getJobResult(jobId: string): Promise<VideoResult> {
  try {
    const res = await api.get<VideoResult>(`/jobs/${jobId}/result`);
    return res.data;
  } catch (err) {
    throw toSeethruError(err);
  }
}

/**
 * Cancel a job. The frontend's Cancel button MUST hit this, or it is a lie that
 * abandons the client while the GPU keeps working (T57/T62).
 */
export async function cancelJob(jobId: string): Promise<void> {
  try {
    await api.delete(`/jobs/${jobId}`);
  } catch (err) {
    throw toSeethruError(err);
  }
}
