import { api, toSeethruError } from "./client";
import type { ImagePrediction, JobAccepted } from "./types";

/**
 * Upload an image for synchronous analysis.
 *
 * `onProgress` reports UPLOAD progress (0..1). axios exposes this via XHR;
 * fetch cannot, which is the reason this app uses axios (T61).
 */
export async function predictImage(
  file: File,
  onProgress?: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<ImagePrediction> {
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await api.post<ImagePrediction>("/predict/image", form, {
      signal,
      onUploadProgress: (e) => {
        if (onProgress && e.total) onProgress(e.loaded / e.total);
      },
    });
    return res.data;
  } catch (err) {
    throw toSeethruError(err);
  }
}

/** Submit a video for async analysis. Returns 202 + a job to poll (T61). */
export async function submitVideo(
  file: File,
  onProgress?: (fraction: number) => void,
  signal?: AbortSignal,
): Promise<JobAccepted> {
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await api.post<JobAccepted>("/predict/video", form, {
      signal,
      onUploadProgress: (e) => {
        if (onProgress && e.total) onProgress(e.loaded / e.total);
      },
    });
    return res.data;
  } catch (err) {
    throw toSeethruError(err);
  }
}
