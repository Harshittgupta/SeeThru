import { useCallback, useRef, useState } from "react";
import { formatBytes } from "../../lib/format";

// Client-side caps MIRROR the server (backend config, T55). Rejecting here saves
// a doomed 200 MB upload, but the server enforces the real limit -- a client
// check is a convenience, never the security boundary.
const IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];
const VIDEO_TYPES = ["video/mp4", "video/quicktime", "video/x-matroska", "video/x-msvideo"];
const MAX_IMAGE = 15 * 1024 * 1024;
const MAX_VIDEO = 200 * 1024 * 1024;

export interface PickedFile {
  file: File;
  kind: "image" | "video";
  previewUrl: string;
}

export function Dropzone({ onPick }: { onPick: (p: PickedFile) => void }) {
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const accept = useCallback(
    (file: File) => {
      setError(null);
      const isImage = IMAGE_TYPES.includes(file.type);
      const isVideo = VIDEO_TYPES.includes(file.type);
      if (!isImage && !isVideo) {
        setError("Please choose a JPEG/PNG/WebP image or an MP4/MOV/MKV video.");
        return;
      }
      const max = isImage ? MAX_IMAGE : MAX_VIDEO;
      if (file.size > max) {
        setError(`That file is ${formatBytes(file.size)}; the limit is ${formatBytes(max)}.`);
        return;
      }
      onPick({ file, kind: isImage ? "image" : "video", previewUrl: URL.createObjectURL(file) });
    },
    [onPick],
  );

  return (
    <div>
      <div
        role="button"
        tabIndex={0}
        aria-label="Upload an image or video"
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files[0];
          if (f) accept(f);
        }}
        className={`flex cursor-pointer flex-col items-center rounded-xl border-2 border-dashed p-12 text-center transition ${
          dragging ? "border-slate-500 bg-slate-50" : "border-slate-300 hover:border-slate-400"
        }`}
      >
        <div className="text-4xl" aria-hidden>⬆️</div>
        <p className="mt-3 font-medium text-slate-700">Drop an image or video, or click to choose</p>
        <p className="mt-1 text-xs text-slate-400">
          Images up to 15 MB · videos up to 200 MB · faces only
        </p>
        <input
          ref={inputRef}
          type="file"
          accept={[...IMAGE_TYPES, ...VIDEO_TYPES].join(",")}
          className="hidden"
          onChange={(e) => e.target.files?.[0] && accept(e.target.files[0])}
        />
      </div>
      {error && (
        <p className="mt-2 text-sm text-fake" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
