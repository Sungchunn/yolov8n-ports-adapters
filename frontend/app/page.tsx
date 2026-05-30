"use client";

import {
  CSSProperties,
  DragEvent,
  KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  API_BASE_URL,
  DetectionRow,
  NormalizedFrame,
  SampleMedia,
  UploadResponse,
  average,
  formatBox,
  formatBytes,
  formatNumber,
  formatPercent,
  mostCommonLabel,
  normalizeFrames,
  rowsFromFrames,
  titleCase,
} from "@/lib/api";

type StatusState = "idle" | "ready" | "loading" | "success" | "error";
type MediaKind = "image" | "video" | "unknown";

type InferredMedia = {
  kind: MediaKind;
  contentType: string;
  extension: string;
};

type AnalysisState =
  | { kind: "empty" }
  | { kind: "result"; payload: UploadResponse; frames: NormalizedFrame[]; rows: DetectionRow[] }
  | { kind: "error"; message: string };

const ACCEPTED_MEDIA =
  ".jpg,.jpeg,.png,.mp4,.mov,.avi,.webm,image/jpeg,image/jpg,image/png,video/mp4,video/quicktime,video/x-msvideo,video/webm";

const SUPPORTED_MEDIA_TYPES: Record<string, MediaKind> = {
  "image/jpeg": "image",
  "image/jpg": "image",
  "image/png": "image",
  "video/mp4": "video",
  "video/quicktime": "video",
  "video/x-msvideo": "video",
  "video/webm": "video",
};

const MEDIA_TYPES_BY_EXTENSION: Record<string, string> = {
  avi: "video/x-msvideo",
  jpeg: "image/jpeg",
  jpg: "image/jpeg",
  mov: "video/quicktime",
  mp4: "video/mp4",
  png: "image/png",
  webm: "video/webm",
};

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [samples, setSamples] = useState<SampleMedia[]>([]);
  const [sampleLoadingId, setSampleLoadingId] = useState<string | null>(null);
  const [status, setStatus] = useState<{ label: string; state: StatusState }>({
    label: "Idle",
    state: "idle",
  });
  const [subtitle, setSubtitle] = useState("Waiting for an upload.");
  const [analysis, setAnalysis] = useState<AnalysisState>({ kind: "empty" });
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  useEffect(() => {
    let active = true;

    async function loadSampleLibrary() {
      try {
        const response = await fetch("/assets/demo/videos.json");
        if (!response.ok) {
          throw new Error(`Sample manifest request failed with ${response.status}`);
        }
        const payload = (await response.json()) as SampleMedia[];
        if (active) {
          setSamples(payload);
        }
      } catch {
        if (active) {
          setSamples([]);
        }
      }
    }

    loadSampleLibrary();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl(null);
      return;
    }

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const metrics = useMemo(() => {
    if (analysis.kind !== "result") {
      return {
        kind: "-",
        detections: "-",
        latency: "-",
        label: "-",
      };
    }

    const totalDetections = analysis.rows.length;
    const averageLatency = average(
      analysis.frames.map((frame) => frame.result.inference_ms),
    );

    return {
      kind: titleCase(analysis.payload.kind),
      detections: String(totalDetections),
      latency: `${averageLatency.toFixed(1)} ms`,
      label: mostCommonLabel(analysis.rows) || "None",
    };
  }, [analysis]);

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function selectFile(file: File) {
    setSelectedFile(fileWithInferredType(file));
    setStatus({ label: "Ready", state: "ready" });
    setSubtitle("Ready to analyze selected media.");
    setAnalysis({ kind: "empty" });
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    const [file] = event.dataTransfer.files;
    if (file) {
      selectFile(file);
    }
  }

  function handleDropzoneKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFilePicker();
    }
  }

  async function selectSample(sample: SampleMedia) {
    setSampleLoadingId(sample.id);

    try {
      const response = await fetch(sample.src);
      if (!response.ok) {
        throw new Error(`Sample request failed with ${response.status}`);
      }

      const blob = await response.blob();
      const fileName = sample.src.split("/").pop() || `${sample.id}.mp4`;
      selectFile(
        new File([blob], fileName, {
          type: sample.contentType || blob.type || "application/octet-stream",
        }),
      );
    } catch (error) {
      renderError(error);
    } finally {
      setSampleLoadingId(null);
    }
  }

  async function analyzeSelectedFile() {
    if (!selectedFile) {
      return;
    }

    setIsAnalyzing(true);
    setStatus({ label: "Analyzing", state: "loading" });
    setSubtitle("Uploading media and running inference.");

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/v1/detect/upload`, {
        method: "POST",
        body: formData,
      });
      const payload = (await response.json()) as UploadResponse | { detail?: string };

      if (!response.ok) {
        throw new Error(
          "detail" in payload && payload.detail
            ? payload.detail
            : `Request failed with ${response.status}`,
        );
      }

      const uploadResponse = payload as UploadResponse;
      const frames = normalizeFrames(uploadResponse);
      const rows = rowsFromFrames(frames);
      setAnalysis({ kind: "result", payload: uploadResponse, frames, rows });
      setSubtitle(
        uploadResponse.kind === "video"
          ? `${frames.length} sampled frames analyzed.`
          : "Still image analyzed.",
      );
      setStatus({ label: "Complete", state: "success" });
    } catch (error) {
      renderError(error);
    } finally {
      setIsAnalyzing(false);
    }
  }

  function renderError(error: unknown) {
    setAnalysis({
      kind: "error",
      message: error instanceof Error ? error.message : "Unknown error",
    });
    setSubtitle("The upload did not complete.");
    setStatus({ label: "Error", state: "error" });
  }

  function clearSelection() {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    setStatus({ label: "Idle", state: "idle" });
    setSubtitle("Waiting for an upload.");
    setAnalysis({ kind: "empty" });
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true">
            VI
          </span>
          <div>
            <h1>Vision Inference</h1>
            <p>Upload-only object detection preview</p>
          </div>
        </div>
        <div className="endpoint-pill">
          {API_BASE_URL || "same origin"} /v1/detect/upload
        </div>
      </header>

      <div className="workspace">
        <section className="panel intake-panel" aria-labelledby="upload-title">
          <div className="panel-heading">
            <div>
              <h2 id="upload-title">Media Intake</h2>
              <p>Images and sampled videos are analyzed through the FastAPI backend.</p>
            </div>
          </div>

          <div
            className={`dropzone${isDragging ? " dragover" : ""}`}
            role="button"
            tabIndex={0}
            aria-label="Choose or drop media file"
            onClick={(event) => {
              if (event.target instanceof HTMLElement && event.target.closest("button")) {
                return;
              }
              openFilePicker();
            }}
            onKeyDown={handleDropzoneKeyDown}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPTED_MEDIA}
              hidden
              onChange={(event) => {
                const [file] = event.currentTarget.files ?? [];
                if (file) {
                  selectFile(file);
                }
              }}
            />
            <div className="drop-icon" aria-hidden="true" />
            <h3>Drop media here</h3>
            <p>JPEG, PNG, MP4, MOV, AVI, or WebM. Default max upload is 20 MB.</p>
            <button className="button primary" type="button" onClick={openFilePicker}>
              Choose File
            </button>
          </div>

          <div className="preview-shell" aria-live="polite">
            <TaggedMediaPreview
              analysis={analysis}
              previewUrl={previewUrl}
              selectedFile={selectedFile}
            />
            <div className="file-meta">
              {selectedFile
                ? `${selectedFile.name} - ${selectedFile.type || "unknown type"} - ${formatBytes(
                    selectedFile.size,
                  )}`
                : "Select a file to start."}
            </div>
          </div>

          <section className="sample-library" aria-labelledby="samples-title">
            <div className="section-heading">
              <h3 id="samples-title">Sample Library</h3>
              <span>
                {samples.length} {samples.length === 1 ? "sample" : "samples"}
              </span>
            </div>
            <div className="sample-list">
              {samples.length === 0 ? (
                <p>No bundled samples were found.</p>
              ) : (
                samples.map((sample) => (
                  <button
                    key={sample.id}
                    className="sample-button"
                    type="button"
                    disabled={sampleLoadingId === sample.id}
                    aria-busy={sampleLoadingId === sample.id}
                    onClick={() => selectSample(sample)}
                  >
                    <strong>{sample.label}</strong>
                    <span>{sample.contentType}</span>
                  </button>
                ))
              )}
            </div>
          </section>

          <div className="action-row">
            <button
              className="button primary"
              type="button"
              disabled={!selectedFile || isAnalyzing}
              onClick={analyzeSelectedFile}
            >
              Analyze
            </button>
            <button
              className="button secondary"
              type="button"
              disabled={!selectedFile || isAnalyzing}
              onClick={clearSelection}
            >
              Clear
            </button>
          </div>
        </section>

        <section className="panel results-panel" aria-labelledby="results-title">
          <div className="panel-heading results-heading">
            <div>
              <h2 id="results-title">Detection Result</h2>
              <p>{subtitle}</p>
            </div>
            <span className={`status-badge ${status.state}`}>{status.label}</span>
          </div>

          <div className="metric-grid" aria-label="Detection summary">
            <MetricTile label="Media" value={metrics.kind} />
            <MetricTile label="Detections" value={metrics.detections} />
            <MetricTile label="Avg Inference" value={metrics.latency} />
            <MetricTile label="Top Label" value={metrics.label} />
          </div>

          {analysis.kind === "result" ? (
            <ResultContent analysis={analysis} />
          ) : (
            <div className="empty-state">
              {analysis.kind === "error" ? (
                <>
                  <h3>Analysis failed</h3>
                  <p className="error-text">{analysis.message}</p>
                </>
              ) : (
                <>
                  <h3>Ready for analysis</h3>
                  <p>
                    Upload a still image or video to see detection labels,
                    confidence scores, bounding box data, and sampled frame timing.
                  </p>
                </>
              )}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

function TaggedMediaPreview({
  analysis,
  previewUrl,
  selectedFile,
}: {
  analysis: AnalysisState;
  previewUrl: string | null;
  selectedFile: File | null;
}) {
  const [currentTime, setCurrentTime] = useState(0);
  const [videoPreviewFailed, setVideoPreviewFailed] = useState(false);
  const inferred = selectedFile ? inferMedia(selectedFile) : null;
  const mediaKind = inferred?.kind ?? "unknown";
  const overlayFrame =
    analysis.kind === "result"
      ? analysis.payload.kind === "video"
        ? frameForPlaybackTime(analysis.frames, currentTime)
        : analysis.frames[0]
      : undefined;

  useEffect(() => {
    setCurrentTime(0);
    setVideoPreviewFailed(false);
  }, [previewUrl]);

  if (!selectedFile || !previewUrl) {
    return (
      <div className="media-preview">
        <span>No media selected</span>
      </div>
    );
  }

  if (mediaKind === "video") {
    const isAvi = inferred?.extension === "avi";

    return (
      <div className="media-preview media-stage">
        <video
          key={previewUrl}
          controls
          muted
          playsInline
          onError={() => setVideoPreviewFailed(true)}
          onLoadedData={() => setVideoPreviewFailed(false)}
          onTimeUpdate={(event) => setCurrentTime(event.currentTarget.currentTime)}
        >
          <source src={previewUrl} type={selectedFile.type || inferred?.contentType} />
          Preview is unavailable for this video format.
        </video>
        {videoPreviewFailed ? (
          <div className="video-preview-fallback" role="status">
            {isAvi
              ? "AVI preview is not supported by this browser. Upload analysis is still available."
              : "Preview is unavailable for this video. Upload analysis is still available."}
          </div>
        ) : null}
        <DetectionOverlay frame={overlayFrame} />
      </div>
    );
  }

  if (mediaKind === "image") {
    return (
      <div className="media-preview media-stage">
        <img src={previewUrl} alt={`Preview of ${selectedFile.name}`} />
        <DetectionOverlay frame={overlayFrame} />
      </div>
    );
  }

  return (
    <div className="media-preview">
      <span>Preview unavailable for this file type</span>
    </div>
  );
}

function inferMedia(file: Pick<File, "name" | "type">): InferredMedia {
  const normalizedType = file.type.toLowerCase();
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  const extensionType = MEDIA_TYPES_BY_EXTENSION[extension] ?? "";
  const canonicalType = normalizedType === "image/jpg" ? "image/jpeg" : normalizedType;
  const contentType =
    canonicalType && SUPPORTED_MEDIA_TYPES[canonicalType]
      ? canonicalType
      : extensionType || canonicalType;

  return {
    kind: SUPPORTED_MEDIA_TYPES[contentType] ?? "unknown",
    contentType,
    extension,
  };
}

function fileWithInferredType(file: File): File {
  const inferred = inferMedia(file);
  if (
    !inferred.contentType ||
    inferred.contentType === file.type ||
    inferred.kind === "unknown"
  ) {
    return file;
  }

  return new File([file], file.name, {
    lastModified: file.lastModified,
    type: inferred.contentType,
  });
}

function DetectionOverlay({ frame }: { frame?: NormalizedFrame }) {
  if (!frame || frame.result.detections.length === 0) {
    return null;
  }

  const { width, height } = frame.result.image;
  if (width <= 0 || height <= 0) {
    return null;
  }

  return (
    <div className="detection-layer" style={overlayLayerStyle(width, height)}>
      {frame.result.detections.map((detection, index) => (
        <div
          className="detection-box"
          key={`${frame.frameIndex}-${detection.label.class_id}-${index}`}
          style={boxStyle(detection.box, width, height)}
        >
          <span className="detection-label">
            {detection.label.name} {formatPercent(detection.confidence)}
          </span>
        </div>
      ))}
    </div>
  );
}

function frameForPlaybackTime(frames: NormalizedFrame[], currentTime: number) {
  if (frames.length === 0) {
    return undefined;
  }

  let activeFrame = frames[0];
  for (const frame of frames) {
    if (frame.timestamp > currentTime) {
      break;
    }
    activeFrame = frame;
  }
  return activeFrame;
}

function overlayLayerStyle(width: number, height: number): CSSProperties {
  const mediaRatio = width / height;
  const stageRatio = 16 / 9;

  if (mediaRatio > stageRatio) {
    const scaledHeight = (stageRatio / mediaRatio) * 100;
    return {
      height: `${scaledHeight}%`,
      left: 0,
      top: `${(100 - scaledHeight) / 2}%`,
      width: "100%",
    };
  }

  const scaledWidth = (mediaRatio / stageRatio) * 100;
  return {
    height: "100%",
    left: `${(100 - scaledWidth) / 2}%`,
    top: 0,
    width: `${scaledWidth}%`,
  };
}

function boxStyle(
  box: NormalizedFrame["result"]["detections"][number]["box"],
  imageWidth: number,
  imageHeight: number,
): CSSProperties {
  const left = clampPercent((box.x1 / imageWidth) * 100);
  const top = clampPercent((box.y1 / imageHeight) * 100);
  const right = clampPercent((box.x2 / imageWidth) * 100);
  const bottom = clampPercent((box.y2 / imageHeight) * 100);

  return {
    height: `${Math.max(0, bottom - top)}%`,
    left: `${left}%`,
    top: `${top}%`,
    width: `${Math.max(0, right - left)}%`,
  };
}

function clampPercent(value: number) {
  return Math.min(100, Math.max(0, value));
}

function MetricTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ResultContent({
  analysis,
}: {
  analysis: Extract<AnalysisState, { kind: "result" }>;
}) {
  const videoFrames = analysis.payload.kind === "video" ? analysis.frames : [];

  return (
    <div className="result-content">
      <section className="result-section" aria-labelledby="detections-title">
        <div className="section-heading">
          <h3 id="detections-title">Detections</h3>
          <span>
            {analysis.rows.length} {analysis.rows.length === 1 ? "row" : "rows"}
          </span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Source</th>
                <th>Label</th>
                <th>Confidence</th>
                <th>Box</th>
                <th>Area</th>
              </tr>
            </thead>
            <tbody>
              {analysis.rows.length === 0 ? (
                <tr>
                  <td colSpan={5}>No detections returned for this media.</td>
                </tr>
              ) : (
                analysis.rows.map((row, index) => (
                  <tr key={`${row.source}-${row.detection.label.name}-${index}`}>
                    <td>{row.source}</td>
                    <td className="label-cell">{row.detection.label.name}</td>
                    <td className="confidence">
                      {formatPercent(row.detection.confidence)}
                    </td>
                    <td>{formatBox(row.detection.box)}</td>
                    <td>{formatNumber(row.detection.box.area)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>

      {videoFrames.length > 0 ? (
        <section className="result-section" aria-labelledby="frames-title">
          <div className="section-heading">
            <h3 id="frames-title">Sampled Frames</h3>
            <span>
              {videoFrames.length} {videoFrames.length === 1 ? "frame" : "frames"}
            </span>
          </div>
          <div className="frames-list">
            {videoFrames.map((frame) => (
              <article className="frame-card" key={frame.frameIndex}>
                <strong>Frame {frame.frameIndex}</strong>
                <span>
                  {frame.timestamp.toFixed(2)}s -{" "}
                  {frame.result.inference_ms.toFixed(1)} ms
                </span>
                <span>
                  {frame.result.detections.length} detections - {frame.result.model}
                </span>
              </article>
            ))}
          </div>
        </section>
      ) : null}
    </div>
  );
}
