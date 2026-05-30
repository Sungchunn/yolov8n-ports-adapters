"use client";

import { DragEvent, KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
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

type AnalysisState =
  | { kind: "empty" }
  | { kind: "result"; payload: UploadResponse; frames: NormalizedFrame[]; rows: DetectionRow[] }
  | { kind: "error"; message: string };

const ACCEPTED_MEDIA =
  "image/jpeg,image/jpg,image/png,video/mp4,video/quicktime,video/x-msvideo,video/webm";

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
    setSelectedFile(file);
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
            <div className="media-preview">
              {selectedFile && previewUrl ? (
                selectedFile.type.startsWith("video/") ? (
                  <video src={previewUrl} controls muted playsInline />
                ) : (
                  <img src={previewUrl} alt={`Preview of ${selectedFile.name}`} />
                )
              ) : (
                <span>No media selected</span>
              )}
            </div>
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
