const fileInput = document.querySelector("#file-input");
const chooseButton = document.querySelector("#choose-button");
const dropzone = document.querySelector("#dropzone");
const analyzeButton = document.querySelector("#analyze-button");
const clearButton = document.querySelector("#clear-button");
const mediaPreview = document.querySelector("#media-preview");
const fileMeta = document.querySelector("#file-meta");
const statusBadge = document.querySelector("#status-badge");
const resultSubtitle = document.querySelector("#result-subtitle");
const emptyState = document.querySelector("#empty-state");
const resultContent = document.querySelector("#result-content");
const detectionsBody = document.querySelector("#detections-body");
const detectionCount = document.querySelector("#detection-count");
const framesSection = document.querySelector("#frames-section");
const framesList = document.querySelector("#frames-list");
const frameCount = document.querySelector("#frame-count");

const metrics = {
  kind: document.querySelector("#metric-kind"),
  detections: document.querySelector("#metric-detections"),
  latency: document.querySelector("#metric-latency"),
  label: document.querySelector("#metric-label"),
};

let selectedFile = null;
let previewUrl = null;

chooseButton.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("click", (event) => {
  if (event.target !== chooseButton) {
    fileInput.click();
  }
});

dropzone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    fileInput.click();
  }
});

dropzone.addEventListener("dragover", (event) => {
  event.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => {
  dropzone.classList.remove("dragover");
});

dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("dragover");
  const [file] = event.dataTransfer.files;
  if (file) {
    selectFile(file);
  }
});

fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  if (file) {
    selectFile(file);
  }
});

analyzeButton.addEventListener("click", async () => {
  if (!selectedFile) {
    return;
  }

  setStatus("Analyzing", "loading");
  analyzeButton.disabled = true;
  resultSubtitle.textContent = "Uploading media and running inference.";

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch("/v1/detect/upload", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail || `Request failed with ${response.status}`);
    }

    renderResult(payload);
    setStatus("Complete", "success");
  } catch (error) {
    renderError(error);
    setStatus("Error", "error");
  } finally {
    analyzeButton.disabled = false;
  }
});

clearButton.addEventListener("click", resetUi);

function selectFile(file) {
  selectedFile = file;
  clearPreviewUrl();

  previewUrl = URL.createObjectURL(file);
  mediaPreview.replaceChildren();

  const mediaElement = file.type.startsWith("video/")
    ? document.createElement("video")
    : document.createElement("img");

  if (mediaElement instanceof HTMLVideoElement) {
    mediaElement.controls = true;
    mediaElement.muted = true;
    mediaElement.playsInline = true;
  } else {
    mediaElement.alt = `Preview of ${file.name}`;
  }

  mediaElement.src = previewUrl;
  mediaPreview.append(mediaElement);
  fileMeta.textContent = `${file.name} - ${file.type || "unknown type"} - ${formatBytes(file.size)}`;
  analyzeButton.disabled = false;
  clearButton.disabled = false;
  setStatus("Ready", "ready");
  resetResults();
  resultSubtitle.textContent = "Ready to analyze selected media.";
}

function renderResult(payload) {
  const normalizedFrames = normalizeFrames(payload);
  const rows = normalizedFrames.flatMap((frame) =>
    frame.result.detections.map((detection) => ({
      source: frame.source,
      detection,
    })),
  );

  const totalDetections = rows.length;
  const averageLatency = average(
    normalizedFrames.map((frame) => frame.result.inference_ms),
  );
  const topLabel = mostCommonLabel(rows);

  metrics.kind.textContent = titleCase(payload.kind);
  metrics.detections.textContent = String(totalDetections);
  metrics.latency.textContent = `${averageLatency.toFixed(1)} ms`;
  metrics.label.textContent = topLabel || "None";

  resultSubtitle.textContent =
    payload.kind === "video"
      ? `${normalizedFrames.length} sampled frames analyzed.`
      : "Still image analyzed.";

  emptyState.hidden = true;
  resultContent.hidden = false;
  renderDetections(rows);
  renderFrames(payload.kind === "video" ? normalizedFrames : []);
}

function renderDetections(rows) {
  detectionsBody.replaceChildren();
  detectionCount.textContent = `${rows.length} ${rows.length === 1 ? "row" : "rows"}`;

  if (rows.length === 0) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.textContent = "No detections returned for this media.";
    row.append(cell);
    detectionsBody.append(row);
    return;
  }

  for (const item of rows) {
    const row = document.createElement("tr");
    row.append(
      tableCell(item.source),
      tableCell(item.detection.label.name, "label-cell"),
      tableCell(formatPercent(item.detection.confidence), "confidence"),
      tableCell(formatBox(item.detection.box)),
      tableCell(formatNumber(item.detection.box.area)),
    );
    detectionsBody.append(row);
  }
}

function renderFrames(frames) {
  framesList.replaceChildren();
  frameCount.textContent = `${frames.length} ${frames.length === 1 ? "frame" : "frames"}`;
  framesSection.hidden = frames.length === 0;

  for (const frame of frames) {
    const card = document.createElement("article");
    card.className = "frame-card";

    const title = document.createElement("strong");
    title.textContent = `Frame ${frame.frameIndex}`;

    const time = document.createElement("span");
    time.textContent = `${frame.timestamp.toFixed(2)}s - ${frame.result.inference_ms.toFixed(1)} ms`;

    const details = document.createElement("span");
    details.textContent = `${frame.result.detections.length} detections - ${frame.result.model}`;

    card.append(title, time, details);
    framesList.append(card);
  }
}

function renderError(error) {
  resetResults();
  emptyState.hidden = false;
  resultContent.hidden = true;
  emptyState.replaceChildren();

  const title = document.createElement("h3");
  title.textContent = "Analysis failed";

  const message = document.createElement("p");
  message.className = "error-text";
  message.textContent = error instanceof Error ? error.message : "Unknown error";

  emptyState.append(title, message);
  resultSubtitle.textContent = "The upload did not complete.";
}

function resetUi() {
  selectedFile = null;
  fileInput.value = "";
  clearPreviewUrl();
  mediaPreview.replaceChildren();

  const emptyPreview = document.createElement("span");
  emptyPreview.textContent = "No media selected";
  mediaPreview.append(emptyPreview);

  fileMeta.textContent = "Select a file to start.";
  analyzeButton.disabled = true;
  clearButton.disabled = true;
  setStatus("Idle", "idle");
  resetResults();
  resultSubtitle.textContent = "Waiting for an upload.";
}

function resetResults() {
  metrics.kind.textContent = "-";
  metrics.detections.textContent = "-";
  metrics.latency.textContent = "-";
  metrics.label.textContent = "-";
  detectionsBody.replaceChildren();
  framesList.replaceChildren();
  framesSection.hidden = true;
  detectionCount.textContent = "0 rows";
  frameCount.textContent = "0 frames";
  resultContent.hidden = true;
  emptyState.hidden = false;
  emptyState.replaceChildren();

  const title = document.createElement("h3");
  title.textContent = "Ready for analysis";

  const message = document.createElement("p");
  message.textContent =
    "Upload a still image or video to see detection labels, confidence scores, bounding box data, and sampled frame timing.";

  emptyState.append(title, message);
}

function normalizeFrames(payload) {
  if (payload.kind === "image") {
    return [
      {
        source: "Image",
        result: payload.result,
        frameIndex: 0,
        timestamp: 0,
      },
    ];
  }

  return payload.frames.map((frame) => ({
    source: `Frame ${frame.frame_index} (${frame.timestamp_seconds.toFixed(2)}s)`,
    result: frame,
    frameIndex: frame.frame_index,
    timestamp: frame.timestamp_seconds,
  }));
}

function tableCell(value, className = "") {
  const cell = document.createElement("td");
  cell.textContent = value;
  if (className) {
    cell.className = className;
  }
  return cell;
}

function setStatus(label, state) {
  statusBadge.textContent = label;
  statusBadge.className = `status-badge ${state}`;
}

function mostCommonLabel(rows) {
  const counts = new Map();
  for (const row of rows) {
    const name = row.detection.label.name;
    counts.set(name, (counts.get(name) || 0) + 1);
  }

  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || "";
}

function average(values) {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function formatBox(box) {
  return `${formatNumber(box.x1)}, ${formatNumber(box.y1)} to ${formatNumber(box.x2)}, ${formatNumber(box.y2)}`;
}

function formatNumber(value) {
  return Number(value).toLocaleString(undefined, {
    maximumFractionDigits: 1,
  });
}

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function formatBytes(bytes) {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  const units = ["KB", "MB", "GB"];
  let size = bytes / 1024;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[unitIndex]}`;
}

function titleCase(value) {
  return `${value.charAt(0).toUpperCase()}${value.slice(1)}`;
}

function clearPreviewUrl() {
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
}
