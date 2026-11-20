"use strict";

const element = (id) => document.getElementById(id);
const ui = {
  form: element("run-form"),
  videoInput: element("video-input"),
  truthInput: element("truth-input"),
  model: element("model-select"),
  profile: element("profile-select"),
  demo: element("demo-button"),
  run: element("run-button"),
  video: element("result-video"),
  dropzone: element("video-drop"),
  evaluationForm: element("evaluation-form"),
  evaluationInput: element("evaluation-input"),
  evaluationButton: element("evaluation-button"),
};

const stateNames = {
  queued: "Queued",
  processing: "Processing",
  complete: "Complete",
  failed: "Failed",
};
const profileNames = {
  balanced: "Balanced",
  recall: "Higher recall",
};
const downloadNames = {
  video: "Annotated video",
  predictions: "Tracking predictions",
  detections: "Raw detections",
};
const timingStages = {
  capture_read: "Read frame",
  capture_to_process: "Capture to processing",
  detector: "Detection (total)",
  model_inference: "Model inference",
  tracker_including_reid: "Tracking (including Re-ID)",
  tracker_excluding_reid: "Tracking (excluding Re-ID)",
  reid: "Re-identification",
  overlay: "Draw overlay",
  video_write: "Write annotated video",
  frame_processing: "Frame processing (total)",
  capture_to_output: "Capture to output",
};
const recoveryNames = {
  recovered: "Recovered",
  switched_identity: "ID switched",
  missed_reappearance: "Return missed",
  untracked_before_gap: "No ID before gap",
};

let options = { models: [], demo_available: false };
let runs = [];
let selectedId = null;
let chosenVideo = null;
let uploading = false;
const sendingEvaluations = new Set();
let evaluationMessage = null;
let historySignature = "";
let activeVideoUrl = "";
let pollTimer = null;

function notice(id, message) {
  const target = element(id);
  target.textContent = message || "";
  target.hidden = !message;
}

async function request(url, init) {
  const response = await fetch(url, { cache: "no-store", ...init });
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = data?.detail;
    const message = typeof detail === "string" ? detail : detail?.[0]?.msg;
    throw new Error(message || `Request failed (${response.status}).`);
  }
  if (data === null) throw new Error("The server returned an unreadable response.");
  return data;
}

function safeFileUrl(value) {
  if (typeof value !== "string" || !value) return "";
  try {
    const url = new URL(value, window.location.href);
    if (url.origin === window.location.origin && ["http:", "https:"].includes(url.protocol)) {
      return url.href;
    }
  } catch {
    return "";
  }
  return "";
}

function formatted(value, decimals = 0) {
  if (!Number.isFinite(value)) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: decimals });
}

function percent(value) {
  return Number.isFinite(value) ? `${formatted(value * 100, 1)}%` : "—";
}

function runDate(value) {
  const date = new Date(value || "");
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

function fileSize(file) {
  if (file.size < 1024 * 1024) return `${formatted(file.size / 1024, 1)} KiB`;
  return `${formatted(file.size / (1024 * 1024), 1)} MiB`;
}

function selectedRun() {
  return runs.find((run) => run.id === selectedId);
}

function setText(id, value) {
  element(id).textContent = value;
}

function updateControls() {
  ui.run.disabled = uploading || !chosenVideo || !ui.model.value;
  ui.demo.disabled = uploading || !options.demo_available;
  ui.model.disabled = !options.models.length;
  ui.profile.disabled = uploading || !ui.model.value;
  const model = options.models.find((model) => model.id === ui.model.value);
  setText("model-hint", model?.description || "");
  setText("profile-hint", ui.profile.value === "recall"
    ? "Retries with overlapping crops when a full frame has no strong detection. Tracks can start at lower confidence. Processing may take longer."
    : "Runs the detector on each full frame.");
}

function chooseVideo(file) {
  let error = "";
  if (file && !/\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(file.name)) {
    error = "Choose an MP4, MOV, AVI, MKV, WebM, or M4V video.";
  } else if (file?.size > 512 * 1024 * 1024) {
    error = "Choose a video up to 512 MiB.";
  }
  if (error) {
    file = null;
    ui.videoInput.value = "";
  }
  chosenVideo = file || null;
  notice("upload-error", error);
  ui.dropzone.classList.toggle("has-file", Boolean(file));
  setText("video-name", file?.name || "Choose a video");
  setText("video-help", file ? `${fileSize(file)} · Click to choose another` : "or drag it here · MP4, MOV, AVI, WebM");
  updateControls();
}

function selectRun(id) {
  if (id !== selectedId) {
    ui.evaluationInput.value = "";
    setText("evaluation-name", "Choose a MOT text file");
    evaluationMessage = null;
  }
  selectedId = id;
  render();
}

function upsertRun(run) {
  const previous = runs.findIndex((item) => item.id === run.id);
  if (previous === -1) {
    runs.unshift(run);
  } else {
    runs[previous] = run;
  }
}

function renderHistory() {
  const signature = JSON.stringify([selectedId, runs.map((run) => [run.id, run.name, run.model, run.status])]);
  if (signature === historySignature) return;
  historySignature = signature;
  const list = element("run-list");
  const scroll = list.scrollTop;
  const focusId = list.contains(document.activeElement) ? document.activeElement.dataset.runId : null;
  const entries = runs.map((run) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.className = "run-chip";
    button.dataset.runId = run.id;
    button.setAttribute("aria-pressed", String(run.id === selectedId));

    const name = document.createElement("span");
    name.className = "run-chip-name";
    name.textContent = run.name;
    name.title = run.name;
    const meta = document.createElement("span");
    meta.className = "run-chip-meta";
    meta.textContent = [run.model, runDate(run.created_at)].filter(Boolean).join(" · ");
    const status = document.createElement("span");
    status.className = `run-chip-state status-${stateNames[run.status] ? run.status : "idle"}`;
    status.textContent = stateNames[run.status] || "Unknown";

    button.append(name, meta, status);
    button.addEventListener("click", () => selectRun(run.id));
    item.append(button);
    return item;
  });
  list.replaceChildren(...entries);
  if (focusId) entries.find((item) => item.firstElementChild.dataset.runId === focusId)?.firstElementChild.focus();
  list.scrollTop = scroll;
}

function showDownloads(files) {
  const area = element("run-files");
  const signature = JSON.stringify(files || {});
  if (area.dataset.files === signature) return;
  area.dataset.files = signature;
  const links = [];
  for (const [key, label] of Object.entries(downloadNames)) {
    const url = safeFileUrl(files?.[key]);
    if (!url) continue;
    const link = document.createElement("a");
    link.className = "download-link";
    link.href = url;
    link.download = "";
    link.textContent = label;
    links.push(link);
  }
  area.replaceChildren(...links);
  area.hidden = !links.length;
}

function showScores(id, values) {
  const children = values.map(([label, value]) => {
    const cell = document.createElement("div");
    const term = document.createElement("dt");
    const amount = document.createElement("dd");
    term.textContent = label;
    amount.textContent = value;
    cell.append(term, amount);
    return cell;
  });
  element(id).replaceChildren(...children);
}

function formatSize(size) {
  const dimensions = Number.isFinite(size) ? [size, size] : size;
  if (!Array.isArray(dimensions) || dimensions.length !== 2 || !dimensions.every(Number.isFinite)) return "—";
  return dimensions.map((dimension) => formatted(dimension)).join(" × ");
}

function formatUnit(value, unit, decimals = 1) {
  return Number.isFinite(value) ? `${formatted(value, decimals)}${unit}` : "—";
}

function renderStatistics(run) {
  const stats = run?.stats;
  const url = safeFileUrl(run?.files?.stats);
  const download = element("statistics-download");
  download.hidden = !url;
  if (url) download.href = url;
  else download.removeAttribute("href");
  element("statistics-content").hidden = !stats;

  let lead = stats
    ? "CPU usage is relative to one core. Processing time excludes browser preview conversion and evaluation."
    : "Statistics appear after a run finishes.";
  if (run?.profile === "recall" && Number.isFinite(stats?.tile_fallback_frames)) {
    const count = stats.tile_fallback_frames;
    lead += ` Retried ${formatted(count)} ${count === 1 ? "frame" : "frames"} with tiles.`;
  }
  setText("statistics-intro", lead);
  if (!stats) return;

  showScores("statistics-metrics", [
    ["Source resolution", formatSize(stats.source_size)],
    ["Source FPS", formatted(stats.source_fps, 2)],
    ["Detector input", stats.detector_mode === "oracle" ? "Annotation replay" : formatSize(stats.detector_input_size)],
    ["Processing time", formatUnit(stats.elapsed_seconds, " s", 2)],
    ["Detector calls", formatted(stats.detector_calls)],
    ["Detection interval", stats.detector_interval === 1 ? "Every frame" : formatUnit(stats.detector_interval, " frames", 0)],
    ["Process CPU", formatUnit(stats.process_cpu_percent, "%")],
    ["Peak memory", formatUnit(stats.peak_rss_mib, " MiB")],
  ]);

  const latency = (value) => value > 0 && value < 0.001 ? "<0.001" : formatted(value, 3);
  const rows = [];
  for (const [stage, timing] of Object.entries(stats.timings_ms || {})) {
    const row = document.createElement("tr");
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = timingStages[stage] || stage.replaceAll("_", " ");
    row.append(heading);
    for (const value of [formatted(timing.count), ...[timing.mean, timing.p50, timing.p95, timing.p99].map(latency)]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    rows.push(row);
  }
  element("timing-rows").replaceChildren(...rows);
  element("timing-details").hidden = !rows.length;
}

function showRecovery(events) {
  const rows = events.map((event) => {
    const row = document.createElement("tr");
    for (const value of [
      formatted(event.truth_id),
      formatted(event.reentry_frame),
      `${formatted(event.gap_frames)} frames`,
      recoveryNames[event.outcome] || String(event.outcome || "—").replaceAll("_", " "),
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    return row;
  });
  element("recovery-rows").replaceChildren(...rows);
  element("recovery-details").hidden = !rows.length;
}

function renderEvaluation(run) {
  const report = run?.evaluation;
  const pending = run?.evaluation_pending || sendingEvaluations.has(run?.id);
  const oracle = run?.stats?.detector_mode === "oracle" || report?.scope === "oracle_detection_diagnostic";
  const url = safeFileUrl(run?.files?.evaluation);
  const download = element("evaluation-download");
  download.hidden = !url;
  if (url) download.href = url;
  else download.removeAttribute("href");

  let lead = "Compare the run with a matching MOT annotation file.";
  if (pending) lead = "Evaluating predictions against ground truth…";
  else if (report && oracle) lead = "Synthetic tracking diagnostic using annotated boxes.";
  else if (report) lead = `Compared with ground truth at IoU ${formatted(report.iou_threshold, 2)}.`;
  else if (run?.status === "complete") lead = "Add matching MOT ground truth to score this run.";
  setText("evaluation-intro", lead);
  element("evaluation-results").hidden = !report;
  element("detection-results").hidden = !report?.detections;
  if (report?.detections) {
    showScores("detection-metrics", [
      ["Precision", percent(report.detections.precision)],
      ["Recall", percent(report.detections.recall)],
      ["F1", percent(report.detections.f1)],
      ["False positives", formatted(report.detections.false_positives)],
    ]);
  }
  if (report) {
    const tracks = report.tracking;
    const truthCount = tracks?.true_positives + tracks?.false_negatives;
    const mota = tracks?.mota ?? (truthCount > 0
      ? 1 - (tracks.false_positives + tracks.false_negatives + tracks.id_switches) / truthCount
      : null);
    setText("submission-scores-title", oracle ? "Oracle tracking scores" : "Submission scores");
    showScores("tracking-metrics", [
      ["Detection F1", percent(tracks?.f1)],
      ["Precision", percent(tracks?.precision)],
      ["Recall", percent(tracks?.recall)],
      ["IDF1", percent(tracks?.idf1)],
      ["ID switches", formatted(tracks?.id_switches)],
      ["MOTA", percent(mota)],
      ["Matched boxes", formatted(tracks?.true_positives)],
      ["IDs recovered", report.recovery ? `${formatted(report.recovery.recovered)} / ${formatted(report.recovery.events)}` : "—"],
    ]);
  }
  showRecovery(report?.recovery?.details || []);

  ui.evaluationForm.hidden = run?.status !== "complete" || oracle;
  ui.evaluationInput.disabled = Boolean(pending);
  ui.evaluationButton.disabled = Boolean(pending) || !ui.evaluationInput.files.length;
  ui.evaluationButton.textContent = pending ? "Evaluating…" : "Evaluate";
  notice("evaluation-error", evaluationMessage || run?.evaluation_error);
}

function render() {
  const run = selectedRun();
  const status = stateNames[run?.status] ? run.status : "idle";
  const statusElement = element("run-status");
  const statusLabel = stateNames[status] || "No run selected";
  statusElement.className = `status status-${status}`;
  if (statusElement.lastElementChild.textContent !== statusLabel) {
    statusElement.lastElementChild.textContent = statusLabel;
  }
  notice("run-meta", run ? [run.name, run.model, profileNames[run.profile]].filter(Boolean).join(" · ") : "");

  const videoUrl = safeFileUrl(run?.files?.video);
  if (videoUrl !== activeVideoUrl) {
    activeVideoUrl = videoUrl;
    notice("video-error", "");
    if (videoUrl) ui.video.src = videoUrl;
    else {
      ui.video.removeAttribute("src");
      ui.video.load();
    }
  }
  ui.video.hidden = !videoUrl;
  element("player-empty").hidden = Boolean(videoUrl);
  setText("player-title", ({
    queued: "Waiting to start",
    processing: "Processing video",
    failed: "This run did not finish",
    complete: "Video preview unavailable",
    idle: "No video processed",
  })[status]);
  setText("player-message", run?.message || ({
    queued: "The test will start when the previous run finishes.",
    processing: "Detection and tracking are running on your machine.",
    failed: "Review the run error, then try another video.",
    complete: "Check the available result files below.",
    idle: "Upload a video or run the synthetic demo.",
  })[status]);
  if (videoUrl && status === "complete" && run.message) notice("video-error", run.message);
  setText("stat-frames", formatted(run?.stats?.frames));
  setText("stat-fps", formatted(run?.stats?.processing_fps, 1));
  setText("stat-tracks", formatted(run?.stats?.tracks_emitted));
  setText("stat-recovered", formatted(run?.stats?.reidentified_events));
  showDownloads(run?.files);
  renderHistory();
  renderStatistics(run);
  renderEvaluation(run);
}

function isPending(run) {
  return run.status === "queued" || run.status === "processing" || run.evaluation_pending;
}

function pollRuns() {
  if (pollTimer !== null || !runs.some(isPending)) return;
  pollTimer = window.setTimeout(async () => {
    try {
      const updates = await Promise.all(runs.filter(isPending).map((run) => request(`/api/runs/${encodeURIComponent(run.id)}`)));
      for (const run of updates) upsertRun(run);
      notice("history-message", "");
      render();
    } catch (error) {
      notice("history-message", `Could not refresh runs. Retrying… ${error.message}`);
    } finally {
      pollTimer = null;
      pollRuns();
    }
  }, 1200);
}

async function submitRun(demo = false) {
  if (uploading || (demo ? !options.demo_available : !chosenVideo || !ui.model.value)) return;
  notice("upload-error", "");
  uploading = true;
  updateControls();
  notice("upload-status", demo ? "Starting synthetic demo…" : "Uploading files…");
  const data = new FormData();
  if (demo) {
    data.append("demo", "true");
  } else {
    data.append("video", chosenVideo);
    data.append("model", ui.model.value);
    data.append("profile", ui.profile.value);
    if (ui.truthInput.files.length) data.append("ground_truth", ui.truthInput.files[0]);
  }

  try {
    const run = await request("/api/runs", { method: "POST", body: data });
    upsertRun(run);
    notice("history-message", "");
    selectRun(run.id);
    if (window.matchMedia("(max-width: 735px)").matches) {
      element("result-title").scrollIntoView({ block: "start" });
    }
    pollRuns();
  } catch (error) {
    notice("upload-error", error.message);
  } finally {
    uploading = false;
    notice("upload-status", "");
    updateControls();
  }
}

ui.videoInput.addEventListener("change", () => chooseVideo(ui.videoInput.files[0]));
function updateTruthChoice() {
  const file = ui.truthInput.files[0];
  setText("truth-name", file?.name || "Choose a MOT text file");
  element("truth-remove").hidden = !file;
  notice("upload-error", "");
}

ui.truthInput.addEventListener("change", updateTruthChoice);
element("truth-remove").addEventListener("click", () => {
  ui.truthInput.value = "";
  updateTruthChoice();
  ui.truthInput.focus();
});
ui.model.addEventListener("change", updateControls);
ui.profile.addEventListener("change", updateControls);
for (const eventName of ["dragenter", "dragover"]) {
  ui.dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    ui.dropzone.classList.add("drag-active");
  });
}
ui.dropzone.addEventListener("dragleave", (event) => {
  if (!ui.dropzone.contains(event.relatedTarget)) ui.dropzone.classList.remove("drag-active");
});
ui.dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  ui.dropzone.classList.remove("drag-active");
  if (!event.dataTransfer?.files.length) return;
  ui.videoInput.value = "";
  chooseVideo(event.dataTransfer.files[0]);
});
ui.form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitRun();
});
ui.demo.addEventListener("click", () => submitRun(true));
ui.video.addEventListener("error", () => {
  if (activeVideoUrl) notice("video-error", "The preview could not play in this browser. Download the annotated video to open it locally.");
});

ui.evaluationInput.addEventListener("change", () => {
  setText("evaluation-name", ui.evaluationInput.files[0]?.name || "Choose a MOT text file");
  evaluationMessage = null;
  renderEvaluation(selectedRun());
});
ui.evaluationForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const run = selectedRun();
  const file = ui.evaluationInput.files[0];
  if (!run || !file || sendingEvaluations.has(run.id) || run.evaluation_pending) return;
  sendingEvaluations.add(run.id);
  evaluationMessage = null;
  renderEvaluation(run);
  const data = new FormData();
  data.append("ground_truth", file);
  try {
    upsertRun(await request(`/api/runs/${encodeURIComponent(run.id)}/evaluation`, { method: "POST", body: data }));
    if (selectedId === run.id) {
      ui.evaluationInput.value = "";
      setText("evaluation-name", "Choose a MOT text file");
    }
  } catch (error) {
    if (selectedId === run.id) evaluationMessage = error.message;
  } finally {
    sendingEvaluations.delete(run.id);
    render();
    pollRuns();
  }
});

async function initialize() {
  const [available, history] = await Promise.allSettled([request("/api/options"), request("/api/runs")]);
  if (available.status === "fulfilled") {
    options = available.value;
    ui.model.replaceChildren();
    for (const model of options.models) {
      ui.model.add(new Option(model.label, model.id));
    }
    if (!options.models.length) {
      ui.model.add(new Option("No detector installed", ""));
      notice("startup-error", "Local model weights were not found. You can still run the synthetic demo.");
    }
  } else {
    notice("startup-error", `Could not load detector options. ${available.reason.message}`);
  }

  if (history.status === "fulfilled") {
    runs = history.value;
    notice("history-message", runs.length ? "" : "Completed and active tests will appear here.");
    selectRun(runs[0]?.id || null);
    pollRuns();
  } else {
    notice("history-message", `Could not load recent runs. ${history.reason.message}`);
  }
  updateControls();
}

initialize();
