// gui/static/app.js
// ---------------------------------------------------------------------------
// Browser-side logic for the S2ST demo page: submits the upload form (or a
// browser-recorded clip) to POST /api/run, then opens an EventSource on
// GET /api/run/<job_id>/stream to render per-stage progress and the final
// result as they arrive.
// ---------------------------------------------------------------------------

const form = document.getElementById("run-form");
const runBtn = document.getElementById("run-btn");
const progressSection = document.getElementById("progress");
const resultsSection = document.getElementById("results");
const errorBox = document.getElementById("error-box");

const recordBtn = document.getElementById("record-btn");
const recordStatus = document.getElementById("record-status");
const audioFileInput = document.getElementById("audio-file");

let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;

// --- Recording -------------------------------------------------------------

recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    };
    mediaRecorder.onstop = () => {
      recordedBlob = new Blob(recordedChunks, { type: "audio/webm" });
      stream.getTracks().forEach((t) => t.stop());
      recordBtn.textContent = "● Record";
      recordBtn.classList.remove("recording");
      recordStatus.textContent = `Recorded ${(recordedBlob.size / 1024).toFixed(0)} KB clip`;
      audioFileInput.value = ""; // recording takes precedence over a stale file selection
    };

    mediaRecorder.start();
    recordBtn.textContent = "■ Stop";
    recordBtn.classList.add("recording");
    recordStatus.textContent = "Recording…";
  } catch (err) {
    showError(`Microphone access failed: ${err.message}`);
  }
});

audioFileInput.addEventListener("change", () => {
  // A fresh file upload takes precedence over a previous recording.
  recordedBlob = null;
  recordStatus.textContent = "";
});

// --- Submit ------------------------------------------------------------------

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();

  const formData = new FormData(form);
  if (recordedBlob) {
    formData.set("audio", recordedBlob, "recording.webm");
  } else if (!audioFileInput.files[0]) {
    showError("Provide English audio: upload a file or record one.");
    return;
  }

  runBtn.disabled = true;
  resetStages();
  progressSection.hidden = false;
  resultsSection.hidden = true;

  try {
    const resp = await fetch("/api/run", { method: "POST", body: formData });
    const data = await resp.json();
    if (!resp.ok) {
      showError(data.error || `Request failed (HTTP ${resp.status})`);
      runBtn.disabled = false;
      return;
    }
    listenToJob(data.job_id);
  } catch (err) {
    showError(`Failed to start pipeline: ${err.message}`);
    runBtn.disabled = false;
  }
});

// --- SSE progress stream -----------------------------------------------------

function listenToJob(jobId) {
  const source = new EventSource(`/api/run/${jobId}/stream`);

  source.addEventListener("stage", (evt) => {
    const { stage, status, message } = JSON.parse(evt.data);
    const el = document.getElementById(`stage-${stage}`);
    if (!el) return;
    el.classList.remove("running", "done", "error");
    if (status === "running") el.classList.add("running");
    if (status === "done") el.classList.add("done");
    if (status === "error") el.classList.add("error");
    el.title = message || "";
  });

  source.addEventListener("result", (evt) => {
    const payload = JSON.parse(evt.data);
    renderResult(jobId, payload);
    source.close();
    runBtn.disabled = false;
  });

  source.onerror = () => {
    showError("Lost connection to the pipeline progress stream.");
    source.close();
    runBtn.disabled = false;
  };
}

function resetStages() {
  ["asr", "mt", "tts", "research"].forEach((s) => {
    const el = document.getElementById(`stage-${s}`);
    el.classList.remove("running", "done", "error");
    el.title = "";
  });
}

function renderResult(jobId, payload) {
  if (payload.error) {
    showError(payload.error);
    return;
  }

  resultsSection.hidden = false;
  document.getElementById("out-asr").textContent = payload.asr_text || "(empty)";
  document.getElementById("out-mt").textContent = payload.mt_text || "(empty)";

  const audioEl = document.getElementById("out-audio");
  audioEl.src = `/api/audio/${jobId}`;

  const baselineBlock = document.getElementById("baseline-audio-block");
  if (payload.has_baseline_audio) {
    baselineBlock.hidden = false;
    document.getElementById("out-audio-baseline").src = `/api/audio/${jobId}/baseline`;
  } else {
    baselineBlock.hidden = true;
  }

  const metrics = payload.metrics || {};
  const metricsEl = document.getElementById("out-metrics");
  if (Object.keys(metrics).length === 0) {
    metricsEl.textContent = "Not available yet — requires eval/ modules (not implemented in this baseline).";
  } else {
    metricsEl.textContent = Object.entries(metrics)
      .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(3) : v}`)
      .join(" · ");
  }

  const timings = payload.stage_timings_sec || {};
  document.getElementById("out-timings").textContent = Object.entries(timings)
    .map(([stage, sec]) => `${stage}: ${sec.toFixed(1)}s`)
    .join(" · ");
}

// --- Errors -------------------------------------------------------------------

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}

function hideError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}
