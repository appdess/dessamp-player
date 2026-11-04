const state = {
  config: null,
  projects: [],
  current: null,
  devices: [],
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-Capture-Token", window.CAPTURE_TOKEN);
  if (options.json !== undefined) {
    headers.set("Content-Type", "application/json");
    options.body = JSON.stringify(options.json);
    delete options.json;
  }
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  const type = response.headers.get("content-type") || "";
  return type.includes("application/json") ? response.json() : response;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  }[character]));
}

function toast(message) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.remove("hidden");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => element.classList.add("hidden"), 5000);
}

async function boot() {
  try {
    state.config = await api("/api/config");
    if (state.config.demo_available) $("#new-demo").classList.remove("hidden");
    await refreshProjects();
    const preferred = localStorage.getItem("dessmetal-capture-project");
    if (preferred && state.projects.some((project) => project.id === preferred)) {
      await selectProject(preferred);
    } else if (state.projects.length) {
      await selectProject(state.projects[0].id);
    }
  } catch (error) {
    $("#service-dot").style.background = "#d95d4f";
    toast(error.message);
  }
}

async function refreshProjects() {
  state.projects = await api("/api/projects");
  const list = $("#project-list");
  list.innerHTML = state.projects.map((project) => `
    <button class="project-link ${state.current?.project?.id === project.id ? "active" : ""}" data-project-id="${escapeHtml(project.id)}">
      ${escapeHtml(project.name)}
      <span>${escapeHtml(project.amp_name)} · ${project.capture_count} captures</span>
    </button>
  `).join("");
  list.querySelectorAll("[data-project-id]").forEach((button) => {
    button.addEventListener("click", () => selectProject(button.dataset.projectId));
  });
  $("#empty-state").classList.toggle("hidden", Boolean(state.current));
}

async function selectProject(projectId) {
  state.current = await api(`/api/projects/${encodeURIComponent(projectId)}`);
  localStorage.setItem("dessmetal-capture-project", projectId);
  $("#empty-state").classList.add("hidden");
  $("#project-view").classList.remove("hidden");
  renderProject();
  await refreshProjects();
}

function renderProject() {
  const { project, recommendations, external_suggestions: externalSuggestions, captures, models, training_jobs: trainingJobs = [] } = state.current;
  $("#project-name").textContent = project.name;
  $("#project-description").textContent = project.description || project.amp_name;
  $("#capture-count").textContent = captures.length;
  $("#training-job-count").textContent = trainingJobs.length;
  $("#model-count").textContent = models.length;
  const pending = recommendations.find((recommendation) => recommendation.status === "pending");
  const queuedLearnerPoints = externalSuggestions.filter((suggestion) => suggestion.status === "queued");
  $("#strategy-badge").textContent = pending?.strategy || (queuedLearnerPoints.length ? "PANAMA queued" : project.acquisition_strategy);
  $("#panama-queue-count").textContent = queuedLearnerPoints.length
    ? `${queuedLearnerPoints.length} learner point${queuedLearnerPoints.length === 1 ? "" : "s"} queued`
    : "No learner points queued";
  const input = project.input_signal_info;
  $("#input-signal-info").innerHTML = input
    ? `<strong>${escapeHtml(input.sample_rate)} Hz · ${escapeHtml(input.sample_width_bits)}-bit · ${escapeHtml(input.duration_seconds.toFixed(3))} s</strong><br><span class="small">SHA-256 ${escapeHtml(input.sha256.slice(0, 16))}…</span>`
    : "No input signal uploaded.";

  $("#recommendation-empty").classList.toggle("hidden", Boolean(pending));
  $("#recommendation-active").classList.toggle("hidden", !pending);
  if (pending) {
    $("#recommended-settings").innerHTML = pending.display_settings.map((setting) => `
      <div class="setting">
        <span>${escapeHtml(setting.label)}</span>
        <strong>${escapeHtml(formatNumber(setting.display_value))}${setting.unit ? ` ${escapeHtml(setting.unit)}` : ""}</strong>
        <small>model value ${escapeHtml(formatNumber(setting.model_value))}</small>
      </div>
    `).join("");
    $("#recommendation-rationale").textContent = pending.rationale;
  }
  const isDemo = project.metadata?.demo_profile;
  $("#replay-tab-button").classList.toggle("hidden", !isDemo);
  $("#use-demo-model").classList.toggle("hidden", !project.metadata?.fixture_model_path);
  renderCaptures(captures);
  renderTrainingJobs(trainingJobs);
  renderModels(models);
}

function renderTrainingJobs(jobs) {
  $("#trainer-badge").textContent = state.config.trainer_available ? "Local worker ready" : "Trainer unavailable";
  $("#prepare-training").disabled = !state.config.trainer_available;
  const container = $("#training-history");
  if (!jobs.length) {
    container.innerHTML = '<p class="muted small">No frozen training jobs yet.</p>';
    return;
  }
  container.innerHTML = jobs.map((job) => {
    const canStart = job.status === "ready";
    const result = job.result?.model_artifact_id ? ` · model ${job.result.model_artifact_id.slice(0, 8)}` : "";
    const esr = Number.isFinite(job.result?.training_metrics?.esr)
      ? ` · ESR ${formatNumber(job.result.training_metrics.esr)}` : "";
    return `<div class="history-item training-item">
      <div class="history-line"><strong>${escapeHtml(job.backend)} · ${escapeHtml(job.options.epochs)} epoch${job.options.epochs === 1 ? "" : "s"}</strong><span class="quality ${escapeHtml(job.status)}">${escapeHtml(job.status)}</span></div>
      <p>${escapeHtml(job.options.alignment_policy)}${job.options.workflow_validation_only ? " · functional validation only" : ""}${escapeHtml(result)}${escapeHtml(esr)}</p>
      ${job.error_message ? `<p>${escapeHtml(job.error_message)}</p>` : ""}
      ${canStart ? `<button class="secondary start-training" data-job-id="${escapeHtml(job.id)}">Start local training</button>` : ""}
      ${["queued", "running"].includes(job.status) ? `<button class="ghost inspect-training" data-job-id="${escapeHtml(job.id)}">Refresh status</button>` : ""}
    </div>`;
  }).join("");
  container.querySelectorAll(".start-training").forEach((button) => {
    button.addEventListener("click", () => startTraining(button.dataset.jobId));
  });
  container.querySelectorAll(".inspect-training").forEach((button) => {
    button.addEventListener("click", () => pollTraining(button.dataset.jobId));
  });
}

function renderCaptures(captures) {
  const container = $("#capture-history");
  if (!captures.length) {
    container.innerHTML = '<p class="muted small">No captures yet.</p>';
    return;
  }
  container.innerHTML = captures.map((capture) => {
    const setting = Object.entries(capture.settings).map(([key, value]) => `${key} ${formatNumber(value)}`).join(", ");
    const warnings = capture.validation.warnings?.join(" · ") || "Format and level checks passed";
    return `<div class="history-item">
      <div class="history-line"><strong>${escapeHtml(setting)}</strong><span class="quality ${escapeHtml(capture.quality_status)}">${escapeHtml(capture.quality_status)}</span></div>
      <p>${escapeHtml(capture.source)} · ${escapeHtml(capture.duration_seconds.toFixed(3))} s</p>
      <p>${escapeHtml(warnings)}</p>
    </div>`;
  }).join("");
}

function renderModels(models) {
  const container = $("#model-history");
  if (!models.length) {
    container.innerHTML = '<p class="muted small">No packaged models yet.</p>';
    return;
  }
  container.innerHTML = models.map((model) => `<div class="history-item">
    <div class="history-line"><strong>${escapeHtml(model.name)}</strong><span class="quality ${escapeHtml(model.validation_status)}">${escapeHtml(model.validation_status)}</span></div>
    <p>${escapeHtml(model.architecture)} · NAM ${escapeHtml(model.model_format_version)} · ${escapeHtml(model.sample_rate)} Hz</p>
    <button class="ghost download-package" data-model-id="${escapeHtml(model.id)}">Download .dess amp</button>
  </div>`).join("");
  container.querySelectorAll(".download-package").forEach((button) => {
    button.addEventListener("click", () => downloadPackage(button.dataset.modelId));
  });
}

function formatNumber(value) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

async function refreshCurrent() {
  if (state.current) await selectProject(state.current.project.id);
}

async function recommendNext() {
  try {
    await api(`/api/projects/${state.current.project.id}/recommendations`, { method: "POST" });
    await refreshCurrent();
  } catch (error) { toast(error.message); }
}

async function dismissRecommendation() {
  const pending = state.current.recommendations.find((item) => item.status === "pending");
  if (!pending) return;
  try {
    await api(`/api/recommendations/${pending.id}/dismiss`, { method: "POST" });
    await refreshCurrent();
  } catch (error) { toast(error.message); }
}

async function importPanamaPoints(file) {
  if (!file || !state.current) return;
  try {
    const points = JSON.parse(await file.text());
    if (!Array.isArray(points)) throw new Error("PANAMA new_points.json must contain a JSON array.");
    const rawRound = prompt("PANAMA round index", "0");
    if (rawRound === null) return;
    const roundIndex = Number(rawRound);
    if (!Number.isInteger(roundIndex) || roundIndex < 0) throw new Error("Round index must be a non-negative integer.");
    const projectOrder = state.current.project.controls.map((control) => control.id);
    const rawOrder = prompt(
      "Control ids in the exact order used for PANAMA's g_vector (comma-separated)",
      projectOrder.join(", "),
    );
    if (rawOrder === null) return;
    const controlOrder = rawOrder.split(",").map((value) => value.trim()).filter(Boolean);
    if (controlOrder.length !== projectOrder.length
        || new Set(controlOrder).size !== projectOrder.length
        || controlOrder.some((value) => !projectOrder.includes(value))) {
      throw new Error(`Control order must contain exactly: ${projectOrder.join(", ")}.`);
    }
    const result = await api(`/api/projects/${state.current.project.id}/recommendations/import-panama`, {
      method: "POST",
      json: { round_index: roundIndex, control_order: controlOrder, points },
    });
    toast(`Queued ${result.queued_count} PANAMA point${result.queued_count === 1 ? "" : "s"} in control order: ${controlOrder.join(", ")}.`);
    await refreshCurrent();
  } catch (error) { toast(error.message); }
  finally { $("#panama-points-file").value = ""; }
}

async function uploadInput(file) {
  if (!file || !state.current) return;
  try {
    await api(`/api/projects/${state.current.project.id}/input-signal`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
    });
    toast("Input signal stored and validated.");
    await refreshCurrent();
  } catch (error) { toast(error.message); }
}

function pendingRecommendation() {
  return state.current?.recommendations.find((item) => item.status === "pending");
}

async function importLogicCapture(file) {
  const pending = pendingRecommendation();
  if (!file || !pending) return toast("Create a recommendation before importing a capture.");
  try {
    const query = new URLSearchParams({ recommendation_id: pending.id });
    await api(`/api/projects/${state.current.project.id}/captures/import?${query}`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
    });
    toast("Capture imported with persistent format and hash evidence.");
    await refreshCurrent();
  } catch (error) { toast(error.message); }
}

async function captureReplay() {
  const pending = pendingRecommendation();
  if (!pending) return toast("Create a recommendation first.");
  try {
    const query = new URLSearchParams({ recommendation_id: pending.id });
    await api(`/api/projects/${state.current.project.id}/captures/replay?${query}`, { method: "POST" });
    toast("Existing WAV accepted as a replay capture.");
    await refreshCurrent();
  } catch (error) { toast(error.message); }
}

async function loadDevices() {
  try {
    state.devices = await api("/api/audio/devices");
    const inputs = state.devices.filter((device) => device.max_input_channels > 0);
    const outputs = state.devices.filter((device) => device.max_output_channels > 0);
    $("#input-device").innerHTML = inputs.map((device) => `<option value="${device.id}" ${device.default_input ? "selected" : ""}>${escapeHtml(device.name)}</option>`).join("");
    $("#output-device").innerHTML = outputs.map((device) => `<option value="${device.id}" ${device.default_output ? "selected" : ""}>${escapeHtml(device.name)}</option>`).join("");
  } catch (error) { toast(error.message); }
}

async function startDirectCapture() {
  const pending = pendingRecommendation();
  if (!pending) return toast("Create a recommendation first.");
  if (!state.current.project.input_signal_info) return toast("Upload the standardized input signal first.");
  if (!confirm("Start full-length playback through the selected output and record the selected input now?")) return;
  try {
    const job = await api(`/api/projects/${state.current.project.id}/captures/direct`, {
      method: "POST",
      json: {
        recommendation_id: pending.id,
        input_device: Number($("#input-device").value),
        output_device: Number($("#output-device").value),
      },
    });
    $("#capture-job").classList.remove("hidden");
    await pollJob(job.id);
  } catch (error) { toast(error.message); }
}

async function pollJob(jobId) {
  try {
    const job = await api(`/api/jobs/${jobId}`);
    $("#capture-job").textContent = job.message;
    if (["queued", "recording"].includes(job.status)) {
      setTimeout(() => pollJob(jobId), 1000);
    } else if (job.status === "complete") {
      toast("Direct capture completed.");
      await refreshCurrent();
    } else {
      toast(job.message);
    }
  } catch (error) { toast(error.message); }
}

async function importModel(file) {
  if (!file || !state.current) return;
  const name = prompt("Name this model artifact", file.name.replace(/\.nam$/i, ""));
  if (!name) return;
  try {
    const query = new URLSearchParams({ name, provenance_json: JSON.stringify({ training_backend: "external-import" }) });
    await api(`/api/projects/${state.current.project.id}/models/import?${query}`, {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
    });
    toast("Model packaged with a player manifest.");
    await refreshCurrent();
  } catch (error) { toast(error.message); }
}

async function useDemoModel() {
  try {
    await api(`/api/projects/${state.current.project.id}/models/demo-fixture`, { method: "POST" });
    toast("Prebuilt model packaged for loader handoff. It was not retrained by this workflow.");
    await refreshCurrent();
  } catch (error) { toast(error.message); }
}

async function prepareTraining() {
  if (!state.current || !state.config.trainer_available) return;
  const workflow = $("#training-profile").value === "workflow";
  const duration = state.current.project.input_signal_info?.duration_seconds || 0;
  const clipDuration = Math.min(5, duration);
  const clipStart = Math.min(20, Math.max(0, duration - clipDuration));
  if (workflow && clipDuration < 1) return toast("The functional profile requires at least one second of input audio.");
  try {
    await api(`/api/projects/${state.current.project.id}/training/jobs`, {
      method: "POST",
      json: workflow ? {
        backend: "local-mps", epochs: 1, batch_size: 8, ny: 2048,
        allow_capture_warnings: true, alignment_policy: "truncate-to-shortest",
        workflow_validation_only: true, clip_start_seconds: clipStart, clip_duration_seconds: clipDuration,
      } : {
        backend: "local-mps", epochs: 50, batch_size: 32, ny: 8192,
        allow_capture_warnings: false, alignment_policy: "strict",
        workflow_validation_only: false, clip_start_seconds: null, clip_duration_seconds: null,
      },
    });
    toast("Training job frozen with ordered controls and source hashes.");
    await refreshCurrent();
  } catch (error) { toast(error.message); }
}

async function startTraining(jobId) {
  if (!confirm("Start this local training job now? It can use substantial compute and may run for a long time.")) return;
  try {
    await api(`/api/training/jobs/${jobId}/start`, { method: "POST" });
    toast("Training queued. The job remains persistent if you navigate away.");
    await refreshCurrent();
    setTimeout(() => pollTraining(jobId), 1200);
  } catch (error) { toast(error.message); }
}

async function pollTraining(jobId) {
  try {
    const job = await api(`/api/training/jobs/${jobId}`);
    if (job.log_tail) $("#training-profile-note").textContent = job.log_tail.split("\n").filter(Boolean).slice(-1)[0];
    await refreshCurrent();
    if (["queued", "running"].includes(job.status)) setTimeout(() => pollTraining(jobId), 2500);
    if (job.status === "completed") toast("Conditioned model exported and packaged for the player.");
    if (job.status === "failed") toast(job.error_message || "Training failed. Inspect training.log.");
  } catch (error) { toast(error.message); }
}

function updateTrainingProfileNote() {
  $("#training-profile-note").textContent = $("#training-profile").value === "workflow"
    ? "Functional E2E uses one epoch and a five-second excerpt. It proves the artifact path, not model quality."
    : "Full training requires warning-free, exactly aligned captures and is intended for model evaluation.";
}

async function downloadPackage(modelId) {
  try {
    const response = await api(`/api/models/${modelId}/package`);
    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "DessMetal-Amp.dess";
    anchor.click();
    URL.revokeObjectURL(url);
  } catch (error) { toast(error.message); }
}

async function createDemo() {
  try {
    const result = await api("/api/demo/dessblock-green", { method: "POST" });
    await refreshProjects();
    await selectProject(result.project.id);
    toast(`Replay project created with ${result.replay_asset_count} existing captures.`);
  } catch (error) { toast(error.message); }
}

function openProjectDialog() {
  $("#project-form").reset();
  $("#control-rows").innerHTML = "";
  addControlRow({ id: "gain", label: "Gain", minimum: 0, maximum: 1, step: 0.1 });
  $("#project-dialog").showModal();
}

function addControlRow(initial = {}) {
  const row = document.createElement("div");
  row.className = "control-row";
  row.innerHTML = `
    <label>Id<input class="control-id" required value="${escapeHtml(initial.id || "")}" placeholder="gain"></label>
    <label>Label<input class="control-label" required value="${escapeHtml(initial.label || "")}" placeholder="Gain"></label>
    <label>Minimum<input class="control-min" required type="number" step="any" value="${escapeHtml(initial.minimum ?? 0)}"></label>
    <label>Maximum<input class="control-max" required type="number" step="any" value="${escapeHtml(initial.maximum ?? 1)}"></label>
    <label>Step<input class="control-step" required type="number" step="any" value="${escapeHtml(initial.step ?? 0.1)}"></label>
    <button type="button" class="ghost remove-control">Remove</button>`;
  row.querySelector(".remove-control").addEventListener("click", () => row.remove());
  $("#control-rows").appendChild(row);
}

async function submitProject(event) {
  event.preventDefault();
  const controls = $$(".control-row").map((row) => ({
    id: row.querySelector(".control-id").value.trim(),
    label: row.querySelector(".control-label").value.trim(),
    minimum: Number(row.querySelector(".control-min").value),
    maximum: Number(row.querySelector(".control-max").value),
    step: Number(row.querySelector(".control-step").value),
  }));
  try {
    const project = await api("/api/projects", {
      method: "POST",
      json: {
        name: $("#form-project-name").value,
        amp_name: $("#form-amp-name").value,
        description: $("#form-description").value,
        controls,
      },
    });
    $("#project-dialog").close();
    await refreshProjects();
    await selectProject(project.id);
  } catch (error) { toast(error.message); }
}

function selectCaptureTab(name) {
  $$("[data-capture-tab]").forEach((tab) => tab.classList.toggle("active", tab.dataset.captureTab === name));
  $$(".capture-tab").forEach((panel) => panel.classList.add("hidden"));
  $(`#capture-tab-${name}`).classList.remove("hidden");
  if (name === "direct" && !state.devices.length) loadDevices();
}

$("#new-project").addEventListener("click", openProjectDialog);
$("#empty-new-project").addEventListener("click", openProjectDialog);
$("#new-demo").addEventListener("click", createDemo);
$("#close-dialog").addEventListener("click", () => $("#project-dialog").close());
$("#cancel-project").addEventListener("click", () => $("#project-dialog").close());
$("#add-control").addEventListener("click", () => addControlRow());
$("#project-form").addEventListener("submit", submitProject);
$("#recommend-next").addEventListener("click", recommendNext);
$("#dismiss-recommendation").addEventListener("click", dismissRecommendation);
$("#panama-points-file").addEventListener("change", (event) => importPanamaPoints(event.target.files[0]));
$("#input-signal-file").addEventListener("change", (event) => uploadInput(event.target.files[0]));
$("#logic-capture-file").addEventListener("change", (event) => importLogicCapture(event.target.files[0]));
$("#capture-replay").addEventListener("click", captureReplay);
$("#refresh-devices").addEventListener("click", loadDevices);
$("#start-direct-capture").addEventListener("click", startDirectCapture);
$("#model-file").addEventListener("change", (event) => importModel(event.target.files[0]));
$("#use-demo-model").addEventListener("click", useDemoModel);
$("#prepare-training").addEventListener("click", prepareTraining);
$("#training-profile").addEventListener("change", updateTrainingProfileNote);
$$('[data-capture-tab]').forEach((button) => button.addEventListener("click", () => selectCaptureTab(button.dataset.captureTab)));

boot();
