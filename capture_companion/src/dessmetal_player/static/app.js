const state = { session: null, input: null, rendering: false, renderPending: false };
const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-Player-Token", window.PLAYER_TOKEN);
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
  return response.json();
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

function formatNumber(value) {
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function displayValue(control, value) {
  if (control.display_minimum === null || control.display_minimum === undefined) return value;
  const ratio = (value - control.minimum) / (control.maximum - control.minimum);
  return control.display_minimum + ratio * (control.display_maximum - control.display_minimum);
}

function renderArtifact() {
  const artifact = state.session.artifact;
  const model = artifact.model;
  $("#model-name").textContent = artifact.name;
  $("#model-detail").textContent = `${model.architecture} · NAM ${model.format_version} · ${model.sample_rate} Hz`;
  $("#parameter-count").textContent = `${model.global_condition_size} learned control${model.global_condition_size === 1 ? "" : "s"}`;
  const mapped = artifact.controls.filter((control) => control.model_parameter_index !== null);
  $("#knobs").innerHTML = mapped.length ? mapped.map((control) => `
    <label class="knob" data-control-id="${escapeHtml(control.id)}">
      <span>${escapeHtml(control.label)}</span>
      <strong class="display-value"></strong>
      <input type="range" min="${escapeHtml(control.minimum)}" max="${escapeHtml(control.maximum)}"
        step="any" data-capture-step="${escapeHtml(control.step)}" value="${escapeHtml(control.default_value)}">
      <small>model <span class="model-value"></span>${control.unit ? ` · ${escapeHtml(control.unit)}` : ""}</small>
    </label>`).join("") : '<p class="muted">This is a fixed snapshot with no learned amplifier controls.</p>';
  $("#knobs").querySelectorAll(".knob").forEach((element) => {
    const control = mapped.find((item) => item.id === element.dataset.controlId);
    const input = element.querySelector("input");
    const update = () => {
      const value = Number(input.value);
      element.querySelector(".display-value").textContent = `${formatNumber(displayValue(control, value))}${control.unit ? ` ${control.unit}` : ""}`;
      element.querySelector(".model-value").textContent = formatNumber(value);
    };
    input.addEventListener("input", update);
    input.addEventListener("change", () => { update(); if (state.input) renderCurrent(); });
    update();
  });
  const quality = model.runtime_quality;
  if (quality.mode === "slimmable") {
    const values = [...quality.breakpoints, 1.0];
    $("#quality").innerHTML = values.map((value, index) => `<option value="${value}" ${value === quality.default_size ? "selected" : ""}>${index === values.length - 1 ? "Full" : `Lite ${index + 1}`} · ${formatNumber(value)}</option>`).join("");
    $("#quality-row").classList.remove("hidden");
    $("#quality").addEventListener("change", () => { if (state.input) renderCurrent(); });
  }
}

function currentSettings() {
  const settings = {};
  $("#knobs").querySelectorAll(".knob").forEach((element) => {
    settings[element.dataset.controlId] = Number(element.querySelector("input").value);
  });
  return settings;
}

function setInput(record) {
  state.input = record;
  const info = record.info;
  $("#input-info").textContent = `${info.sample_rate} Hz · ${info.sample_width_bits}-bit · ${info.duration_seconds.toFixed(2)} s`;
  $("#render").disabled = false;
  $("#render-status").textContent = "Ready to render through the loaded model.";
}

async function uploadInput(file) {
  if (!file) return;
  try {
    $("#render-status").textContent = "Validating input…";
    const record = await api("/api/input", {
      method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file,
    });
    setInput(record);
    await renderCurrent();
  } catch (error) { toast(error.message); $("#render-status").textContent = "Input rejected."; }
  finally { $("#input-file").value = ""; }
}

async function renderCurrent() {
  if (!state.input) return;
  if (state.rendering) {
    state.renderPending = true;
    return;
  }
  state.rendering = true;
  state.renderPending = false;
  $("#render").disabled = true;
  $("#render-status").textContent = "Rendering with NAM Core…";
  try {
    const quality = state.session.artifact.model.runtime_quality.mode === "slimmable" ? Number($("#quality").value) : null;
    const result = await api("/api/render", {
      method: "POST",
      json: { input_id: state.input.id, settings: currentSettings(), quality },
    });
    $("#output-audio").src = `${result.audio_url}?v=${encodeURIComponent(result.id)}`;
    $("#output-info").textContent = `peak ${formatNumber(result.output.peak_dbfs)} dBFS · ${result.output.duration_seconds.toFixed(2)} s`;
    $("#output-block").classList.remove("hidden");
    $("#render-status").textContent = `Rendered ${Object.entries(result.settings).map(([key, value]) => `${key} ${formatNumber(value)}`).join(", ") || "snapshot"}.`;
  } catch (error) { toast(error.message); $("#render-status").textContent = "Render failed."; }
  finally {
    state.rendering = false;
    $("#render").disabled = !state.input;
    if (state.renderPending) {
      state.renderPending = false;
      void renderCurrent();
    }
  }
}

async function boot() {
  try {
    state.session = await api("/api/session");
    renderArtifact();
    if (state.session.input) setInput(state.session.input);
    $("#service-dot").classList.add("ready");
  } catch (error) { toast(error.message); $("#service-state").textContent = "Unavailable"; }
}

$("#input-file").addEventListener("change", (event) => uploadInput(event.target.files[0]));
$("#render").addEventListener("click", renderCurrent);
boot();
