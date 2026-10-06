/* ============================================================================
   Map Creator — frontend
   Map-first workflow: find a place, draw a selection, tune the model,
   generate, preview the real STL, download it.
   No build step, no framework, no third-party 3D library: the preview renders
   the exact bytes the server exported.
   ========================================================================== */

(function () {
  "use strict";

  // -- Constants --------------------------------------------------------------

  var POLL_INTERVAL_MS = 400;
  var MIN_SELECTION_M = 60;   // matches the backend's minimum extent
  var HANDLE_PX = 9;
  var STAGE_LABELS = {
    elevation: "Elevation",
    features: "Map features",
    mesh: "Mesh",
    export: "Export",
    validate: "Validate"
  };

  // WGS84, mirroring app/utils/projection.py so readouts match the backend.
  var WGS84_A = 6378137.0;
  var WGS84_E2 = 6.69437999014e-3;

  var ERROR_HINTS = {
    network: "The app could not reach the server. Check that it is still running.",
    srtm: "NASA's SRTM dataset covers 60°S to 56°N only. Pick a location inside that band.",
    timeout: "This run took too long. Try a coarser grid detail or a smaller area.",
    overpass: "OpenStreetMap's Overpass service is busy or rate limiting. Try again shortly."
  };

  // -- State ------------------------------------------------------------------

  var map = null;
  var selection = null;      // {west, south, east, north}
  var job = null;            // {pollTimer, token, cancelled}
  var lastResult = null;

  var els = {};

  // -- Small helpers ----------------------------------------------------------

  function $(id) { return document.getElementById(id); }

  function cacheElements() {
    [
      "search-form", "search-input", "search-btn", "search-status", "data-badge",
      "data-badge-text", "map", "zoom-in", "zoom-out", "selection-empty",
      "sel-state", "sel-size", "sel-centre", "sel-corners", "clear-selection",
      "use-viewport", "width-mm", "depth-mm", "min-mm", "max-mm", "base-mm",
      "resolution", "exaggeration", "exag-out", "include-roads",
      "include-buildings", "include-contours", "contour-fields",
      "contour-interval", "contours-engraved", "generate", "generate-hint",
      "progress", "progress-fill", "progress-label", "stages", "result-panel",
      "preview", "preview-canvas", "preview-loading", "preview-fallback",
      "stats", "validation", "validation-label", "validation-list", "download",
      "result-hint", "error-panel", "error-message", "error-hint",
      "error-dismiss"
    ].forEach(function (id) { els[id] = $(id); });
  }

  function clamp(value, low, high) {
    return Math.min(high, Math.max(low, value));
  }

  function debounce(fn, wait) {
    var timer = null;
    return function () {
      var args = arguments, self = this;
      clearTimeout(timer);
      timer = setTimeout(function () { fn.apply(self, args); }, wait);
    };
  }

  /** Metres spanned by a selection, using the same WGS84 radii as the backend. */
  function selectionMetres(bounds) {
    var lat = (bounds.south + bounds.north) / 2;
    var rad = lat * Math.PI / 180;
    var sinLat = Math.sin(rad);
    var rLat = WGS84_A * (1 - WGS84_E2) / Math.pow(1 - WGS84_E2 * sinLat * sinLat, 1.5);
    var rLon = WGS84_A * Math.cos(rad) / Math.sqrt(1 - WGS84_E2 * sinLat * sinLat);
    var height = (bounds.north - bounds.south) * Math.PI / 180 * rLat;
    var width = (bounds.east - bounds.west) * Math.PI / 180 * rLon;
    return { width: width, height: height };
  }

  function formatDistance(metres) {
    if (metres >= 1000) return (metres / 1000).toFixed(2) + " km";
    if (metres >= 100) return Math.round(metres) + " m";
    return metres.toFixed(1) + " m";
  }

  function formatDegrees(value) {
    return value.toFixed(5) + "°";
  }

  function formatNumber(value) {
    if (typeof value !== "number" || !isFinite(value)) return "—";
    if (value >= 1e6) return (value / 1e6).toFixed(2) + "M";
    if (value >= 1e4) return (value / 1e3).toFixed(1) + "k";
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toFixed(2);
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  // -- Status messaging -------------------------------------------------------

  function setStatus(message, state) {
    els["search-status"].textContent = message || "";
    els["search-status"].dataset.state = state || "";
    if (message) els["search-status"].classList.remove("sr-only");
  }

  function showError(message, hint) {
    els["error-message"].textContent = message;
    els["error-hint"].textContent = hint || "";
    els["error-panel"].hidden = false;
    els["error-panel"].scrollIntoView({ block: "nearest", behavior: "smooth" });
    setStatus(message, "error");
  }

  function clearError() {
    els["error-panel"].hidden = true;
    els["error-message"].textContent = "";
    els["error-hint"].textContent = "";
  }

  /** Turn an API failure into a message a map-maker can act on. */
  function describeFailure(status, detail) {
    if (status === 0 || status === 503) {
      return { message: "Lost contact with the server.", hint: ERROR_HINTS.network };
    }
    if (typeof detail === "string") {
      var text = detail;
      var lower = text.toLowerCase();
      if (lower.indexOf("srtm") !== -1) {
        return { message: text, hint: ERROR_HINTS.srtm };
      }
      if (lower.indexOf("overpass") !== -1 || lower.indexOf("features") !== -1) {
        return { message: text, hint: ERROR_HINTS.overpass };
      }
      return { message: text, hint: "" };
    }
    if (detail && typeof detail === "object") {
      var messages = [];
      if (detail.message) messages.push(detail.message);
      if (Array.isArray(detail.issues) && detail.issues.length) {
        messages.push(detail.issues.join("; "));
      }
      return { message: messages.join(" — ") || "The request was rejected.", hint: "" };
    }
    return { message: "The server rejected the request (HTTP " + status + ").", hint: "" };
  }

  async function apiFetch(url, options) {
    var response;
    try {
      response = await fetch(url, options);
    } catch (err) {
      return { status: 0, data: null };
    }

    var isJson = (response.headers.get("content-type") || "").indexOf("json") !== -1;
    var payload = null;
    if (isJson) {
      try { payload = await response.json(); } catch (err) { payload = null; }
    }
    return { status: response.status, data: payload };
  }

  // -- Map --------------------------------------------------------------------

  function initMap() {
    map = L.map("map", {
      center: [37.7459, -119.5937],
      zoom: 13,
      zoomControl: false,
      attributionControl: true,
      preferCanvas: false,
      worldCopyJump: true
    });

    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 19,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);

    attachSelectionInteraction();

    map.on("moveend zoomend resize", renderSelection);
  }

  // -- Selection overlay ------------------------------------------------------

  var overlay = null;

  function attachSelectionInteraction() {
    var container = map.getContainer();

    container.addEventListener("pointerdown", onPointerDown);
    container.addEventListener("keydown", onMapKeyDown);

    // Keep the overlay sized to the map viewport.
    map.on("moveend zoomend resize", syncOverlaySize);
    syncOverlaySize();
  }

  function syncOverlaySize() {
    ensureOverlay();
    var size = map.getSize();
    overlay.setAttribute("width", size.x);
    overlay.setAttribute("height", size.y);
    overlay.setAttribute("viewBox", "0 0 " + size.x + " " + size.y);
    renderSelection();
  }

  function ensureOverlay() {
    if (overlay) return overlay;
    overlay = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    overlay.setAttribute("class", "sel-overlay");
    overlay.style.position = "absolute";
    overlay.style.top = "0";
    overlay.style.left = "0";
    overlay.style.pointerEvents = "none";
    overlay.style.zIndex = "450";
    map.getContainer().appendChild(overlay);
    return overlay;
  }

  function selectionPixels(bounds) {
    var topLeft = map.latLngToContainerPoint([bounds.north, bounds.west]);
    var bottomRight = map.latLngToContainerPoint([bounds.south, bounds.east]);
    return {
      x: Math.min(topLeft.x, bottomRight.x),
      y: Math.min(topLeft.y, bottomRight.y),
      width: Math.abs(bottomRight.x - topLeft.x),
      height: Math.abs(bottomRight.y - topLeft.y)
    };
  }

  function renderSelection() {
    var hasSelection = !!selection;
    els["selection-empty"].hidden = hasSelection;
    ensureOverlay();

    if (!hasSelection) {
      overlay.innerHTML = "";
      updateSelectionReadout();
      return;
    }

    var box = selectionPixels(selection);
    var metres = selectionMetres(selection);

    var handleSize = HANDLE_PX;
    var handles = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];
    var html = '<g class="sel-box">';
    html += '<rect class="sel-box__fill" x="' + box.x + '" y="' + box.y +
            '" width="' + box.width + '" height="' + box.height + '" rx="2"/>';
    html += '<rect class="sel-box__edge" x="' + box.x + '" y="' + box.y +
            '" width="' + box.width + '" height="' + box.height + '" rx="2"/>';

    handles.forEach(function (name) {
      var point = handlePoint(box, name);
      html += '<rect class="sel-box__handle sel-box__handle--' + name +
              '" data-handle="' + name + '" tabindex="0" role="button"' +
              ' aria-label="Resize from ' + name + '"' +
              ' x="' + (point.x - handleSize / 2) + '" y="' + (point.y - handleSize / 2) +
              '" width="' + handleSize + '" height="' + handleSize + '"/>';
    });

    html += '<text class="sel-box__size" x="' + (box.x + 6) + '" y="' + (box.y - 6) + '">' +
            formatDistance(metres.width) + " × " + formatDistance(metres.height) +
            "</text>";
    html += "</g>";
    overlay.innerHTML = html;

    updateSelectionReadout();
  }

  function handlePoint(box, name) {
    var midX = box.x + box.width / 2;
    var midY = box.y + box.height / 2;
    switch (name) {
      case "nw": return { x: box.x, y: box.y };
      case "n":  return { x: midX, y: box.y };
      case "ne": return { x: box.x + box.width, y: box.y };
      case "e":  return { x: box.x + box.width, y: midY };
      case "se": return { x: box.x + box.width, y: box.y + box.height };
      case "s":  return { x: midX, y: box.y + box.height };
      case "sw": return { x: box.x, y: box.y + box.height };
      default:   return { x: box.x, y: midY };
    }
  }

  function updateSelectionReadout() {
    if (!selection) {
      els["sel-state"].textContent = "None";
      els["sel-state"].dataset.state = "empty";
      els["sel-size"].textContent = "—";
      els["sel-centre"].textContent = "—";
      els["sel-corners"].textContent = "—";
      els["clear-selection"].disabled = true;
      els["use-viewport"].disabled = true;
      updateGenerateAvailability();
      return;
    }

    var metres = selectionMetres(selection);
    var tooSmall = metres.width < MIN_SELECTION_M || metres.height < MIN_SELECTION_M;

    els["sel-state"].textContent = tooSmall ? "Too small" : "Ready";
    els["sel-state"].dataset.state = tooSmall ? "empty" : "ready";
    els["sel-size"].textContent =
      formatDistance(metres.width) + " × " + formatDistance(metres.height);
    els["sel-centre"].textContent =
      formatDegrees((selection.south + selection.north) / 2) + ", " +
      formatDegrees((selection.west + selection.east) / 2);
    els["sel-corners"].textContent =
      formatDegrees(selection.west) + " / " + formatDegrees(selection.south);
    els["clear-selection"].disabled = false;
    els["use-viewport"].disabled = false;
    updateGenerateAvailability();
  }

  function setSelection(bounds) {
    selection = bounds ? {
      west: bounds.west, south: bounds.south,
      east: bounds.east, north: bounds.north
    } : null;
    renderSelection();
    return selection;
  }

  /** Normalise a dragged rectangle so west<east and south<north. */
  function normaliseBounds(start, end) {
    return {
      west: Math.min(start.lng, end.lng),
      east: Math.max(start.lng, end.lng),
      south: Math.min(start.lat, end.lat),
      north: Math.max(start.lat, end.lat)
    };
  }

  /** Grow a rectangle dragged inward from a corner so it never inverts. */
  function growBounds(base, corner) {
    var bounds = {
      west: Math.min(base.west, corner.lng),
      east: Math.max(base.east, corner.lng),
      south: Math.min(base.south, corner.lat),
      north: Math.max(base.north, corner.lat)
    };
    return bounds;
  }

  var drag = null;

  function onPointerDown(event) {
    if (event.button !== undefined && event.button !== 0) return;
    if (event.target.closest && event.target.closest(".leaflet-control")) return;

    var handleName = event.target.getAttribute && event.target.getAttribute("data-handle");
    var container = map.getContainer();
    var point = map.mouseEventToContainerPoint(event);
    var latLng = map.containerPointToLatLng(point);

    drag = {
      pointerId: event.pointerId,
      handle: handleName || null,
      origin: latLng,
      startBounds: selection ? Object.assign({}, selection) : null,
      moved: false,
      mode: selection ? (handleName ? "resize" : "move") : "create"
    };

    if (drag.mode === "create") {
      // A click without a drag clears the selection.
      drag.startBounds = { west: latLng.lng, east: latLng.lng, south: latLng.lat, north: latLng.lat };
      renderSelection();
    }

    container.setPointerCapture(event.pointerId);
    container.classList.add("leaflet-dragging");
    event.preventDefault();
  }

  function onPointerMove(event) {
    if (!drag || event.pointerId !== drag.pointerId) return;

    var point = map.mouseEventToContainerPoint(event);
    var latLng = map.containerPointToLatLng(point);
    drag.moved = true;

    if (drag.mode === "create") {
      setSelection(growBounds(drag.startBounds, latLng));
    } else if (drag.mode === "move" && drag.startBounds) {
      var dLat = latLng.lat - drag.origin.lat;
      var dLng = latLng.lng - drag.origin.lng;
      var height = drag.startBounds.north - drag.startBounds.south;
      var width = drag.startBounds.east - drag.startBounds.west;
      setSelection({
        west: drag.startBounds.west + dLng,
        east: drag.startBounds.east + dLng,
        south: drag.startBounds.south + dLat,
        north: drag.startBounds.north + dLat
      });
    } else if (drag.mode === "resize" && drag.startBounds) {
      resizeFromHandle(drag.handle, drag.startBounds, latLng);
    }
    event.preventDefault();
  }

  function resizeFromHandle(handle, base, corner) {
    var next = {
      west: base.west, east: base.east,
      south: base.south, north: base.north
    };
    if (handle.indexOf("w") !== -1) next.west = corner.lng;
    if (handle.indexOf("e") !== -1) next.east = corner.lng;
    if (handle.indexOf("n") !== -1) next.north = corner.lat;
    if (handle.indexOf("s") !== -1) next.south = corner.lat;

    // Never let a drag invert the box: clamp the moving edge to the fixed one.
    var minSize = 1e-6;
    if (next.east - next.west < minSize) {
      if (handle.indexOf("w") !== -1) next.west = next.east - minSize;
      else next.east = next.west + minSize;
    }
    if (next.north - next.south < minSize) {
      if (handle.indexOf("s") !== -1) next.south = next.north - minSize;
      else next.north = next.south + minSize;
    }
    setSelection(next);
  }

  function onPointerUp(event) {
    if (!drag || event.pointerId !== drag.pointerId) return;

    var container = map.getContainer();
    container.releasePointerCapture(event.pointerId);
    container.classList.remove("leaflet-dragging");

    if (!drag.moved && drag.mode === "create") {
      setSelection(null);
    }
    drag = null;
  }

  function onMapKeyDown(event) {
    // Arrow keys nudge the whole selection when the map itself has focus.
    if (!selection || event.target !== map.getContainer()) return;
    var step = event.shiftKey ? 5 : 1;
    var delta = {
      ArrowUp: [step, 0], ArrowDown: [-step, 0],
      ArrowLeft: [0, -step], ArrowRight: [0, step]
    }[event.key];
    if (!delta) return;
    event.preventDefault();
    nudgeSelection(delta[0] * 0.0005, delta[1] * 0.0008);
  }

  function nudgeSelection(dLat, dLng) {
    setSelection({
      west: selection.west + dLng, east: selection.east + dLng,
      south: selection.south + dLat, north: selection.north + dLat
    });
  }

  /** Move only the edges a resize handle owns, keeping the box valid. */
  function nudgeHandle(handle, dLat, dLng) {
    var next = {
      west: selection.west, east: selection.east,
      south: selection.south, north: selection.north
    };
    if (handle.indexOf("n") !== -1) next.north += dLat;
    if (handle.indexOf("s") !== -1) next.south += dLat;
    if (handle.indexOf("w") !== -1) next.west += dLng;
    if (handle.indexOf("e") !== -1) next.east += dLng;

    if (next.north - next.south < 1e-6) {
      if (handle.indexOf("s") !== -1) next.south = next.north - 1e-6;
      else next.north = next.south + 1e-6;
    }
    if (next.east - next.west < 1e-6) {
      if (handle.indexOf("w") !== -1) next.west = next.east - 1e-6;
      else next.east = next.west + 1e-6;
    }
    setSelection(next);
  }

  // -- Form state -------------------------------------------------------------

  function numberField(id) {
    var input = els[id];
    var value = parseFloat(input.value);
    return isFinite(value) ? value : null;
  }

  function readForm() {
    return {
      bounds: selection,
      width: numberField("width-mm"),
      depth: numberField("depth-mm"),
      minHeight: numberField("min-mm"),
      maxHeight: numberField("max-mm"),
      base: numberField("base-mm"),
      resolution: numberField("resolution"),
      exaggeration: numberField("exaggeration"),
      roads: els["include-roads"].checked,
      buildings: els["include-buildings"].checked,
      contours: els["include-contours"].checked,
      contourInterval: numberField("contour-interval"),
      engrave: els["contours-engraved"].checked
    };
  }

  function updateGenerateAvailability() {
    var form = readForm();
    var metres = form.bounds ? selectionMetres(form.bounds) : { width: 0, height: 0 };

    var problem = null;
    if (!form.bounds) {
      problem = "Draw a selection on the map to begin.";
      els["generate-hint"].textContent = problem;
    } else if (metres.width < MIN_SELECTION_M || metres.height < MIN_SELECTION_M) {
      problem = "Selection is too small — it must be at least " + MIN_SELECTION_M + " m across.";
      els["generate-hint"].textContent = problem;
    } else if (form.minHeight === null || form.maxHeight === null) {
      problem = "Enter a starting and ending relief height.";
    } else if (form.minHeight >= form.maxHeight) {
      problem = "Relief height must increase from “from” to “to”.";
      els["generate-hint"].textContent = problem;
    } else if (form.width === null || form.depth === null) {
      problem = "Enter a model width and depth.";
      els["generate-hint"].textContent = problem;
    } else if (form.width < 10 || form.width > 600 || form.depth < 10 || form.depth > 600) {
      problem = "Model size must be between 10 and 600 mm.";
      els["generate-hint"].textContent = problem;
    } else {
      var ratio = Math.max(metres.width / form.width, metres.height / form.depth);
      els["generate-hint"].textContent = "Ground " + formatDistance(metres.width) + " × " +
        formatDistance(metres.height) + " at " + form.resolution + " m detail.";
    }

    var busy = job !== null;
    els["generate"].disabled = busy || !!problem;
    els["generate"].querySelector(".btn__label").textContent =
      busy ? "Generating…" : "Generate model";
  }

  // -- Search -----------------------------------------------------------------

  async function onSearchSubmit(event) {
    event.preventDefault();
    var query = els["search-input"].value.trim();
    if (!query) {
      setStatus("Type a place name to search for.", "error");
      els["search-input"].focus();
      return;
    }

    els["search-btn"].disabled = true;
    setStatus('Searching for "' + query + '"…', "busy");

    var result = await apiFetch("/api/geocode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ q: query })
    });

    els["search-btn"].disabled = false;

    if (result.status !== 200) {
      var failure = describeFailure(result.status, result.data && result.data.detail);
      setStatus(failure.message, "error");
      showError(failure.message, failure.hint);
      return;
    }

    var place = result.data;
    var bounds = place.bounds;
    setSelection(bounds);
    map.fitBounds([[bounds.south, bounds.west], [bounds.north, bounds.east]],
                  { padding: [24, 24] });
    setStatus(place.display_name, "ok");
    clearError();
  }

  // -- Generation -------------------------------------------------------------

  async function onGenerate() {
    if (job) return;
    clearError();

    var form = readForm();
    if (!form.bounds) return;

    hideResult();
    startProgress();

    var payload = {
      west: form.bounds.west,
      south: form.bounds.south,
      east: form.bounds.east,
      north: form.bounds.north,
      model_width_mm: form.width,
      model_depth_mm: form.depth,
      resolution_m: form.resolution,
      min_altitude_mm: form.minHeight,
      max_altitude_mm: form.maxHeight,
      vertical_exaggeration: form.exaggeration,
      base_thickness_mm: form.base,
      include_roads: form.roads,
      include_buildings: form.buildings,
      include_contours: form.contours,
      contour_interval_m: form.contourInterval,
      contours_engraved: form.engrave
    };

    var accepted = await apiFetch("/api/generate/stl", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    if (accepted.status !== 202) {
      stopProgress();
      updateGenerateAvailability();
      var failure = describeFailure(accepted.status, accepted.data && accepted.data.detail);
      showError(failure.message, failure.hint);
      return;
    }

    job = { token: accepted.data.token, cancelled: false };
    // The server reports the real stage order for this run, so mirror it and
    // show the first stage as underway from the moment the job is accepted.
    syncStages(accepted.data.stages);
    markStages(accepted.data.stages[0]);
    updateGenerateAvailability();
    pollProgress();
  }

  function startProgress() {
    els["progress"].hidden = false;
    els["progress-label"].textContent = "Starting…";
    // Stage chips stay hidden until the server reports this run's real stages.
    els["stages"].innerHTML = "";
    setProgress(0, "");
  }

  function stopProgress() {
    els["progress"].hidden = true;
    setProgress(0, "");
  }

  function setProgress(percent, message) {
    els["progress-fill"].style.width = clamp(percent, 0, 100) + "%";
    els["progress"].querySelector(".progress__bar").setAttribute("aria-valuenow", String(percent));
    if (message) els["progress-label"].textContent = message;
  }

  /** Rebuild the stage chips from the stages the server actually reported. */
  function syncStages(stages) {
    els["stages"].innerHTML = stages.map(function (stage) {
      return '<li data-stage="' + stage + '">' + (STAGE_LABELS[stage] || stage) + "</li>";
    }).join("");
  }

  function markStages(currentStage) {
    var items = els["stages"].querySelectorAll("li");
    var reached = -1;
    Array.prototype.forEach.call(items, function (item, index) {
      if (item.dataset.stage === currentStage) reached = index;
    });

    Array.prototype.forEach.call(items, function (item, index) {
      if (currentStage === null) {
        item.dataset.status = "";
      } else if (currentStage === "__done__") {
        item.dataset.status = "done";
      } else if (index < reached) {
        item.dataset.status = "done";
      } else if (index === reached) {
        item.dataset.status = "active";
      } else {
        item.dataset.status = "";
      }
    });
  }

  function pollProgress() {
    if (!job || job.cancelled) return;

    apiFetch("/api/progress/" + job.token).then(function (response) {
      if (!job || job.cancelled) return;

      if (response.status !== 200 || !response.data) {
        finishWithError(describeFailure(response.status, response.data && response.data.detail));
        return;
      }

      var progress = response.data;
      setProgress(progress.percent, progress.message);
      markStages(progress.stage);

      if (progress.done) {
        if (progress.error) {
          finishWithError({ message: progress.error, hint: guessHint(progress.error) });
        } else if (progress.result) {
          finishWithSuccess(progress.result);
        } else {
          finishWithError({ message: "The server finished without returning a model.", hint: "" });
        }
        return;
      }

      job.pollTimer = setTimeout(pollProgress, POLL_INTERVAL_MS);
    });
  }

  function guessHint(message) {
    var lower = (message || "").toLowerCase();
    if (lower.indexOf("srtm") !== -1) return ERROR_HINTS.srtm;
    if (lower.indexOf("features") !== -1 || lower.indexOf("overpass") !== -1) {
      return ERROR_HINTS.overpass;
    }
    if (lower.indexOf("timeout") !== -1) return ERROR_HINTS.timeout;
    return "";
  }

  function releaseJob() {
    if (job && job.pollTimer) clearTimeout(job.pollTimer);
    job = null;
    updateGenerateAvailability();
  }

  function finishWithSuccess(result) {
    lastResult = result;
    stopProgress();
    markStages("__done__");   // every stage completed
    releaseJob();

    showResult(result);
    setStatus("Model ready — " + formatNumber(result.triangles) + " triangles, " +
              formatBytes(result.file_size_bytes) + ".", "ok");
  }

  function finishWithError(failure) {
    stopProgress();
    releaseJob();
    hideResult();
    showError(failure.message, failure.hint);
  }

  // -- Result panel -----------------------------------------------------------

  function hideResult() {
    els["result-panel"].hidden = true;
    els["download"].hidden = true;
    els["result-hint"].hidden = true;
    els["preview-fallback"].hidden = true;
    lastResult = null;
  }

  function statRow(label, value) {
    return '<div><dt>' + label + "</dt><dd>" + value + "</dd></div>";
  }

  function showResult(result) {
    els["result-panel"].hidden = false;

    var validation = result.validation || {};
    var elevation = result.elevation || {};
    var features = result.features || {};

    els["stats"].innerHTML = [
      statRow("Model", result.width_mm + " × " + result.depth_mm + " mm"),
      statRow("Height", result.height_range_mm.toFixed(1) + " mm"),
      statRow("Triangles", formatNumber(result.triangles)),
      statRow("Volume", formatNumber(validation.volume_mm3 ? Math.round(validation.volume_mm3) : null) + " mm³"),
      statRow("Elevation", formatNumber(elevation.min_m) + " – " + formatNumber(elevation.max_m) + " m"),
      statRow("Samples", elevation.rows + " × " + elevation.cols),
      statRow("Features", featureSummary(features)),
      statRow("File", formatBytes(result.file_size_bytes)),
      statRow("Took", (result.duration_ms / 1000).toFixed(1) + " s")
    ].join("");

    renderValidation(validation);
    els["download"].hidden = false;

    var warnings = [];
    var requestedFeatures = result.requested_features;
    if (requestedFeatures && !features.roads && !features.buildings && !features.contours) {
      warnings.push("OpenStreetMap has no roads or buildings mapped in this area.");
    }
    if (result.triangles > 250000) {
      warnings.push("This model is very detailed — some slicers may struggle with it.");
    }
    if (warnings.length) {
      els["result-hint"].textContent = warnings.join(" ");
      els["result-hint"].hidden = false;
    } else {
      els["result-hint"].hidden = true;
    }

    els["result-panel"].scrollIntoView({ block: "nearest", behavior: "smooth" });
    renderPreview(result.token);
  }

  function featureSummary(features) {
    var parts = [];
    if (features.roads) parts.push(formatNumber(features.roads) + " road cells");
    if (features.buildings) parts.push(formatNumber(features.buildings) + " buildings");
    if (features.contours) parts.push(formatNumber(features.contours) + " contour cells");
    return parts.length ? parts.join(", ") : "none";
  }

  function renderValidation(validation) {
    var checks = [
      { label: "Watertight", value: validation.watertight },
      { label: "Manifold", value: validation.manifold },
      { label: "Valid STL", value: validation.valid_stl },
      { label: "Open edges", value: validation.boundary_edges, invert: true },
      { label: "Non-manifold edges", value: validation.non_manifold_edges, invert: true },
      { label: "Degenerate faces", value: validation.degenerate_triangles, invert: true }
    ];

    var failed = checks.filter(function (check) {
      if (check.value === undefined || check.value === null) return false;
      return check.invert ? check.value > 0 : check.value !== true;
    });

    var state = failed.length === 0 ? "pass" : "fail";
    els["validation"].dataset.state = state;
    els["validation-label"].textContent = failed.length === 0
      ? "Mesh passed all checks"
      : "Mesh failed " + failed.length + " check" + (failed.length > 1 ? "s" : "");

    els["validation-list"].innerHTML = checks.map(function (check) {
      var pass;
      if (check.invert) {
        pass = check.value === 0 || check.value === undefined;
      } else {
        pass = check.value === true;
      }
      var display = check.invert ? (check.value === undefined ? "—" : check.value) :
        (check.value === undefined ? "—" : (check.value ? "yes" : "no"));
      return '<li data-pass="' + pass + '"><span>' + check.label + "</span><b>" + display + "</b></li>";
    }).join("");

    if (validation.issues && validation.issues.length) {
      els["validation-list"].innerHTML += validation.issues.map(function (issue) {
        return '<li data-pass="false"><span>' + issue + "</span><b>!</b></li>";
      }).join("");
    }
  }

  // -- 3D preview -------------------------------------------------------------
  // Parses the binary STL the server just produced and draws it with an
  // orthographic camera. This previews the actual downloaded file.

  function parseBinaryStl(buffer) {
    var view = new DataView(buffer);
    var count = view.getUint32(80, true);
    var triangles = new Float32Array(count * 9);
    var normals = new Float32Array(count * 3);

    for (var i = 0; i < count; i++) {
      var base = 84 + i * 50;
      normals[i * 3 + 0] = view.getFloat32(base, true);
      normals[i * 3 + 1] = view.getFloat32(base + 4, true);
      normals[i * 3 + 2] = view.getFloat32(base + 8, true);
      for (var v = 0; v < 3; v++) {
        var offset = base + 12 + v * 12;
        triangles[i * 9 + v * 3 + 0] = view.getFloat32(offset, true);
        triangles[i * 9 + v * 3 + 1] = view.getFloat32(offset + 4, true);
        triangles[i * 9 + v * 3 + 2] = view.getFloat32(offset + 8, true);
      }
    }
    return { count: count, positions: triangles, normals: normals };
  }

  async function renderPreview(token) {
    var canvas = els["preview-canvas"];
    var context = canvas.getContext("2d");
    if (!context) {
      els["preview-fallback"].hidden = false;
      return;
    }

    els["preview-loading"].hidden = false;
    els["preview-fallback"].hidden = true;

    try {
      var response = await fetch("/api/download/" + token);
      if (!response.ok) throw new Error("download failed");
      var buffer = await response.arrayBuffer();
      var mesh = parseBinaryStl(buffer);
      drawMesh(context, mesh);
      els["preview-canvas"].setAttribute(
        "aria-label",
        "3D preview of a terrain model with " + mesh.count.toLocaleString() + " triangular faces"
      );
    } catch (err) {
      context.clearRect(0, 0, canvas.width, canvas.height);
      els["preview-fallback"].hidden = false;
    } finally {
      els["preview-loading"].hidden = true;
    }
  }

  function drawMesh(context, mesh) {
    var canvas = context.canvas;
    var width = canvas.width;
    var height = canvas.height;
    var positions = mesh.positions;
    var count = mesh.count;

    if (!count) return;

    var minX = Infinity, minY = Infinity, minZ = Infinity;
    var maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (var i = 0; i < positions.length; i += 3) {
      var x = positions[i], y = positions[i + 1], z = positions[i + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }

    var sizeX = maxX - minX || 1;
    var sizeY = maxY - minY || 1;
    var sizeZ = maxZ - minZ || 1;
    var span = Math.max(sizeX, sizeY);

    // Centre the model, normalise to unit size, then apply an isometric view.
    var cx = (minX + maxX) / 2;
    var cy = (minY + maxY) / 2;
    var cz = (minZ + maxZ) / 2;
    var cos30 = Math.cos(Math.PI / 6);
    var sin30 = Math.sin(Math.PI / 6);

    var pad = 26;
    // sx/sy below are dimensionless (model normalised to ~1 across), so convert
    // to pixels with span*scale.
    var scale = Math.min((width - pad * 2) / (span * cos30 * 2),
                         (height - pad * 2) / (span * (cos30 + sin30) + sizeZ));
    if (!isFinite(scale) || scale <= 0) scale = 1;
    var pixelsPerUnit = span * scale;

    var screenX = new Float32Array(count * 3);
    var screenY = new Float32Array(count * 3);
    var depth = new Float32Array(count);
    var order = new Uint32Array(count);

    for (var t = 0; t < count; t++) {
      var triDepth = 0;
      for (var v = 0; v < 3; v++) {
        var index = (t * 3 + v) * 3;
        var nx = (positions[index] - cx) / span;
        var ny = (positions[index + 1] - cy) / span;
        var nz = (positions[index + 2] - cz) / span;
        var sx = (nx - ny) * cos30;
        var sy = (nx + ny) * sin30 - nz;
        screenX[t * 3 + v] = width / 2 + sx * pixelsPerUnit;
        screenY[t * 3 + v] = height / 2 + sy * pixelsPerUnit;
        triDepth += nx + ny + nz;
      }
      depth[t] = triDepth / 3;
      order[t] = t;
    }

    // Painter's algorithm: farthest faces first.
    var sorted = Array.prototype.slice.call(order).sort(function (a, b) {
      return depth[b] - depth[a];
    });

    context.clearRect(0, 0, width, height);
    context.lineJoin = "round";
    context.lineWidth = 0.5;

    var lightX = 0.42, lightY = -0.62, lightZ = 0.66;
    var lightLength = Math.sqrt(lightX * lightX + lightY * lightY + lightZ * lightZ);
    lightX /= lightLength; lightY /= lightLength; lightZ /= lightLength;

    for (var s = 0; s < sorted.length; s++) {
      var face = sorted[s];
      var nIndex = face * 3;
      var shade = mesh.normals[nIndex] * lightX +
                  mesh.normals[nIndex + 1] * lightY +
                  mesh.normals[nIndex + 2] * lightZ;
      var intensity = 0.42 + 0.58 * Math.max(0, shade);
      // Terrain tint: cool slate in shadow, warm highlight on sunlit slopes.
      var red = Math.round(38 + 205 * intensity);
      var green = Math.round(74 + 165 * intensity);
      var blue = Math.round(92 + 140 * intensity);

      context.beginPath();
      context.moveTo(screenX[face * 3], screenY[face * 3]);
      context.lineTo(screenX[face * 3 + 1], screenY[face * 3 + 1]);
      context.lineTo(screenX[face * 3 + 2], screenY[face * 3 + 2]);
      context.closePath();
      context.fillStyle = "rgb(" + red + "," + green + "," + blue + ")";
      context.fill();
      context.strokeStyle = context.fillStyle;
      context.stroke();
    }
  }

  // -- Wiring -----------------------------------------------------------------

  function bindEvents() {
    els["search-form"].addEventListener("submit", onSearchSubmit);

    els["zoom-in"].addEventListener("click", function () { map.zoomIn(1); });
    els["zoom-out"].addEventListener("click", function () { map.zoomOut(1); });

    els["clear-selection"].addEventListener("click", function () {
      setSelection(null);
      setStatus("Selection cleared.");
    });

    els["use-viewport"].addEventListener("click", function () {
      var bounds = map.getBounds();
      setSelection({
        west: bounds.getWest(), south: bounds.getSouth(),
        east: bounds.getEast(), north: bounds.getNorth()
      });
      setStatus("Selected the whole map view.");
    });

    ["width-mm", "depth-mm", "min-mm", "max-mm", "base-mm", "resolution"]
      .forEach(function (id) {
        els[id].addEventListener("input", updateGenerateAvailability);
      });

    els["exaggeration"].addEventListener("input", function () {
      els["exag-out"].textContent = parseFloat(els["exaggeration"].value).toFixed(1) + "×";
      updateGenerateAvailability();
    });

    els["include-contours"].addEventListener("change", function () {
      els["contour-fields"].hidden = !els["include-contours"].checked;
    });

    els["generate"].addEventListener("click", onGenerate);

    els["download"].addEventListener("click", function () {
      if (!lastResult) return;
      var link = document.createElement("a");
      link.href = "/api/download/" + lastResult.token;
      link.download = "terrain-" + lastResult.token + ".stl";
      document.body.appendChild(link);
      link.click();
      link.remove();
      setStatus("Download started.", "ok");
    });

    els["error-dismiss"].addEventListener("click", function () {
      clearError();
      setStatus("");
    });

    var container = map.getContainer();
    container.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);

    // Resize handles respond to the keyboard too. Capture phase so this wins
    // over Leaflet's own arrow-key panning.
    els["map"].addEventListener("keydown", function (event) {
      if (!selection) return;
      var handle = event.target.getAttribute && event.target.getAttribute("data-handle");
      if (!handle) return;
      var step = event.shiftKey ? 0.01 : 0.002;
      var delta = {
        ArrowUp: [step, 0], ArrowDown: [-step, 0],
        ArrowLeft: [0, -step], ArrowRight: [0, step]
      }[event.key];
      if (!delta) return;
      event.preventDefault();
      event.stopPropagation();
      nudgeHandle(handle, delta[0], delta[1]);
    }, true);

    window.addEventListener("resize", debounce(function () {
      map.invalidateSize();
    }, 150));
  }

  async function checkHealth() {
    var response = await apiFetch("/api/health");
    var ok = response.status === 200 && response.data && response.data.ok;
    els["data-badge"].dataset.state = ok ? "ok" : "down";
    els["data-badge-text"].textContent = ok ? "Live data sources" : "Server unreachable";
  }

  // -- Boot -------------------------------------------------------------------

  function init() {
    cacheElements();
    initMap();
    bindEvents();
    els["contour-fields"].hidden = !els["include-contours"].checked;
    updateGenerateAvailability();
    checkHealth();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();