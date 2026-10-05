/** Map-Creator frontend — vanilla JS + Leaflet (no framework) */

/* eslint-env browser */
/* global L fetch document window console Promise XMLHttpRequest XMLHttpRequest */

(function() {
  "use strict";

  // ---- DOM refs -----------------------------------------------------------
  const mapEl = document.getElementById("map-container");
  const queryInput = document.getElementById("query-input");
  const searchBtn = document.getElementById("search-btn");
  const searchForm = document.getElementById("search-form");
  const searchStatus = document.getElementById("search-status");
  const locationInfo = document.getElementById("location-info");
  const locNameEl = document.getElementById("location-name");
  const centerLatEl = document.getElementById("center-lat");
  const centerLonEl = document.getElementById("center-lon");
  const wsnTextEl = document.getElementById("wsn-text");
  const zoomInBtn = document.getElementById("zoom-in-btn");
  const zoomOutBtn = document.getElementById("zoom-out-btn");
  const clearSelectionBtn = document.getElementById("clear-selection-btn");
  const widthInput = document.getElementById("width-input");
  const depthInput = document.getElementById("depth-input");
  const resSelect = document.getElementById("res-input");
  const minHeightInp = document.getElementById("min-height");
  const maxHeightInp = document.getElementById("max-height");
  const exagInput = document.getElementById("exag-input");
  const includeRoads = document.getElementById("include-roads");
  const includeBuild = document.getElementById("include-buildings");
  const includeContours = document.getElementById("include-contours");
  const areaSummary = document.getElementById("area-summary");
  const generateBtn = document.getElementById("generate-btn");
  const genStatus = document.getElementById("generation-status");

  // ---- State --------------------------------------------------------------
  let map = null;
  let rectangle = null;
  let selectedBounds = null;
  let searchMarker = null;
  let isDragging = false;
  let dragStartLat = null, dragStartLon = null;

  // Defaults
  const DEFAULT_CENTER = [37.7749, -122.4194]; // San Francisco
  const DEFAULT_ZOOM = 12;

  // ---- Init ---------------------------------------------------------------
  initMap();
  populateResolutionSelect();
  bindEvents();

  function initMap() {
    map = L.map(mapEl).setView(DEFAULT_CENTER, DEFAULT_ZOOM);
    // Use OSM tiles (ODbL licensed)
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; <a href=\"https://www.openstreetmap.org/copyright\">OpenStreetMap</a> contributors",
      maxZoom: 18,
    }).addTo(map);

    // Draw rectangle on mouse drag
    map.on("mousedown", function(e) {
      if (e.originalEvent.buttons !== 1) return;
      isDragging = true;
      dragStartLat = e.latlng.lat;
      dragStartLon = e.latlng.lng;
    });

    map.on("mousemove", function(e) {
      if (!isDragging) return;
      if (rectangle) map.removeLayer(rectangle);
      rectangle = L.rectangle(
        [[dragStartLat, dragStartLon], [e.latlng.lat, e.latlng.lng]],
        { color: "#4a7cff", weight: 2, opacity: 0.8, fillColor: "#4a7cff", fillOpacity: 0.15 }
      ).addTo(map);
    });

    map.on("mouseup", function(e) {
      if (!isDragging) return;
      isDragging = false;
      finalizeSelection();
    });

    // Double-click to zoom and clear selection
    map.on("dblclick", function() {
      map.zoomIn(1);
    });
  }

  function populateResolutionSelect() {
    const options = [25, 50, 64, 100, 128, 200];
    resSelect.innerHTML = "";
    options.forEach(function(r) {
      var opt = document.createElement("option");
      opt.value = r;
      opt.textContent = r + " m";
      if (r === 64) opt.selected = true;
      resSelect.appendChild(opt);
    });
  }

  function bindEvents() {
    searchForm.addEventListener("submit", onSearchSubmit);
    clearSelectionBtn.addEventListener("click", clearSelection);
    zoomInBtn.addEventListener("click", function() { map.zoomIn(1); });
    zoomOutBtn.addEventListener("click", function() { map.zoomOut(1); });
    generateBtn.addEventListener("click", onStartGeneration);

    // Recalculate area summary when dimensions change
    [widthInput, depthInput].forEach(function(inp) {
      inp.addEventListener("change", recalcSummary);
    });
  }

  async function onSearchSubmit(ev) {
    ev.preventDefault();
    var query = queryInput.value.trim();
    if (!query) return;

    showStatus(searchStatus, "Searching...", "");
    generateBtn.disabled = true;

    try {
      // Call backend geocoding endpoint
      var resp = await apiPost("/api/geocode", {q: query});
      if (resp.error) throw new Error(resp.error);

      var data = resp.data;
      updateSearchResult(data);
      hideStatus(searchStatus);

      // Show on map — use returned bounds or default
      showLocationOnMap(data.bounds || null, [data.center_lat, data.center_lon]);
      searchStatus.textContent = "Found: " + data.display_name;
    } catch (err) {
      searchStatus.textContent = "Search failed: " + err.message;
      searchStatus.className = "status-message error";
    } finally {
      generateBtn.disabled = false;
    }
  }

  function updateSearchResult(data) {
    locNameEl.textContent = data.display_name || "Unknown location";
    centerLatEl.textContent = data.center_lat.toFixed(4);
    centerLonEl.textContent = data.center_lon.toFixed(4);

    if (data.bounds) {
      wsnTextEl.textContent = [
        "N: +", Math.max(data.bounds.north, 0).toFixed(4),
        " W: ", Math.abs(data.bounds.west).toFixed(4),
        " S: ", Math.min(Math.abs(data.bounds.south), 90).toFixed(4),
        " E: ", Math.abs(data.bounds.east).toFixed(4)
      ].join("");

      selectedBounds = {
        north: data.bounds.north, south: data.bounds.south,
        east: data.bounds.east, west: data.bounds.west
      };
      locationInfo.classList.remove("hidden");
    } else {
      wsnTextEl.textContent = "No precise bounds — using point";
      selectedBounds = null;
      locationInfo.classList.remove("hidden");
    }
  }

  function showLocationOnMap(bounds, center) {
    if (rectangle) map.removeLayer(rectangle);
    rectangle = null;
    selectedBounds = null;

    // Place marker at the returned center
    if (searchMarker) map.removeLayer(searchMarker);
    if (center && bounds) {
      var llbounds = [[bounds.south, bounds.west], [bounds.north, bounds.east]];
      rectangle = L.rectangle(llbounds, { color: "#9b59b6", weight: 2, opacity: 0.8, fillColor: "#9b59b6", fillOpacity: 0.12 }).addTo(map);
      map.fitBounds(llbounds);

      // Draw marker at center
      var marker = L.marker(center).addTo(map).bindPopup(center[0].toFixed(4) + ", " + center[1].toFixed(4));
      marker.bindTooltip("Center point", { permanent: false, direction: "top" });
      return;
    }

    // No bounds — just zoom to center
    map.setView(center, DEFAULT_ZOOM);
    locationInfo.classList.add("hidden");
  }

  function finalizeSelection() {
    if (!rectangle) return;

    var bounds = rectangle.getBounds();
    selectedBounds = {
      north: Math.max(bounds.getNorth(), -90).toFixed(6),
      south: Math.min(bounds.getSouth(), 90).toFixed(6),
      east: Math.min(bounds.getEast(), 180).toFixed(6),
      west: Math.max(bounds.getWest(), -180).toFixed(6),
    };

    var llbounds = [[selectedBounds.south, selectedBounds.west], [selectedBounds.north, selectedBounds.east]];

    // Update rectangle styling for "active" state
    rectangle.setStyle({ color: "#27ae60", fillColor: "#27ae60", fillOpacity: 0.18 });

    recalcSummary();
    showStatus(searchStatus, "Selection active — click Clear or draw new", "success");
  }

  function clearSelection() {
    if (rectangle) map.removeLayer(rectangle);
    rectangle = null;
    selectedBounds = null;
    locationInfo.classList.add("hidden");
    areaSummary.classList.add("hidden");
    hideStatus(searchStatus);
  }

  function recalcSummary() {
    if (!selectedBounds) return;

    var w = parseFloat(widthInput.value) || 300;
    var d = parseFloat(depthInput.value) || 300;
    var latSpan = Math.abs(selectedBounds.north - selectedBounds.south);
    var lonSpan = Math.abs(selectedBounds.east - selectedBounds.west);
    var centerLat = (parseFloat(selectedBounds.north) + parseFloat(selectedBounds.south)) / 2;

    // Rough meters conversion
    var widthMeters = lonSpan * 111320 * Math.cos(Math.pow(centerLat, 2) * Math.PI / 648000);
    var heightMeters = latSpan * 111320;

    // Grid estimates
    var res = parseInt(resSelect.value);
    var cols = Math.max(1, Math.round(Math.abs(widthMeters) / res));
    var rows = Math.max(1, Math.round(heightMeters / res));
    var cells = cols * rows;

    areaSummary.classList.remove("hidden");
    areaSummary.textContent = [
      "Area: ", latSpan.toFixed(4), "° lat × ", lonSpan.toFixed(4), "° lon\n",
      "Approx: ", widthMeters.toFixed(0), "m × ", heightMeters.toFixed(0), "m (",
      ((widthMeters / 1000 * heightMeters / 1000).toFixed(2))
    ].join("");

    return { rows: cols, cols: rows, res: res, widthM: widthMeters, heightM: heightMeters };
  }

  async function onStartGeneration() {
    if (!selectedBounds) {
      showStatus(genStatus, "❌ Please select an area on the map first.", "error");
      return;
    }

    // Gather settings
    var w = parseFloat(widthInput.value) || 300;
    var d = parseFloat(depthInput.value) || 300;
    var res = parseFloat(resSelect.value);
    var minH = parseFloat(minHeightInp.value) || 2;
    var maxH = parseFloat(maxHeightInp.value) || 50;
    var exag = parseFloat(exagInput.value) || 1.0;

    // Validate
    if (minH >= maxH) {
      showStatus(genStatus, "Min height must be < Max height", "error");
      return;
    }

    // Disable button during generation
    generateBtn.disabled = true;
    generateBtn.innerHTML = "<span class='spinner'></span>Generating…";
    genStatus.textContent = "";

    try {
      var payload = {
        west: selectedBounds.west, south: selectedBounds.south,
        east: selectedBounds.east, north: selectedBounds.north,
        model_width_mm: w, model_depth_mm: d,
        resolution_m: res, min_altitude_mm: minH, max_altitude_mm: maxH,
        vertical_exaggeration: exag,
        base_thickness_mm: 3, base_extension_mm: 0,
        include_roads: includeRoads.checked,
        include_buildings: includeBuild.checked,
        include_contours: includeContours.checked,
      };

      // This is a potentially long-running request (tile fetches + mesh build)
      var resp = await apiFetchWithTimeout("/api/generate/stl", {
        method: "POST", body: JSON.stringify(payload), headers: {"Content-Type": "application/json"},
      }, 120000);

      if (resp.error) throw new Error(resp.error);

      var meshInfo = resp.data;

      // Store the binary data in localStorage (for download button) or fetch immediately
      showStatus(genStatus, "Mesh generated! Click below to download.", "success");
      generateBtn.innerHTML = "✔ Mesh Ready — Download";
      generateBtn.disabled = false;
      window._meshToken = meshInfo.token;
      window._meshDownloadUrl = "/api/download/" + meshInfo.token;

    } catch (err) {
      genStatus.textContent = "Error: " + err.message;
      genStatus.className = "status-message error";
    } finally {
      // Restore button after a delay
      setTimeout(function() {
        if (!window._meshToken) return;
      }, 2000);
    }
  }

  // ---- Helpers ------------------------------------------------------------

  function showStatus(el, msg, type) {
    el.textContent = msg;
    el.className = "status-message" + (type ? " " + type : "");
  }

  function hideStatus(el) {
    el.textContent = "";
    el.className = "status-message";
  }

  async function apiPost(url, data) {
    var resp = await fetch(url, {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(data),
    });
    if (!resp.ok) return { error: "HTTP " + resp.status + ": " + await resp.text() };
    return { ok: true, data: await resp.json() };
  }

  async function apiFetchWithTimeout(url, opts, timeoutMs) {
    var ctrl = new AbortController();
    var id = setTimeout(function(){ ctrl.abort(); }, timeoutMs);
    var resp = await fetch(url, Object.assign({}, opts || {}, { signal: ctrl.signal }));
    clearTimeout(id);
    if (!resp.ok) return { error: "HTTP " + resp.status + ": " + (await resp.text()) };
    return { ok: true, data: await resp.json() };
  }

})();
