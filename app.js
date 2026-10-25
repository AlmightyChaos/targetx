/* ==========================================================================
   TargetX — real-time bullet hole detection & auto-scoring
   --------------------------------------------------------------------------
   Pipeline:
     camera (lightweight preview) ──► ArUco 4-corner lock ──► perspective warp
     mic (Web Audio peak) ──► shot trigger ──► capture frame ──► warp
                             ──► absdiff vs baseline ──► blob filter ──► score
   ========================================================================== */
(() => {
  "use strict";

  /* ------------------------------ constants ------------------------------ */
  const CV_URL = "https://docs.opencv.org/4.9.0/opencv.js";
  const MARKER_CELLS = 6;      // 4x4 dict + 1-cell black border each side
  const MARKER_CELL_PX = 8;    // sampling resolution per cell
  const DETECT_SCALE_MAX = 640; // downscale long edge for cheap marker detection
  const MIN_MARKER_AREA = 120; // px^2 at detection scale

  /* ------------------------------ DOM refs ------------------------------- */
  const $ = (id) => document.getElementById(id);
  const video = $("video");
  const overlay = $("overlay");
  const overlayCtx = overlay.getContext("2d");
  const stage = $("stage");
  const statusChip = $("statusChip");
  const audioMeterBar = $("audioMeterBar");
  const readLastShot = $("readLastShot");
  const readGroup = $("readGroup");
  const readMoa = $("readMoa");
  const readShots = $("readShots");
  const alignHint = $("alignHint");
  const cvStatus = $("cvStatus");
  const debugInfo = $("debugInfo");

  const btnCamera = $("btnCamera");
  const btnLock = $("btnLock");
  const btnTest = $("btnTest");
  const btnRotate = $("btnRotate");
  const btnClear = $("btnClear");
  const btnSettings = $("btnSettings");
  const btnCloseDrawer = $("btnCloseDrawer");
  const drawer = $("drawer");
  const drawerScrim = $("drawerScrim");
  const sensitivity = $("sensitivity");
  const sensVal = $("sensVal");
  const caliber = $("caliber");
  const caliberVal = $("caliberVal");
  const breakLine = $("breakLine");
  const resolution = $("resolution");

  /* ------------------------------ state ---------------------------------- */
  const state = {
    cv: null,
    cvReady: false,
    stream: null,
    audioCtx: null,
    analyser: null,
    audioBuf: null,
    running: false,          // audio monitoring active
    locked: false,           // homography established
    homography: null,        // cv.Mat 3x3, video -> warped (target plane)
    homographyInv: null,     // cv.Mat 3x3, warped -> video
    baseline: null,          // cv.Mat grayscale (warped, baseline frame)
    shots: [],               // [{x, y, mmX, mmY, score, ring, ts}]
    rotationOffset: 0,       // corner assignment rotation (0..3)
    learnedMarkers: null,    // Map code -> corner index ("tl"/"tr"/"br"/"bl")
    processing: false,       // shot pipeline mutex
    lastShotAt: 0,
    cfg: null,
    w: 0, h: 0,
    lastMarkers: [],         // markers in video coords (for overlay)
    overlayHoles: [],        // persistent hole markers in video coords
  };

  const DEFAULT_CFG = {
    target: { widthMm: 170, heightMm: 170, centerXmm: 85, centerYmm: 85, distanceM: 10, warpWidthPx: 1024, warpHeightPx: 1024 },
    bullet: { caliberMm: 4.5, breakTheLine: true, minHoleAreaPx: 20, maxHoleAreaPx: 3000, maxAspectRatio: 2.6 },
    trigger: { threshold: 0.14, shotDelayMs: 70, cooldownMs: 650, rmsWindow: 96 },
    morph: { blurKernel: 5, thresholdMethod: "otsu", fixedThreshold: 28, closeKernel: 3, diffScale: 0.6 },
    rings: [
      { label: "X", diameterMm: 5, score: 10.9 },
      { label: "10", diameterMm: 11.5, score: 10 },
      { label: "9", diameterMm: 27.5, score: 9 },
      { label: "8", diameterMm: 43.5, score: 8 },
      { label: "7", diameterMm: 59.5, score: 7 },
      { label: "6", diameterMm: 75.5, score: 6 },
      { label: "5", diameterMm: 91.5, score: 5 },
      { label: "4", diameterMm: 107.5, score: 4 },
      { label: "3", diameterMm: 123.5, score: 3 },
      { label: "2", diameterMm: 139.5, score: 2 },
      { label: "1", diameterMm: 155.5, score: 1 },
    ],
  };

  /* --------------------------- helpers ----------------------------------- */
  function log(...args) {
    // eslint-disable-next-line no-console
    console.log("[TargetX]", ...args);
  }

  function setStatus(text, kind) {
    statusChip.textContent = text;
    statusChip.className = "chip chip--" + kind;
  }

  function setDebug(text) {
    debugInfo.textContent = text;
  }

  function sleep(ms) {
    return new Promise((r) => setTimeout(r, ms));
  }

  function scalar0(s) {
    if (s == null) return 0;
    if (typeof s[0] === "number") return s[0];
    if (s.val && typeof s.val[0] === "number") return s.val[0];
    if (typeof s.get === "function") return s.get(0);
    if (s.data64F && s.data64F.length) return s.data64F[0];
    return 0;
  }

  function vibrate(pattern) {
    try { if (navigator.vibrate) navigator.vibrate(pattern); } catch (_) { /* noop */ }
  }

  function beep(freq = 1400, dur = 0.07, type = "square", gain = 0.12) {
    try {
      if (!state.audioCtx) return;
      const osc = state.audioCtx.createOscillator();
      const g = state.audioCtx.createGain();
      osc.type = type;
      osc.frequency.value = freq;
      g.gain.setValueAtTime(gain, state.audioCtx.currentTime);
      g.gain.exponentialRampToValueAtTime(0.0001, state.audioCtx.currentTime + dur);
      osc.connect(g).connect(state.audioCtx.destination);
      osc.start();
      osc.stop(state.audioCtx.currentTime + dur);
    } catch (_) { /* noop */ }
  }

  /* --------------------------- config loading ---------------------------- */
  async function loadConfig() {
    try {
      const res = await fetch("./target_config.json", { cache: "no-cache" });
      if (res.ok) {
        state.cfg = JSON.parse(await res.text());
        log("config loaded:", state.cfg.name);
        return state.cfg;
      }
    } catch (e) {
      log("config load failed, using defaults", e);
    }
    state.cfg = DEFAULT_CFG;
    return state.cfg;
  }

  function applyConfigToUI() {
    const t = state.cfg.trigger;
    sensitivity.min = String(t.minThreshold ?? 0.02);
    sensitivity.max = String(t.maxThreshold ?? 0.9);
    sensitivity.value = String(t.threshold);
    sensVal.textContent = t.threshold.toFixed(2);
    caliber.value = String(state.cfg.bullet.caliberMm);
    caliberVal.textContent = state.cfg.bullet.caliberMm.toFixed(1);
    breakLine.checked = !!state.cfg.bullet.breakTheLine;
  }

  /* --------------------------- OpenCV.js --------------------------------- */
  function loadOpenCV() {
    return new Promise((resolve, reject) => {
      if (state.cvReady) return resolve(state.cv);
      const existing = document.querySelector("script[data-opencv]");
      if (existing) {
        if (window.cv && window.cv.Mat) { state.cv = window.cv; state.cvReady = true; return resolve(window.cv); }
        existing.addEventListener("load", onLoad);
        existing.addEventListener("error", onErr);
        return;
      }
      const s = document.createElement("script");
      s.src = CV_URL;
      s.async = true;
      s.setAttribute("data-opencv", "1");
      s.addEventListener("load", onLoad);
      s.addEventListener("error", onErr);
      document.head.appendChild(s);

      function onLoad() {
        cvStatus.textContent = "OpenCV.js: initializing…";
        let done = false;
        const ready = () => {
          if (done) return;
          done = true;
          state.cv = window.cv;
          state.cvReady = true;
          cvStatus.textContent = "OpenCV.js: ready ✓";
          resolve(window.cv);
        };
        if (window.cv && window.cv.Mat) return ready();
        try {
          if (window.cv && "onRuntimeInitialized" in window.cv) window.cv.onRuntimeInitialized = ready;
        } catch (_) { /* noop */ }
        // Safety net: poll until the WASM runtime exposes cv.Mat.
        let n = 0;
        const iv = setInterval(() => {
          if (window.cv && window.cv.Mat) { clearInterval(iv); ready(); }
          else if (++n > 400) { clearInterval(iv); if (!done) onErr(); }
        }, 50);
      }
      function onErr() {
        cvStatus.textContent = "OpenCV.js: failed to load";
        reject(new Error("OpenCV.js load failed"));
      }
    });
  }

  /* --------------------------- camera ------------------------------------ */
  function idealConstraints(choice) {
    const specs = {
      ideal: { width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 30 } },
      "1080": { width: { ideal: 1920 }, height: { ideal: 1080 } },
      "720": { width: { ideal: 1280 }, height: { ideal: 720 } },
      "480": { width: { ideal: 640 }, height: { ideal: 480 } },
    };
    const base = specs[choice] || specs.ideal;
    return {
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        ...base,
      },
    };
  }

  async function startCamera() {
    setStatus("STARTING…", "ready");
    try {
      // Load OpenCV in parallel with requesting the camera.
      const cvPromise = loadOpenCV();

      const constraints = idealConstraints(resolution.value);
      let stream = null;
      try { stream = await navigator.mediaDevices.getUserMedia(constraints); }
      catch (e) {
        log("primary constraints failed, falling back", e);
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } },
        });
      }
      state.stream = stream;
      video.srcObject = stream;
      await video.play();

      state.w = video.videoWidth;
      state.h = video.videoHeight;
      sizeOverlay();
      btnCamera.textContent = "■ STOP CAMERA";
      btnCamera.classList.add("is-active");
      btnLock.disabled = false;
      btnTest.disabled = false;
      btnRotate.disabled = false;

      // Request the microphone inside the same user gesture as the camera,
      // otherwise iOS Safari may refuse the permission prompt later.
      startAudio();

      await cvPromise;
      setStatus("SEARCHING MARKERS…", "ready");
      startMarkerLoop();
    } catch (e) {
      log("camera error", e);
      setStatus("CAMERA ERROR", "err");
      setDebug(String(e && e.message ? e.message : e));
    }
  }

  function stopCamera() {
    if (state.stream) {
      state.stream.getTracks().forEach((t) => t.stop());
      state.stream = null;
    }
    markerLoopRunning = false;
    video.srcObject = null;
    state.locked = false;
    state.baseline = null;
    if (state.homography) { state.homography.delete(); state.homography = null; }
    if (state.homographyInv) { state.homographyInv.delete(); state.homographyInv = null; }
    state.running = false;
    setStatus("OFFLINE", "idle");
    btnCamera.textContent = "▶ START CAMERA";
    btnCamera.classList.remove("is-active");
    btnLock.disabled = true;
    btnTest.disabled = true;
    btnRotate.disabled = true;
    btnLock.textContent = "◈ LOCK TARGET";
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  }

  function sizeOverlay() {
    overlay.width = video.videoWidth || window.innerWidth;
    overlay.height = video.videoHeight || window.innerHeight;
  }

  // Read the current video frame into an RGBA cv.Mat via an offscreen canvas.
  // (More reliable across iOS Safari than cv.VideoCapture on a live element.)
  // `maxDim` optionally downscales at drawImage time to avoid full-res copies.
  const captureCanvas = document.createElement("canvas");
  const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });

  function readFrame(maxDim) {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh || !state.cvReady) return null;
    let w = vw, h = vh;
    if (maxDim && Math.max(vw, vh) > maxDim) {
      const s = maxDim / Math.max(vw, vh);
      w = Math.round(vw * s);
      h = Math.round(vh * s);
    }
    if (captureCanvas.width !== w) captureCanvas.width = w;
    if (captureCanvas.height !== h) captureCanvas.height = h;
    captureCtx.drawImage(video, 0, 0, w, h);
    return state.cv.imread(captureCanvas);
  }

  /* --------------------------- audio trigger ----------------------------- */
  async function startAudio() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
      });
      const AC = window.AudioContext || window.webkitAudioContext;
      state.audioCtx = new AC();
      const src = state.audioCtx.createMediaStreamSource(stream);
      state.analyser = state.audioCtx.createAnalyser();
      state.analyser.fftSize = 2048;
      state.analyser.smoothingTimeConstant = 0.55;
      src.connect(state.analyser);
      state.audioBuf = new Float32Array(state.analyser.fftSize);
      state.running = true;
      audioTick();
    } catch (e) {
      log("audio unavailable", e);
      setDebug((debugInfo.textContent || "") + "\n⚠ no mic — use TEST button");
    }
  }

  let audioLoopId = 0;
  function audioTick() {
    if (!state.running) return;
    const myId = ++audioLoopId;
    const loop = () => {
      if (!state.running || myId !== audioLoopId) return;

      const cfg = state.cfg.trigger;
      const windowSize = Math.min(cfg.rmsWindow, state.audioBuf.length);
      state.analyser.getFloatTimeDomainData(state.audioBuf);

      let peak = 0;
      let sumSq = 0;
      for (let i = 0; i < windowSize; i++) {
        const v = state.audioBuf[i];
        const a = Math.abs(v);
        if (a > peak) peak = a;
        sumSq += v * v;
      }
      const rms = Math.sqrt(sumSq / windowSize);
      const level = Math.max(peak, rms * 1.6);

      const meter = Math.min(100, (level / 0.6) * 100);
      audioMeterBar.style.width = meter.toFixed(1) + "%";

      const th = parseFloat(sensitivity.value);
      if (level >= th) onTrigger();

      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  function onTrigger() {
    const cfg = state.cfg.trigger;
    const now = performance.now();
    if (now - state.lastShotAt < cfg.cooldownMs) return;
    state.lastShotAt = now;
    captureShot();
  }

  /* --------------------------- marker detection -------------------------- */
  // Dictionary-agnostic ArUco 4x4 detection. Returns array of
  // { center:[x,y], corners:[[tl],[tr],[br],[bl]], code, orientation }.
  function detectMarkers(srcGray) {
    const cv = state.cv;
    const bin = new cv.Mat();
    const markers = [];

    cv.adaptiveThreshold(srcGray, bin, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 15, 6);

    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(bin, contours, hierarchy, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);

    const imgArea = srcGray.rows * srcGray.cols;
    const minArea = MIN_MARKER_AREA;
    const maxArea = imgArea * 0.12;

    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const perimeter = cv.arcLength(cnt, true);
      if (perimeter < 40) { cnt.delete(); continue; }
      const approx = new cv.Mat();
      cv.approxPolyDP(cnt, approx, 0.04 * perimeter, true);
      if (approx.rows !== 4 || !cv.isContourConvex(approx)) { approx.delete(); cnt.delete(); continue; }
      const area = cv.contourArea(approx);
      if (area < minArea || area > maxArea) { approx.delete(); cnt.delete(); continue; }

      const pts = orderPoints(approx);
      approx.delete();
      cnt.delete();
      if (!pts) continue;

      const decoded = sampleMarker(srcGray, pts);
      if (decoded) {
        markers.push({
          center: [
            (pts[0][0] + pts[1][0] + pts[2][0] + pts[3][0]) / 4,
            (pts[0][1] + pts[1][1] + pts[2][1] + pts[3][1]) / 4,
          ],
          corners: pts,
          code: decoded.code,
          orientation: decoded.orientation,
          area,
        });
      }
    }

    contours.delete();
    hierarchy.delete();
    bin.delete();
    return markers;
  }

  function orderPoints(approxMat) {
    const pts = [];
    for (let i = 0; i < 4; i++) {
      const x = approxMat.data32S[i * 2];
      const y = approxMat.data32S[i * 2 + 1];
      pts.push([x, y]);
    }
    // TL = min sum, BR = max sum
    const sums = pts.map((p) => p[0] + p[1]);
    const rect = [null, null, null, null];
    rect[0] = pts[sums.indexOf(Math.min(...sums))];
    rect[2] = pts[sums.indexOf(Math.max(...sums))];
    // TR = min (y - x), BL = max (y - x)
    const rest = pts.filter((p) => p !== rect[0] && p !== rect[2]);
    if (rest.length !== 2) return null;
    rect[1] = rest[0][1] - rest[0][0] <= rest[1][1] - rest[1][0] ? rest[0] : rest[1];
    rect[3] = rest[0] === rect[1] ? rest[1] : rest[0];
    return rect;
  }

  function sampleMarker(srcGray, corners) {
    const cv = state.cv;
    const N = MARKER_CELLS * MARKER_CELL_PX;
    const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
      corners[0][0], corners[0][1],
      corners[1][0], corners[1][1],
      corners[2][0], corners[2][1],
      corners[3][0], corners[3][1],
    ]);
    const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, N - 1, 0, N - 1, N - 1, 0, N - 1]);
    const M = cv.getPerspectiveTransform(srcPts, dstPts);
    const warped = new cv.Mat();
    cv.warpPerspective(srcGray, warped, M, new cv.Size(N, N), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(128));

    // Read 6x6 cell means.
    const cell = new Array(MARKER_CELLS);
    for (let r = 0; r < MARKER_CELLS; r++) {
      cell[r] = new Array(MARKER_CELLS);
      for (let c = 0; c < MARKER_CELLS; c++) {
        const roi = warped.roi(new cv.Rect(c * MARKER_CELL_PX, r * MARKER_CELL_PX, MARKER_CELL_PX, MARKER_CELL_PX));
        cell[r][c] = scalar0(cv.mean(roi));
        roi.delete();
      }
    }

    M.delete();
    srcPts.delete();
    dstPts.delete();
    warped.delete();

    // Border cells (outer ring) must be dark; inner must have contrast.
    let borderSum = 0, borderN = 0, innerSum = 0, innerN = 0;
    for (let r = 0; r < MARKER_CELLS; r++) {
      for (let c = 0; c < MARKER_CELLS; c++) {
        const isBorder = r === 0 || r === MARKER_CELLS - 1 || c === 0 || c === MARKER_CELLS - 1;
        if (isBorder) { borderSum += cell[r][c]; borderN++; }
        else { innerSum += cell[r][c]; innerN++; }
      }
    }
    const borderMean = borderSum / borderN;
    const innerMean = innerSum / innerN;
    if (borderMean > innerMean - 18) return null; // border should be clearly darker

    const innerVals = [];
    for (let r = 1; r <= 4; r++) for (let c = 1; c <= 4; c++) innerVals.push(cell[r][c]);
    const minV = Math.min(...innerVals);
    const maxV = Math.max(...innerVals);
    if (maxV - minV < 30) return null; // no bit contrast

    const thr = innerMean;
    const bits = [];
    for (let r = 1; r <= 4; r++) {
      const row = [];
      for (let c = 1; c <= 4; c++) row.push(cell[r][c] < thr ? 1 : 0);
      bits.push(row);
    }

    // Canonicalize across rotations; orientation = rotation of minimal code.
    let best = null;
    for (let rot = 0; rot < 4; rot++) {
      const code = codeOf(bits, rot);
      if (best === null || code < best.code) best = { code, orientation: rot };
    }
    return { code: best.code, orientation: best.orientation };
  }

  function codeOf(bits, rot) {
    // rot: 0=0°, 1=90° CW, 2=180°, 3=270° CW
    let v = 0, idx = 0;
    for (let r = 0; r < 4; r++) {
      for (let c = 0; c < 4; c++) {
        let rr = r, cc = c;
        if (rot === 1) { const t = rr; rr = cc; cc = 3 - t; }
        else if (rot === 2) { rr = 3 - r; cc = 3 - c; }
        else if (rot === 3) { const t = rr; rr = 3 - cc; cc = t; }
        v = (v << 1) | bits[rr][cc];
        idx++;
      }
    }
    return v;
  }

  /* --------------------------- corner assignment ------------------------- */
  // Select the 4 best markers and map them to TL/TR/BR/BL target corners.
  function assignCorners(markers) {
    if (!markers || markers.length < 4) return null;

    const best = [...markers].sort((a, b) => b.area - a.area).slice(0, 4);
    const cx = best.reduce((s, m) => s + m.center[0], 0) / 4;
    const cy = best.reduce((s, m) => s + m.center[1], 0) / 4;
    const ordered = best
      .map((m) => ({ m, angle: Math.atan2(m.center[1] - cy, m.center[0] - cx) }))
      .sort((a, b) => a.angle - b.angle)
      .map((o) => o.m);

    // Try to use learned marker codes for robust, rotation-invariant assignment.
    const learned = loadLearnedMarkers();
    if (learned) {
      const byCorner = { tl: null, tr: null, br: null, bl: null };
      for (const m of best) {
        const c = learned.get(m.code);
        if (c && !byCorner[c]) byCorner[c] = m;
      }
      if (byCorner.tl && byCorner.tr && byCorner.br && byCorner.bl) {
        return { tl: byCorner.tl, tr: byCorner.tr, br: byCorner.br, bl: byCorner.bl, ordered };
      }
    }

    // Fallback: geometric assignment, rotationOffset cycles the 4 possibilities.
    const k = ((state.rotationOffset % 4) + 4) % 4;
    return {
      tl: ordered[k],
      tr: ordered[(k + 1) % 4],
      br: ordered[(k + 2) % 4],
      bl: ordered[(k + 3) % 4],
      ordered,
    };
  }

  const LEARN_KEY = "targetx.markers.v1";
  function loadLearnedMarkers() {
    if (state.learnedMarkers !== null) return state.learnedMarkers.size ? state.learnedMarkers : null;
    try {
      const raw = localStorage.getItem(LEARN_KEY);
      if (!raw) { state.learnedMarkers = new Map(); return null; }
      const obj = JSON.parse(raw);
      const m = new Map();
      for (const [code, corner] of Object.entries(obj)) m.set(parseInt(code, 10), corner);
      state.learnedMarkers = m;
      return m.size ? m : null;
    } catch (_) {
      state.learnedMarkers = new Map();
      return null;
    }
  }

  function saveLearnedMarkers(assignment) {
    const obj = {
      [assignment.tl.code]: "tl",
      [assignment.tr.code]: "tr",
      [assignment.br.code]: "br",
      [assignment.bl.code]: "bl",
    };
    try { localStorage.setItem(LEARN_KEY, JSON.stringify(obj)); } catch (_) { /* noop */ }
    state.learnedMarkers = new Map(Object.entries(obj).map(([k, v]) => [parseInt(k, 10), v]));
  }

  /* --------------------------- homography -------------------------------- */
  function computeHomography(assignment) {
    const cv = state.cv;
    const W = state.cfg.target.warpWidthPx;
    const H = state.cfg.target.warpHeightPx;
    const src = cv.matFromArray(4, 1, cv.CV_32FC2, [
      assignment.tl.center[0], assignment.tl.center[1],
      assignment.tr.center[0], assignment.tr.center[1],
      assignment.br.center[0], assignment.br.center[1],
      assignment.bl.center[0], assignment.bl.center[1],
    ]);
    const dst = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, W - 1, 0, W - 1, H - 1, 0, H - 1]);
    const M = cv.getPerspectiveTransform(src, dst);
    src.delete();
    dst.delete();
    return M;
  }

  /* --------------------------- marker loop ------------------------------- */
  let markerLoopRunning = false;
  async function startMarkerLoop() {
    if (markerLoopRunning) return;
    markerLoopRunning = true;
    let last = 0;
    while (state.stream && markerLoopRunning) {
      await sleep(150);
      const now = performance.now();
      if (now - last < 150) continue;
      last = now;
      try {
        runMarkerDetection();
      } catch (e) {
        // ignore transient detection errors
      }
    }
  }

  function runMarkerDetection() {
    if (!state.cvReady || !state.stream) return;
    const cv = state.cv;
    const src = readFrame(DETECT_SCALE_MAX);
    if (!src) return;

    const scale = src.cols / video.videoWidth;
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

    const markers = detectMarkers(gray);
    state.lastMarkers = markers.map((m) => ({
      center: [m.center[0] / scale, m.center[1] / scale],
      corners: m.corners.map((c) => [c[0] / scale, c[1] / scale]),
    }));

    const assignment = assignCorners(markers);
    const wasLocked = state.locked;
    if (assignment) {
      const m = computeHomography(assignment);
      if (state.homography) state.homography.delete();
      if (state.homographyInv) state.homographyInv.delete();
      state.homography = m;
      state.homographyInv = invertHomography(m);
      state.locked = true;
      alignHint.classList.add("is-hidden");
      if (!wasLocked) {
        setStatus("TARGET LOCKED", "locked");
        btnLock.textContent = "◈ RE-LOCK";
        beep(880, 0.08, "sine", 0.08);
        // Learn marker codes on the first stable lock for rotation robustness.
        saveLearnedMarkers(assignment);
      }
    } else {
      if (state.locked) {
        state.locked = false;
        state.baseline = null;
        setStatus("SEARCHING MARKERS…", "ready");
        alignHint.classList.remove("is-hidden");
      }
    }

    drawMarkers();

    gray.delete();
    src.delete();
  }

  function invertHomography(M) {
    const cv = state.cv;
    const inv = new cv.Mat();
    cv.invert(M, inv, cv.DECOMP_LU);
    return inv;
  }

  /* --------------------------- shot pipeline ----------------------------- */
  async function captureShot() {
    if (state.processing) return;
    if (!state.cvReady || !state.stream) return;
    if (!state.locked || !state.homography) {
      setDebug("Lock the four markers before firing.");
      beep(300, 0.12, "square", 0.06);
      return;
    }
    state.processing = true;
    try {
      const cfg = state.cfg;
      // Delay to let muzzle blast / target oscillation settle.
      await sleep(cfg.trigger.shotDelayMs);

      const cv = state.cv;
      const src = readFrame();
      if (!src) return;

      // Warp current frame.
      const warped = new cv.Mat();
      cv.warpPerspective(src, warped, state.homography, new cv.Size(cfg.target.warpWidthPx, cfg.target.warpHeightPx), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
      const gray = new cv.Mat();
      cv.cvtColor(warped, gray, cv.COLOR_RGBA2GRAY);

      let holes = [];
      if (state.baseline) {
        holes = diffAndFindHoles(gray, state.baseline);
      }
      // Store current frame as next baseline (whether or not baseline existed).
      if (state.baseline) state.baseline.delete();
      state.baseline = gray.clone();

      // Cleanup temporaries.
      warped.delete();
      gray.delete();
      src.delete();

      if (holes.length) {
        const shot = scoreHole(holes[0]);
        registerShot(shot);
        // draw preview
        drawShotPreview(shot, holes[0]);
        vibrate(60);
        beep(1800, 0.08, "square", 0.10);
      } else {
        beep(500, 0.1, "square", 0.08);
        setDebug("Shot captured — no new hole detected (baseline reset).");
      }
    } catch (e) {
      log("shot pipeline error", e);
      setDebug("Pipeline error: " + (e && e.message ? e.message : e));
    } finally {
      state.processing = false;
    }
  }

  function diffAndFindHoles(curGray, baseGray) {
    const cv = state.cv;
    const m = state.cfg.morph;
    const b = state.cfg.bullet;

    const scale = m.diffScale || 1;
    const W = Math.max(64, Math.round(curGray.cols * scale));
    const H = Math.max(64, Math.round(curGray.rows * scale));
    const a = new cv.Mat(), bb = new cv.Mat();
    cv.resize(curGray, a, new cv.Size(W, H), 0, 0, cv.INTER_AREA);
    cv.resize(baseGray, bb, new cv.Size(W, H), 0, 0, cv.INTER_AREA);

    const diff = new cv.Mat();
    cv.absdiff(a, bb, diff);
    a.delete(); bb.delete();

    const blur = new cv.Mat();
    const k = m.blurKernel || 5;
    cv.GaussianBlur(diff, blur, new cv.Size(k, k), 0);
    diff.delete();

    const bin = new cv.Mat();
    if (m.thresholdMethod === "fixed") {
      cv.threshold(blur, bin, m.fixedThreshold ?? 28, 255, cv.THRESH_BINARY);
    } else {
      cv.threshold(blur, bin, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    }
    blur.delete();

    // Morphological close to remove speckle and bridge tear marks.
    if ((m.closeKernel || 0) > 0) {
      const kc = m.closeKernel;
      const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kc, kc));
      const closed = new cv.Mat();
      cv.morphologyEx(bin, closed, cv.MORPH_CLOSE, kernel);
      bin.delete();
      kernel.delete();
      return findHoles(closed, b, scale);
    }
    return findHoles(bin, b, scale);
  }

  function findHoles(bin, bullet, scale) {
    const cv = state.cv;
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();
    cv.findContours(bin, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    const holes = [];
    const minA = bullet.minHoleAreaPx * scale * scale;
    const maxA = bullet.maxHoleAreaPx * scale * scale;

    for (let i = 0; i < contours.size(); i++) {
      const cnt = contours.get(i);
      const area = cv.contourArea(cnt);
      if (area < minA || area > maxA) { cnt.delete(); continue; }
      const rect = cv.boundingRect(cnt);
      const aspect = rect.width > rect.height ? rect.width / rect.height : rect.height / rect.width;
      if (aspect > bullet.maxAspectRatio) { cnt.delete(); continue; }
      const m = cv.moments(cnt);
      const cx = m.m10 / (m.m00 || 1);
      const cy = m.m01 / (m.m00 || 1);
      holes.push({
        xPx: cx / scale,
        yPx: cy / scale,
        wPx: rect.width / scale,
        hPx: rect.height / scale,
        areaPx: area / (scale * scale),
      });
      cnt.delete();
    }
    contours.delete();
    hierarchy.delete();
    bin.delete();

    // Sort by area descending; the freshest bullet hole is usually the largest.
    holes.sort((x, y) => y.areaPx - x.areaPx);
    return holes;
  }

  /* --------------------------- scoring ----------------------------------- */
  function scoreHole(hole) {
    const cfg = state.cfg;
    const t = cfg.target;
    const b = cfg.bullet;

    // pixel -> mm
    const pxPerMmX = t.warpWidthPx / t.widthMm;
    const pxPerMmY = t.warpHeightPx / t.heightMm;

    const mmX = hole.xPx / pxPerMmX;
    const mmY = hole.yPx / pxPerMmY;
    const dx = mmX - t.centerXmm;
    const dy = mmY - t.centerYmm;
    let dist = Math.hypot(dx, dy);

    // Break-the-line: bullet radius counts toward the higher ring.
    let effectiveDist = dist;
    if (b.breakTheLine) effectiveDist = Math.max(0, dist - b.caliberMm / 2);

    // Rings are sorted innermost-first in config.
    let ring = cfg.rings[cfg.rings.length - 1];
    let score = ring.score;
    for (const r of cfg.rings) {
      if (effectiveDist <= r.diameterMm / 2) { ring = r; score = r.score; break; }
    }

    return {
      xPx: hole.xPx,
      yPx: hole.yPx,
      mmX,
      mmY,
      distMm: dist,
      ring: ring.label,
      score,
    };
  }

  function registerShot(shot) {
    state.shots.push(shot);
    readShots.textContent = String(state.shots.length);
    readLastShot.textContent = shot.ring + " · " + (Number.isInteger(shot.score) ? shot.score : shot.score.toFixed(1));
    readLastShot.classList.add("is-hit");

    if (state.shots.length >= 2) {
      const spread = groupSpreadMm();
      readGroup.textContent = spread.toFixed(1) + " mm";
      readMoa.textContent = "MOA " + groupSizeToMoa(spread, state.cfg.target.distanceM).toFixed(2);
    } else {
      readGroup.textContent = "—";
      readMoa.textContent = "—";
    }
    stage.classList.remove("is-flash");
    void stage.offsetWidth;
    stage.classList.add("is-flash");
  }

  function groupSpreadMm() {
    const pts = state.shots.map((s) => [s.mmX, s.mmY]);
    let maxD = 0;
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        maxD = Math.max(maxD, Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]));
      }
    }
    return maxD;
  }

  function groupSizeToMoa(spreadMm, distanceM) {
    if (!distanceM || spreadMm <= 0) return 0;
    // 1 MOA ≈ 1.047 inch at 100 yd ≈ 0.2909 mm at 1 m
    return spreadMm / (distanceM * 0.2909);
  }

  /* --------------------------- overlay drawing --------------------------- */
  // The overlay canvas shares the video's intrinsic resolution and the same
  // CSS object-fit:cover, so drawing coordinates map 1:1 onto video pixels.
  function drawMarkers() {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

    // Detected markers (green).
    for (const m of state.lastMarkers) {
      const p = m.corners;
      overlayCtx.strokeStyle = "rgba(56,224,160,0.9)";
      overlayCtx.lineWidth = 2;
      overlayCtx.beginPath();
      overlayCtx.moveTo(p[0][0], p[0][1]);
      for (let i = 1; i < 4; i++) overlayCtx.lineTo(p[i][0], p[i][1]);
      overlayCtx.closePath();
      overlayCtx.stroke();
    }

    // Locked target outline (dashed green), warped back to screen space.
    if (state.locked && state.homography) {
      const t = state.cfg.target;
      const corners = [[0, 0], [t.warpWidthPx - 1, 0], [t.warpWidthPx - 1, t.warpHeightPx - 1], [0, t.warpHeightPx - 1]];
      const pts = corners.map((c) => mapWarpedToScreen(c[0], c[1]));
      overlayCtx.strokeStyle = "rgba(56,224,160,0.55)";
      overlayCtx.setLineDash([6, 6]);
      overlayCtx.lineWidth = 1.5;
      overlayCtx.beginPath();
      overlayCtx.moveTo(pts[0][0], pts[0][1]);
      for (let i = 1; i < 4; i++) overlayCtx.lineTo(pts[i][0], pts[i][1]);
      overlayCtx.closePath();
      overlayCtx.stroke();
      overlayCtx.setLineDash([]);

      // Bullseye reference.
      const ctr = mapWarpedToScreen(t.warpWidthPx * (t.centerXmm / t.widthMm), t.warpHeightPx * (t.centerYmm / t.heightMm));
      overlayCtx.strokeStyle = "rgba(56,224,160,0.4)";
      overlayCtx.lineWidth = 1;
      overlayCtx.beginPath();
      overlayCtx.arc(ctr[0], ctr[1], 6, 0, Math.PI * 2);
      overlayCtx.stroke();
    }

    // Persistent shot markers (amber).
    for (const h of state.overlayHoles) {
      const r = Math.max(8, h.r);
      overlayCtx.strokeStyle = "rgba(255,176,32,0.95)";
      overlayCtx.lineWidth = 2.5;
      overlayCtx.beginPath();
      overlayCtx.arc(h.x, h.y, r, 0, Math.PI * 2);
      overlayCtx.stroke();
      overlayCtx.lineWidth = 1.5;
      overlayCtx.beginPath();
      overlayCtx.moveTo(h.x - r - 5, h.y); overlayCtx.lineTo(h.x + r + 5, h.y);
      overlayCtx.moveTo(h.x, h.y - r - 5); overlayCtx.lineTo(h.x, h.y + r + 5);
      overlayCtx.stroke();
      overlayCtx.fillStyle = "rgba(255,176,32,0.95)";
      overlayCtx.font = "bold 14px monospace";
      overlayCtx.fillText(h.ring, h.x + r + 6, h.y - r - 6);
    }
  }

  function mapWarpedToScreen(wx, wy) {
    // Inverse homography: warped coords -> source (video) coords.
    const M = state.homographyInv && state.homographyInv.data64F;
    if (!M || M.length < 9) return [wx, wy];
    const w = M[6] * wx + M[7] * wy + M[8] || 1;
    return [(M[0] * wx + M[1] * wy + M[2]) / w, (M[3] * wx + M[4] * wy + M[5]) / w];
  }

  function drawShotPreview(shot, hole) {
    const p = mapWarpedToScreen(hole.xPx, hole.yPx);
    state.overlayHoles.push({
      x: p[0],
      y: p[1],
      r: Math.max(8, hole.wPx),
      ring: shot.ring,
    });
    drawMarkers();
  }

  /* --------------------------- UI wiring --------------------------------- */
  function openDrawer() { drawer.classList.add("is-open"); drawerScrim.classList.add("is-open"); }
  function closeDrawer() { drawer.classList.remove("is-open"); drawerScrim.classList.remove("is-open"); }

  btnCamera.addEventListener("click", () => {
    if (state.stream) stopCamera();
    else startCamera();
  });

  btnLock.addEventListener("click", () => {
    if (!state.cvReady) return;
    if (state.locked) {
      // Re-learn the corner↔marker mapping on a new/rotated target.
      state.learnedMarkers = new Map();
      try { localStorage.removeItem(LEARN_KEY); } catch (_) { /* noop */ }
    }
    runMarkerDetection();
    if (state.locked) {
      setDebug("Locked. Baseline will reset on next shot.");
    }
  });

  btnTest.addEventListener("click", () => onTrigger());

  btnRotate.addEventListener("click", () => {
    state.rotationOffset = (state.rotationOffset + 1) % 4;
    if (state.cvReady) runMarkerDetection();
  });

  btnClear.addEventListener("click", () => {
    state.shots = [];
    state.baseline = null;
    state.overlayHoles = [];
    readShots.textContent = "0";
    readLastShot.textContent = "—";
    readLastShot.classList.remove("is-hit");
    readGroup.textContent = "—";
    readMoa.textContent = "—";
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
    beep(700, 0.06, "sine", 0.06);
  });

  btnSettings.addEventListener("click", openDrawer);
  btnCloseDrawer.addEventListener("click", closeDrawer);
  drawerScrim.addEventListener("click", closeDrawer);

  sensitivity.addEventListener("input", () => {
    sensVal.textContent = parseFloat(sensitivity.value).toFixed(2);
  });
  caliber.addEventListener("input", () => {
    const v = parseFloat(caliber.value).toFixed(1);
    caliberVal.textContent = v;
    if (state.cfg) state.cfg.bullet.caliberMm = parseFloat(v);
  });
  breakLine.addEventListener("change", () => {
    if (state.cfg) state.cfg.bullet.breakTheLine = breakLine.checked;
  });

  /* --------------------------- boot -------------------------------------- */
  window.addEventListener("resize", sizeOverlay);
  window.addEventListener("orientationchange", () => setTimeout(sizeOverlay, 300));

  async function boot() {
    await loadConfig();
    applyConfigToUI();
    sizeOverlay();
    // Preload OpenCV in the background so first lock is fast.
    loadOpenCV().catch((e) => {
      log("OpenCV preload failed", e);
      cvStatus.textContent = "OpenCV.js: load failed — check connection";
    });
    if ("serviceWorker" in navigator) {
      try { await navigator.serviceWorker.register("./sw.js"); } catch (e) { log("sw register failed", e); }
    }
  }

  boot();
})();
