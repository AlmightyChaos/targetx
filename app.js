/* ==========================================================================
   TargetX — real-time bullet hole detection & auto-scoring (main thread)
   --------------------------------------------------------------------------
   The main thread is responsible ONLY for: camera, microphone, UI, overlay
   drawing, and pure-JS math (homography solve + scoring). Every heavy
   OpenCV operation runs inside cv-worker.js, so the UI never blocks on the
   ~10 MB OpenCV WASM compile.
   ========================================================================== */
(() => {
  "use strict";

  const DETECT_SCALE_MAX = 960; // preserve enough marker pixels for reliable long-range detection
  const LOCK_LOSS_GRACE_MS = 1200;

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
  const btnCenter = $("btnCenter");
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
  const detectionMode = $("detectionMode");
  const resolution = $("resolution");
  const recentShots = $("recentShots");
  const recentVal = $("recentVal");
  const loading = $("loading");
  const loadingText = $("loadingText");

  /* ------------------------------ state ---------------------------------- */
  const state = {
    stream: null,
    audioCtx: null,
    analyser: null,
    audioBuf: null,
    running: false,
    locked: false,
    H: null,             // video(full-res) -> warped, 3x3 as number[9]
    Hinv: null,          // warped -> video(full-res), 3x3 as number[9]
    targetCenter: null,  // detected bullseye and scale in warped pixels
    pickingCenter: false,
    hasBaseline: false,  // worker holds a diff baseline
    shots: [],
    rotationOffset: 0,
    learnedMarkers: null,
    processing: false,
    lastShotAt: 0,
    cfg: null,
    lastMarkers: [],     // markers in full-res video coords
    overlayHoles: [],
    liveLaserSpot: null,
    laserCandidates: [],
    lockCenters: null,   // last assigned marker centers
    lockMissSince: null,
    highlightCount: 10,  // latest N shots highlighted + used for grouping
    mode: "hole",        // hole | red | green
  };

  const DEFAULT_CFG = {
    target: { widthMm: 170, heightMm: 170, centerXmm: 85, centerYmm: 85, distanceM: 10, warpWidthPx: 1024, warpHeightPx: 1024 },
    detection: { mode: "hole" },
    laser: {
      minAreaPx: 3, maxAreaPx: 1200, maxAspectRatio: 2.5, minCircularity: 0.28,
      openKernel: 1, closeKernel: 3,
      breakTheLine: false, spotDiameterMm: 0,
      red: { hLow1: 0, hHigh1: 12, hLow2: 160, hHigh2: 180, sMin: 85, vMin: 100 },
      green: { hLow: 35, hHigh: 95, sMin: 75, vMin: 95 },
    },
    bullet: { caliberMm: 4.5, breakTheLine: true, minHoleAreaPx: 20, maxHoleAreaPx: 3000, maxAspectRatio: 2.6 },
    trigger: { threshold: 0.14, shotDelayMs: 70, cooldownMs: 650, rmsWindow: 96 },
    morph: { blurKernel: 5, thresholdMethod: "otsu", fixedThreshold: 28, closeKernel: 3, diffScale: 0.6, darknessMax: 120 },
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
  function log(...args) { console.log("[TargetX]", ...args); }

  function setStatus(text, kind) {
    statusChip.textContent = text;
    statusChip.className = "chip chip--" + kind;
  }
  function setDebug(text) { debugInfo.textContent = text; }
  function setLoading(show, text) {
    if (!loading) return;
    loading.classList.toggle("is-visible", !!show);
    if (text) loadingText.textContent = text;
  }
  function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
  function vibrate(pattern) { try { if (navigator.vibrate) navigator.vibrate(pattern); } catch (_) {} }
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
    } catch (_) {}
  }

  /* --------------------------- voice feedback ---------------------------- */
  const NUM_WORDS = { 0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve" };

  function speak(text) {
    try {
      if (!("speechSynthesis" in window)) return;
      window.speechSynthesis.cancel();
      const u = new SpeechSynthesisUtterance(text);
      u.lang = "en-US";
      u.rate = 1.0;
      u.pitch = 1.0;
      window.speechSynthesis.speak(u);
    } catch (_) { /* noop */ }
  }

  function speakScore(shot) {
    let text;
    if (shot.ring === "M" || shot.score === 0) text = "miss";
    else if (shot.ring === "X" || !Number.isInteger(shot.score)) text = "inner ten";
    else text = NUM_WORDS[shot.score] || String(shot.score);

    // Add clock direction when it is meaningful (not dead center).
    if (shot.clockHour && shot.distMm > 3) {
      text += ", " + NUM_WORDS[shot.clockHour] + " o'clock";
    }
    speak(text);
  }

  /* --------------------------- config ------------------------------------ */
  async function loadConfig() {
    try {
      const res = await fetch("./target_config.json", { cache: "no-cache" });
      if (res.ok) {
        const parsed = JSON.parse(await res.text());
        const laser = parsed.laser || {};
        state.cfg = {
          ...parsed,
          detection: { ...DEFAULT_CFG.detection, ...(parsed.detection || {}) },
          laser: {
            ...DEFAULT_CFG.laser,
            ...laser,
            red: { ...DEFAULT_CFG.laser.red, ...(laser.red || {}) },
            green: { ...DEFAULT_CFG.laser.green, ...(laser.green || {}) },
          },
        };
        return state.cfg;
      }
    } catch (e) { log("config load failed, using defaults", e); }
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
    const configuredMode = state.cfg.detection && state.cfg.detection.mode;
    state.mode = ["hole", "red", "green"].includes(configuredMode) ? configuredMode : "hole";
    if (detectionMode) detectionMode.value = state.mode;
  }

  /* --------------------------- worker ------------------------------------ */
  let worker = null;
  let workerReadyPromise = null;
  let workerReadyTimer = null;
  let reqId = 0;
  const pendingCallbacks = new Map();
  let workerReadyResolver = null;
  let workerReadyRejecter = null;

  function failWorker(error) {
    const message = error && error.message ? error.message : String(error || "Unknown worker error");
    log("OpenCV worker failed", message);
    cvStatus.textContent = "OpenCV.js: failed — " + message;
    setLoading(false);

    if (workerReadyTimer) {
      clearTimeout(workerReadyTimer);
      workerReadyTimer = null;
    }
    if (workerReadyRejecter) {
      workerReadyRejecter(new Error(message));
      workerReadyResolver = null;
      workerReadyRejecter = null;
    }
    if (worker) {
      worker.terminate();
      worker = null;
    }
    workerReadyPromise = null;
  }

  function onWorkerMessage(m) {
    if (m.type === "ready") {
      if (workerReadyTimer) {
        clearTimeout(workerReadyTimer);
        workerReadyTimer = null;
      }
      if (workerReadyResolver) workerReadyResolver();
      workerReadyResolver = null;
      workerReadyRejecter = null;
      cvStatus.textContent = "OpenCV.js: ready ✓ (worker)";
      setLoading(false);
      return;
    }
    if (m.type === "status") { setLoading(true, m.message); return; }
    if (m.type === "error" && m.id == null) {
      failWorker(new Error(m.message || "OpenCV.js failed to initialize"));
      return;
    }
    const cb = pendingCallbacks.get(m.id);
    if (!cb) return;
    clearTimeout(cb.t);
    pendingCallbacks.delete(m.id);
    if (m.type === "error") cb.reject(new Error(m.message));
    else cb.resolve(m);
  }

  function ensureWorker() {
    if (worker) return workerReadyPromise;
    workerReadyPromise = new Promise((resolve, reject) => {
      workerReadyResolver = resolve;
      workerReadyRejecter = reject;
    });
    try {
      worker = new Worker("./cv-worker.js");
    } catch (e) {
      failWorker(e);
      return workerReadyPromise;
    }
    workerReadyTimer = setTimeout(() => {
      failWorker(new Error("Timed out loading OpenCV.js. Check your internet connection and try again."));
    }, 45000);
    worker.addEventListener("message", (e) => onWorkerMessage(e.data));
    worker.addEventListener("error", (e) => {
      failWorker(new Error(e.message || "Could not load the OpenCV.js worker"));
    });
    return workerReadyPromise;
  }

  function callWorker(type, payload, transferables) {
    return new Promise((resolve, reject) => {
      const id = ++reqId;
      const t = setTimeout(() => {
        pendingCallbacks.delete(id);
        reject(new Error("worker timeout"));
      }, 30000);
      pendingCallbacks.set(id, { resolve, reject, t });
      worker.postMessage({ id, type, ...payload }, transferables || []);
    });
  }

  /* --------------------------- frame reading ----------------------------- */
  const captureCanvas = document.createElement("canvas");
  const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true });

  // Reads the current video frame into an RGBA ArrayBuffer. `maxDim`
  // downscales at drawImage time. Returns { buf, width, height }.
  function readFrameBuf(maxDim) {
    const vw = video.videoWidth, vh = video.videoHeight;
    if (!vw || !vh) return null;
    let w = vw, h = vh;
    if (maxDim && Math.max(vw, vh) > maxDim) {
      const s = maxDim / Math.max(vw, vh);
      w = Math.round(vw * s);
      h = Math.round(vh * s);
    }
    if (captureCanvas.width !== w) captureCanvas.width = w;
    if (captureCanvas.height !== h) captureCanvas.height = h;
    captureCtx.drawImage(video, 0, 0, w, h);
    const img = captureCtx.getImageData(0, 0, w, h);
    return { buf: img.data.buffer, width: w, height: h };
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
    return { audio: false, video: { facingMode: { ideal: "environment" }, ...base } };
  }

  async function startCamera() {
    setStatus("STARTING…", "ready");
    try {
      // Create the worker first so OpenCV downloads+compiles on its own
      // thread while we request the camera. The UI stays responsive.
      setLoading(true, "Loading OpenCV.js…");
      const cvPromise = ensureWorker();
      cvPromise.catch(() => {}); // Camera permission may take longer than worker startup.

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

      sizeOverlay();
      btnCamera.textContent = "■ STOP CAMERA";
      btnCamera.classList.add("is-active");
      btnLock.disabled = false;
      btnCenter.disabled = true;
      btnTest.disabled = false;
      btnRotate.disabled = false;

      // Microphone within the same user gesture.
      startAudio();

      // Reset diff baseline for a fresh session.
      state.hasBaseline = false;
      callWorker("clear", {}).catch(() => {});

      await cvPromise;
      setLoading(false);
      setStatus("SEARCHING MARKERS…", "ready");
      startMarkerLoop();
    } catch (e) {
      log("camera error", e);
      if (state.stream) {
        state.stream.getTracks().forEach((track) => track.stop());
        state.stream = null;
      }
      video.srcObject = null;
      state.running = false;
      btnCamera.textContent = "▶ START CAMERA";
      btnCamera.classList.remove("is-active");
      btnLock.disabled = true;
      btnCenter.disabled = true;
      btnTest.disabled = true;
      btnRotate.disabled = true;
      setLoading(false);
      setStatus("CAMERA ERROR", "err");
      setDebug("Camera/OpenCV startup failed: " + (e && e.message ? e.message : e) + ". Check your internet connection and retry.");
    }
  }

  function stopCamera() {
    if (state.stream) {
      state.stream.getTracks().forEach((t) => t.stop());
      state.stream = null;
    }
    markerLoopRunning = false;
    stopLaserTracking();
    state.pickingCenter = false;
    overlay.classList.remove("is-picking-center");
    btnCenter.disabled = true;
    video.srcObject = null;
    state.locked = false;
    state.H = null;
    state.Hinv = null;
    state.targetCenter = null;
    state.liveLaserSpot = null;
    state.laserCandidates = [];
    state.hasBaseline = false;
    state.lockCenters = null;
    state.lockMissSince = null;
    state.running = false;
    callWorker("clear", {}).catch(() => {});
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
      let peak = 0, sumSq = 0;
      for (let i = 0; i < windowSize; i++) {
        const a = Math.abs(state.audioBuf[i]);
        if (a > peak) peak = a;
        sumSq += state.audioBuf[i] * state.audioBuf[i];
      }
      const rms = Math.sqrt(sumSq / windowSize);
      const level = Math.max(peak, rms * 1.6);
      audioMeterBar.style.width = Math.min(100, (level / 0.6) * 100).toFixed(1) + "%";
      if (level >= parseFloat(sensitivity.value)) onTrigger();
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  function onTrigger() {
    const cfg = state.cfg.trigger;
    const now = performance.now();
    if (now - state.lastShotAt < cfg.cooldownMs) return;
    state.lastShotAt = now;
    setDebug(state.mode === "hole"
      ? "Audio trigger detected — capturing target frame."
      : "Shot sound detected — checking recent " + state.mode + " laser frames.");
    captureShot();
  }

  /* --------------------------- homography (pure JS) ---------------------- */
  function solvePerspective(src, dst) {
    const A = [], b = [];
    for (let i = 0; i < 4; i++) {
      const [u, v] = src[i];
      const [x, y] = dst[i];
      A.push([u, v, 1, 0, 0, 0, -u * x, -v * x]); b.push(x);
      A.push([0, 0, 0, u, v, 1, -u * y, -v * y]); b.push(y);
    }
    const n = 8;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
      let pivot = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
      if (Math.abs(M[pivot][col]) < 1e-12) return null;
      [M[col], M[pivot]] = [M[pivot], M[col]];
      const d = M[col][col];
      for (let c = col; c <= n; c++) M[col][c] /= d;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col];
        if (f !== 0) for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
      }
    }
    const h = M.map((row) => row[n]);
    return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
  }

  function applyH(M, x, y) {
    const w = M[6] * x + M[7] * y + M[8];
    return [(M[0] * x + M[1] * y + M[2]) / w, (M[3] * x + M[4] * y + M[5]) / w];
  }

  function invert3x3(m) {
    const [a, b, c, d, e, f, g, h, i] = m;
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const D = -(b * i - c * h), E = a * i - c * g, F = -(a * h - b * g);
    const G = b * f - c * e, H = -(a * f - c * d), I = a * e - b * d;
    const det = a * A + b * B + c * C;
    if (Math.abs(det) < 1e-12) return null;
    return [A / det, D / det, G / det, B / det, E / det, H / det, C / det, F / det, I / det];
  }

  function computeHomography(assignment) {
    const t = state.cfg.target;
    const src = [assignment.tl.center, assignment.tr.center, assignment.br.center, assignment.bl.center];
    const dst = [[0, 0], [t.warpWidthPx - 1, 0], [t.warpWidthPx - 1, t.warpHeightPx - 1], [0, t.warpHeightPx - 1]];
    const H = solvePerspective(src, dst);
    if (!H) return null;
    const Hinv = invert3x3(H);
    if (!Hinv) return null;
    return { H, Hinv };
  }

  /* --------------------------- corner assignment ------------------------- */
  function assignCorners(markers) {
    if (!markers || markers.length < 4) return null;
    const distinctCodes = new Map();
    for (const marker of markers) {
      const existing = distinctCodes.get(marker.code);
      if (!existing || marker.area > existing.area) distinctCodes.set(marker.code, marker);
    }
    const best = [...distinctCodes.values()].sort((a, b) => b.area - a.area).slice(0, 4);
    const cx = best.reduce((s, m) => s + m.center[0], 0) / 4;
    const cy = best.reduce((s, m) => s + m.center[1], 0) / 4;
    const ordered = best
      .map((m) => ({ m, angle: Math.atan2(m.center[1] - cy, m.center[0] - cx) }))
      .sort((a, b) => a.angle - b.angle)
      .map((o) => o.m);

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

    const k = ((state.rotationOffset % 4) + 4) % 4;
    return {
      tl: ordered[k],
      tr: ordered[(k + 1) % 4],
      br: ordered[(k + 2) % 4],
      bl: ordered[(k + 3) % 4],
      ordered,
    };
  }

  const LEARN_KEY = "targetx.markers.v3";
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
    } catch (_) { state.learnedMarkers = new Map(); return null; }
  }

  function saveLearnedMarkers(assignment) {
    const obj = {
      [assignment.tl.code]: "tl",
      [assignment.tr.code]: "tr",
      [assignment.br.code]: "br",
      [assignment.bl.code]: "bl",
    };
    try { localStorage.setItem(LEARN_KEY, JSON.stringify(obj)); } catch (_) {}
    state.learnedMarkers = new Map(Object.entries(obj).map(([k, v]) => [parseInt(k, 10), v]));
  }

  /* --------------------------- marker loop ------------------------------- */
  let markerLoopRunning = false;
  let detecting = false;

  async function startMarkerLoop() {
    if (markerLoopRunning) return;
    markerLoopRunning = true;
    while (state.stream && markerLoopRunning) {
      await sleep(state.locked && state.mode !== "hole" ? 800 : 200);
      if (detecting) continue;
      detecting = true;
      try { await runMarkerDetection(); } catch (e) { /* transient */ }
      finally { detecting = false; }
    }
  }

  async function runMarkerDetection() {
    if (!worker || !state.stream) return;
    const frame = readFrameBuf(DETECT_SCALE_MAX);
    if (!frame) return;
    const scaleBackX = video.videoWidth / frame.width;
    const scaleBackY = video.videoHeight / frame.height;

    let res;
    try {
      res = await callWorker("detect", {
        buf: frame.buf, width: frame.width, height: frame.height,
      }, [frame.buf]);
    } catch (e) {
      log("detect error", e);
      return;
    }

    const markers = (res.markers || []).map((m) => ({
      code: m.code,
      area: m.area,
      center: [m.center[0] * scaleBackX, m.center[1] * scaleBackY],
      corners: m.corners.map((c) => [c[0] * scaleBackX, c[1] * scaleBackY]),
    }));
    state.lastMarkers = markers;

    const assignment = assignCorners(markers);
    const wasLocked = state.locked;
    const hom = assignment ? computeHomography(assignment) : null;

    if (assignment && hom) {
      state.lockMissSince = null;
      state.H = hom.H;
      state.Hinv = hom.Hinv;

      const centers = [assignment.tl.center, assignment.tr.center, assignment.br.center, assignment.bl.center];
      if (!wasLocked) {
        state.hasBaseline = false;
        state.targetCenter = null;
        callWorker("clear", {}).catch(() => {});
        calibrateTargetCenter();
        if (state.mode === "hole") primeBaseline();
      }
      state.lockCenters = centers;

      state.locked = true;
      btnCenter.disabled = false;
      if (!wasLocked && state.mode !== "hole") startLaserTracking();
      alignHint.classList.add("is-hidden");
      if (!wasLocked) {
        setStatus("TARGET LOCKED", "locked");
        btnLock.textContent = "◈ RE-LOCK";
        beep(880, 0.08, "sine", 0.08);
        saveLearnedMarkers(assignment);
      }
    } else if (wasLocked) {
      if (state.lockMissSince === null) state.lockMissSince = performance.now();
      if (performance.now() - state.lockMissSince >= LOCK_LOSS_GRACE_MS) {
        state.locked = false;
        btnCenter.disabled = true;
        state.lockMissSince = null;
        state.hasBaseline = false;
        state.lockCenters = null;
        setStatus("SEARCHING MARKERS…", "ready");
        alignHint.classList.remove("is-hidden");
      }
    } else {
      state.locked = false;
    }

    drawMarkers();
  }

  let targetCalibrationPromise = null;
  async function calibrateTargetCenter() {
    if (targetCalibrationPromise) return targetCalibrationPromise;
    if (!worker || !state.stream || !state.H || !state.cfg) return;
    targetCalibrationPromise = (async () => {
      try {
        const frame = readFrameBuf();
        if (!frame) return;
        const result = await callWorker("calibrate", {
          buf: frame.buf,
          width: frame.width,
          height: frame.height,
          H: [...state.H],
          warpW: state.cfg.target.warpWidthPx,
          warpH: state.cfg.target.warpHeightPx,
          target: state.cfg.target,
          rings: state.cfg.rings,
        }, [frame.buf]);
        if (!state.targetCenter || !state.targetCenter.manual) state.targetCenter = result.center || null;
        drawMarkers();
        setDebug(state.targetCenter
          ? (state.targetCenter.manual
            ? "Bullseye center set manually."
            : "Bullseye found automatically (confidence " + state.targetCenter.confidence.toFixed(2) + ").")
          : "Bullseye not found. Keep the complete printed rings visible and relock the target.");
      } catch (e) {
        log("target center calibration error", e);
        setDebug("Could not locate bullseye: " + e.message);
      } finally {
        targetCalibrationPromise = null;
      }
    })();
    return targetCalibrationPromise;
  }

  // Warp the current frame and store it as the diff baseline (no diff, no
  // shot registration), so the very first shot after a lock is detected.
  let baselinePromise = null;
  async function primeBaseline() {
    if (state.mode !== "hole") return; // laser dots need no diff baseline
    if (!worker || !state.stream || !state.H) return;
    if (baselinePromise) return baselinePromise;
    if (state.processing) return;
    state.processing = true;
    baselinePromise = (async () => {
      try {
        const frame = readFrameBuf();
        if (!frame) return;
        await callWorker("capture", {
          buf: frame.buf,
          width: frame.width,
          height: frame.height,
          H: state.H,
          warpW: state.cfg.target.warpWidthPx,
          warpH: state.cfg.target.warpHeightPx,
          morph: state.cfg.morph,
          bullet: state.cfg.bullet,
          hasBaseline: false,
        }, [frame.buf]);
        state.hasBaseline = true;
      } catch (e) {
        log("prime baseline error", e);
      } finally {
        state.processing = false;
        baselinePromise = null;
      }
    })();
    return baselinePromise;
  }

  let laserTrackActive = false;
  let laserFrameBusy = false;
  let lastLaserFrameAt = 0;
  let lastLaserVideoTime = -1;
  function startLaserTracking() {
    if (laserTrackActive || !state.stream || !state.locked || state.mode === "hole") return;
    laserTrackActive = true;
    requestAnimationFrame(laserTrackingTick);
  }

  function stopLaserTracking() {
    laserTrackActive = false;
    state.liveLaserSpot = null;
  }

  function scaleHomography(H, scaleX, scaleY) {
    return [
      H[0] * scaleX, H[1] * scaleY, H[2],
      H[3] * scaleX, H[4] * scaleY, H[5],
      H[6] * scaleX, H[7] * scaleY, H[8],
    ];
  }

  function laserTrackingTick() {
    if (!laserTrackActive || !state.stream || !state.locked || state.mode === "hole") {
      stopLaserTracking();
      return;
    }
    const now = performance.now();
    if (!laserFrameBusy && !detecting && now - lastLaserFrameAt >= 25 && video.currentTime !== lastLaserVideoTime) {
      const frame = readFrameBuf(DETECT_SCALE_MAX);
      if (frame) {
        const scaleX = video.videoWidth / frame.width;
        const scaleY = video.videoHeight / frame.height;
        const jobMode = state.mode;
        const H = scaleHomography(state.H, scaleX, scaleY);
        laserFrameBusy = true;
        lastLaserFrameAt = now;
        lastLaserVideoTime = video.currentTime;
        callWorker("laserFrame", {
          buf: frame.buf,
          width: frame.width,
          height: frame.height,
          H,
          warpW: 512,
          warpH: 512,
          mode: jobMode,
          laser: state.cfg.laser,
        }, [frame.buf]).then((result) => {
          const timestamp = performance.now();
          const spot = (result.spots || [])[0];
          if (laserTrackActive && state.stream && spot && jobMode === state.mode) {
            const candidate = {
              xPx: spot.xPx * state.cfg.target.warpWidthPx / 512,
              yPx: spot.yPx * state.cfg.target.warpHeightPx / 512,
              color: spot.color,
              quality: spot.quality,
              time: timestamp,
            };
            state.laserCandidates.push(candidate);
            state.laserCandidates = state.laserCandidates.filter((item) => timestamp - item.time < 700).slice(-40);
            state.liveLaserSpot = candidate;
          } else if (state.liveLaserSpot && timestamp - state.liveLaserSpot.time > 250) {
            state.liveLaserSpot = null;
          }
          drawMarkers();
        }).catch((e) => {
          log("laser frame error", e);
        }).finally(() => {
          laserFrameBusy = false;
        });
      }
    }
    requestAnimationFrame(laserTrackingTick);
  }

  async function captureLaserShot() {
    state.processing = true;
    const triggerTime = performance.now();
    try {
      if (targetCalibrationPromise) await targetCalibrationPromise;
      await sleep(100);
      const latestTime = performance.now();
      const candidates = state.laserCandidates.filter((candidate) =>
        candidate.time >= triggerTime - 500 && candidate.time <= latestTime
      );
      candidates.sort((a, b) =>
        Math.abs(a.time - triggerTime) - Math.abs(b.time - triggerTime)
        || b.quality - a.quality
      );
      const spot = candidates[0];
      if (!spot) {
        setDebug("Shot sound detected, but no " + state.mode + " laser spot appeared in the recent camera frames.");
        return;
      }
      if (!state.targetCenter) {
        setDebug("Laser found, but bullseye center is not calibrated. Find the printed rings or tap SET CENTER.");
        return;
      }
      const scoring = {
        target: { ...state.cfg.target },
        bullet: { ...state.cfg.bullet },
        laser: state.cfg.laser,
        rings: state.cfg.rings,
      };
      const shot = scoreHole(spot, scoring);
      registerShot(shot);
      drawShotPreview(shot, spot, state.Hinv);
      state.laserCandidates = state.laserCandidates.filter((item) => item.time > spot.time);
      vibrate(60);
    } catch (e) {
      log("laser shot capture error", e);
      setDebug("Laser capture error: " + e.message);
    } finally {
      state.processing = false;
    }
  }

  /* --------------------------- shot pipeline ----------------------------- */
  async function captureShot(simulate) {
    if (state.processing) {
      if (!simulate && state.mode === "hole" && baselinePromise) await baselinePromise;
      if (state.processing) {
        setDebug("Shot trigger skipped — target frame is still processing. Wait before the next shot.");
        return;
      }
    }
    if (!worker || !state.stream) return;
    if (!state.locked || !state.H) {
      setDebug("Lock the four markers before firing.");
      beep(300, 0.12, "square", 0.06);
      return;
    }
    if (state.mode !== "hole" && !simulate) {
      return captureLaserShot();
    }
    state.processing = true;
    try {
      if (!simulate) await sleep(state.cfg.trigger.shotDelayMs);
      if (state.mode === "hole" && targetCalibrationPromise) await targetCalibrationPromise;

      const frame = readFrameBuf();
      if (!frame) return;
      const H = [...state.H];
      const Hinv = state.Hinv ? [...state.Hinv] : null;
      const target = { ...state.cfg.target };
      const bullet = { ...state.cfg.bullet };
      const scoring = {
        target,
        bullet,
        laser: state.cfg.laser,
        rings: state.cfg.rings,
      };
      const mode = state.mode;
      const res = await callWorker("capture", {
        buf: frame.buf,
        width: frame.width,
        height: frame.height,
        H,
        warpW: target.warpWidthPx,
        warpH: target.warpHeightPx,
        morph: state.cfg.morph,
        bullet,
        mode,
        laser: state.cfg.laser,
        hasBaseline: mode === "hole" ? state.hasBaseline : false,
        simulate: simulate || undefined,
      }, [frame.buf]);

      if (mode === "hole") state.hasBaseline = true;
      const holes = res.holes || [];

      if (holes.length && !state.targetCenter) {
        setDebug("New hole found, but bullseye center is not calibrated. Find the printed rings or tap SET CENTER before scoring.");
        return;
      }

      if (holes.length) {
        const detectedShots = mode === "hole" ? holes : holes.slice(0, 1);
        for (const hole of detectedShots) {
          const shot = scoreHole(hole, scoring);
          registerShot(shot);
          drawShotPreview(shot, hole, Hinv);
        }
        vibrate(60);
      } else {
        beep(500, 0.1, "square", 0.08);
        setDebug(mode === "hole"
          ? (state.hasBaseline
            ? "Shot captured — no new dark hole found. Check focus, lighting, and target baseline."
            : "Shot captured — baseline was not ready. Lock the target again.")
          : "Captured — no " + mode + " laser dot detected.");
      }
    } catch (e) {
      log("shot pipeline error", e);
      setDebug("Pipeline error: " + (e && e.message ? e.message : e));
    } finally {
      state.processing = false;
    }
  }

  // Synthetic shot: inject a fake bullet hole at a random position inside the
  // target so the full diff→detect→score pipeline can be tested at home with
  // no paper damage and no gunshot sound.
  function simulateShot() {
    if (!worker || !state.stream) return;
    if (!state.locked || !state.H) {
      setDebug("Lock the four markers before simulating.");
      beep(300, 0.12, "square", 0.06);
      return;
    }
    const t = state.cfg.target;
    const pxPerMm = state.targetCenter ? state.targetCenter.pxPerMmX : t.warpWidthPx / t.widthMm;
    const cx = state.targetCenter ? state.targetCenter.xPx : t.warpWidthPx * (t.centerXmm / t.widthMm);
    const cy = state.targetCenter ? state.targetCenter.yPx : t.warpHeightPx * (t.centerYmm / t.heightMm);

    // Random radius inside the scoring area (~5..65 mm from center).
    const ang = Math.random() * Math.PI * 2;
    const radMm = 5 + Math.random() * 60;
    const x = cx + Math.cos(ang) * radMm * pxPerMm;
    const y = cy + Math.sin(ang) * radMm * pxPerMm;
    const isLaser = state.mode !== "hole";
    const r = isLaser
      ? Math.max(8, 3 * pxPerMm) // ~6 mm laser spot
      : Math.max(6, (state.cfg.bullet.caliberMm / 2) * pxPerMm);
    let color;
    if (isLaser) {
      if (state.mode === "red") color = "red";
      else if (state.mode === "green") color = "green";
      else color = Math.random() < 0.5 ? "red" : "green";
    }

    captureShot({ x, y, r, color });
  }

  /* --------------------------- scoring ----------------------------------- */
  function scoreHole(hole, cfg = state.cfg) {
    const t = cfg.target;
    const isLaser = !!hole.color;
    const l = cfg.laser || {};
    const b = cfg.bullet;
    const pxPerMmX = state.targetCenter ? state.targetCenter.pxPerMmX : t.warpWidthPx / t.widthMm;
    const pxPerMmY = state.targetCenter ? state.targetCenter.pxPerMmY : t.warpHeightPx / t.heightMm;
    const centerXpx = state.targetCenter ? state.targetCenter.xPx : t.warpWidthPx * (t.centerXmm / t.widthMm);
    const centerYpx = state.targetCenter ? state.targetCenter.yPx : t.warpHeightPx * (t.centerYmm / t.heightMm);
    const mmX = t.centerXmm + (hole.xPx - centerXpx) / pxPerMmX;
    const mmY = t.centerYmm + (hole.yPx - centerYpx) / pxPerMmY;
    const dist = Math.hypot(mmX - t.centerXmm, mmY - t.centerYmm);

    // For lasers we score the center of the spot; for bullets we can apply
    // break-the-line radius compensation.
    const breakLineEnabled = isLaser ? !!l.breakTheLine : b.breakTheLine;
    const diameterMm = isLaser ? (l.spotDiameterMm || 0) : b.caliberMm;
    let eff = dist;
    if (breakLineEnabled && diameterMm > 0) eff = Math.max(0, dist - diameterMm / 2);

    // Miss by default; match the innermost ring whose radius covers the shot.
    let ring = { label: "M", score: 0 };
    for (const r of cfg.rings) {
      if (eff <= r.diameterMm / 2) { ring = r; break; }
    }
    const clockHour = clockFromDxDy(mmX - t.centerXmm, mmY - t.centerYmm);
    return { xPx: hole.xPx, yPx: hole.yPx, mmX, mmY, distMm: dist, ring: ring.label, score: ring.score, clockHour, kind: isLaser ? "laser" : "hole", color: hole.color || null };
  }

  // Clock-face bearing (shooting convention): 12 = up, 3 = right, 6 = down,
  // 9 = left, measured clockwise. dx is right-positive, dy is down-positive.
  function clockFromDxDy(dx, dy) {
    let ang = Math.atan2(dx, -dy);
    if (ang < 0) ang += Math.PI * 2;
    return (Math.round(ang / (Math.PI / 6)) % 12) || 12;
  }

  function registerShot(shot) {
    state.shots.push(shot);
    readShots.textContent = String(state.shots.length);
    const tag = shot.color ? (shot.color === "red" ? "R" : "G") : "";
    readLastShot.textContent = shot.ring === "M" ? "MISS" : (shot.ring + " · " + (Number.isInteger(shot.score) ? shot.score : shot.score.toFixed(1)) + (tag ? " · " + tag : ""));
    readLastShot.classList.add("is-hit");
    updateGrouping();
    stage.classList.remove("is-flash");
    void stage.offsetWidth;
    stage.classList.add("is-flash");
    speakScore(shot);
  }

  // Group size / MOA over the latest N shots (a magazine), not the whole
  // session, matching how shooters review grouping per string.
  function updateGrouping() {
    const recent = state.shots.slice(-state.highlightCount);
    if (recent.length >= 2) {
      const spread = groupSpreadMm(recent);
      readGroup.textContent = spread.toFixed(1) + " mm";
      readMoa.textContent = "MOA " + groupSizeToMoa(spread, state.cfg.target.distanceM).toFixed(2);
    } else {
      readGroup.textContent = "—";
      readMoa.textContent = "—";
    }
  }

  function groupSpreadMm(shots) {
    const pts = (shots || state.shots).map((s) => [s.mmX, s.mmY]);
    let maxD = 0;
    for (let i = 0; i < pts.length; i++)
      for (let j = i + 1; j < pts.length; j++)
        maxD = Math.max(maxD, Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]));
    return maxD;
  }

  function groupSizeToMoa(spreadMm, distanceM) {
    if (!distanceM || spreadMm <= 0) return 0;
    return spreadMm / (distanceM * 0.2909);
  }

  /* --------------------------- overlay drawing --------------------------- */
  function mapWarpedToScreen(wx, wy) {
    if (!state.Hinv) return [wx, wy];
    return applyH(state.Hinv, wx, wy);
  }

  function drawMarkers() {
    overlayCtx.clearRect(0, 0, overlay.width, overlay.height);

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

    if (state.locked && state.Hinv) {
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

      const center = state.targetCenter || {
        xPx: t.warpWidthPx * (t.centerXmm / t.widthMm),
        yPx: t.warpHeightPx * (t.centerYmm / t.heightMm),
      };
      const ctr = mapWarpedToScreen(center.xPx, center.yPx);
      overlayCtx.strokeStyle = state.targetCenter ? "rgba(56,224,160,0.95)" : "rgba(255,176,32,0.6)";
      overlayCtx.lineWidth = 2;
      overlayCtx.beginPath();
      overlayCtx.arc(ctr[0], ctr[1], 8, 0, Math.PI * 2);
      overlayCtx.moveTo(ctr[0] - 12, ctr[1]);
      overlayCtx.lineTo(ctr[0] + 12, ctr[1]);
      overlayCtx.moveTo(ctr[0], ctr[1] - 12);
      overlayCtx.lineTo(ctr[0], ctr[1] + 12);
      overlayCtx.stroke();
    }

    if (state.liveLaserSpot && state.Hinv) {
      const spot = state.liveLaserSpot;
      const p = mapWarpedToScreen(spot.xPx, spot.yPx);
      overlayCtx.strokeStyle = spot.color === "red" ? "#ff504f" : "#40ff80";
      overlayCtx.lineWidth = 2;
      overlayCtx.beginPath();
      overlayCtx.arc(p[0], p[1], 10, 0, Math.PI * 2);
      overlayCtx.stroke();
    }

    // Shot markers: a small dot at the hole centroid + the ring label.
    // The latest `highlightCount` shots are bright amber; older ones stay
    // visible but dimmed.
    const n = state.overlayHoles.length;
    state.overlayHoles.forEach((h, i) => {
      const recent = i >= n - state.highlightCount;
      const col = h.color === "red" ? (recent ? "255,80,80" : "160,80,80")
                : h.color === "green" ? (recent ? "64,255,128" : "80,160,100")
                : (recent ? "255,176,32" : "150,162,180");
      const alpha = recent ? 0.95 : 0.45;

      overlayCtx.fillStyle = `rgba(${col},${alpha})`;
      overlayCtx.beginPath();
      overlayCtx.arc(h.x, h.y, recent ? 4.5 : 3, 0, Math.PI * 2);
      overlayCtx.fill();

      overlayCtx.fillStyle = `rgba(${col},${alpha})`;
      overlayCtx.font = recent ? "bold 14px monospace" : "12px monospace";
      overlayCtx.fillText(h.ring, h.x + 9, h.y - 9);
    });
  }

  function drawShotPreview(shot, hole, Hinv) {
    const p = Hinv ? applyH(Hinv, hole.xPx, hole.yPx) : mapWarpedToScreen(hole.xPx, hole.yPx);
    const dir = shot.clockHour && shot.distMm > 3 ? shot.clockHour + "h" : "C";
    state.overlayHoles.push({ x: p[0], y: p[1], ring: shot.ring + dir, color: shot.color });
    drawMarkers();
  }

  /* --------------------------- UI wiring --------------------------------- */
  function openDrawer() { drawer.classList.add("is-open"); drawerScrim.classList.add("is-open"); }
  function closeDrawer() { drawer.classList.remove("is-open"); drawerScrim.classList.remove("is-open"); }

  btnCamera.addEventListener("click", () => { if (state.stream) stopCamera(); else startCamera(); });

  btnLock.addEventListener("click", async () => {
    if (!worker) return;
    if (state.locked) {
      state.learnedMarkers = new Map();
      try { localStorage.removeItem(LEARN_KEY); } catch (_) {}
      state.locked = false;
      state.lockCenters = null;
    }
    state.hasBaseline = false;
    callWorker("clear", {}).catch(() => {});
    await runMarkerDetection();
    if (state.locked) {
      setDebug("Locked. Baseline captured.");
      await primeBaseline();
    }
  });

  btnTest.addEventListener("click", () => simulateShot());

  btnRotate.addEventListener("click", () => {
    state.rotationOffset = (state.rotationOffset + 1) % 4;
    runMarkerDetection();
  });

  btnCenter.addEventListener("click", () => {
    if (!state.locked || !state.H) return;
    state.pickingCenter = true;
    overlay.classList.add("is-picking-center");
    setDebug("Tap the center of the printed bullseye in the camera view.");
  });

  overlay.addEventListener("click", (event) => {
    if (!state.pickingCenter || !state.H) return;
    const rect = overlay.getBoundingClientRect();
    const scale = Math.max(rect.width / video.videoWidth, rect.height / video.videoHeight);
    const offsetX = (rect.width - video.videoWidth * scale) / 2;
    const offsetY = (rect.height - video.videoHeight * scale) / 2;
    const x = (event.clientX - rect.left - offsetX) / scale;
    const y = (event.clientY - rect.top - offsetY) / scale;
    const [xPx, yPx] = applyH(state.H, x, y);
    state.targetCenter = {
      xPx,
      yPx,
      pxPerMmX: state.cfg.target.warpWidthPx / state.cfg.target.widthMm,
      pxPerMmY: state.cfg.target.warpHeightPx / state.cfg.target.heightMm,
      confidence: 1,
      manual: true,
    };
    state.pickingCenter = false;
    overlay.classList.remove("is-picking-center");
    setDebug("Bullseye center set manually. Target scale still uses configured dimensions.");
    drawMarkers();
  });

  btnClear.addEventListener("click", () => {
    state.shots = [];
    state.hasBaseline = false;
    state.overlayHoles = [];
    state.laserCandidates = [];
    callWorker("clear", {}).catch(() => {});
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

  sensitivity.addEventListener("input", () => { sensVal.textContent = parseFloat(sensitivity.value).toFixed(2); });
  caliber.addEventListener("input", () => {
    const v = parseFloat(caliber.value).toFixed(1);
    caliberVal.textContent = v;
    if (state.cfg) state.cfg.bullet.caliberMm = parseFloat(v);
  });
  breakLine.addEventListener("change", () => {
    if (state.cfg) state.cfg.bullet.breakTheLine = breakLine.checked;
  });

  detectionMode.addEventListener("change", () => {
    state.mode = detectionMode.value;
    if (state.cfg) state.cfg.detection = { ...(state.cfg.detection || {}), mode: state.mode };
    state.hasBaseline = false;
    callWorker("clear", {}).catch(() => {});
    state.laserCandidates = [];
    state.liveLaserSpot = null;
    stopLaserTracking();
    if (state.mode === "hole" && state.locked) primeBaseline();
    else if (state.mode !== "hole" && state.locked) startLaserTracking();
    setDebug(state.mode === "hole" ? "Mode: bullet hole detection" : "Tracking " + state.mode + " laser continuously.");
  });

  recentShots.addEventListener("input", () => {
    state.highlightCount = Math.max(1, parseInt(recentShots.value, 10) || 1);
    recentVal.textContent = String(state.highlightCount);
    updateGrouping();
    drawMarkers();
  });

  /* --------------------------- boot -------------------------------------- */
  window.addEventListener("resize", sizeOverlay);
  window.addEventListener("orientationchange", () => setTimeout(sizeOverlay, 300));

  async function boot() {
    await loadConfig();
    applyConfigToUI();
    sizeOverlay();
    if ("serviceWorker" in navigator) {
      try { await navigator.serviceWorker.register("./sw.js"); } catch (e) { log("sw register failed", e); }
    }
  }

  boot();
})();
