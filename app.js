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

  const DETECT_SCALE_MAX = 640; // downscale long edge for cheap detection

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
    hasBaseline: false,  // worker holds a diff baseline
    shots: [],
    rotationOffset: 0,
    learnedMarkers: null,
    processing: false,
    lastShotAt: 0,
    cfg: null,
    lastMarkers: [],     // markers in full-res video coords
    overlayHoles: [],
    lockCenters: null,   // last assigned marker centers (for drift reset)
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

  /* --------------------------- config ------------------------------------ */
  async function loadConfig() {
    try {
      const res = await fetch("./target_config.json", { cache: "no-cache" });
      if (res.ok) { state.cfg = JSON.parse(await res.text()); return state.cfg; }
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
  }

  /* --------------------------- worker ------------------------------------ */
  let worker = null;
  let workerReadyPromise = null;
  let reqId = 0;
  const pendingCallbacks = new Map();

  function onWorkerMessage(m) {
    if (m.type === "ready") {
      if (workerReadyResolver) workerReadyResolver();
      cvStatus.textContent = "OpenCV.js: ready ✓ (worker)";
      setLoading(false);
      return;
    }
    if (m.type === "status") { setLoading(true, m.message); return; }
    const cb = pendingCallbacks.get(m.id);
    if (!cb) return;
    clearTimeout(cb.t);
    pendingCallbacks.delete(m.id);
    if (m.type === "error") cb.reject(new Error(m.message));
    else cb.resolve(m);
  }

  let workerReadyResolver = null;

  function ensureWorker() {
    if (worker) return workerReadyPromise;
    worker = new Worker("./cv-worker.js");
    workerReadyPromise = new Promise((resolve) => { workerReadyResolver = resolve; });
    worker.addEventListener("message", (e) => onWorkerMessage(e.data));
    worker.addEventListener("error", (e) => {
      log("worker error", e);
      cvStatus.textContent = "OpenCV.js worker: failed";
      setLoading(false);
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
      setLoading(false);
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
    state.H = null;
    state.Hinv = null;
    state.hasBaseline = false;
    state.lockCenters = null;
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
    const best = [...markers].sort((a, b) => b.area - a.area).slice(0, 4);
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
      await sleep(200);
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
    const scaleBack = video.videoWidth / frame.width;

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
      center: [m.center[0] * scaleBack, m.center[1] * scaleBack],
      corners: m.corners.map((c) => [c[0] * scaleBack, c[1] * scaleBack]),
    }));
    state.lastMarkers = markers;

    const assignment = assignCorners(markers);
    const wasLocked = state.locked;

    if (assignment) {
      const hom = computeHomography(assignment);
      if (hom) {
        state.H = hom.H;
        state.Hinv = hom.Hinv;

        // Detect drift of the assigned markers to reset the diff baseline.
        const centers = [assignment.tl.center, assignment.tr.center, assignment.br.center, assignment.bl.center];
        if (!wasLocked || markerDrift(centers) > 14) {
          state.hasBaseline = false;
          callWorker("clear", {}).catch(() => {});
          primeBaseline();
        }
        state.lockCenters = centers;

        state.locked = true;
        alignHint.classList.add("is-hidden");
        if (!wasLocked) {
          setStatus("TARGET LOCKED", "locked");
          btnLock.textContent = "◈ RE-LOCK";
          beep(880, 0.08, "sine", 0.08);
          saveLearnedMarkers(assignment);
        }
      } else {
        state.locked = false;
      }
    } else if (wasLocked) {
      state.locked = false;
      state.hasBaseline = false;
      state.lockCenters = null;
      setStatus("SEARCHING MARKERS…", "ready");
      alignHint.classList.remove("is-hidden");
    }

    drawMarkers();
  }

  function markerDrift(centers) {
    if (!state.lockCenters) return Infinity;
    let max = 0;
    for (let i = 0; i < 4; i++) {
      max = Math.max(max, Math.hypot(centers[i][0] - state.lockCenters[i][0], centers[i][1] - state.lockCenters[i][1]));
    }
    return max;
  }

  // Warp the current frame and store it as the diff baseline (no diff, no
  // shot registration), so the very first shot after a lock is detected.
  async function primeBaseline() {
    if (!worker || !state.stream || !state.H || state.processing) return;
    state.processing = true;
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
    }
  }

  /* --------------------------- shot pipeline ----------------------------- */
  async function captureShot(simulate) {
    if (state.processing) return;
    if (!worker || !state.stream) return;
    if (!state.locked || !state.H) {
      setDebug("Lock the four markers before firing.");
      beep(300, 0.12, "square", 0.06);
      return;
    }
    state.processing = true;
    try {
      if (!simulate) await sleep(state.cfg.trigger.shotDelayMs);

      const frame = readFrameBuf(); // full res (matches H coordinate space)
      if (!frame) return;

      let res;
      try {
        res = await callWorker("capture", {
          buf: frame.buf,
          width: frame.width,
          height: frame.height,
          H: state.H,
          warpW: state.cfg.target.warpWidthPx,
          warpH: state.cfg.target.warpHeightPx,
          morph: state.cfg.morph,
          bullet: state.cfg.bullet,
          hasBaseline: state.hasBaseline,
          simulate: simulate || undefined,
        }, [frame.buf]);
      } catch (e) {
        log("capture error", e);
        setDebug("Pipeline error: " + (e && e.message ? e.message : e));
        return;
      }

      state.hasBaseline = true;
      const holes = res.holes || [];

      if (holes.length) {
        const shot = scoreHole(holes[0]);
        registerShot(shot);
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
    const pxPerMm = t.warpWidthPx / t.widthMm;
    const cx = t.warpWidthPx * (t.centerXmm / t.widthMm);
    const cy = t.warpHeightPx * (t.centerYmm / t.heightMm);

    // Random radius inside the scoring area (~5..65 mm from center).
    const ang = Math.random() * Math.PI * 2;
    const radMm = 5 + Math.random() * 60;
    const x = cx + Math.cos(ang) * radMm * pxPerMm;
    const y = cy + Math.sin(ang) * radMm * pxPerMm;
    const r = Math.max(6, (state.cfg.bullet.caliberMm / 2) * pxPerMm);

    captureShot({ x, y, r });
  }

  /* --------------------------- scoring ----------------------------------- */
  function scoreHole(hole) {
    const cfg = state.cfg;
    const t = cfg.target;
    const b = cfg.bullet;
    const pxPerMmX = t.warpWidthPx / t.widthMm;
    const pxPerMmY = t.warpHeightPx / t.heightMm;
    const mmX = hole.xPx / pxPerMmX;
    const mmY = hole.yPx / pxPerMmY;
    const dist = Math.hypot(mmX - t.centerXmm, mmY - t.centerYmm);
    let eff = dist;
    if (b.breakTheLine) eff = Math.max(0, dist - b.caliberMm / 2);
    let ring = cfg.rings[cfg.rings.length - 1];
    let score = ring.score;
    for (const r of cfg.rings) {
      if (eff <= r.diameterMm / 2) { ring = r; score = r.score; break; }
    }
    return { xPx: hole.xPx, yPx: hole.yPx, mmX, mmY, distMm: dist, ring: ring.label, score };
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

      const ctr = mapWarpedToScreen(t.warpWidthPx * (t.centerXmm / t.widthMm), t.warpHeightPx * (t.centerYmm / t.heightMm));
      overlayCtx.strokeStyle = "rgba(56,224,160,0.4)";
      overlayCtx.lineWidth = 1;
      overlayCtx.beginPath();
      overlayCtx.arc(ctr[0], ctr[1], 6, 0, Math.PI * 2);
      overlayCtx.stroke();
    }

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

  function drawShotPreview(shot, hole) {
    const p = mapWarpedToScreen(hole.xPx, hole.yPx);
    state.overlayHoles.push({ x: p[0], y: p[1], r: Math.max(8, hole.wPx), ring: shot.ring });
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
    }
    state.hasBaseline = false;
    callWorker("clear", {}).catch(() => {});
    await runMarkerDetection();
    if (state.locked) {
      setDebug("Locked. Baseline captured.");
      primeBaseline();
    }
  });

  btnTest.addEventListener("click", () => simulateShot());

  btnRotate.addEventListener("click", () => {
    state.rotationOffset = (state.rotationOffset + 1) % 4;
    runMarkerDetection();
  });

  btnClear.addEventListener("click", () => {
    state.shots = [];
    state.hasBaseline = false;
    state.overlayHoles = [];
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
