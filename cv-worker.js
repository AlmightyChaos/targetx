/* ==========================================================================
   TargetX — OpenCV.js worker
   --------------------------------------------------------------------------
   All OpenCV work runs here so the main thread never blocks on the ~10 MB
   WASM compile. The worker is stateless with respect to homography (the main
   thread sends it with each capture); it only persists the diff baseline.

   Messages (main -> worker):
     { id, type: "detect", buf, width, height }           -> detect markers
     { id, type: "capture", buf, width, height, H, warpW, warpH, morph, bullet, hasBaseline }
                                                          -> warp + diff + score
     { id, type: "clear" }                                -> drop baseline

   Replies (worker -> main):
     { id, type: "detect", markers: [{code, center, corners}] }
     { id, type: "capture", holes: [{xPx, yPx, wPx, hPx, areaPx}] }
     { id, type: "ready" }
     { id, type: "error", message }
   ========================================================================== */

/* --------------------------- constants --------------------------- */
const MARKER_CELLS = 6;      // 4x4 dict + 1-cell black border each side
const MARKER_CELL_PX = 8;    // sampling resolution per cell
const MIN_MARKER_AREA = 120; // px^2

let cv = null;
let baseline = null; // grayscale warped Mat
const pending = new Map(); // id -> resolve/reject

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === "detect") handleDetect(msg);
  else if (msg.type === "capture") handleCapture(msg);
  else if (msg.type === "clear") handleClear(msg);
};

function reply(msg) {
  self.postMessage(msg);
}

function scalar0(s) {
  if (s == null) return 0;
  if (typeof s[0] === "number") return s[0];
  if (s.val && typeof s.val[0] === "number") return s.val[0];
  if (typeof s.get === "function") return s.get(0);
  if (s.data64F && s.data64F.length) return s.data64F[0];
  return 0;
}

function rgbaMatFromBuffer(buf, width, height) {
  // buf is an ArrayBuffer of RGBA bytes (row-major, width*height*4).
  const data = new Uint8ClampedArray(buf, 0, width * height * 4);
  return cv.matFromImageData({ data, width, height });
}

/* --------------------------- ArUco detection --------------------------- */
function detectMarkers(srcGray) {
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
    pts.push([approxMat.data32S[i * 2], approxMat.data32S[i * 2 + 1]]);
  }
  const sums = pts.map((p) => p[0] + p[1]);
  const rect = [null, null, null, null];
  rect[0] = pts[sums.indexOf(Math.min(...sums))];
  rect[2] = pts[sums.indexOf(Math.max(...sums))];
  const rest = pts.filter((p) => p !== rect[0] && p !== rect[2]);
  if (rest.length !== 2) return null;
  rect[1] = rest[0][1] - rest[0][0] <= rest[1][1] - rest[1][0] ? rest[0] : rest[1];
  rect[3] = rest[0] === rect[1] ? rest[1] : rest[0];
  return rect;
}

function sampleMarker(srcGray, corners) {
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
  if (borderMean > innerMean - 18) return null;

  const innerVals = [];
  for (let r = 1; r <= 4; r++) for (let c = 1; c <= 4; c++) innerVals.push(cell[r][c]);
  if (Math.max(...innerVals) - Math.min(...innerVals) < 30) return null;

  const thr = innerMean;
  const bits = [];
  for (let r = 1; r <= 4; r++) {
    const row = [];
    for (let c = 1; c <= 4; c++) row.push(cell[r][c] < thr ? 1 : 0);
    bits.push(row);
  }

  let best = null;
  for (let rot = 0; rot < 4; rot++) {
    const code = codeOf(bits, rot);
    if (best === null || code < best.code) best = { code, orientation: rot };
  }
  return { code: best.code, orientation: best.orientation };
}

function codeOf(bits, rot) {
  let v = 0;
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      let rr = r, cc = c;
      if (rot === 1) { const t = rr; rr = cc; cc = 3 - t; }
      else if (rot === 2) { rr = 3 - r; cc = 3 - c; }
      else if (rot === 3) { const t = rr; rr = 3 - cc; cc = t; }
      v = (v << 1) | bits[rr][cc];
    }
  }
  return v;
}

/* --------------------------- diff & holes --------------------------- */
function diffAndFindHoles(curGray, morph, bullet) {
  const scale = morph.diffScale || 1;
  const W = Math.max(64, Math.round(curGray.cols * scale));
  const H = Math.max(64, Math.round(curGray.rows * scale));
  const a = new cv.Mat(), b = new cv.Mat();
  cv.resize(curGray, a, new cv.Size(W, H), 0, 0, cv.INTER_AREA);
  cv.resize(baseline, b, new cv.Size(W, H), 0, 0, cv.INTER_AREA);

  const diff = new cv.Mat();
  // Asymmetric diff: keep only regions where the CURRENT frame is darker
  // than the baseline (a freshly punched hole). absdiff is symmetric and
  // would also flag things that got brighter or disappeared, producing
  // false "holes" from camera shake, exposure drift, or simulated holes.
  cv.subtract(b, a, diff); // b - a > 0 where current (a) is darker
  b.delete(); // keep `a` (resized current frame) for the darkness check

  const blur = new cv.Mat();
  const k = morph.blurKernel || 5;
  cv.GaussianBlur(diff, blur, new cv.Size(k, k), 0);
  diff.delete();

  const bin = new cv.Mat();
  if (morph.thresholdMethod === "fixed") {
    cv.threshold(blur, bin, morph.fixedThreshold ?? 28, 255, cv.THRESH_BINARY);
  } else {
    cv.threshold(blur, bin, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  }
  blur.delete();

  const darknessMax = morph.darknessMax ?? 120;
  if ((morph.closeKernel || 0) > 0) {
    const kc = morph.closeKernel;
    const kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(kc, kc));
    const closed = new cv.Mat();
    cv.morphologyEx(bin, closed, cv.MORPH_CLOSE, kernel);
    bin.delete();
    kernel.delete();
    const holes = findHoles(closed, a, bullet, scale, darknessMax);
    a.delete();
    return holes;
  }
  const holes = findHoles(bin, a, bullet, scale, darknessMax);
  a.delete();
  return holes;
}

function findHoles(bin, curFrame, bullet, scale, darknessMax) {
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
    // A real hole is dark in the current frame. Reject blobs that are only
    // "slightly darker than baseline" but actually bright (e.g. shifted ring
    // edges from camera shake), which the asymmetric diff can still surface.
    const roi = curFrame.roi(new cv.Rect(rect.x, rect.y, rect.width, rect.height));
    const curMean = scalar0(cv.mean(roi));
    roi.delete();
    if (curMean > darknessMax) { cnt.delete(); continue; }
    const m = cv.moments(cnt);
    holes.push({
      xPx: (m.m10 / (m.m00 || 1)) / scale,
      yPx: (m.m01 / (m.m00 || 1)) / scale,
      wPx: rect.width / scale,
      hPx: rect.height / scale,
      areaPx: area / (scale * scale),
    });
    cnt.delete();
  }
  contours.delete();
  hierarchy.delete();
  bin.delete();

  holes.sort((x, y) => y.areaPx - x.areaPx);
  return holes;
}

/* --------------------------- handlers --------------------------- */
function handleDetect(msg) {
  try {
    const src = rgbaMatFromBuffer(msg.buf, msg.width, msg.height);
    const gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    src.delete();
    const markers = detectMarkers(gray);
    gray.delete();
    reply({ id: msg.id, type: "detect", markers });
  } catch (e) {
    reply({ id: msg.id, type: "error", message: String(e && e.message ? e.message : e) });
  }
}

function handleCapture(msg) {
  try {
    const src = rgbaMatFromBuffer(msg.buf, msg.width, msg.height);
    const H = cv.matFromArray(3, 3, cv.CV_64FC1, msg.H);

    const warped = new cv.Mat();
    cv.warpPerspective(src, warped, H, new cv.Size(msg.warpW, msg.warpH), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar(0, 0, 0, 0));
    src.delete();
    H.delete();

    // Clean grayscale of the current frame — always used as the next baseline.
    const cleanGray = new cv.Mat();
    cv.cvtColor(warped, cleanGray, cv.COLOR_RGBA2GRAY);

    let holes = [];
    const isSimulate = !!(msg.simulate && Number.isFinite(msg.simulate.x));
    if (isSimulate) {
      // Draw a fake dark "bullet hole" (with a torn-paper ring), then diff
      // that frame against the clean baseline so the real pipeline detects it.
      const sim = msg.simulate;
      cv.circle(warped, new cv.Point(sim.x, sim.y), Math.round(sim.r || 10), new cv.Scalar(0, 0, 0, 255), -1, cv.LINE_AA);
      cv.circle(warped, new cv.Point(sim.x, sim.y), Math.round((sim.r || 10) * 1.6), new cv.Scalar(90, 90, 90, 255), 2, cv.LINE_AA);
      const simGray = new cv.Mat();
      cv.cvtColor(warped, simGray, cv.COLOR_RGBA2GRAY);
      if (msg.hasBaseline && baseline) holes = diffAndFindHoles(simGray, msg.morph, msg.bullet);
      simGray.delete();
    } else if (msg.hasBaseline && baseline) {
      holes = diffAndFindHoles(cleanGray, msg.morph, msg.bullet);
    }
    warped.delete();

    // Baseline is always the CLEAN current frame, so a synthetic hole never
    // pollutes it and a later real shot isn't flagged as a "disappeared" hole.
    if (baseline) baseline.delete();
    baseline = cleanGray.clone();
    cleanGray.delete();

    reply({ id: msg.id, type: "capture", holes });
  } catch (e) {
    reply({ id: msg.id, type: "error", message: String(e && e.message ? e.message : e) });
  }
}

function handleClear(msg) {
  if (baseline) { baseline.delete(); baseline = null; }
  reply({ id: msg.id, type: "cleared" });
}

/* --------------------------- boot --------------------------- */
try {
  importScripts("https://docs.opencv.org/4.9.0/opencv.js");
  const waitReady = () => {
    if (cv && cv.Mat) { reply({ type: "ready" }); return; }
    if (self.cv && self.cv.Mat) { cv = self.cv; reply({ type: "ready" }); return; }
    setTimeout(waitReady, 50);
  };
  if (self.cv && self.cv.onRuntimeInitialized) {
    const prev = self.cv.onRuntimeInitialized;
    self.cv.onRuntimeInitialized = () => {
      if (typeof prev === "function") prev();
      waitReady();
    };
    waitReady();
  } else {
    waitReady();
  }
} catch (e) {
  reply({ type: "error", message: String(e && e.message ? e.message : e) });
}
