# TargetX — Shot Monitor (PWA)

Real-time bullet-hole detection and auto-scoring for ArUco-anchored paper
targets, running fully client-side as an installable full-screen PWA for
iPhone (Safari / Chrome) and Android.

## How it works

1. **Camera** — `getUserMedia` with `facingMode: environment` streams a
   lightweight preview (`playsinline` + `muted` so iOS never forces native
   fullscreen). No per-frame full-resolution processing.
2. **ArUco lock** — four `DICT_4X4_50` markers on the target corners are
   detected (dictionary-agnostic bit decoder) and used to compute a
   perspective homography. The live overlay draws a green alignment box.
3. **Audio trigger** — a Web Audio `AnalyserNode` watches microphone
   RMS/peak. A gunshot peak waits ~70 ms (muzzle blast / target oscillation
   settle), then captures the current frame.
4. **Diff & score** — the captured frame is warped to a flat 1024×1024 view,
   compared against the previous baseline (`absdiff → GaussianBlur → Otsu →
   morphological close → findContours`), filtered by area/aspect ratio, and
   the centroid is converted to physical mm and scored against the configured
   rings (with optional break-the-line bullet-radius compensation).

## Printing the target

Print the target sheet with one ArUco marker (dictionary `DICT_4X4_50`,
ids 0/1/2/3) centered at each of the four corners. Any marker generator works
(OpenCV, https://chev.me/arucogen, etc.). The default config assumes a
**170 × 170 mm** target with markers centered on the corners.

> The marker IDs only need to be unique per corner. The app learns the
> corner↔marker mapping automatically on the first successful lock (stored in
> `localStorage`) and is then rotation-invariant. Before the first lock, use
> the **↻** button to cycle the corner assignment if the overlay appears
> rotated.

## Local development

```bash
# any static server works; these keep the PWA (HTTPS) requirements happy:
npx serve .
# or
python -m http.server 8080
```

Open `http://localhost:PORT` on your phone (same Wi-Fi) or use a tunnel. Note
that `getUserMedia` requires HTTPS (or `localhost`).

## Configuration

Edit [`target_config.json`](./target_config.json):

- `target` — physical width/height (mm), bullseye center, warp resolution,
  and shooting distance (for MOA).
- `rings` — innermost-first list of `{ label, diameterMm, score }`.
- `bullet` — caliber (mm) for break-the-line compensation and hole-area /
  aspect-ratio filters.
- `trigger` — audio threshold, shot delay, cooldown.
- `morph` — diff blur/threshold/morphology knobs.

A JSON Schema is provided at [`target_config.schema.json`](./target_config.schema.json)
for editor validation/autocomplete.

## Deployment (GitHub Pages)

```bash
git init
git add .
git commit -m "feat: setup target scoring PWA with ArUco perspective and audio trigger"

# create the remote (GitHub CLI)
gh repo create targetx --public --source=. --remote=origin --push

# or manually:
# git remote add origin https://github.com/<user>/targetx.git
# git push -u origin main
```

Then enable Pages:

1. Repository → **Settings → Pages**.
2. Source: **GitHub Actions** (the included
   [`.github/workflows/deploy.yml`](./.github/workflows/deploy.yml) handles
   build + deploy automatically).
3. After the action runs, open:
   `https://<username>.github.io/targetx/`

On iPhone: open the URL in Safari → **Share → Add to Home Screen** → it
launches in standalone fullscreen with `manifest.json` + service worker
offline caching.

## Notes & limitations

- OpenCV.js (~8 MB WASM) loads from the OpenCV CDN; the first load needs a
  network connection and a few seconds.
- The audio threshold must be tuned to your firearm/range (use the
  **⚙ sensitivity** slider). For dry testing press **⦿ TEST**.
- The diff pipeline intentionally scores the single largest new blob per
  shot; very tight double-taps within the cooldown window may merge.
