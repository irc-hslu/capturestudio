# Interactive Web Viewer

An interactive web application for visualizing **dynamic point clouds** and **Gaussian splats** produced by the capturestudio pipeline.  
Built with [Next.js (App Router)](https://nextjs.org) and [three.js](https://threejs.org/) for real-time rendering.

---

## Table of Contents

1. [Features](#features)  
2. [Requirements](#requirements)  
3. [Quick Start](#quick-start)  
4. [App Directory Structure](#app-directory-structure)  
   - [Routes & Behavior](#routes--behavior)  
   - [Key Components](#key-components)  
5. [Reconstruction File Structure](#reconstruction-file-structure)  
   - [`manifest.json` — Detailed Schema](#manifestjson--detailed-schema)  
   - [Teaser Naming & Discovery](#teaser-naming--discovery)  
   - [Per-camera Frame Layout](#per-camera-frame-layout)  
6. [Configuration & Serving](#configuration--serving)  
7. [Development](#development)  
8. [Performance Notes](#performance-notes)  
9. [Troubleshooting](#troubleshooting)  
10. [Roadmap / Progress Tracking](#roadmap--progress-tracking)  
11. [License](#license)

---

## Features

- Landing grid of **sessions** with synchronized teaser playback and hover/focus highlighting.  
- Per-session **ranges** grid with **global** and **per-card** loading indicators and synchronized teasers.  
- Three.js **viewer** for dynamic point clouds / Gaussian splats with multi-camera support.  
- Camera **frusta** visualization and an orbit path attached to ground-truth camera trajectories.  
- Multiple **depth sources** and reconstruction types selectable via URL parameters.  
- Robust handling of missing/failed assets with graceful fallbacks.  
- Global sticky header with a top-right logo.

---

## Requirements

- **Node.js ≥ 18** (LTS recommended)  
- A modern browser with **WebGL2** support (Chromium, Firefox, Safari TP recommended)  
- Datasets placed under `public/reconstructions/` (you may symlink to external storage)

---

## Quick Start

1) **Install & run:**
```bash
npm i
npm run dev
# or: yarn dev / pnpm dev / bun dev
```

2) **Provide data:**  
Place (or symlink) your sessions under:

```
public/reconstructions/
  <Session_1>/
  <Session_2>/
  ...
```

3) Open **http://localhost:3000**.

> Fonts: uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to load [Geist](https://vercel.com/font).

---

## App Directory Structure

```
app/
  layout.tsx                       # global layout + top-right logo/header
  page.tsx                         # "/": landing grid of sessions (synced teasers)
  session/
    [session]/
      page.tsx                     # "/session/:session": ranges grid for that session (synced teasers)
  viewer/
    [session]/
      [start]/
        [total]/
          page.tsx                 # "/viewer/:session/:start/:total": 3D viewer (Three.js)

components/
  ReconGrid.tsx                    # landing grid (hover/focus highlight, loading, sync)
  SyncedTeaserGrid.tsx             # per-session grid (hover/focus highlight, loading, sync)
  ThreeViewer.tsx                  # main viewer component

public/
  reconstructions/                 # served at /reconstructions/*
    <Session_X>/
      manifest.json
      ...
  logo.svg                         # logo rendered by layout.tsx
```

### Routes & Behavior

- **`/`** — Lists sessions (subfolders of `public/reconstructions`).  
  Each card shows `teaser_grid_*.mp4` if present. Teasers start **together** when ready and remain **time-synchronized**.

- **`/session/:session`** — Lists reconstruction **ranges** for the session.  
  Each card prefers a **range-level** teaser and falls back to a session-level grid teaser. Shows **global** and **per-card** loading states; all teasers start **in sync**.

- **`/viewer/:session/:start/:total?recon=...&depth=...&modality=...`** — Interactive 3D viewer.  
  URL params:
  - `recon`: reconstruction type (e.g., `pcd`, `gs`) — default: first available.  
  - `depth`: depth key (e.g., `bilateral_temporal`) — default: `defaultDepthSource` or first available.  
  - `modality`: visualization modality (e.g., `color`) — default: `color`.

### Key Components

- **`ReconGrid`** — Landing grid with synchronized teaser playback, hover/focus highlighting, and global progress bar.  
- **`SyncedTeaserGrid`** — Per-session grid with the same synchronization and loading features.  
- **`ThreeViewer`** — Three.js canvas, dynamic resource loading, camera frusta, and orbit path.

---

## Reconstruction File Structure

The app expects a **session-based** layout under `public/reconstructions`:

```
public/reconstructions/
├─ Session_A/
│  ├─ manifest.json
│  ├─ teaser_grid_0001_120.mp4                # optional session-level grid teaser(s)
│  ├─ pcd_bilateral_temporal/                  # example: <recon>_<depthKey> = rootDir
│  │  ├─ teaser_0001_120.mp4                   # range-level teaser for [frameStart,totalFrames]
│  │  ├─ cam01/
│  │  │   ├─ 0001.ply
│  │  │   ├─ 0002.ply
│  │  │   └─ ...
│  │  ├─ cam02/
│  │  │   └─ ...
│  │  └─ ... (other cameras)
│  ├─ gs_bilateral_temporal/
│  │  ├─ teaser_0001_120.mp4
│  │  └─ cam01/ ... (format depends on GS exporter)
│  └─ ... (other recon/depth combinations)
└─ Session_B/
   ├─ manifest.json
   └─ ...
```

### `manifest.json` — Detailed Schema

Each session includes a `manifest.json` with **session metadata**, **camera rig**, and **reconstruction ranges**.  
Below is the schema used by the webapp, with field purposes and constraints.

#### Top-level fields

| Field             | Type / Example                           | Required     | Purpose |
|-------------------|------------------------------------------|--------------|---------|
| `sessionId`       | `"Session_A"`                            | Optional     | Internal identifier (if omitted, the folder name is used). |
| `title`           | `"Performance Title"`                    | Recommended  | Human-readable title shown in the UI. |
| `performers`      | `["Artist A", "Artist B"]`               | Optional     | Shown in the viewer overlay. |
| `location`        | `"Venue Name"`                           | Optional     | Shown in the viewer overlay. |
| `date`            | `"YYYY-MM-DD"`                           | Optional     | Shown in the viewer overlay (e.g., `"2025-05-01"`). |
| `time`            | `"HH:MM"`                                | Optional     | Shown in the viewer overlay (e.g., `"16:38"`). |
| `fps`             | `30`                                     | Recommended  | Default playback rate for ranges (can be range-specific in the future). |
| `pointBudget`     | `300000`                                 | Optional     | Viewer hint for max points to render concurrently. |
| `cameras`         | `Camera[]`                               | Recommended  | Camera rig used for frusta/orbits and multi-cam behavior. |
| `reconstructions` | `Range[]`                                | **Required** | Defines playable ranges and their storage layout. |

#### `Camera` object

```json
{
  "id": "cam04",
  "index": "04",
  "position": [0.0, 1.7, 2.3],
  "rotation": [0.0, 0.0, 0.0, 1.0],
  "intrinsics": {
    "fx": 1066.81, "fy": 1065.84,
    "cx": 490.27,  "cy": 526.23,
    "w": 1024, "h": 1024
  }
}
```

- `id`: unique camera identifier (used as folder name under a rootDir).  
- `index`: optional string/number index (for human ordering).  
- `position`: world-space camera origin in meters `[x,y,z]`.  
- `rotation`: quaternion `[x,y,z,w]` defining camera orientation (match your exporter’s convention).  
- `intrinsics`: pinhole parameters and image dimensions. Distortion parameters can be added if needed; they are currently ignored by the viewer.

#### `Range` object

```json
{
  "frameStart": 1,
  "totalFrames": 120,
  "recon_types": ["pcd", "gs"],
  "defaultDepthSource": "bilateral_temporal",
  "depthSources": {
    "bilateral_temporal": { "label": "camera depth", "dir": "pcd_bilateral_temporal" },
    "raw": { "label": "raw depth", "dir": "pcd_raw" }
  }
}
```

- `frameStart` / `totalFrames`: define the contiguous frame window for this range.  
- `recon_types`: available reconstruction types for this range (e.g., **pcd** for point clouds, **gs** for Gaussian splats).  
- `defaultDepthSource`: key into `depthSources` used by default in the UI.  
- `depthSources`: map of **depth keys** to display labels and directories.  
  - `dir`: **root directory** name for files; if the name already starts with `<recon>_`, it is used as-is.  
  - If `dir` does **not** start with `<recon>_`, the app will probe `<recon>_<depthKey>` (e.g., `pcd_bilateral_temporal`) when looking for teasers and frames.

> **Minimum viable manifest:** `title`, `fps`, and `reconstructions[]` with at least one `Range` containing `frameStart`, `totalFrames`, `recon_types[]`, and one `depthSources` entry with a valid `dir`. Cameras are recommended for frusta/orbit features but not strictly required for basic playback.

### Teaser Naming & Discovery

The grids search for MP4 teasers using the following patterns:

- **Session-level grid teaser** (used on `/` and as a fallback in `/session/:session`):
  ```
  public/reconstructions/${session}/teaser_grid_${pad4(frameStart)}_${pad3(totalFrames)}.mp4
  # e.g., teaser_grid_0001_120.mp4
  ```

- **Range-level teaser** (preferred on `/session/:session`):
  ```
  public/reconstructions/${session}/${rootDir}/teaser_${pad4(frameStart)}_${pad3(totalFrames)}.mp4
  # e.g., pcd_bilateral_temporal/teaser_0001_120.mp4
  ```

Where:
- `pad4(n)` = `String(n).padStart(4,"0")` (for `frameStart`)  
- `pad3(n)` = `String(n).padStart(3,"0")` (for `totalFrames`)

### Per-camera Frame Layout

Per recon/depth **rootDir** (e.g., `pcd_bilateral_temporal/`), frames are grouped by **camera id**:

```
<rootDir>/
  cam01/
    0001.ply
    0002.ply
    ...
  cam02/
    ...
```

The viewer resolves these paths based on the selected `recon`, `depth`, and current frame.

---

## Configuration & Serving

Assets under `public/reconstructions/` are served **statically** at:

```
/reconstructions/<Session>/<...>
```

Tips:

- For large datasets stored elsewhere, create a **symlink** into `public/reconstructions`:
  ```bash
  ln -s /data/my-datasets/Session_A public/reconstructions/Session_A
  ```
- Filenames are case-sensitive on most servers—keep naming consistent with the manifest.
- If you cannot place files under `public/`, implement a **route handler** that maps `/reconstructions/*` to your external storage; the UI paths remain the same.

---

## Development

- **Scripts**
  ```bash
  npm i            # install
  npm run dev      # start dev server
  npm run build    # production build
  npm start        # run production server (after build)
  ```

- **Styling & UI**
  - Tailwind utility classes for layout, spacing, rings, and shadows.  
  - Loading UX: **global progress** bar + **per-card spinner** overlays.  
  - Cards are fully clickable (`<Link>` wraps the card), and keyboard focus is indicated via `focus-visible` rings.

- **Accessibility**
  - All interactive cards are keyboard navigable.  
  - Videos are `muted` + `playsInline` to satisfy autoplay policies.

---

## Performance Notes

- **Synchronized playback:** teasers start together once all are ready; micro-drift is corrected against a master clock.  
- **Autoplay policy:** keep teasers `muted` and `playsInline`.  
- **Load shedding:** many simultaneous videos can be heavy; consider fewer items per page or lower-res teasers.

---

## Troubleshooting

- **Nothing appears in `/`** — Ensure your sessions are under `public/reconstructions/` and contain `manifest.json`.  
- **Teasers don’t play or never start** — Verify teaser filenames match the naming convention and are readable by the server.  
- **Viewer shows no geometry** — Confirm per-camera folders and frame files exist under the selected `rootDir` (`<recon>_<depthKey>`).  
- **Autoplay blocked** — Confirm video tags are `muted` and `playsInline`.  
- **Misaligned cameras/frusta** — Check quaternion order `[x,y,z,w]` and metric units.

---

## Roadmap / Progress Tracking

- [x] Loading of point cloud out of the main thread  
- [x] Rendering of a single point cloud  
- [x] Background loading of sequence of point clouds and swapping  
- [x] Initial support for multi-camera point clouds sequences  
- [x] Swapping of active camera based on proximity  
- [x] Support for multiple depth sources  
- [x] Display of camera frusta  
- [x] Virtual orbit attached to GT camera positions and orientations  
- [x] UI for selection of performance for visualization  
- [x] Metadata display  
- [ ] Keyframe interpolation of point clouds  
- [ ] Support of Gaussian splat clouds  
- [ ] Support for compressed formats

---

### License and Attribution

The source code is licensed under **Apache License 2.0**.  
Redistributions must preserve the LICENSE and the contents of the **NOTICE** file.

- SPDX: `Apache-2.0`
- See: `LICENSE` and `NOTICE`


