# icecamera

Development workspace for classifying sea ice in the CCGS *Amundsen*'s camera
photos: backfilling the photo archive, training and checking the seawater
filter, reviewing results, and experimenting with new triage methods. The live
classifier runs inside the [underway](https://github.com/rec3141/underway)
dashboard; this repository feeds it historical results and tested code, and is
never needed at runtime.

## How the camera products are made and used

1. **Photos.** The ship's 360° camera writes a frame every two minutes to
   `Data/Camera_360/<leg>/<day>/…/Camera360_<UTC stamp>_cam_3.jpg`; camera 3
   looks over the side at the sea surface.
2. **Live classification (underway).** `dashboard.ice_worker` queues each new
   photo after its start boundary and crops a sea region. A conservative
   seawater filter (image guards plus a tree model, `camera_seawater`) and a
   batched Gemma yes/no pass (`camera_batch`) set aside open water as 0% ice;
   anything uncertain, and every filter failure, goes on to Gemma 4 26B
   (llama.cpp on the ship server's GPU, shared with crew chat). Gemma sees the
   full frame, a reference key of real ice examples and the enlarged sea
   region, and returns surface percentages for six ice types (grease ice,
   nilas, thin ice floe, icy bits, brash ice, thick ice floe) plus water
   classes. A sample of filtered photos is re-checked by Gemma and reported,
   never acted on automatically.
3. **Products.** Results go to `ice.sqlite` under the product root
   (`/data/underway_server/ice`): per photo its time, status (pending,
   filtered, gemma), total ice percentage, the six type percentages and the
   full response, with cached source, sea-region and slice JPEGs.
4. **Use.** The dashboard serves them at `/api/ice/*` and plots them in its
   Ice camera group (concentration, composition, fingerprints, sea regions,
   slices), aboard only. The cruise report takes the camera's concentration
   and ice types at each station arrival and can colour its map track by them.
   Filtered means an estimated 0%; pending is unknown, never 0; all values are
   exploratory.

The live worker classifies new arrivals only. Older photos are classified
here and brought into the product store with
`python -m dashboard.ice_import --source <results> --since <ISO time>` (from
the underway checkout), which takes completed results and never queues photos.

## Layout

| Path | Holds |
|---|---|
| `tools/` | backfill workers, training, review pages, projections and experiments (below) |
| `dashboard/` | the camera modules shared with underway (`camera_batch`, `camera_features`, `camera_live`, `camera_model`, `camera_seawater` are identical to underway's), plus `camera_completion` and `ice_classifier`, used only here; not the web application |
| `tests/` | unit tests for the camera modules and queues |
| `tools/systemd/` | the hourly projection-refresh unit and timer |
| `ops/` | machine-local service units (git-ignored); read before installing |

## Tools

**Archive backfill and queues.** `ice-camera-arrivals` syncs unseen originals
from the ship share and puts them first; `ice-2026-queue`, `ice-cache-2026`,
`ice-leg4-queue`, `ice-overnight-queue` and `ice-round-robin-queue` build and
reorder queues; `ice-select-backfill` ranks human-labelled clusters by their
ice fraction to choose what to classify next; `ice-gemma-full`,
`ice-overnight`, `ice-parallel-classes` and `ice-monitored-review` run the
classification with thermal cutoffs; `ice-teacher-watch` restarts stalled
runs, cool-only.

**Seawater filter.** `train-camera-seawater`, `train-seawater-v2`,
`train-camera-presence` and `train-camera-area` train against human labels
with day holdouts; `ice-seawater-v2-review`, `ice-seawater-recheck` and
`ice-seawater-proofsheet` review thresholds and filtered photos.

**Triage experiments (shadow only; none makes live decisions).**
`ice-image-similarity`, `ice-patch-similarity`, `ice-native-triage`,
`ice-batch-triage-pilot` and `ice-batch-consensus`.

**Projections and review.** `ice-full-leg-tsne`, `ice-projections`,
`ice-feature-pcs`, `ice-refresh-2026-projection`, `ice-refresh-annotations`,
`ice-cluster-representatives`, `ice-region-explorer`, `ice-tree-review`,
`ice-camera-track` and `ice-environment-summary`.

**Prompt, key and model trials.** `ice-reference-key`, `ice-key-candidates`,
`ice-key-hires`, `ice-context-prompt`, `ice-resolution-test`,
`ice-rotated-preview`, `ice-gradient-trial`, `ice-model-comparison`,
`ice-vision-review`, `ice-muse-smoke`, `ice-response-quality`,
`ice-profile-demo`, `ice-texture` and `ice-taxonomy` (the versioned label set).

**Machine health.** `ice-temperature-page` samples CPU and GPU temperature
into a rolling page.

Each tool's docstring says what it reads and writes.

## Data

The tools use machine-local paths, set at the top of each script:

- 2026 originals: `/data/scratch/camera-originals` (synced from
  `/mnt/ship/Data/Camera_360`); 2025 originals: a local T7 SSD.
- Results, labels, queues and viewers:
  `/home/cryomics/Downloads/amundsen-ice-gemma-2026` and neighbouring
  directories under `~/Downloads`.

Models, photo caches and results are not in the repository (see
`.gitignore`). No tool downloads a model or sends a photo off the machine; the
model server is shared with crew chat, so runs load nothing extra.

## Development

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/miniforge3/bin/python3 -m unittest discover -s tests
```

Code moves to underway as a single script with explicit input and output
paths, no dependency on this repository, and a filter that fails open (an
error routes the photo to Gemma, never to 0%).

Development attribution: Codex (GPT-6 Astra).
