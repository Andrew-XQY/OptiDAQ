# OptiDAQ

## Overview

OptiDAQ is built around multi-camera acquisition and stream handling for experiments at CERN CLEAR. An orchestrator runs actuator scans while camera capture and storage continue in background threads. The pipeline carries immutable frames, with actuator logic kept separate and storage backends sharing one interface. `experiment_v1` is the active raw acquisition workflow for real experiments and beam time: each camera's unmodified frame bytes are appended to `.raw` files alongside timing, format and run metadata. `experiment_lab_v1` has a separate configuration and runner for lab point-motor, DMD or camera-only acquisition, saving PNG images and `dataset.db`.

## Minimal example

Acquisition runs in a CERN acc-py environment with `pyda`, `pyda_rda3`, `pyrbac` and NumPy. Offline export needs Python and NumPy. From the repository root, install the package:

```bash
python -m pip install --no-build-isolation -e .
```

On the PC driving the ViALUX DMD, install `-e ".[dmd]"` instead to add ALP4lib and Pillow. The SGM generator is bundled and needs only NumPy. Outside an acc-py environment, add the CERN packages with `-e ".[cern]" --index-url https://acc-py-repo.cern.ch/repository/vr-py-releases/simple` on Python 3.9, 3.11 or 3.14; extras combine, for example `-e ".[cern,dmd]"`.

For the beam-time raw workflow, edit `configs/experiment_v1.json`, including `run.data_root`, camera settings and motor/reference settings. It defaults to the existing continuous motor row sweep; the runner also retains its magnet and camera-only modes. It does not run DMD scans or save PNGs during acquisition.

Start acquisition:

```bash
python experiments/experiment_v1.py --config configs/experiment_v1.json
```

For lab acquisition, edit `configs/experiment_lab_v1.json`. Set `run.data_root` to your storage directory, for example `/opt/data/CLEAR2026`, and check the camera and DMD settings. This profile selects cam1/cam2, an SGM-only DMD stream and PNG output. `dmd.count` is the number of displayed patterns; `scan.frames_per_point` is the saved-frame quota per camera per pattern.

```bash
python experiments/experiment_lab_v1.py --config configs/experiment_lab_v1.json
```

DMD/camera-only lab runs are stored under `run.data_root/runs/lab`; beam-time raw runs remain under `run.data_root/runs`. The lab run folder contains per-camera PNGs, image descriptors, timing sidecars, `resolved_config.json` and `dataset.db`. Keep these files together. DMD/camera-only modes retain independent camera capture and DMD settling/count waits.

For camera-only lab collection, set `actuator.mode` to `"noop"` and
`postprocess.actions` to `[]`. `actuator.points` defaults to 1; each point waits
for the configured `scan.frames_per_point` saved-frame quota per camera.

### Position-by-position motor raster

Use the same runner with the dedicated motor configuration:

```bash
python experiments/experiment_lab_v1.py --config configs/experiment_lab_motor_raster.json
```

This selects a 100 × 100 serpentine grid spanning X −5500 through +5500 and
Y −15000 through +15000 raw counts about `motors.reference_counters` (the saved
CLEAR2026 reference: X79912/Y19908). The absolute target ranges are X74412–85412
and Y4908–34908. Change only `scan.nx` and `scan.ny` to choose another grid
density over this same rectangle. `scan.span_counts` stores the full widths:
X11000 and Y30000. Both endpoints are included, with evenly spaced positions
rounded symmetrically to integer counts. At 100 × 100, X steps are 111 or 112
counts and Y steps are 303 or 304 counts, producing 10000 distinct positions.
Changing the range does not fix the observed 16-count positioning/readback
discrepancy. Preflight checks every target against controller limits; the new
optical coverage needs a bench check.
The starting live counters and scan reference are recorded separately. Omitting
`motors.reference_counters` centers the scan on the live start. The alternative
`scan.stride_counts` plus `scan.start_offset_counts` geometry remains supported;
do not combine it with `span_counts`. Raw-counter mode preserves the actual
`position_valid` flag; it performs no homing or counter reset.

For each point the runner moves and verifies idle/settled counters, opens a fresh
capture request, selects one eligible pair from the continuously Line1-triggered
cameras, verifies the held counters, saves both PNGs, verifies the counters again,
and commits the pair before moving on. Frames arriving outside an active request
are discarded. The guard is 2 s and the image-ready timing-proxy tolerance is
1 ms. This selects images in software; it does not gate exposure or prove that
the cameras exposed on the same hardware pulse. An exposure longer than the
guard is rejected. Keep settings fixed during acquisition.

The motor profile applies the last unclipped laser-test settings: cam1 100 ms,
cam2 1 ms, gain 0, Mono12, with the laser ND filter out. Check these against the
current optical setup. Existing crops/average binning produce 256 × 256 uint16
PNGs. Compression is lossless; the configured spatial transforms still apply.
To monitor cam2 without changing the acquisition settings, use preview with
both `--no-configure --no-control`.

Motor runs go under `run.data_root/runs/lab_motor`. `capture_commits.jsonl` records
each pair's capture ID, row/column, requested counters, actual ready/held
readbacks, camera source numbers and PNG paths. The same capture metadata is in
both camera sidecars and `dataset.db`'s `images.sync_json.capture`.
Database `sample_id = step_id + 1`; finalization uses committed capture IDs and
does not re-pair images by timestamp. `run_summary.json` records completion and
return-to-start results. Successful collection returns both axes to their live
starting counters. Cancellation, motor/camera errors or failed saves abort the
run, stop commanded motors when needed, retain the evidence, and skip database
publication. No automatic recovery motion follows a failed run.

The existing DMD-only analysis action is disabled in this profile. Capture-ID
database finalization still runs after a successful collection.

Preview and background acquisition accept either profile. To use the lab camera and PNG settings explicitly:

```bash
python experiments/camera_preview.py --config configs/experiment_lab_v1.json
python experiments/background_acquire.py --config configs/experiment_lab_v1.json
```

### Independent exposure probe

With the cameras already in the lab's Mono12/Line1 mode, manual exposure and
gain 0, run the exposure-only probe on the lab machine:

```bash
python experiments/exposure_probe.py --source laser
```

For a held DMD pattern, supply its **actual active grayscale period**:

```bash
python experiments/exposure_probe.py --source dmd --dmd-period-us 40000
```

The probe does not operate motors or the DMD. It prompts you to block the selected
source for a background ladder, unblock it and hold each scene, then block it
again for a final background ladder. Keep ambient lighting and optics unchanged.
Use `--scenes bright,edge,weak` to include several representative states; set each
state manually at its prompt. A single held scene cannot establish coverage of
the whole motor scan or every DMD pattern. Camera gamma and automatic exposure/
gain must already be disabled. Only exposure is written, then the original values
are restored and read back, including after an interruption or failed write.
Recommendations are recorded, never automatically applied to cameras/configs.

Defaults test five independent exposures per camera with three repeats at each
setting. Laser ladders are cam1 25/50/100/150/200 ms and cam2
0.25/0.5/1/1.5/2 ms. DMD ladders cover cam1 5/10/20/30/40 and cam2 2/5/10/15/24
whole display periods. `--cam1-us` and `--cam2-us` accept equal-length increasing
comma-separated ladders in microseconds, with at least three settings and at most
2 s exposure. The supplied DMD period is recorded as operator-provided, not a
device readback. Camera limits and applied settings are checked before capture.

Each camera is selected independently. A recommendation requires zero clipping
in every native pixel (including outside the saved crop), brightest-scene raw
peak 70–85% of 4095, and corrected signal p99.9 at least 20% of full scale in every
tested scene. Screening also requires contrast/repeat-background-noise >=10,
background drift and repeated-signal RMS <=10% of corrected contrast, and
background-subtracted signal/exposure agreement within 10% over at least three
usable exposures per scene. These are explicit screening heuristics, not
calibrated photon-SNR limits. If they fail, the report gives review flags and a
clearly provisional unsaturated candidate; it does not assert a valid setting.

The output directory (default `/tmp/clear26_exposure_<timestamp>`, or `--out`)
contains native-frame NPZs, one `probe.json` with settings/timing/provenance,
measurements and recommendations, and a compact `report.txt`. Keep that directory
together. Exposure scaling is `(raw - matching background) * 1000000 / exposure_us`
for **each** camera separately, producing DN/s at fixed gain. Raw frames are not
altered. This allows different fibre/reference exposure ratios; it does not
prove laser/DMD optical equivalence, linear optical superposition, or reconstruction
accuracy, and does not change downstream training preprocessing.

### After collection

The DMD lab config enables `postprocess.actions: ["data_analysis"]`. Install its
optional dependencies in the Python environment used for acquisition:

```bash
python -m pip install --no-build-isolation -e ".[analysis]"
```

After all planned points finish, files close, hardware cleanup completes and
the database and pattern associations are ready, the lab experiment starts analysis
as a separate Python process and waits for it. The analysis receives only the
saved run folder. It does not import CERN controls or access hardware.
Set `postprocess.actions` to `[]` to turn optional actions off; omitted means off.
Existing dataset finalization still runs independently of this setting.

The first supported action is `data_analysis`, for PNG DMD runs with saved
patterns, the DMD frame index, `stats` image metadata and explicit
`image_ready_ns` database pairing. Unsupported action/config combinations are
rejected before hardware access. Interrupted or unsuccessful collection skips
automatic analysis. A missing dependency or analysis error is reported separately
and does not change the saved images or the collection result.

Analysis writes two files under the run's `data_analysis/`:

- `results.npz`: one reusable numerical cache containing file hashes, validation
  results, per-image measurements, pairing times, pixel counts and maximum images.
- `report.pdf`: a one-page table of checks and core measurements, followed by two
  publication-sized figure pages. Fonts are embedded Arial; plot lines and text
  remain vector graphics, while camera images retain their measured pixels.

The PDF uses fixed calculations and explicit review rules.
File/record/pairing checks determine PASS or FAIL. Separate review flags identify
gaps longer than three times the median, identical repeated images, and flat
images. These flags do not establish dropped data or optical faults. Brightness
changes are reported as measurements because the DMD patterns intentionally vary.
Rows of the maximum-image figure are cameras; columns are linear/log color scales.

Arial must be installed in the analysis environment. If needed, point
`CLEAR2026_ARIAL_FONT` at a licensed local `Arial.ttf`; the code does not copy or
bundle the font and does not silently substitute another typeface.

The automatic launcher separately keeps `action.log` and `action_status.json`
for execution status, not a second copy of the measurements. A remaining
`running` status means completion was not recorded; check before retrying.

Run the same analysis manually, including on an older run:

```bash
python -m optidaq.analysis.analyze_run /path/to/run
```

Unchanged inputs reuse `results.npz` after source SHA256 verification, skipping
image decoding and database analysis. Changed inputs or computation code rebuild
the cache; plot-only edits reuse measurements. `--refresh` forces recomputation.
Images are processed one at a time, keeping a maximum image and a pixel histogram
per camera instead of holding the full image stack in memory. The PDF is always
generated from the same stored measurements.

`--out /temporary/output` optionally generates a review copy. Analysis never
modifies original files. It stages the PDF/cache and rechecks source hashes before
replacing them. It returns a nonzero exit code when dataset checks fail. Legacy
JSON/CSV/Markdown/PNG analysis files are no longer generated; unrelated existing
files are never automatically deleted. A manual run does not update the automatic
launcher's log/status. To add another action, register its module in
`src/optidaq/after_collection.py`; actions run sequentially in config order.

For raw runs from `experiment_v1`, export them offline instead. Replace the example run path with the folder printed by acquisition and choose an empty output directory:

```bash
python parse/export_scientific_png.py --config parse/export_config_full.json --run-dir /opt/data/CLEAR2026/runs/run_YYYYMMDDTHHMMSSZ --out-dir ./outputs/first-dataset
```

This export groups camera frames by timestamp, crops and bins them to 256-by-256 PNGs, and writes `dataset.db`. Images are stored in one folder per active camera (`cam1` and `cam2` with the current config). To preserve full-resolution pixels instead, set `crop_enabled` to `false` in the export config.

## Workflow

### Direct PNG acquisition

The lab runner uses PNG output, configured under `output.png` in
`configs/experiment_lab_v1.json`. Lab and background acquisition share the same
PNG save backend. Background acquisition follows the selected profile's output
format; `experiment_v1` remains raw-only. The lab runner rejects raw output
instead of silently changing storage modes.

- `transforms`: ordered image hooks, either a shared list or a mapping of camera
  aliases to lists. The lab config applies the per-camera crops below, then
  averages and rounds to 256×256 uint16. Empty means no image processing.
- `metadata_hooks`: the same configuration form, returning JSON-compatible
  dictionaries. The lab config enables `stats` and `centroid` on stored pixels,
  plus `crop_energy` on the original frame and its crop ROI.
- `compress_level`: 0–9; the lab config uses 1, matching the reference export.
  Compression preserves pixel values.
- `mode`: `raw16` preserves native values, including 12-bit values in a 16-bit
  PNG. Legacy `output.png_mode` still works when the nested mode is absent.
- `sync_alignment`: PNG database pairing source and tolerance. The lab config
  uses the scientific exporter's `image_ready_ns` matching with a 50 ms window.
  This is separate from acquisition's `sync` policy. Omitting this block keeps
  the original behavior of using that acquisition policy.
- `pairing`: the motor profile requires `"capture_id"`; its lab-only finalizer
  validates committed camera pairs instead of using `sync_alignment`.

The cam1/cam2 settings match those recorded in the reference dataset
`405_stimuation_large_256/dataset.db` (the current standalone export config has
a different cam2 crop):

| Camera | Crop `(x1, y1)–(x2, y2)` | Average bin | Stored image |
| --- | --- | --- | --- |
| cam1 | `(96, 96)–(1120, 1120)` | 4×4 | 256×256 uint16 |
| cam2 | `(335, 345)–(847, 857)` | 2×2 | 256×256 uint16 |

Crop coordinates are relative to the received camera image, with exclusive
upper bounds. Cropping discards pixels outside the box; binning averages and
rounds. The original full frames are not saved by this PNG profile.

Image hooks return a nonempty uint8/uint16 image with fixed dimensions per camera
run. Set `avg_bin`'s `dtype` to `uint16` to round its mean explicitly. Optional
`to_uint8` with `max_value: 4095` discards 12-bit precision. Custom hooks use
`{"name": "package.module:function", "params": {...}}` and the existing
`function(image, context, **params)` interface. Metadata hooks receive read-only
pixels. `crop_energy` supports one initial crop followed by bin/value transforms.

Each run contains `<camera>/*.png`, `<camera>.sync.jsonl`, a `<camera>.png.json`
source/stored-image descriptor and `resolved_config.json`. After the writers
close, acquisition builds `dataset.db` using the existing scientific export
schema, without reading or processing the pixels again. For DMD/camera-only
runs, the exporter alignment helper assigns `sample_id`, including incomplete camera groups.
Every saved image gets a row; missing timestamps or groups with duplicate frames
from one camera leave those rows' `sample_id` null, without invalidating good
groups elsewhere in the run. Effective processing and pairing settings are also
recorded in `export_config_json`. Existing DMD frame
associations are included when that index is enabled. Keep the run files together.
The parser uses stored dimensions and skips repeat image processing by default
for transformed PNG runs; `--process-png` explicitly enables further processing.
Use the delivered `dataset.db` for these sample IDs. `experiments/parse_process.py`
can regroup frames using its own sync settings when creating another dataset;
it does not currently reuse the PNG database's pairing. For motor rasters, use
the delivered database to preserve committed capture IDs and coordinate labels.

Hooks and compression run synchronously in the existing writer, so their cost
can fill its queue at high capture rates in independent capture modes. Motor
capture waits for durable saves before advancing. A failed/interrupted run retains
its PNGs and sidecars. After checking the capture error, DMD/camera-only databases
can be rebuilt with:

```python
from optidaq.data.scientific_png_export import finalize_png_dataset
finalize_png_dataset("/path/to/run")
```

Motor databases instead use `optidaq.data.lab_png_finalize.finalize_lab_png_dataset`.
That function rejects orphan images, missing commits and inconsistent coordinate
records; it does not repair or relabel a failed capture.

The quantitative quad-scan viewer requires empty PNG image transforms: binning
changes physical pixel calibration and cropping can change the measured beam
profile. It rejects processed PNGs instead of reporting misleading beam sizes.

`experiments/experiment_v1.py` connects the beam-time scan controllers to raw DAQ. `experiments/experiment_lab_v1.py` connects the lab controllers to PNG DAQ. DMD/camera-only modes reuse the existing camera pipeline; motor raster mode uses the separate lab capture coordinator and a lab adapter around the existing camera decoding. `parse/export_scientific_png.py` handles offline raw export.

![OptiDAQ workflow: control, acquisition, saved run data and offline processing](docs/assets/workflow.svg)

## DMD stream and timing

The lab config uses the bundled `dynamic_patterns` source in
`optidaq.extensions.sgm`: one Gaussian on a 256×256 canvas, with the configured
shape, position and amplitude ranges. Each source has its own NumPy `RandomState`,
preserving the original single-stream seeded sequence without changing global
random state or coupling separate streams. No XFlow or SciPy installation is needed.
`to_uint8.max_value=0.16` maps the upper amplitude to full DMD brightness using a
fixed scale, preserving relative brightness across patterns. It does not
normalize every pattern independently.

Display preparation stays separate from generation: upscale, fit to the native
DMD frame, then rotate 45° counterclockwise and flip vertically. This lab
orientation was checked with an asymmetric projected marker on the input camera.
Null `fit_center` height/width resolve from the device
during preflight; explicit sizes are still supported. Check the projected size,
rotation and intensity on the bench. The DMD picture time is 25 ms and camera
exposure is 50 ms; integer-cycle timing does not synchronize pattern changes to
camera exposures.

The controller shows a pattern, waits 0.1 s, then waits for ten additional saved
frames per camera before advancing; `dmd.count=10` requests 10 such steps.
Cameras capture independently throughout. There is no camera exposure gate,
hardware-trigger handshake or deliberate black interval. Queued older frames can
satisfy the saved-count wait, and transition/extra frames are retained. This is
not a guarantee of exactly N clean, pattern-labelled pairs.

A black interval alone would still allow partial-brightness and black camera
frames. Exposure-safe collection needs a trigger/exposure-complete handshake,
or selection using verified exposure timing and stable-pattern intervals. Current
DMD associations below remain explicitly approximate.

## DMD metadata

DMD scans append `actuator.events.jsonl` inside the normal configured run folder, alongside the camera files and `dmd_patterns/`. Each event records a schema version, step ID, host Unix timestamp in nanoseconds, and source metadata. `change_begin` precedes applying a pattern; `settle_complete` records when the software settling wait finished; `end` precedes halting the DMD. Settled events include the source information and, when pattern saving is enabled, a relative PNG path. Those PNGs are the source patterns before `display_transforms`, which remain recorded in `resolved_config.json`. Events are written as they happen, including during runs that later fail.

With `postprocess.build_aligned_index` enabled, a completed DMD run also creates `actuator_frame_index.jsonl`. Each row identifies an existing camera frame by `device` and `frame_index` (the raw record index or PNG filename index) and gives its DMD `step_id`, or `null` when unmatched. The reader compares camera `local_receive_ns` with closed intervals from `settle_complete` to the next `change_begin` or `end`. Transition periods, missing receive times and unfinished final intervals remain unmatched. A backward event clock is an error; a backward receive clock leaves that camera unmatched.

Every association is explicitly `method: host_receive_time`, `quality: approximate`: arrival time does not prove exposure time or synchronize the devices. Keep the event log and camera sidecars together. The offline reader also supports older raw DMD datasets using the same association format. Rebuild the index offline, including for an interrupted run with readable logs:

```bash
python -m optidaq.daq.actuator_events /path/to/run_YYYYMMDDTHHMMSSZ
```
