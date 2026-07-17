Usage
=====

## Building Your Own Pipeline


#### What Is Required?

Different parts of the pipeline need different channels:

| Goal | Required channels | Optional channels | Notes |
|------|-------------------|-------------------|-------|
| Load and inspect channels | at least one supported image file | any channel can be skipped | Produces a channel overview for loaded channels. |
| Whole-embryo segmentation | `brightfield` and `DAPI` | RNA channels | The current embryo segmentation function uses brightfield for the embryo boundary and DAPI for nuclei. |
| Cell segmentation and blastomere classification | `brightfield` and `DAPI` | RNA channels | Classification only runs when segmentation detects the expected 2-cell or 4-cell mask count. |
| Spot detection in segmented embryo/cells | at least one RNA channel plus a segmentation mask | additional RNA channels | Best for total or per-cell/region molecule counts. |
| Spot detection with DAPI only, no brightfield | one RNA channel and `DAPI` | additional RNA channels | WormLib uses nuclear masks, so counts are nuclear-region counts, not whole-cell counts. |
| Spot detection with RNA only | one RNA channel | all segmentation channels | WormLib creates a whole-image mask, so only whole-image counts are meaningful. |

For true whole-embryo segmentation, **brightfield/reference and DAPI are both
necessary** in the current pipeline. If either one is missing, WormLib can still
run limited spot-detection workflows, but it cannot produce a true embryo/cell
mask.




### 4. Minimal Channel Workflows

WormLib can run with missing optional channels:

| Available channels | Example config | What happens |
|--------------------|----------------|--------------|
| brightfield + DAPI + two RNAs | `config/examples/two_rna_full.yml` | Full standard pipeline. |
| brightfield + DAPI + one RNA | `config/examples/one_rna_full.yml` | Segmentation and one-channel spot detection run; CSVs include only the available RNA. |
| brightfield + DAPI, no RNA | remove all `channels.rna` entries and set `spot_detection: false` | Segmentation/reporting can run, but no molecule counts are generated. |
| DAPI + one RNA, no brightfield | `config/examples/dapi_one_rna_no_brightfield.yml` | Nuclear-only segmentation and one-channel spot detection run. |
| one RNA only | `config/examples/rna_only_spot_detection.yml` | Spot detection runs with one whole-image mask; use only whole-image counts. |
| brightfield + one RNA, no DAPI | set `channels.nuclei.name: null` | True embryo/cell segmentation will not run; spot detection falls back to a whole-image mask. |

For a DAPI + single-RNA DeltaVision stack where RNA is channel 0 and DAPI is
channel 3, the config looks like:

```yaml
input:
  path: /path/to/embryo_001_R3D.dv
  output_directory: /path/to/results/embryo_001

channels:
  nuclei:
    name: DAPI
    index: 3
  brightfield:
    name: null
    index: null
  rna:
    - name: my_gene
      fluorophore: Alexa647
      index: 0
      spot_radius_nm: [1409, 340, 340]

pipeline:
  cell_segmentation: true
  cell_classification: false
  spot_detection: true
  heatmaps: false
  rna_density: false
  line_scan: false
```

In this mode, the mask is nuclear rather than whole-cell or whole-embryo. That is
appropriate for nuclear spot counts, but it is not a substitute for cytoplasmic
or whole-embryo RNA quantification.

### 5. Configure the Example Script Instead

If you prefer editing a Python file, open `examples/wormlib_example.py`. The
CLI/config path above is recommended for most users, but the example script still
shows how to call library functions directly. In Section 1, set paths and
channels:

```python
# Define image path and microscope parameters
image_path = main_dir / "data/08_dv/230521_N2_08_R3D.dv"
image_ref = main_dir / "data/08_dv/230521_N2_08_R3D_REF.dv"
output_directory = current_dir / "output_temp"

voxel_size = (1448, 450, 450)        # Z, Y, X in nm
spot_radius_ch0 = (1409, 340, 340)   # PSF for Cy5 channel
spot_radius_ch1 = (1283, 310, 310)   # PSF for mCherry channel

# Channel assignments (set to None to skip a channel)
channel_names = {
    "Cy5": "mRNA1",
    "mCherry": "mRNA2",
    "FITC": None,
    "DAPI": "DAPI",
    "brightfield": "brightfield",
}

# Zero-based channel indices in the image stack
channel_indices = {
    "Cy5": 0,
    "mCherry": 1,
    "FITC": 2,
    "DAPI": 3,
    "brightfield": None,
}

# Enable pipeline steps
run_cell_segmentation = True
run_cell_classifier = True
run_spot_detection = True
run_mRNA_heatmaps = True
run_rna_density_analysis = True
run_line_scan_analysis = True
```

Then run:

```bash
python examples/wormlib_example.py
```




---

## Analysis Pipeline

```text
                     ┌──────────────┐
                     │  YAML Config │
                     │  or env vars │
                     └──────┬───────┘
                            ▼
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Image I/O   │───▶│  Segmentation    │───▶│  Classification │
│  DV/ND2/TIFF │    │  Cell / Embryo / │    │  AB/P1 or 4-cell│
└──────────────┘    │  Nuclear / None  │    └────────┬────────┘
                    └──────────────────┘             │
                    ┌──────────────────┐             │
                    │  Spot Detection  │◀────────────┘
                    │  smFISH (BigFISH)│
                    │  N RNA channels  │
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌─────────────┐ ┌────────────┐ ┌────────────┐
     │  Heatmaps   │ │  Density   │ │  Line Scan │
     │  mRNA grid  │ │  AP axis   │ │  ROI-based │
     └──────┬──────┘ └─────┬──────┘ └─────┬──────┘
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                   ┌──────────────┐
                   │  PDF Report  │
                   │  + CSV Export│
                   └──────────────┘
```

**Segmentation fallback chain:**

```text
Cell segmentation (bf + DAPI)
  ├─ success → classifier → spot detection
  └─ fail ──▶ Whole-embryo segmentation (bf + DAPI)
                ├─ success → spot detection (no per-cell labels)
                └─ fail ──▶ Nuclear-only segmentation (DAPI only)
                              ├─ success → spot detection (nuclear regions)
                              └─ fail ──▶ Whole-image mask
                                            └─ spot detection (image-level counts)
```
