Building your pipeline
==============================

A single config file specifies input/output paths, microscope parameters, channel definitions, and which pipeline steps to run.

---
.. code-block:: python
    # ============================================================================
    # 1.1 Input Configuration
    # ============================================================================

    # Image path
    folder_name = main_dir / "data/230713_Lp306_L4440_11"
    image_ref = folder_name / "230713_Lp306_L4440_11_R3D_REF.dv"
    image_path = folder_name / "230713_Lp306_L4440_11_R3D.dv"

    # # Microscope parameters
    voxel_size = (1448, 450, 450)  # Z, Y, X in nm
    spot_radius_ch0 = (1409, 340, 340)  # PSF for channel 0 (Cy5)
    spot_radius_ch1 = (1283, 310, 310)  # PSF for channel 1 (mCherry)

    # Channel names (set to None if the channel does not exist)
    ch0 = "set3_mRNA"  # (Q670)
    ch1 = "erm1_mRNA"  # (Q610)
    ch2 = "membrane"  # (GFP)
    ch3 = "DAPI"
    brightfield = "brightfield"

    channel_names = {
            'Cy5': ch0,
            'mCherry': ch1,
            'FITC': ch2,
            'DAPI': ch3,
            'brightfield': None,
        }

    channel_indices = {
            'Cy5': 0,
            'mCherry': 1,
            'FITC': 2,
            'DAPI': 3,
            'brightfield': None,
        }

    # Pipeline flags
    run_cell_segmentation = True
    cell_diameter = 250
    nuclei_diameter = 30
    run_cell_classifier = True

    run_spot_detection = True
    run_mRNA_heatmaps = True
    normalize_heatmap_scale = True  # Set to False to keep raw per-heatmap counts
    run_line_scan_analysis = True

    print(f"Pipeline configuration loaded.")



Example pipelines
----------------

WormLib includes example notebooks in ``examples/``:

- ``1 - Single-cell spot detection.ipynb`` — Two RNA channels with cell classification



### 3. Run Your Own Image Locally

From the repository root, copy one of the example configs, edit the paths and
channel indices, then run it:

```bash
conda activate wormlib
cd /path/to/WormLib

cp config/examples/two_rna_full.yml config/my_experiment.yml
# edit config/my_experiment.yml for your image paths and channel positions
python src/wormlib.py --config config/my_experiment.yml
```

You can override the input and output paths from the command line while keeping
the rest of the config:

```bash
python src/wormlib.py --config config/my_experiment.yml \
  /path/to/my_experiment/input/embryo_001 \
  /path/to/results/embryo_001
```

WormLib automatically detects `.dv`, `.nd2`, `.tif`, or `.tiff` files in the
input path.

The old environment-variable interface still works for existing scripts, but
YAML configs are the recommended interface for new users.




Running an Analysis
-------------------

**1. Run the Bundled Example**

.. code-block:: bash

    python examples/1 - Single-cell spot detection.ipynb
The example analyzes the sample DeltaVision image in ``data/230713_Lp306_L4440_11/`` and writes
figures, CSV files, and ``report.pdf`` to ``output/``. This is the quickest
way to confirm that your Python environment, Cellpose, BigFISH, and model files
are working on your computer.

**2. Prepare Your Own Images**

For the full pipeline, WormLib expects microscopy data with:

- a reference/brightfield image for cell or embryo segmentation
- one or more smFISH RNA channels
- a nuclear channel, typically DAPI




#### Choosing Channels

WormLib uses semantic channel roles. A YAML config file tells WormLib which
image channel is nuclei, which is brightfield/reference, and which are RNA
channels.

| Setting | Meaning | Example |
|---------|---------|---------|
| `channels.rna[].name` | The RNA label used in output files and plots | `par-3` |
| `channels.rna[].fluorophore` | The physical fluorophore or microscope channel name, for human readability | `Alexa647` |
| `channels.rna[].index` | The zero-based position of that RNA in the image stack | `0` |
| `channels.rna[].spot_radius_nm` | PSF radius in nm as `[Z, Y, X]` | `[1409, 340, 340]` |
| `channels.rna[].detection_color` | Color used for spot detection overlays | `red` |
| `channels.nuclei.index` | The zero-based channel position used for nuclei segmentation | `3` |
| `channels.brightfield.index` | The zero-based channel position used for brightfield/reference, if it is inside the stack | `4` |

The RNA channel names are fully user-defined. They do not need to be `Cy5` or
`mCherry`. For example, if your only RNA channel is physically Alexa 647, name it
after the target gene and record the fluorophore separately:

```yaml
channels:
  rna:
    - name: par-3
      fluorophore: Alexa647
      index: 0
      spot_radius_nm: [1409, 340, 340]
      detection_color: red
```

To skip a channel, omit it from the config or set its `name` to `null`:

```yaml
channels:
  nuclei:
    name: DAPI
    index: 3
  brightfield:
    name: null
    index: null
  rna:
    - name: my_gene
      index: 0
      spot_radius_nm: [1409, 340, 340]
```

If a channel exists but is in a different position than the bundled example,
change only the index:

```yaml
channels:
  nuclei:
    name: DAPI
    index: 0
  rna:
    - name: my_gene
      index: 1
      spot_radius_nm: [1409, 340, 340]
```

The channel indices must match the order in the loaded image stack. DeltaVision
data are treated as `channel, z, y, x`; ND2 data are treated as `time/z, channel,
y, x`.



