Building your pipeline
======================

Example pipelines
-----------------

WormLib includes example notebooks in ``examples/``:

- ``1 - Single-cell spot detection.ipynb`` - Two RNA channels with cell classification

**Run Your Own Image Locally**

Copy one of the example notebooks, edit the paths and
channel indices, then run it.


What Is Required?
------------------

Different parts of the pipeline need different channels:

.. list-table::
    :header-rows: 1

    * - Goal
      - Required channels
      - Optional channels
      - Notes
    * - Load and inspect channels
      - at least one supported image file
      - any channel can be skipped
      - Produces a channel overview for loaded channels.
    * - Whole-embryo segmentation
      - ``brightfield`` and ``DAPI``
      - RNA channels
      - The current embryo segmentation function uses brightfield for the embryo boundary and DAPI for nuclei.
    * - Cell segmentation and blastomere classification
      - ``brightfield`` and ``DAPI``
      - RNA channels
      - Classification only runs when segmentation detects the expected 2-cell or 4-cell mask count.
    * - Spot detection in segmented embryo/cells
      - at least one RNA channel plus a segmentation mask
      - additional RNA channels
      - Best for total or per-cell/region molecule counts.
    * - Spot detection only
      - at least one RNA channel
      - WormLib creates a whole-image mask, all spots in the image will be counted, in or outside the embryo.

For true whole-embryo segmentation, **brightfield/reference and DAPI are both
necessary** in the current pipeline. If either one is missing, WormLib models may not be able to produce a true embryo/cell mask.

4. Run the example notebooks
----------------------------

Select the example notebook that matches your analysis needs. The chunks in a notebook are designed to be run in order, but you can also copy a notebook and edit it to match your own data.:

Open ``examples/1-single-cell-spot-detection.ipynb`` in JupyterLab or Jupyter Notebook. 
In Section 1, specify the paths, channel indices and microscope parameters to match your own data, then run the cells in order.

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



    YAML 
    ~~~~~~~~~~~~~~~~~


.. list-table::
    :header-rows: 1

    * - Available channels
      - Example config
      - What happens
    * - brightfield + DAPI + two RNAs
      - ``config/examples/1-single-cell-spot-detection.ipynb``
      - Full standard pipeline.

For a DAPI + single-RNA DeltaVision stack where RNA is channel 0 and DAPI is
channel 3, the config looks like:

.. code-block:: yaml

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

In this mode, the mask is nuclear rather than whole-cell or whole-embryo. That
is appropriate for nuclear spot counts, but it is not a substitute for
cytoplasmic or whole-embryo RNA quantification.

Choosing chanels 
~~~~~~~~~~~~~~~~~

WormLib uses semantic channel roles. A YAML config file tells WormLib which
image channel is nuclei, which is brightfield/reference, and which are RNA
channels.

.. list-table::
    :header-rows: 1

    * - Setting
      - Meaning
      - Example
    * - ``channels.rna[].name``
      - The RNA label used in output files and plots
      - ``par-3``
    * - ``channels.rna[].fluorophore``
      - The physical fluorophore or microscope channel name, for human readability
      - ``Alexa647``
    * - ``channels.rna[].index``
      - The zero-based position of that RNA in the image stack
      - ``0``
    * - ``channels.rna[].spot_radius_nm``
      - PSF radius in nm as ``[Z, Y, X]``
      - ``[1409, 340, 340]``
    * - ``channels.rna[].detection_color``
      - Color used for spot detection overlays
      - ``red``
    * - ``channels.nuclei.index``
      - The zero-based channel position used for nuclei segmentation
      - ``3``
    * - ``channels.brightfield.index``
      - The zero-based channel position used for brightfield/reference, if it is inside the stack
      - ``4``

The RNA channel names are fully user-defined. They do not need to be ``Cy5``
or ``mCherry``. For example, if your only RNA channel is physically Alexa 647,
name it after the target gene and record the fluorophore separately:

.. code-block:: yaml

    channels:
      rna:
        - name: par-3
          fluorophore: Alexa647
          index: 0
          spot_radius_nm: [1409, 340, 340]
          detection_color: red

To skip a channel, omit it from the config or set its ``name`` to ``null``:

.. code-block:: yaml

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

If a channel exists but is in a different position than the bundled example,
change only the index:

.. code-block:: yaml

    channels:
      nuclei:
        name: DAPI
        index: 0
      rna:
        - name: my_gene
          index: 1
          spot_radius_nm: [1409, 340, 340]

The channel indices must match the order in the loaded image stack. 




Analysis Pipeline
-----------------

.. code-block:: text


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



Pipeline Selection
-----------------

Control which analysis steps run (true/false):

- **cell_segmentation** — Segment individual cells
- **cell_classification** — Predict cell identity (AB, ABa, EMS, etc.)
- **embryo_segmentation** — Segment whole embryo
- **spot_detection** — Detect individual mRNA spots
- **heatmaps** — Generate grid-based abundance heatmaps
- **rna_density** — Analyze RNA density along embryo axis
- **line_scan** — Generate line scan intensity plots

Disable steps to save computation time if not needed.



Segmentation fallback chain:

.. code-block:: text

    Cell segmentation (bf + DAPI)
      ├─ success → classifier → spot detection
      └─ fail ──▶ Whole-embryo segmentation (bf + DAPI)
                    ├─ success → spot detection (no per-cell labels)
                    └─ fail ──▶ Nuclear-only segmentation (DAPI only)
                                  ├─ success → spot detection (nuclear regions)
                                  └─ fail ──▶ Whole-image mask
                                                └─ spot detection (image-level counts)

