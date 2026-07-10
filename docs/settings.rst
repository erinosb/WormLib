Configuration (YAML Settings)
==============================

WormLib uses YAML configuration files to define analysis pipelines. A single config file specifies input/output paths, microscope parameters, channel definitions, and which pipeline steps to run.

---

Configuration Structure
------------------------

**Minimal Example:**

.. code-block:: yaml

    input:
      path: ./data/my_images/
      output_directory: ./output/

    microscope:
      voxel_size_nm: [1448, 450, 450]  # Z, Y, X in nanometers

    channels:
      nuclei:
        name: DAPI
        index: 3

    pipeline:
      cell_segmentation: true
      spot_detection: true


**Full Example:**

.. code-block:: yaml

    input:
      path: ../../data/my_data_folder
      output_directory: ../../output_temp/results

    microscope:
      voxel_size_nm: [1448, 450, 450]  # Z, Y, X

    channels:
      nuclei:
        name: DAPI
        index: 3
      brightfield:
        name: brightfield
        source: ref_file          # Load from _R3D_REF file
        index: null
      rna:
        - name: mRNA1
          fluorophore: Cy5
          index: 0
          spot_radius_nm: [1409, 340, 340]
          detection_color: red
        - name: mRNA2
          fluorophore: mCherry
          index: 1
          spot_radius_nm: [1283, 310, 310]
          detection_color: blue

    segmentation:
      embryo_diameter: 500
      nuclei_diameter: 70

    pipeline:
      cell_segmentation: true
      cell_classification: true
      embryo_segmentation: false
      spot_detection: true
      heatmaps: true
      rna_density: true
      line_scan: true


---

Input Section
--------------

**path** — Directory containing microscopy files (.dv, .nd2, .tiff)

**output_directory** — Where analysis results will be saved


Microscope Section
-------------------

**voxel_size_nm** — ``[Z, Y, X]`` voxel dimensions in nanometers

For *C. elegans* 4-cell embryos on DeltaVision: ``[1448, 450, 450]``


Channels Section
----------------

Define how to extract channels from your images.

**nuclei** — Nuclear stain (required for cell segmentation)

- ``name`` — Display name (e.g., "DAPI")
- ``index`` — Channel index in the image file (0, 1, 2, 3, ...)

**brightfield** — Optional brightfield/transmission image

- ``name`` — Display name
- ``source`` — ``"ref_file"`` to load from DeltaVision reference file (_R3D_REF)
- ``index`` — Set to ``null`` for brightfield

**rna** — One or more mRNA channels to detect

For each RNA channel:

- ``name`` — Gene name (e.g., "set3_mRNA")
- ``fluorophore`` — Dye label (e.g., "Cy5", "mCherry")
- ``index`` — Channel index
- ``spot_radius_nm`` — ``[Z, Y, X]`` PSF (Point Spread Function) in nm
- ``detection_color`` — Visualization color for detection plots ("red", "blue", etc.)


Segmentation Section
---------------------

**embryo_diameter** — Expected embryo diameter in pixels (for whole-embryo segmentation)

**nuclei_diameter** — Expected nucleus diameter in pixels (for nuclear segmentation)

These values help Cellpose optimize segmentation. Typical values:

- Embryo diameter: 375–500 pixels
- Nuclei diameter: 30–70 pixels


Pipeline Section
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


Example Configs
----------------

WormLib includes example configurations in ``config/examples/``:

- ``one_rna_full.yml`` — Single channel with all steps enabled
- ``two_rna_full.yml`` — Two RNA channels with cell classification
- ``rna_only_spot_detection.yml`` — Only spot detection, skip segmentation
- ``dapi_one_rna_no_brightfield.yml`` — Nuclear and mRNA, no brightfield


Using Configuration Files
---------------------------

**Via Jupyter notebook:**

.. code-block:: python

    import yaml
    with open('config/examples/two_rna_full.yml') as f:
        config = yaml.safe_load(f)
    
    # Access settings
    voxel_size = config['microscope']['voxel_size_nm']
    output_dir = config['input']['output_directory']

**Via command line (future):**

.. code-block:: bash

    python src/wormlib.py --config config/examples/two_rna_full.yml

---

Next Steps
----------

- See :doc:`inputs` to understand supported image formats
- See :doc:`outputs` to understand generated result files
