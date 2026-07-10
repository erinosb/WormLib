Input Formats and File Organization
====================================

Supported Image Formats
------------------------

WormLib supports three microscopy file formats:

**DeltaVision (.dv)**
- Applied Precision DeltaVision microscope format
- Multi-channel volumetric images (Z-stacks)
- Native support for 4D data (X, Y, Z, time) and channel extraction
- Recommended for *C. elegans* embryo imaging

**Nikon ND2 (.nd2)**
- Nikon microscope format
- Multi-channel support
- Requires ``nd2`` package

**TIFF (.tif, .tiff)**
- Multi-page TIFF with channel stacking
- Must have explicit channel indexing in configuration


DeltaVision Naming Convention
------------------------------

DeltaVision files follow a specific naming pattern for reference vs. color images:

**Reference image (brightfield):**

.. code-block:: text

    230713_Lp306_L4440_11_R3D_REF.dv

Contains:
- 2D brightfield transmission image for segmentation
- Single channel per Z-slice

**Color image (fluorescence):**

.. code-block:: text

    230713_Lp306_L4440_11_R3D.dv

Contains:
- 4D volumetric fluorescence data
- 4 channels (Cy5, mCherry, FITC, DAPI) × Z-slices
- Each channel is a separate index (0, 1, 2, 3)

**Critical Pattern:**

.. code-block:: text

    _R3D_REF  → brightfield reference (2D)
    _R3D      → color/fluorescence (4D multi-channel)

Do NOT confuse these! The ``_R3D`` file must not include ``_REF`` in the name.

---

File Organization Best Practice
--------------------------------

Organize your data in a clear folder structure:

.. code-block:: text

    data/
    ├── 230713_Lp306_L4440_11/
    │   ├── 230713_Lp306_L4440_11_R3D_REF.dv
    │   └── 230713_Lp306_L4440_11_R3D.dv
    ├── 230713_Lp306_L4440_12/
    │   ├── 230713_Lp306_L4440_12_R3D_REF.dv
    │   └── 230713_Lp306_L4440_12_R3D.dv
    └── 230713_Lp306_L4440_13/
        ├── 230713_Lp306_L4440_13_R3D_REF.dv
        └── 230713_Lp306_L4440_13_R3D.dv

**Benefits:**
- Both reference and color files in same folder
- WormLib auto-detects and loads both
- Easy batch processing with wildcards
- Output can mirror input structure

---

Channel Indexing
-----------------

**For DeltaVision files:**

Standard *C. elegans* 4-cell embryo imaging:

- Channel 0 — Cy5 (set3_mRNA, PSF: 1409/340/340 nm)
- Channel 1 — mCherry (erm1_mRNA, PSF: 1283/310/310 nm)
- Channel 2 — GFP (membrane/protein)
- Channel 3 — DAPI (nuclei)

Specify these in your YAML config:

.. code-block:: yaml

    channels:
      nuclei:
        index: 3          # DAPI
      rna:
        - name: set3_mRNA
          fluorophore: Cy5
          index: 0
          spot_radius_nm: [1409, 340, 340]
        - name: erm1_mRNA
          fluorophore: mCherry
          index: 1
          spot_radius_nm: [1283, 310, 310]

---

Loading Images in Code
------------------------

**Automatic DeltaVision detection:**

.. code-block:: python

    import wormlib
    from pathlib import Path
    
    # Pass path to ANY file in the folder, or the color file
    image_path = Path("data/230713_Lp306_L4440_11/230713_Lp306_L4440_11_R3D.dv")
    
    result = wormlib.load_images(
        image_path=str(image_path),
        output_directory="output/",
        channel_names={
            'Cy5': 'set3_mRNA',
            'mCherry': 'erm1_mRNA',
            'FITC': 'membrane',
            'DAPI': 'DAPI',
            'brightfield': 'brightfield'
        },
        channel_indices={
            'Cy5': 0,
            'mCherry': 1,
            'FITC': 2,
            'DAPI': 3,
            'brightfield': None
        }
    )

WormLib automatically:
1. Detects ``.dv`` extension
2. Finds both ``_R3D_REF`` and ``_R3D`` files in the same folder
3. Loads brightfield from the reference file
4. Loads channels from the color file
5. Returns organized image data

**Result structure:**

.. code-block:: python

    {
        'image_type': 'DeltaVision',
        'image_name': '230713_Lp306_L4440_11',
        'bf': array(...),              # brightfield (1024, 1024)
        'image_Cy5': array(...),       # Channel 0 max projection
        'image_mCherry': array(...),   # Channel 1 max projection
        'image_FITC': array(...),      # Channel 2 max projection
        'image_nuclei': array(...),    # Channel 3 max projection
        'Cy5_array': array(...),       # Channel 0 full 3D (Z, Y, X)
        'mCherry_array': array(...),   # Channel 1 full 3D
        'FITC_array': array(...),      # Channel 2 full 3D
        'nuclei_array': array(...),    # Channel 3 full 3D
        'grid_width': 80,
        'grid_height': 80
    }

---

Next Steps
----------

- Configure your analysis in :doc:`settings`
- Run analysis and check :doc:`outputs`
