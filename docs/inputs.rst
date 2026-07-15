Input
====================================

Supported Image Formats
------------------------

File formats supported by WormLib:

**DeltaVision (.dv)**
**Nikon ND2 (.nd2)**
**TIFF (.tif, .tiff)**



DeltaVision files
------------------------
**Critical Pattern:**

.. code-block:: text


  **Reference image (brightfield):**
    _R3D_REF  → brightfield reference (2D)
  Contains:
- Single channel 2D brightfield
- used for embryo segmentation and cell identity prediction

  **Color image (fluorescence):**
   _R3D      → color/fluorescence (4D multi-channel)
  Contains:
- 4D (C,Z,Y,X) (C = channel, Z = z-slice, Y = height, X = width)
- usually query channel: used for smFISH spot detection and spatial mRNA analysis

Do NOT confuse these! The ``_R3D`` file must NOT include ``_D3D`` in the name and must be a different file from ``_R3D_REF``. 

WormLib automatically detects ``.dv`` extension and finds both ``_R3D_REF`` and ``_R3D`` files in the same folder. 
It can load brightfield from the reference file (R3D_REF.dv) and channels from the color file (R3D.dv) and returns organized image data.

---

Nikon files
------------------------
**Critical Pattern:**

.. code-block:: text


  **image_01.nd2:**

---

TIFF files
------------------------
**Critical Pattern:**

.. code-block:: text


  **image_01.tif:**


File Organization Best Practice
--------------------------------
Give your files informative names at the time of acquisition. Example: ``230713_Lp306_L4440_11_R3D_REF.dv`` = ``date_strain_condition_replicate/``.
Organize your data in a clear folder structure:

data>image_subdirectory>image_files

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
- Both reference and color files in same folder allows WormLib to auto-detect and load both
- Easy batch processing 
- Image output subdirectory automatically created in the same folder as input images
---



Loading Images in Code
------------------------

**Automatic file type detection.** 
DeltaVision, Nikon and TIFF files supported.

.. code-block:: python

    import wormlib

    # Path to image subdirectory
    image_path = Path("data/230713_Lp306_L4440_11")
    output_directory = Path("output/")
    
    result = wormlib.load_images(
        image_path=str(image_path),
        output_directory=str(output_directory),
        channel_names={
            'Cy5': 'set3_mRNA', # Describe what mRNA is in this channel
            'mCherry': 'erm1_mRNA', # Describe what mRNA is in this channel
            'FITC': 'membrane', # Describe what marker is in this channel
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


