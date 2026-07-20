Dictionary
==============================

Image Data Dictionary
---------------------

.. list-table:: Image Data Dictionary
   :widths: 25 20 55
   :header-rows: 1

   * - Key
     - Value/Type
     - Description
   * - image_type
     - str
     - Image format type (e.g., 'DeltaVision')
   * - image_name
     - str
     - Image filename without extension (e.g., '230713_Lp306_L4440_11')
   * - bf
     - numpy array
     - Brightfield (transmission) 2D image (1024, 1024)
   * - image_Cy5
     - numpy array
     - Channel 0 max projection (Cy5 fluorophore)
   * - image_mCherry
     - numpy array
     - Channel 1 max projection (mCherry fluorophore)
   * - image_FITC
     - numpy array
     - Channel 2 max projection (FITC/GFP fluorophore)
   * - image_nuclei
     - numpy array
     - Channel 3 max projection (DAPI/nuclei stain)
   * - Cy5_array
     - numpy array
     - Channel 0 full 3D volume (Z, Y, X)
   * - mCherry_array
     - numpy array
     - Channel 1 full 3D volume (Z, Y, X)
   * - FITC_array
     - numpy array
     - Channel 2 full 3D volume (Z, Y, X)
   * - nuclei_array
     - numpy array
     - Channel 3 full 3D volume (Z, Y, X)
   * - grid_width
     - int
     - Grid width in pixels (80)
   * - grid_height
     - int
     - Grid height in pixels (80)


Input Terms
--------------

**path** — Directory containing microscopy files (.dv, .nd2, .tiff)

**output_directory** — Where analysis results will be saved


Microscope Terms
-------------------

**voxel_size_nm** — ``[Z, Y, X]`` voxel dimensions in nanometers

For *C. elegans* 4-cell embryos on DeltaVision: ``[1448, 450, 450]``


Channels Terms
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


Segmentation Terms
---------------------

**embryo_diameter** — Expected embryo diameter in pixels (for whole-embryo segmentation)

**nuclei_diameter** — Expected nucleus diameter in pixels (for nuclear segmentation)

These values help Cellpose optimize segmentation. Typical values:

- Embryo diameter: 375–500 pixels
- Nuclei diameter: 30–70 pixels

If cell segmentation or blastomere classification is not reliable for an image,
WormLib falls back to whole-embryo segmentation and skips per-cell labels. If
neither cell nor embryo segmentation succeeds, spot detection can still run
using a whole-image mask.

