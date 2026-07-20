Dictionary
==============================


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


