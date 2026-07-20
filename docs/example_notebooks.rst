Example notebooks
======================

WormLib includes example notebooks in ``examples/``:

- ``1 - Single-cell spot detection.ipynb`` - Two RNA channels with cell classification

**Run Your Own Image Locally**

Copy one of the example notebooks, edit the paths and
channel indices, then run it.

Set up input in the example notebooks
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



