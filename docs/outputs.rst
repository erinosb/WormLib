Outputs
=======

WormLib generates organized output organized by image name. Each analysis produces visualization PNGs, quantification CSVs, and binary segmentation masks. Outputs are saved flat (no subdirectories) in the image-specific output directory.

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

Output Files Reference
----------------------
All output is automatically saved in the image subdirectory after single cell spot detection analysis on ``230713_Lp306_L4440_11``:


.. list-table::
   :header-rows: 1
   :widths: 30 15 40

   * - Generic Pattern
     - Type
     - Description
   * - ``channels_{image_name}.png``
     - PNG
     - Raw channel images displayed side-by-side for visual inspection
   * - ``{channel_name}_detection_{image_name}.png``
     - PNG
     - Detected mRNA spots marked as red/blue points on max projection. One file per RNA channel.
   * - ``{channel_name}_threshold_{image_name}.png``
     - PNG
     - Visualization of threshold applied during spot detection. One file per RNA channel.
   * - ``{channel_name}_heatmap_{image_name}.png``
     - PNG
     - Side-by-side heatmap figure: (left) max projection with spot density overlay; (right) 80×80 grid showing spot abundance per region. One file per RNA channel.
   * - ``{channel_name}_line_scan_{image_name}.png``
     - PNG
     - 1D intensity profile along embryo anterior-posterior axis. One file per RNA channel.
   * - ``{channel_name}_line_ROI_{image_name}.png``
     - PNG
     - Visual representation of the line scan region of interest on the embryo. One file per RNA channel.
   * - ``masks_cytosol.tif``
     - TIF
     - Binary segmentation mask of all detected cells/regions (multi-channel format)
   * - ``centroid_position_plot_{image_name}.png``
     - PNG
     - Visualization of cell/region centroids overlaid on embryo image
   * - ``cell_confidence_plot_{image_name}.png``
     - PNG
     - Plot showing classification confidence scores for each predicted cell identity
   * - ``predicted_label_{image_name}.png``
     - PNG
     - Overlay of predicted cell identity labels on segmented embryo image
   * - ``total_mRNA_counts_{image_name}.csv``
     - CSV
     - Wide format summary: total mRNA spot counts per channel for the entire image (useful for bulk statistics)
   * - ``per_region_mRNA_counts_{image_name}.csv``
     - CSV
     - Long format: spot counts per region, including region_id, channel counts, predicted label, and confidence scores (for statistical analysis and per-cell quantification)
   * - ``features_df_{image_name}.csv``
     - CSV
     - Classification features used by Random Forest: centroid position (X, Y), cell area, eccentricity, and other morphological properties
   * - ``{channel_name}_line_density_data_{image_name}.csv``
     - CSV
     - Quantitative spot density values sampled along the line ROI. One file per RNA channel.
   * - ``{channel_name}_line_scan_data_{image_name}.csv``
     - CSV
     - Raw intensity values sampled along the line scan profile. One file per RNA channel.


CSV Data Format Examples
------------------------

**total_mRNA_counts_{image_name}.csv** — Quick summary of total counts:

.. code-block:: text

    Image ID,set3_mRNA total molecules,erm1_mRNA total molecules
    230713_Lp306_L4440_11,989,2848


**per_region_mRNA_counts_{image_name}.csv** — Per-region analysis data:

.. code-block:: text

    Image ID,region_id,set3_mRNA,erm1_mRNA,label,confidence
    230713_Lp306_L4440_11,1,125,89,ABa,0.987
    230713_Lp306_L4440_11,2,98,72,P2,0.954
    230713_Lp306_L4440_11,3,156,104,ABp,0.923
    230713_Lp306_L4440_11,4,168,37,EMS,0.891


**features_df_{image_name}.csv** — Morphological features for classification:

.. code-block:: text

    label,area,centroid_y,centroid_x,eccentricity,...
    1,19956.0,502.04,276.28,0.769,...
    2,31203.0,559.68,449.80,0.386,...
    3,18288.0,557.60,145.62,0.683,...
    4,20427.0,622.99,283.62,0.798,...



Interpreting Results
--------------------

**Robust spot detection**

- Threshold PNG shows thresholding applied. Verify here if appropriate.
- Not too noisy (false positives from background)
- Not too conservative (missing real signal)
- Adjust ``spot_radius_nm`` in your input configutation if results look off

Example PSF values for a DeltaVision microscope:

spot_radius_ch0 = (1409, 340, 340)  # PSF for channel 0 (Cy5)
spot_radius_ch1 = (1283, 310, 310)  # PSF for channel 1 (mCherry)

**Meaningful heatmaps**

- Grid heatmap shows clear spatial patterns (not uniform)
- High-abundance regions correspond to visible spots in image
- Compare between channels to identify differential localization

**Cell classification accuracy**

- We encourage all users to inspect the ``predicted_label`` overlay and ``cell_confidence_plot`` to verify that cell identities are assigned correctly. 
- Check ``confidence`` column in per_region CSV. Confidence > 0.9 is generally reliable.
- Lower confidence suggests ambiguous cell identity or segmentation artifact
- If the confidence is low, consider retraining the classifier with your own representative training data.
- Classification accuracy is directly impacted by the quality of segmentation.

**Segmentation quality**

- Check ``masks_cytosol.tif`` to verify cell boundaries
- Inspect ``centroid_position_plot`` to confirm region centroids
- If segmentation is poor, consider adjusting ``segmentation_threshold`` or ``min_region_size`` in your input configuration.
- Consider improving the ce-embryo single cell segmentation model by re-training on your own images.

6. Understand the Outputs
-------------------------

Each output directory can include:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Output
     - Description
   * - ``channels_*.png``
     - Loaded channel overview
   * - ``cell_segmentation_*.png`` or ``embryo_segmentation_*.png``
     - Segmentation result
   * - ``nuclear_segmentation_*.png``
     - Nuclear-only segmentation (when no brightfield is available)
   * - ``features_df_*.csv``
     - Cell morphology and classifier features
   * - ``total_mRNA_counts_*.csv``
     - Total molecule counts per RNA channel
   * - ``per_cell_mRNA_counts_*.csv``
     - Per-cell counts and predicted labels, when classification succeeds
   * - ``quantification_cell_*.csv``
     - Compatibility copy of per-cell counts for batch aggregation
   * - ``per_region_mRNA_counts_*.csv``
     - Region-level counts when classification is disabled or unavailable
   * - ``*_detection_*.png``, ``*_threshold_*.png``
     - BigFISH detection diagnostics
   * - ``*_AP_profile_data_*.csv``
     - RNA density along the AP axis
   * - ``*_line_scan_data_*.csv`` and ``*_line_density_data_*.csv``
     - ROI line-scan outputs
   * - ``report.pdf``
     - Summary report with figures and tables

If cell segmentation or blastomere classification is not reliable for an image,
WormLib falls back to whole-embryo segmentation and skips per-cell labels. If
neither cell nor embryo segmentation succeeds, spot detection can still run
using a whole-image mask.
