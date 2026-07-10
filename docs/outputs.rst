Output Files and Results Structure
===================================

WormLib generates organized output organized by image name. Each analysis produces visualization PNGs, quantification CSVs, and a compiled PDF report.

---

Output Directory Structure
----------------------------

After running analysis on ``230713_Lp306_L4440_11``:

.. code-block:: text

    output/230713_Lp306_L4440_11/
    ├── Segmentation_Masks/
    │   ├── cytosol_mask.png
    │   ├── nuclei_mask.png
    │   └── nuclei_outlines.png
    ├── Spot_Detection/
    │   ├── set3_mRNA_detection_230713_Lp306_L4440_11.png
    │   └── erm1_mRNA_detection_230713_Lp306_L4440_11.png
    ├── Heatmaps/
    │   ├── set3_mRNA_heatmap_230713_Lp306_L4440_11.png
    │   └── erm1_mRNA_heatmap_230713_Lp306_L4440_11.png
    ├── Line_Scans/
    │   ├── set3_mRNA_line_scan_230713_Lp306_L4440_11.png
    │   └── erm1_mRNA_line_scan_230713_Lp306_L4440_11.png
    ├── per_cell_mRNA_counts_230713_Lp306_L4440_11.csv
    ├── total_mRNA_counts_230713_Lp306_L4440_11.csv
    ├── Classification_Report_230713_Lp306_L4440_11.csv
    └── 230713_Lp306_L4440_11_report.pdf


Visualization Outputs (PNG)
----------------------------

**Segmentation Masks**

- ``cytosol_mask.png`` — Individual cell outlines (one per segmented cell)
- ``nuclei_mask.png`` — Nucleus boundaries
- ``nuclei_outlines.png`` — Nuclear outlines overlaid on brightfield

Used to verify segmentation quality before analysis.

**Spot Detection**

- ``{channel_name}_detection_{image_name}.png`` — Red/blue points marking detected spots

Shows spot locations in max projection view. One file per RNA channel.

**Heatmaps**

- ``{channel_name}_heatmap_{image_name}.png`` — Side-by-side figure

Left panel: Max projection image with spot density overlay
Right panel: 80×80 grid heatmap showing spot abundance per cell

Quantifies spatial distribution of mRNA across the embryo.

**Line Scans**

- ``{channel_name}_line_scan_{image_name}.png`` — 1D intensity profile along embryo axis

Shows RNA intensity variation along anterior-posterior axis with embryo cell labels (if classifier enabled).


Quantification Outputs (CSV)
-----------------------------

**Wide Format: ``total_mRNA_counts_{image_name}.csv``**

Summary counts per image:

.. code-block:: text

    Image ID,set3_mRNA total molecules,erm1_mRNA total molecules
    230713_Lp306_L4440_11,547,302

Useful for quick statistics across many images.

**Long Format: ``per_cell_mRNA_counts_{image_name}.csv``**

Per-cell spot counts:

.. code-block:: text

    Image ID,region_id,set3_mRNA,erm1_mRNA,label,confidence
    230713_Lp306_L4440_11,1,125,89,AB,0.987
    230713_Lp306_L4440_11,2,98,72,P1,0.954
    230713_Lp306_L4440_11,3,156,104,ABa,0.923
    230713_Lp306_L4440_11,4,168,37,EMS,0.891

Columns:

- ``Image ID`` — Image filename
- ``region_id`` — Cell number (1, 2, 3, ...)
- ``{channel_name}`` — Spot count per channel
- ``label`` — Predicted cell identity (if classifier enabled)
- ``confidence`` — Prediction confidence (0.0–1.0)

Use this for statistical analysis, correlation tests, etc.

**Classification Report: ``Classification_Report_{image_name}.csv``**

Cell identity predictions with feature values:

.. code-block:: text

    Cell_ID,label,prediction_confidence,centroid_x,centroid_y,area,eccentricity
    1,AB,0.987,512.5,256.3,18500,0.45
    2,P1,0.954,520.1,389.2,22100,0.52
    ...

Features used by the Random Forest classifier:
- Centroid position (X, Y)
- Cell area (pixels²)
- Eccentricity (elongation)
- Other morphological features


PDF Report
----------

**``{image_name}_report.pdf``**

Compiled analysis report containing:

- Segmentation visualization (masks overlay)
- Spot detection for each channel
- Heatmaps (left: image, right: grid)
- Line scan plots
- Summary statistics (total spots, per-cell distributions)
- Classification results table (if enabled)
- Processing parameters (voxel size, PSF, segmentation settings)

Automatically generated at the end of analysis. Suitable for sharing with collaborators.

---

Interpreting Results
---------------------

**High-quality segmentation**

- Clear cell boundaries in segmentation mask
- No merged cells or fragments
- Expected count: 2-cell or 4-cell embryos only

**Robust spot detection**

- Red/blue dots in spot detection PNG match obvious fluorescent puncta
- Not too noisy (false positives from background)
- Not too conservative (missing real signal)
- Adjust ``spot_radius_nm`` in config if results look off

**Meaningful heatmaps**

- Grid heatmap shows clear spatial patterns (not uniform)
- High-abundance regions correspond to visible spots in image
- Compare between channels to identify differential localization

**Cell classification accuracy**

- Check ``confidence`` column in per_cell CSV
- Confidence > 0.9 is generally reliable
- Lower confidence suggests ambiguous cell identity or segmentation artifact

---

Troubleshooting Output Issues
------------------------------

**No output files generated**

- Check pipeline flags in config (e.g., ``spot_detection: true``)
- Verify output_directory exists and is writable
- Check console output for error messages

**Blank or noisy visualizations**

- Verify channel_indices are correct
- Check image data (plot in Jupyter to inspect)
- Confirm PSF values are appropriate for your microscope

**CSV files missing**

- Spot detection must run before quantification CSVs are generated
- Classifier must be enabled for ``label`` and ``confidence`` columns
- Review pipeline flags

---

Next Steps
----------

- See :doc:`models` to understand pre-trained classifiers
- See :doc:`settings` to configure your analysis
