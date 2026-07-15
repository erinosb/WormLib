Troubleshooting Output Issues
------------------------------
WormLib generates organized output organized by image name. Each analysis produces visualization PNGs, quantification CSVs, and binary segmentation masks. Outputs are saved flat (no subdirectories) in the image-specific output directory.


**No output files generated**

- Check pipeline flags in config (e.g., ``spot_detection: true``)
- Verify output_directory exists and is writable
- Check console output for error messages

**Blank or noisy visualizations**

- Verify channel_indices are correct
- Check image data (plot in Jupyter to inspect)
- Confirm PSF values are appropriate for your microscope
- Inspect ``*_threshold_{image_name}.png`` to check if threshold is too aggressive/conservative

**CSV files missing**

- Spot detection must run before quantification CSVs are generated
- Classifier must be enabled for ``label`` and ``confidence`` columns in per_region CSV
- Review pipeline flags in config

**Line scan files missing**

- Ensure ``line_scan_enabled`` is set to ``true`` in config
- Check that line ROI was properly defined for your embryo orientation

---

Next Steps
----------

- See :doc:`models` to understand pre-trained classifiers
- See :doc:`settings` to configure your analysis



