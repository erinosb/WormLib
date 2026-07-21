Troubleshooting
------------------------------
We recommend running the example notebook to verify that WormLib is installed and working correctly. 
If you encounter issues, please open an issue on GitHub with details of your problem, including error messages and a description of your data.

Most common issues:

**No output files generated**

- Check the conda environment is activated and all dependencies are installed
- Check that the pipeline ran successfully (no errors in console)
- Check pipeline flags in input (e.g., ``spot_detection: true``)
- Verify output_directory exists and is writable
- Check console output for error messages


**Inaccurate spot detection**

- Check you are using a non-deconvolved image for spot detection
- Check image data (plot in Jupyter/VS Code to inspect)
- Verify channel_indices are correct
- Confirm PSF values are appropriate for your microscope

**Inaccurate segmentation**

- Check you are using the correct model for your data (ce-embryo vs. cyto)
- Verify channel_indices are correct
- Adjust segmentation parameters (e.g., threshold, min_size, max_size) to better fit your data
- Adjust cell/embryo diameter parameters


**Inaccurate classification**

- Inspect segmentation masks for obvious errors