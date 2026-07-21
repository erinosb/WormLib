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

