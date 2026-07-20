Troubleshooting
------------------------------

**No output files generated**

- Check the conda environment is activated and all dependencies are installed
- Check that the pipeline ran successfully (no errors in console)
- Check pipeline flags in config (e.g., ``spot_detection: true``)
- Verify output_directory exists and is writable
- Check console output for error messages
- Verify channel_indices are correct
- Check image data (plot in Jupyter/VS Code to inspect)
- Confirm PSF values are appropriate for your microscope

We recommend running the example notebook to verify that WormLib is installed and working correctly. 
If you encounter issues, please open an issue on GitHub with details of your problem, including error messages and a description of your data.

