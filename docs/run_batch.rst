### HPC Batch Processing (SLURM)

```bash
# Submit one array task per input subfolder.
# Example: if input/ has 20 embryo folders, use --array=0-19.
sbatch --array=0-N examples/run-WormLib.sh /path/to/my_experiment

# The script automatically:
# 1. Processes each image subdirectory
# 2. Saves timestamped script snapshots
# 3. Combines per-image CSVs into aggregate outputs
```

Before submitting, edit `examples/run-WormLib.sh` to match your microscope
calibration, channel names, channel indices, and desired pipeline switches.
The SLURM script uses the legacy environment-variable interface.

