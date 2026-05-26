# WormLib <img src="docs/WormLib_logo.png" alt="WormLib Logo" width="150" align="right" />

**Authors:** Erin Osborne Nishimura and contributors

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/) [![BigFISH](https://img.shields.io/badge/smFISH-BigFISH-orange.svg)](https://github.com/fish-quant/big-fish) [![Cellpose](https://img.shields.io/badge/segmentation-Cellpose-green.svg)](https://github.com/MouseLand/cellpose)

## About

**WormLib** is a modular open-source image analysis library for quantifying *Caenorhabditis elegans* microscopy images. It provides an end-to-end pipeline from image loading through embryo segmentation, cell identity prediction, single-molecule FISH (smFISH) spot detection, and spatial mRNA analysis.

<p align="center">
  <img src="docs/WormLib_logo.png" alt="WormLib" width="300" />
</p>

---

## Features

- **Image I/O**: Load DeltaVision (.dv), Nikon (.nd2), and TIFF images with multi-channel extraction
- **Embryo segmentation**: Cellpose-based whole-embryo detection with size-outlier filtering
- **Cell segmentation**: Automated nucleus-cytosol pairing with diameter optimization (≤4-cell embryos)
- **Cell classification**: Random Forest classifiers for blastomere identity prediction
  - **2-cell stage**: AB vs P1 with proximity fail-safe
  - **4-cell stage**: ABa, ABp, EMS, P2 with ellipse-based positional assignment
- **smFISH spot detection**: BigFISH pipeline with LoG filtering, automated thresholding, and dense region decomposition
- **Cluster detection**: Identify transcription sites and mRNA clusters
- **Per-cell quantification**: Spot counting per segmented cell with classifier labels
- **Spatial mRNA analysis**:
  - Grid-based mRNA abundance heatmaps
  - RNA density profiles along the anterior-posterior (AP) axis
  - Line scan intensity analysis with ROI restriction
- **PDF report generation**: Automated reports with figures, tables, and analysis logs
- **HPC batch processing**: SLURM array job support for high-throughput analysis

---

## Installation

### Quick Install

```bash
# Create and activate conda environment
conda create -n wormlib python=3.10 -y
conda activate wormlib

# Clone the repository
git clone https://github.com/erinosb/WormLib.git
cd WormLib

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Recommended)

For GPU-accelerated Cellpose segmentation:

```bash
# NVIDIA GPU (Linux/Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Apple Silicon (macOS) — MPS support is automatic with:
pip install torch torchvision
```

### Verify Installation

```python
import bigfish
import cellpose
from scipy.ndimage import label
from skimage import measure
print("WormLib dependencies OK")
```

---

## Usage

### Pipeline Example Script

```bash
cd examples
python wormlib_example.py
```

Configure parameters in the script (Section 1):

```python
# Define image path and microscope parameters
image_path = main_dir / "data/08_dv/230521_N2_08_R3D.dv"
voxel_size = (1448, 450, 450)        # Z, Y, X in nm
spot_radius_ch0 = (1409, 340, 340)   # PSF for Cy5 channel
spot_radius_ch1 = (1283, 310, 310)   # PSF for mCherry channel

# Channel assignments
Cy5 = "mRNA1"
mCherry = "mRNA2"
DAPI = "DAPI"
brightfield = "brightfield"

# Enable pipeline steps
run_cell_segmentation = True
run_cell_classifier = True
run_spot_detection = True
run_mRNA_heatmaps = True
run_rna_density_analysis = True
run_line_scan_analysis = True
```

### HPC Batch Processing (SLURM)

```bash
# Submit array job for all images in an experiment
sbatch --array=0-N examples/run-WormLib.sh /path/to/experiment

# The script automatically:
# 1. Processes each image subdirectory
# 2. Saves timestamped script snapshots
# 3. Combines per-image CSVs into aggregate outputs
```

### Command-Line Execution

```bash
cd examples

# Set environment variables (see run-WormLib.sh for full list)
export FOLDER_NAME="/path/to/input"
export OUTPUT_DIRECTORY="/path/to/output"
export DV_IMAGES="True"

python ../src/wormlib.py
```

---

## Analysis Pipeline

```text
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Image I/O   │───▶│  Segmentation    │───▶│  Classification │
│  DV/ND2/TIFF │    │  Embryo + Cells  │    │  AB/P1 or 4-cell│
└──────────────┘    └──────────────────┘    └────────┬────────┘
                                                     │
                    ┌──────────────────┐              │
                    │  Spot Detection  │◀─────────────┘
                    │  smFISH (BigFISH)│
                    └────────┬─────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌─────────────┐ ┌────────────┐ ┌────────────┐
     │  Heatmaps   │ │  Density   │ │  Line Scan │
     │  mRNA grid  │ │  AP axis   │ │  ROI-based │
     └──────┬──────┘ └─────┬──────┘ └─────┬──────┘
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                   ┌──────────────┐
                   │  PDF Report  │
                   │  + CSV Export│
                   └──────────────┘
```

---

## Project Structure

```text
WormLib/
├── src/
│   └── wormlib.py                # Main analysis engine
├── examples/
│   ├── wormlib_example.py        # Pipeline example script
│   └── run-WormLib.sh            # SLURM batch script
├── models/                       # Trained ML classifiers
│   ├── 2-cell_classification_RFmodel.joblib
│   ├── 4-cell_classification_RFmodel.joblib
│   └── ce-embryo                 # Cellpose pretrained embryo model
├── data/                         # Sample microscopy images
│   ├── 04_dv/                    # DeltaVision samples
│   ├── 05_dv/
│   ├── 08_dv/
│   └── 1886_nd2/                 # Nikon ND2 samples
├── docs/                         # Documentation and assets
│   └── WormLib_logo.png
├── .gitignore
├── requirements.txt              # Python dependencies
└── LICENSE                       # MIT License
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| [BigFISH](https://github.com/fish-quant/big-fish) | smFISH spot detection & analysis |
| [Cellpose](https://github.com/MouseLand/cellpose) | Deep learning cell segmentation |
| [scikit-image](https://scikit-image.org/) | Image processing & morphology |
| [scikit-learn](https://scikit-learn.org/) | Random Forest classifiers |
| [PyTorch](https://pytorch.org/) | GPU backend for Cellpose |
| [OpenCV](https://opencv.org/) | Contour & ellipse fitting |
| [nd2](https://github.com/tlambert03/nd2) | Nikon ND2 file reader |
| [tifffile](https://github.com/cgohlke/tifffile) | TIFF file I/O |
| [ReportLab](https://www.reportlab.com/) | PDF report generation |

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use WormLib in your research, please cite:

> **Nishimura EO et al.** *WormLib: A Modular Image Analysis Library for Quantifying C. elegans Microscopy.* (In preparation)

---

## Support

- **Issues & Contributions**: [GitHub](https://github.com/erinosb/WormLib/issues)
- **Repository**: [github.com/erinosb/WormLib](https://github.com/erinosb/WormLib)
