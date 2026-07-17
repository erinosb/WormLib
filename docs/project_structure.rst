---

## Project Structure

```text
WormLib/
├── src/
│   └── wormlib.py                # Main analysis engine and CLI entry point
├── examples/
│   ├── wormlib_example.py        # Pipeline example script
│   └── run-WormLib.sh            # SLURM batch script
├── config/
│   └── examples/                 # Copy-and-edit YAML pipeline configs
│       ├── two_rna_full.yml
│       ├── one_rna_full.yml
│       ├── dapi_one_rna_no_brightfield.yml
│       └── rna_only_spot_detection.yml
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
├── installation/                 # Conda environment files
│   ├── wormlib.yml               # Default CPU/macOS environment
│   └── wormlib_cuda.yml          # NVIDIA CUDA environment
├── .gitignore
├── requirements.txt              # Python dependencies
└── LICENSE                       # MIT License
```

---