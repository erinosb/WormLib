Project Structure
=================

.. code-block:: text

	WormLib/
	├── src/
	│   └── wormlib.py                # Main analysis engine and CLI entry point
	├── examples/
	│   ├── 1 - Single_cell_spot_detection.ipynb        # Pipeline example as Jupyter notebook
	│   └── run-WormLib.sh            # SLURM batch script to run the pipeline on HPC clusters
	├── models/                       # Trained ML classifiers
	│   ├── 2-cell_classification_RFmodel.joblib
	│   ├── 4-cell_classification_RFmodel.joblib
	│   └── ce-embryo                 # Cellpose pretrained embryo model
	├── data/                         # Sample microscopy images
	│   ├── 04_dv/                    # DeltaVision samples
	│   └── 1886_nd2/                 # Nikon ND2 samples
	├── docs/                         # Documentation and assets
	│   ├── WormLib_logo.png
	│   └── READTHEDOCS.rst
	├── installation/                 # Conda environment files
	│   ├── wormlib.yml               # Default CPU/macOS environment
	│   └── wormlib_cuda.yml          # NVIDIA CUDA environment
	├── .gitignore
	├── requirements.txt              # Python dependencies
	└── LICENSE                       # MIT License



    