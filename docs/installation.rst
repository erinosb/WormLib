Installation
=============

Quick Install with Conda
------------------------

Follow these steps to install WormLib using conda. This method is recommended for most users as it handles dependencies automatically.

**1. Clone the repository:**

.. code-block:: bash

    git clone https://github.com/erinosb/WormLib.git
    cd WormLib

**2. Create the conda environment:**

For CPU-based installation (recommended for most users):

.. code-block:: bash

    conda env create -f installation/wormlib.yml
    conda activate wormlib

For GPU acceleration (CUDA 11.8):

.. code-block:: bash

    conda env create -f installation/wormlib_cuda.yml
    conda activate wormlib

**3. Verify installation:**

.. code-block:: python

    import wormlib
    print(f"WormLib version: {wormlib.__version__}")

If the import succeeds, you're ready to go!

---

Core Dependencies
------------------

| Package | Version | Purpose |
|---------|---------|---------|
| [BigFISH](https://github.com/fish-quant/big-fish) | 0.6.2 | smFISH spot detection & analysis with LoG filtering and automated thresholding|
| [Cellpose](https://github.com/MouseLand/cellpose) | 3.1.0 | Deep learning-based cell and embryo segmentation |
| [scikit-image](https://scikit-image.org/) | 0.23.2 | Image processing & morphology |
| [scikit-learn](https://scikit-learn.org/) | Conda-managed | Random Forest classifiers (transitive via joblib) |
| [PyTorch](https://pytorch.org/) | 2.4.1 | GPU backend for Cellpose |
| [OpenCV](https://opencv.org/) | 4.10.0.84 | Image manipulation without GUI, Contour & ellipse fitting |
| [nd2](https://github.com/tlambert03/nd2) | 0.10.3 | Nikon ND2 file reader |
| [tifffile](https://github.com/cgohlke/tifffile) | 2025.6.11 | TIFF file I/O |
| [PyYAML](https://pyyaml.org/) | ≥ 6.0.1 | YAML configuration parsing |
| [ReportLab](https://www.reportlab.com/) | ≥ 4.0.8 | PDF report generation |
| [Pillow](https://python-pillow.org/) | ≥ 10.0 | Image handling for PDF reports |
| [Python](https://www.python.org/) | 3.11+ | Core programming language |

Full dependency list is maintained in ``requirements.txt`` and environment files.

---

Troubleshooting
----------------

**Environment creation fails**

If conda environment creation fails:

.. code-block:: bash

    # Clear conda cache
    conda clean --all
    
    # Try creating environment again
    conda env create -f installation/wormlib.yml --force-reinstall

**Import errors after activation**

Make sure you've activated the correct environment:

.. code-block:: bash

    conda activate wormlib
    python -c "import wormlib; print(wormlib.__version__)"

**GPU support not working**

If CUDA installation fails or torch doesn't detect your GPU, use CPU I installation instead.

.. code-block:: bash

    # Verify CUDA availability
    python -c "import torch; print(torch.cuda.is_available())"
    
    # If False, use CPU version instead
    conda deactivate
    conda env remove -n wormlib
    conda env create -f installation/wormlib.yml

**Missing data or models**
There are currently no example datasets included in the repository. Please download your own dv, nd2 or tiff files for analysis.
The repository includes pre-trained classifiers in the ``models/`` directory. Verify if files are missing:

.. code-block:: bash

    # Verify model files exist
    ls models/
    # Expected output:
    # 2-cell_classification_RFmodel.joblib
    # 4-cell_classification_RFmodel.joblib
    # ce-embryo/

---

Next Steps
----------

Once installed, explore the example notebook:

.. code-block:: bash

    cd examples
    jupyter notebook "1 - Single-cell spot detection.ipynb"

Or refer to :doc:`settings` to learn how to configure analysis pipelines with YAML.
