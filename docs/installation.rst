Installation
=============

Quick Install with Conda
------------------------

WormLib requires Python 3.11+ and uses conda for dependency management. Follow these steps to get started:

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

WormLib relies on several key scientific Python packages:

- **big-fish** — Single-molecule FISH spot detection with LoG filtering and automated thresholding
- **cellpose** — Deep learning-based cell and embryo segmentation
- **nd2** — Nikon microscopy file format support
- **scikit-image** — Image processing utilities
- **opencv-python-headless** — Image manipulation without GUI
- **reportlab** — PDF report generation
- **PyYAML** — Configuration file parsing

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

If CUDA installation fails or torch doesn't detect your GPU:

.. code-block:: bash

    # Verify CUDA availability
    python -c "import torch; print(torch.cuda.is_available())"
    
    # If False, use CPU version instead
    conda deactivate
    conda env remove -n wormlib
    conda env create -f installation/wormlib.yml

**Missing data or models**

The repository includes pre-trained classifiers in the ``models/`` directory. If files are missing:

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
