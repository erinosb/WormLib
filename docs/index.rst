.. WormLib documentation master file, created by
   sphinx-quickstart on Fri Jul 10 16:26:06 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

WormLib documentation
=====================

WormLib is a modular open-source image analysis library for quantifying microscopy images of Caenorhabditis elegans embryos. It provides an end-to-end pipeline from image loading, embryo segmentation, cell identity prediction, single-molecule FISH (smFISH) spot detection, and spatial mRNA analysis.

.. toctree::
   :maxdepth: 3
   :caption: BASICS:

   installation
   inputs
   outputs
   dictionary
   troubleshooting
  

.. toctree::
   :maxdepth: 3
   :caption: PRE-TRAINED MODELS:

   cellpose_model
   rf_models
   model_limitations
   training_models
   
.. toctree::
   :maxdepth: 3
   :caption: EXAMPLE NOTEBOOKS:

   example_notebooks
   run_batch

   
.. toctree::
   :maxdepth: 3
   :caption: CITATION:

   citation

   .. toctree::
   :maxdepth: 3
   :caption: PROJECT STRUCTURE:
   
   project_structure
