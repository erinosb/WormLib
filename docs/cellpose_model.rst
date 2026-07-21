Segmentation Models (Cellpose)
==================

WormLib includes pre-trained deep learning Cellpose models for cell segmentation located in the ``models/`` directory.

Cellpose Segmentation Model
----------------------------

**Pre-trained model: ``ce-embryo/``**

Custom Cellpose model trained on brightfield images of 2- and 4-cell *C. elegans* embryo.


**What it does:**

- Segments individual cells in early-stage embryos from brightfield microscopy
- Uses diameter optimization (default: 250 pixels)
- Outputs mask labels (one integer per cell)
- Separates cells from background and each other

**Architecture:** Cellpose "cyto" model fine-tuned using embryo images

**Input:** Brightfield image (2D or max projection)

**Output:** Segmentation mask (same shape as input, integer labels per cell)

**To use:**

Enable in config:

.. code-block:: yaml

    pipeline:
      cell_segmentation: true

**Performance:**

- Typical accuracy: ~90% for 2-cell and 4-cell stages
- Works best for embryos with clear cell boundaries
- Fails if cells are heavily overlapped or out-of-focus

---

**Using Models in Code**

**Cell Segmentation:**

.. code-block:: python

    import wormlib
    
    # Segmentation happens automatically when enabled
    if run_cell_segmentation:
        print("Running cell segmentation...")
        image_cytosol = bf
        second_image_cytosol = image_nuclei  # DAPI channel
        
        masks_cytosol, masks_nuclei, _, _ = wormlib.segmentation(
            image_cytosol, 
            image_nuclei,  # nuclei image
            second_image_cytosol,
            output_directory=output_directory,
        )
        

---

