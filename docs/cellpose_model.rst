Segmentation Models (Cellpose)
==================

WormLib includes pre-trained machine learning models for cell segmentation and classification. These models are optimized for early-stage *C. elegans* embryo imaging and are located in the ``models/`` directory.

Cellpose Segmentation Model
----------------------------

**Pre-trained model: ``ce-embryo/``**

Custom Cellpose model trained on *C. elegans* embryo brightfield images.


**What it does:**

- Segments individual cells in early-stage embryos from brightfield microscopy
- Uses diameter optimization (default: 250 pixels)
- Outputs mask labels (one integer per cell)
- Separates cells from background and each other

**Architecture:** Cellpose "cyto" model fine-tuned using embryo images

**Input:** Brightfield image (2D or max projection)

**Output:** Segmentation mask (same shape as input, integer labels per cell)

**When to use:**

Enable in config:

.. code-block:: yaml

    pipeline:
      cell_segmentation: true

**Performance:**

- Typical accuracy: ~90% for 2-cell and 4-cell stages
- Works best for embryos with clear cell boundaries
- Fails if cells are heavily overlapped or out-of-focus

---

## Using Model in Code

**Cell Segmentation:**

.. code-block:: python

    import wormlib
    
    # Segmentation happens automatically when enabled
    masks_cytosol, masks_nuclei, _, _ = wormlib.segmentation(
        image_cytosol=brightfield_image,
        image_nuclei=dapi_image,
        second_image_cytosol=dapi_image,
        output_directory='output/'
    )

---

