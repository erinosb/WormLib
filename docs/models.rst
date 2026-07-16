Pre-Trained Models
==================

WormLib includes pre-trained machine learning models for cell segmentation and classification. These models are optimized for early-stage *C. elegans* embryo imaging and are located in the ``models/`` directory.

---
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

Training Custom Models
-----------------------

To train your own cellpose segmentation model, follow these steps:

1. Visit the original [Cellpose documentation](https://cellpose.readthedocs.io/en/latest/index.html) for detailed instructions on training custom models.
2. Upload ce-embryo model on Cellpose GUI.
3. Retrain with your own images by manually segmenting cells to set ground truth.
5. Save model
6. Reference in your analysis

Cell Classification Models (Random Forest)
-------------------------------------------

Used to predict cell identity (blastomere name) based on morphological features.

**2-Cell Stage: ``2-cell_classification_RFmodel.joblib``**

Classifies cells into:

- **AB** — Anterior blastomere
- **P1** — Posterior blastomere

Features used:

- Cell position (centroid X, Y)
- Cell area
- Eccentricity (elongation)
- Proximity to predicted embryo center

**Accuracy:** ~95% on test data

**When to use:** Set ``pipeline.cell_classification: true`` for 2-cell stage embryos

**4-Cell Stage: ``4-cell_classification_RFmodel.joblib``**

Classifies cells into:

- **ABa** — Anterior-left blastomere
- **ABp** — Anterior-right blastomere
- **EMS** — Endoderm/mesoderm precursor
- **P2** — Posterior blastomere

Features used:

- Cell position (X, Y relative to embryo center)
- Cell area
- Eccentricity
- Ellipse-based spatial assignment

**Accuracy:** ~92% on test data

**When to use:** Set ``pipeline.cell_classification: true`` for 4-cell stage embryos

**Output:**

Classification results are saved in:

- ``Classification_Report_{image_name}.csv`` — Full feature set + prediction
- ``per_cell_mRNA_counts_{image_name}.csv`` — Label and confidence per cell

Example output:

.. code-block:: text

    Cell_ID,label,prediction_confidence
    1,AB,0.987
    2,P1,0.954
    3,ABa,0.923
    4,ABp,0.891




## Using Models in Code
**Cell Classification:**

.. code-block:: python

    import wormlib
    from pathlib import Path
    
    models_dir = Path('models')
    model_2cell = models_dir / '2-cell_classification_RFmodel.joblib'
    
    features_df = wormlib.classify_2cell(
        masks_cytosol=segmentation_mask,
        bf=brightfield_image,
        image_name='sample_image',
        output_directory='output/',
        model_path=str(model_2cell),
        verbose=True
    )
    # Returns DataFrame with cell IDs, labels, and confidence scores

---

Disabling Classification
------------------------

To skip cell classification (e.g., for embryos at stages not covered by models):

.. code-block:: yaml

    pipeline:
      cell_classification: false

This will:

- Skip the classifier step
- Still perform segmentation
- Per-cell CSV will have no ``label`` or ``confidence`` columns
- Use generic ``region_id`` instead of cell names

---

Model Limitations and Best Practices
-------------------------------------

**Known limitations:**

- 2-cell and 4-cell models only (not trained for other stages)
- Assumes brightfield is of reasonable quality (in-focus, proper exposure)
- May fail on abnormal embryo morphology
- Classifier confidence varies by image quality

**Best practices:**

1. **Always inspect segmentation masks** before trusting results

   .. code-block:: python

       import matplotlib.pyplot as plt
       plt.imshow(masks_cytosol)
       plt.title('Segmentation Mask')
       plt.show()

2. **Check classifier confidence** in output CSVs

   - Confidence > 0.9 is reliable
   - 0.7–0.9 suggests ambiguous cells (review manually)
   - < 0.7 is unreliable (consider disabling classifier)

3. **Validate on a few images first** before batch processing

4. **Classification is directly impacted by segmentation quality. If results are poor**, consider:

   - Re-optimizing cell_diameter in config
   - Checking image quality (brightness, contrast, focus)
   - Manually validating a subset of results
   - Retraining the model with your own labeled data


Training Custom Models
-----------------------

To train your own classifiers:

1. Acquire images
2. Manually label cells to set ground truth for training (segmentation masks + cell identities)
3. Extract morphological features
4. Train Random Forest in scikit-learn
5. Save model with joblib
6. Reference in your analysis

---


