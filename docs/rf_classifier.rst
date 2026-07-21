Cell Classification Models (Random Forest)
==========================================

Used to predict cell identity (blastomere name) based on morphological features.

2-cell classifier
------------------

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



4-cell classifier
------------------
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




Using Cell Classification Models in Code
--------------------

**Cell Classification:**
**To use:** Set ``pipeline.cell_classification: true`` for 2-cell stage embryos


.. code-block:: python

    import wormlib
    # Cell classification
    if run_cell_classifier and masks_cytosol is not None:
        models_dir = main_dir / "models"
        model_2cell_path = models_dir / "2-cell_classification_RFmodel.joblib"
        model_4cell_path = models_dir / "4-cell_classification_RFmodel.joblib"
        
        if cell_stage == "2-cell" and model_2cell_path.exists():
            print("Running 2-cell classifier...")
            features_df = wormlib.classify_2cell(
                masks_cytosol=masks_cytosol, 
                bf=bf,
                image_name=image_name, 
                output_directory=output_directory,
                model_path=str(model_2cell_path), 
                verbose=True,
            )
        elif cell_stage == "4-cell" and model_4cell_path.exists():
            print("Running 4-cell classifier...")
            features_df = wormlib.classify_4cell(
                masks_cytosol=masks_cytosol, 
                bf=bf,
                image_name=image_name, 
                output_directory=output_directory,
                model_path=str(model_4cell_path), 
                verbose=True,
            )
        else:
            print(f"Classifier not available for stage '{cell_stage}'")
            run_cell_classifier = False
            
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