Training Custom Models
-----------------------

To train your own cellpose segmentation model, follow these steps:

1. Visit the original [Cellpose documentation](https://cellpose.readthedocs.io/en/latest/index.html) for detailed instructions on training custom models.
2. Upload ce-embryo model on Cellpose GUI.
3. Retrain with your own images by manually segmenting cells to set ground truth.
4. Save model
5. Reference in your analysis


To train your own classifier:

1. Acquire images
2. Manually label cells to set ground truth for training (segmentation masks + cell identities)
3. Extract morphological features
4. Train Random Forest in scikit-learn
5. Save model with joblib
6. Reference in your analysis

---
