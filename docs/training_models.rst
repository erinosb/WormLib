Training Custom Models
-----------------------

To train your own cellpose segmentation model:

1. Visit the original [Cellpose documentation](https://cellpose.readthedocs.io/en/latest/index.html) for detailed instructions on training custom models.
2. Upload ce-embryo model on Cellpose GUI.
3. Retrain with your own images by manually segmenting cells to set ground truth.
4. Save model
5. Update model path to the new model in your analysis pipeline


To train your own classifier:

1. Manually label cells to set ground truth for training (segmentation masks + cell identities)
2. Extract morphological features
3. Train Random Forest in scikit-learn
4. Save model with joblib
5. Update model path to the new model in your analysis pipeline

---
