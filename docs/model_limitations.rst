WormLib Models Limitations and Best Practices
==================

**Known limitations:**

- The cell segmentation model included in WormLib was trained on images of 2-cell and 4-cell embryos only, doesn't perform well for other stages
- Assumes brightfield is of reasonable quality (in-focus, proper exposure)
- May fail on abnormal embryo morphology
- Classification performance is directly impacted by segmentation quality
- Classifier confidence can vary by image quality
- The model may not generalize well to images from different microscopes or imaging conditions than those used for training
- Classifier is disabled if cell segmentation is set to False or fails (e.g., if no cells are detected)

**Best practices:**

1. **Always inspect segmentation masks** before trusting results

   .. code-block:: python

       import matplotlib.pyplot as plt
       plt.imshow(masks_cytosol)
       plt.title('Segmentation Mask')
       plt.show()

2. **Classification is directly impacted by segmentation quality. If results are poor**, consider:

   - Manually validating a subset of results
   - Re-optimizing cell_diameter in config
   - Checking image quality (brightness, contrast, focus)
   - Retraining the model with your own labeled data


3. **Check classifier confidence** in output CSVs

   - Confidence > 0.9 is reliable
   - 0.7–0.9 suggests ambiguous cells (review manually)
   - < 0.7 is unreliable (consider disabling classifier)



4. **Finally, validate on a few images first** before batch processing
---

