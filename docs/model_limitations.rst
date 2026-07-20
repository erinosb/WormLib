WormLib Models Limitations and Best Practices
==================

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


---

