## A Multi-Modal Neural Network for Autism Detection and Classification

We have used two modalities - fMRI (functional connectivity) and Phenotypic (from .csv file) to create an embedding fusion based architecture to perform autism detection.

**Dataset used :** ABIDE-1 (855 subjects, after removing corrupted files)

**fMRI branch :** Pearson correlation based functional connectivity is extracted from the CC200 based time-series files (.1D files). For preprocessing, we have performed bandpass filtering but no global signal regression. 

**Phenotypic branch:** six phenotypic features were selected- [VIQ, PIQ, func_mean_fd, func_perc_fd, func_num_fd, func_quality] from the .csv file. 

The model takes these 2 branches and fuses their embeddings, before passing it to the classifier.

Our model has achieved a reasonable accuracy of 71.46% over a 5-fold cross validation. This is on par with most method and papers that have been published over the last few years (2020-24).

