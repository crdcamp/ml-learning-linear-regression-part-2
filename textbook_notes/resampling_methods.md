# Resampling Methods

Resampling methods involve repeatedly drawing samples from a training set and refitting a model of interest on each sample in order to obtain additional information about the fitted model.

Resampling approaches can be computationally expensive, because they involve fitting the same statistical method multiple times using different subsets of training data. However, more recent advances in computing power have made this not as big of an issue.

In this chapter, we discuss two of the most commonly used resampling methods, **cross-validation** and the **bootstrap**. 

The process of 