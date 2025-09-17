# Resampling Methods

Resampling methods involve repeatedly drawing samples from a training set and refitting a model of interest on each sample in order to obtain additional information about the fitted model.

Resampling approaches can be computationally expensive, because they involve fitting the same statistical method multiple times using different subsets of training data. However, more recent advances in computing power have made this not as big of an issue.

In this chapter, we discuss two of the most commonly used resampling methods, **cross-validation** and the **bootstrap**. 

The process of evaluating a model's performance is known as **model assessment**, whereas the process of selecting the proper level of flexibility for a model is known as **model selection**.

# Cross-Validation

In the absence of a very large designated test set that can be used to directly estimate the test error rate, a number of techniques can be used to estimate this quantity using the available training data. Some methods make a mathematical adjustment to the training error in order to estimate the test error rate (discussed in Chapter 6). Instead, we'll consider a class of methods that estimate the test error rate by *holding out* a subset of the training observations from the fitting process, and then applying the statistical learning method to those held out observations.

## The Validation Set Approach

Suppose that we would like to estimate the test error associated with fitting a particular statistical learning method on a set of observations. The **validation set approach** is a very simple strategy for this task.

It involves randomly dividing the available set of observations into two parts, a **training set** and **validation or holdout** set. The model is fit on the training set, and the fitted model is used to predict the responses in the validation set. This is where we get our estimate of the test error rate.

We'll illustrate all these explanations using the `Auto` dataset. Recall from much earlier that there appears to be a non-linear relationship between `mpg` and `horsepower`, and that a model that predicts `mpg` using `horsepower` and `horsepower**2` gives better results than a model that uses only a linear term. It's natural to wonder whether a cubic or higher-order fit might provide even better results. We answer this question in Chapter 3 by looking at the --values associated with a cubic term and higher-order polynomial terms in a linear regression. But we could also answer this question using the validation method.

We randomly split the 392 observations into two sets, a training set containing 196 of the data points, and a validation set containing the remaining 196 observations. 

In the graph below, the left graph contains validation error estimates for a single split into training and validation data sets. The right graph shows the validation method repeated ten times, each time using a different random split of the observations into a training and validation set.

![Alt image](../images/validation_set_graph.png)

This illustrates the variability in the estimated test MSE that results from this approach. Based on the variability of these curves, all that we can conclude with any confidence is that the attempted linear fit in these graphs is not adequate for this data.

As you might've notices, the validation set approach has two potential drawbacks:
1. As is shown in the right-hand panel, the validation estimate of the test error rate can be highly variable, depending on precisely which observations are included in the training set and which observations are included in the validation set.
1. In the validation approach, only a subset of the observations-those that are included in the training set rather than in the validation set-are used to fit the model. Since statistical methods tend to perform worse when trained on fewer observations, this suggests that validations set error rate may tend to **overestimate** the test error rate for the model fit on the entire data set.

## Leave-One-Out-Cross-Validation

