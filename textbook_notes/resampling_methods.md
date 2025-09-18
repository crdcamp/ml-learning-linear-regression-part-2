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

Also called LOOCV (what a terrible acronym) attempts to address the validation set approach's drawbacks.

Like the validation set approach, LOOCV involves splitting the set of observations into two parts. However, increase of creating two subsets of comparable size, a single observation (*x1*, *y1*) is used for the validation set, and the remaining observations make up the training set. The statistical method is fit on the *n* -1 training observations, and a prediction `y_hat1` is made for the excluded observation, using its value *x1*. Since (*x1*, *y1*) was not used in the fitting process, `MSE1 = (y1 - y_hat1)**2` provides an approximately unbiased estimate for the test error. But even though MSE1 is unbiased for the test error, it's a poor estimate because it's highly variable, since it's based upon a single observation.

Essentially, you're taking one observation to predict and using the rest of the data as training data to predict that one given observation. You repeat this process for all observations. It's a pretty brute force approach.

The LOOCV estimate for the test MSE is the average of the *n* test error estimates:

![Alt image](../images/loocv_formula.png)

LOOCV has a couple of major advantages over the validation set approach. First, it has far less bias, and tends to not overestimate the test error rate as much as the validation set approach does. Second, in contrast to the validation approach whiick will yield different results when applied repeatedly due to randomness in the training/validation set splits, performing LOOCV multiple times will always yield the same results.

As you've already guessed, LOOCV has the potential to be expensive to implement. With the least squares linear or polynomial regression, an amazing shortcut makes the cost of LOOCV the same as that of a single model fit. The following formula holds:

![Alt image](../images/loocv_formula_2.png)

where `y_hati` is the *i*th value from the original least squares fit, and *h*i is the leverage.

**Reminder:** Leverage refers to the input value. High leverage indicates an unusual input value.

This is like the ordinary MSE, except that the *i*th residual is divided by 1 - *h*. The leverage lies between 1/*n* and 1, and reflects the amount that an observation influences its own fit. Hence, the residuals for high-leverage points are inflated in this formula by exactly the right amount for this equality to hold.

LOOCV is a very general method, and can be used with any kind of predictive modeling. We could use it with logistic regression or linear discriminant analysis, or any of the methods discussed in later chapters.

## k-Fold Cross-Validation

An alternative to LOOCV is k-fold CV. This approach involves randomly dividing the set of observations into *k* groups, or *folds*, of approximately equal size. The first fold is treated as a validation set, and the method is fit on the remaining *k* - 1 folds. The mean squared error, MSE1, is then computed on the observations in the held-out fold. This procedure is repeated *k* times, each time, a different group of observations is treated as a validation set. This process results in *k* estimates of the test error, MSE1, MSE2,...., MSEk. The *k*-fold CV estimate is computed by averaging these values,

![Alt image](../images/k_fold_formula.png)

LOOCV is a special case of *k*-fold CV in which *k* is set to equal *n*. **In practice, one typically performs *k*-fold CV using *k* = 5 or *k* = 10**.

The right hand panel below displays nine different 10-fold CV estimates for the `Auto` dataset, each resulting from a different random split of the observations into ten folds.

![Alt image](../images/10_fold_k_cv.png)

As we can see, there is a much lower variability in these results compared to LOOCV. 

When we perform cross-validation, our goal might be to determine how well a given statistical learning procedure can be expected to perform on independent data. But, at other times we're only interested in the location of the minimum point in the estimated test MSE curve. This is because **we might be performing cross-validation on a number of statistical learning methods, or on a single method using different levels of flexibility, in order to identify the methods that results in the lowest test error. For this reason, the location of the minimum point in the estimated test MSE curve is important, but the actual value of the estimated test MSE is not**.

## Bias-Variance Trade-Off for *k*-Fold Cross Validation

We mentioned in a previous section that *k*-fold CV with k < n has a computational advantage to LOOCV. A less obvious advantage but potentially more important advantage of *k*-fold CV is that it often gives more accurate estimates of the test error rate than does LOOCV. This has to do with a bias-variance trad-off.

It was mentioned earlier that the validation set approach can lead to overestimates of the test error rate, since in this approach the training set used to fit the statistical learning method contains only half the observations of the entire data set. Using LOOCV gives approximately unbiased estimates of the test error, while *k*-fold is said to have an intermediate level of bias, since each training set contains fewer than in the LOOCV approach, but substantially more than the validation set approach. Therefore, **from the perspective of bias reduction, LOOCV is to preferred to k-fold CV**.

However, we know that bias is not the only source for concern in estimating procedure; we must also consider the procedure's variance. It turns out that **LOOCV has a higher variance than does *k*-fold CV with *k* < *n***. 

Why is this the case? When we perform LOOCV, we are in effect averaging the outputs of *n* fitted models, each of which is trained on an almost identical set of observations; therefore, these outputs are highly (positively) correlated with each other. In contrast, when we perform *k*-fold CV with *k* < *n*, we are averaging the outputs of *k* fitted models that are somewhat less correlated with each other, since the overlap between the training sets in each model is smaller. Since the mean of many highly correlated quantities has higher variance than does the mean of many quantities that are not as highly correlated, the test error estimate resulting from LOOCV tends to have higher variance than does the test error estimate resulting from the *k*-fold CV.

**Note:** The textbook then continues onto cross-validation on classification problems, but we'll revisit that once linear regression is complete (page 216).

