# Understanding the Bias-Variance Trade-Off

[Resource](https://medium.com/data-science/understanding-the-bias-variance-tradeoff-165e6942b229)

To reiterate what you already know...

**What is bias?**

Bias is the difference between the average prediction of our model and the correct value which we are trying to predict. A model with high bias pays very little attention to the training data and oversimplifies the model. It always leads to high error on training and test data.

**What is variance?**

Variance is the variability of model prediction for a given data point or a value which tells us the spread of our data. A model with high variance pays a lot of attention to training data and doesn't generalize on the data which is hasn't seen before. As a result, such models perform very well on training data but has high error rates on test data.

**Mathematically**

Let the variable we're trying to predict be Y and other coefficients are X. We assume there is a relationship between the two such that

Y=f(X) + e

Where e is the error term and it's normally distributed with a mean of 0.

We will make a model f^(X) of f(X) using linear regression or any other modeling technique.

