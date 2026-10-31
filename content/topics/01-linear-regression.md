---
id: linear-regression
title: Linear Regression
group: Supervised Learning
difficulty: 1
order: 1
summary: Predict a number by fitting the best straight line (or plane) through your data.
prerequisites:
---

## What it is and why it matters
Linear regression predicts a continuous number, such as a house price, from one or more input features. It is the simplest supervised model and introduces ideas used everywhere in machine learning: loss functions, gradient descent, regularisation and the bias-variance trade-off.

## Intuition
Plot house size against price. The points rise but never line up perfectly. Linear regression draws the single straight line that stays as close to all the points as possible. To predict a new house, read its price off the line. With many features the line becomes a flat surface (a hyperplane), but the idea is the same.

## Key ideas
- **Features and target**: the inputs $x$ and the number $y$ we want to predict.
- **Weights and bias**: the slopes $w$ and the intercept $b$ that the model learns from data.
- **Prediction**: $\hat{y} = w^\top x + b$, a weighted sum of the features.
- **Residual**: the gap $y - \hat{y}$ between the truth and the prediction.
- **Loss function**: one number that says how wrong the model is. Here it is mean squared error (MSE).
- **Ordinary least squares (OLS)**: choose $w, b$ that minimise the sum of squared residuals. It has a closed-form answer, the normal equation.
- **Gradient descent**: repeatedly step the weights downhill on the loss. It works when the closed form is too costly.
- **Learning rate**: the step size. Too large overshoots and diverges; too small crawls.
- **Feature scaling**: putting features on similar ranges so gradient descent behaves well.
- **Regularisation**: a penalty on large weights. Ridge (L2) shrinks weights smoothly; lasso (L1) can set some exactly to zero.
- **Overfitting and underfitting**: fitting noise (low training error, high test error) versus being too simple to capture the pattern.
- **$R^2$**: the share of the variation in $y$ that the model explains.
- **Assumptions**: a linear relationship, independent errors with constant variance, and features that are not almost copies of each other (multicollinearity).

## The math
Mean squared error over $n$ examples:

$$J(w,b) = \frac{1}{n}\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2$$

With a column of ones added to $X$ for the bias, the normal equation gives the best weights directly:

$$w = (X^\top X)^{-1} X^\top y$$

Gradient descent instead repeats $w \leftarrow w - \eta \, \frac{2}{n} X^\top (Xw - y)$ where $\eta$ is the learning rate. Ridge regression adds $\lambda \lVert w \rVert^2$ to the loss, giving $w = (X^\top X + \lambda I)^{-1} X^\top y$.

## Worked example
Three points: $(1,2)$, $(2,3)$, $(3,5)$. The means are $\bar{x}=2$ and $\bar{y}=10/3$. The slope is $\sum (x-\bar{x})(y-\bar{y}) / \sum (x-\bar{x})^2$. The numerator is $(-1)(-4/3) + 0 + (1)(5/3) = 3$ and the denominator is $2$, so the slope is $1.5$. The intercept is $\bar{y} - 1.5\bar{x} = 1/3$. The prediction at $x=4$ is $1.5 \times 4 + 1/3 \approx 6.33$.

## Code
```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

rng = np.random.default_rng(0)
X = rng.uniform(0, 10, size=(200, 1))
y = 3.0 * X[:, 0] + 2.0 + rng.normal(0, 1.5, 200)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
model = LinearRegression().fit(X_tr, y_tr)
print("slope, intercept:", model.coef_[0], model.intercept_)
print("test R^2:", r2_score(y_te, model.predict(X_te)))
ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)   # L2-regularised version
```

## Common mistakes
- Judging the model on the training data instead of held-out test data.
- Forgetting to scale features before gradient descent or regularisation.
- Trusting a straight line when the residual plot shows a curve.
- Including near-duplicate features, which makes the weights unstable.
- Reading a large weight as "important" when the features are on different scales.

## Summary
- Linear regression fits $\hat{y} = w^\top x + b$ by minimising squared error.
- Solve it with the normal equation or with gradient descent.
- Ridge and lasso regularisation fight overfitting; $R^2$ and test error measure fit.
- Check the assumptions with residual plots.

## Questions
Q: Which quantity does ordinary least squares minimise?
A) The sum of absolute residuals
B) The sum of squared residuals
C) The largest single residual
D) The cross-entropy between labels and predictions
ANSWER: B
WHY: OLS picks the weights that make the sum of squared residuals as small as possible.

Q: In $\hat{y} = w^\top x + b$, what does the bias $b$ represent?
A) The prediction when all features are zero
B) The learning rate
C) The number of features
D) The regularisation strength
ANSWER: A
WHY: With every feature at zero the weighted sum vanishes, so the output is just $b$, the intercept.

Q: What usually happens if the learning rate in gradient descent is far too large?
A) Training converges faster and more smoothly
B) The loss oscillates or diverges
C) The weights become exactly zero
D) The model switches to the normal equation
ANSWER: B
WHY: Large steps overshoot the minimum, so the loss can bounce around or grow.

Q: Which penalty does ridge regression add to the loss?
A) The sum of absolute weights
B) The number of non-zero weights
C) The sum of squared weights
D) The largest weight
ANSWER: C
WHY: Ridge adds $\lambda \lVert w \rVert^2$, the squared L2 norm, which shrinks weights smoothly.

Q: Which regulariser can drive some weights exactly to zero?
A) Ridge (L2)
B) Lasso (L1)
C) Early stopping only
D) Feature scaling
ANSWER: B
WHY: The L1 penalty has a corner at zero, so lasso produces sparse weights and acts as feature selection.

Q: What does $R^2 = 0$ tell you?
A) The model is perfect
B) The model does no better than always predicting the mean of $y$
C) All residuals are negative
D) The model has zero weights and zero bias
ANSWER: B
WHY: $R^2$ compares the model with the mean predictor; zero means no improvement over it.

Q: Very low training error but high test error indicates what?
A) Underfitting
B) Overfitting
C) A perfect model
D) A learning rate that is too small
ANSWER: B
WHY: The model memorised the training data, including noise, and does not generalise.

Q: Why scale features before running gradient descent?
A) To make the loss negative
B) To remove the bias term
C) So all features have similar ranges and the steps behave well
D) To increase the number of features
ANSWER: C
WHY: Very different ranges stretch the loss surface, which slows and destabilises gradient descent.

Q: What can break the normal equation $(X^\top X)^{-1}$?
A) Having too few outliers
B) Perfectly or nearly collinear features
C) Using a bias term
D) Having a small learning rate
ANSWER: B
WHY: Collinear features make $X^\top X$ singular or nearly so, so it cannot be inverted reliably.

Q: A residual plot shows a clear U-shaped curve. What does that suggest?
A) The linearity assumption is violated
B) The learning rate is too high
C) The model is perfect
D) There is no bias term
ANSWER: A
WHY: Structured residuals mean the straight line misses a pattern, such as a curve, in the data.
