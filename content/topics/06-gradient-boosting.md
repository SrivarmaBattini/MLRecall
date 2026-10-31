---
id: gradient-boosting
title: Gradient Boosting
group: Supervised Learning
difficulty: 2
order: 6
summary: Build trees one after another, each correcting the mistakes of the ones before it.
prerequisites: decision-trees, linear-regression
---

## What it is and why it matters
Gradient boosting builds a strong model by adding many small, shallow trees one at a time. Each new tree focuses on the errors left by the current model. Libraries such as XGBoost, LightGBM and CatBoost use it, and it is often the most accurate method on tabular data.

## Intuition
Imagine a student fixing an essay in rounds. The first draft is rough. Each round, an editor looks only at what is still wrong and makes a small fix. After many rounds the essay is good. The editor never rewrites everything; each correction is deliberately small.

## Key ideas
- **Boosting**: models are built sequentially, each one improving the ensemble.
- **Weak learner**: a simple model, usually a shallow tree of depth 3 to 6.
- **Residuals**: the errors left by the current prediction.
- **Pseudo-residuals**: the negative gradient of the loss; for squared error they equal the plain residuals.
- **Additive model**: the final prediction is the sum of all trees' contributions.
- **Learning rate (shrinkage)**: scales each tree's contribution; smaller values need more trees but generalise better.
- **Number of trees**: too many overfits, so use early stopping on a validation set.
- **Subsampling**: training each tree on a random fraction of rows (stochastic gradient boosting).
- **Tree regularisation**: limit depth, minimum samples per leaf and add L1 or L2 penalties on leaf values.
- **Loss functions**: squared error, absolute error, log loss and others; the framework is generic.
- **Bagging versus boosting**: bagging reduces variance with parallel trees; boosting reduces bias with sequential trees.
- **AdaBoost**: an earlier boosting method that re-weights misclassified samples.

## The math
Start from a constant model $F_0$. At round $m$ compute pseudo-residuals

$$r_{im} = -\left.\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right|_{F = F_{m-1}}$$

fit a tree $h_m$ to them, and update

$$F_m(x) = F_{m-1}(x) + \eta \, h_m(x)$$

where $\eta$ is the learning rate. For squared error $L = \tfrac{1}{2}(y - F)^2$ the pseudo-residual is $y - F$.

## Worked example
Targets are $y = [3, 5, 7]$. Start with the mean, $F_0 = 5$ for every point. Residuals are $[-2, 0, 2]$. Suppose the tree fits them perfectly and $\eta = 0.1$. Then $F_1 = 5 + 0.1 \times [-2, 0, 2] = [4.8, 5, 5.2]$. The new residuals are $[-1.8, 0, 1.8]$, which are 10% smaller. Repeating the step shrinks them further.

## Code
```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

gb = HistGradientBoostingRegressor(learning_rate=0.1, max_depth=4,
                                   early_stopping=True, random_state=0).fit(X_tr, y_tr)
print("test RMSE:", mean_squared_error(y_te, gb.predict(X_te)) ** 0.5)
```

## Common mistakes
- Using a large learning rate with many trees and overfitting.
- Making the trees deep, which turns weak learners into strong, noisy ones.
- Skipping early stopping and tuning the number of trees blindly.
- Tuning on the test set instead of a validation set.
- Expecting boosting to be robust to very noisy labels; it chases errors.

## Summary
- Boosting adds shallow trees sequentially, each fitting the current errors.
- The gradient view lets any differentiable loss be used.
- Learning rate, number of trees and tree depth trade off against each other.
- Use early stopping and subsampling to control overfitting.

## Questions
Q: What does each new tree in gradient boosting try to fit?
A) The original targets
B) The pseudo-residuals, the negative gradient of the loss
C) Random noise
D) The features' means
ANSWER: B
WHY: Each tree is trained to predict the direction that reduces the current loss.

Q: For squared-error loss, the pseudo-residual equals what?
A) $y - F(x)$, the ordinary residual
B) $F(x) - y$ squared
C) The learning rate
D) The tree depth
ANSWER: A
WHY: The negative gradient of $\tfrac{1}{2}(y - F)^2$ with respect to $F$ is $y - F$.

Q: What does a smaller learning rate usually require?
A) Fewer trees
B) More trees to reach the same fit, often with better generalisation
C) Deeper trees
D) No validation set
ANSWER: B
WHY: Smaller steps make slower progress per tree, so more trees are needed, but the model tends to generalise better.

Q: Which description fits bagging versus boosting?
A) Bagging is sequential; boosting is parallel
B) Bagging trains independent models in parallel; boosting trains sequentially
C) They are the same
D) Neither uses trees
ANSWER: B
WHY: Bagging averages independent models, while boosting builds each model on the errors of earlier ones.

Q: Why are the trees in boosting usually shallow?
A) Deep trees cannot be trained
B) They are weak learners, and many small corrections generalise better
C) Shallow trees are always more accurate alone
D) Depth must equal the number of features
ANSWER: B
WHY: Boosting adds many small corrections; deep trees would overfit quickly.

Q: What is early stopping in boosting?
A) Stopping adding trees when validation error stops improving
B) Removing the first tree
C) Using only one tree
D) Stopping the feature scaling
ANSWER: A
WHY: It halts training at the number of trees where validation performance is best, preventing overfitting.

Q: What does subsampling rows (stochastic gradient boosting) add?
A) Randomness that reduces overfitting and speeds training
B) A larger learning rate
C) Extra features
D) A second loss function
ANSWER: A
WHY: Training each tree on a random subset decorrelates trees and acts as regularisation.

Q: Boosting mainly reduces which source of error?
A) Bias, by sequentially correcting the model's mistakes
B) Only variance from the data
C) Irreducible noise
D) Label noise to zero
ANSWER: A
WHY: Each round lowers the remaining error, which reduces bias; regularisation then keeps variance in check.

Q: What is the starting model $F_0$ for squared error usually?
A) The mean of the targets
B) Zero
C) A deep tree
D) The median of features
ANSWER: A
WHY: The constant that minimises squared error is the mean of $y$.

Q: Which of these is a gradient-boosting library?
A) XGBoost
B) NumPy
C) Matplotlib
D) Pandas
ANSWER: A
WHY: XGBoost, LightGBM and CatBoost are popular implementations of gradient-boosted trees.
