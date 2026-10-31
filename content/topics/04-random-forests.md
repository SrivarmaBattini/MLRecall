---
id: random-forests
title: Random Forests
group: Supervised Learning
difficulty: 2
order: 4
summary: Average many different decision trees to get a stable, accurate model.
prerequisites: decision-trees
---

## What it is and why it matters
A random forest trains many decision trees on slightly different versions of the data and combines their votes. It fixes the main weakness of a single tree, high variance, and is a strong, low-tuning baseline for tabular data.

## Intuition
Ask one expert and you may get an odd answer. Ask a hundred experts who each saw different examples and considered different clues, and the average is far more reliable. The key is that the experts must disagree in useful ways, so the forest deliberately makes the trees different from each other.

## Key ideas
- **Ensemble**: a group of models whose predictions are combined.
- **Bagging (bootstrap aggregating)**: train each tree on a bootstrap sample, drawn with replacement from the training set.
- **Random feature subsets**: at every split, only a random subset of features (`max_features`) may be considered. This decorrelates the trees.
- **Voting and averaging**: majority vote for classification, mean for regression.
- **Variance reduction**: averaging many noisy but different trees cancels much of the noise.
- **Out-of-bag (OOB) error**: each tree never saw about 37% of the data, so those samples give a free validation estimate.
- **Feature importance**: impurity-based importance and permutation importance show which features matter.
- **Key hyperparameters**: `n_estimators` (more is rarely harmful), `max_depth`, `min_samples_leaf`, `max_features`.
- **Parallelism**: trees are independent, so training parallelises easily.
- **Trade-offs**: less interpretable than one tree, bigger and slower to predict, and cannot extrapolate outside the training range.

## The math
A bootstrap sample of size $n$ leaves out each example with probability

$$\left(1 - \frac{1}{n}\right)^n \approx e^{-1} \approx 0.368$$

If $B$ trees each have variance $\sigma^2$ and pairwise correlation $\rho$, the variance of their average is

$$\rho \sigma^2 + \frac{1 - \rho}{B}\sigma^2$$

Adding trees shrinks the second term, while random feature subsets shrink $\rho$, which is why both ingredients matter.

## Worked example
Three trees classify a customer as A, B and A. The forest predicts A by majority vote. For regression, three trees predict 10, 12 and 14, so the forest predicts their mean, 12. With 1,000 training rows, each tree skips roughly 368 of them, and those rows are used to compute the OOB score for that tree.

## Code
```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0, stratify=y)

rf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                            oob_score=True, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
print("OOB accuracy:", rf.oob_score_)
print("test accuracy:", rf.score(X_te, y_te))
```

## Common mistakes
- Using all features at every split, which makes the trees too similar.
- Believing more trees can overfit; they mainly add compute.
- Treating impurity-based importance as unbiased when features have many distinct values.
- Expecting a forest to extrapolate a trend beyond the training range.
- Ignoring OOB score and wasting data on a separate validation set unnecessarily.

## Summary
- A forest is many decorrelated trees averaged together.
- Bootstrap sampling and random feature subsets create the diversity.
- Averaging cuts variance; OOB error gives a free performance estimate.
- It is robust and needs little tuning, but is harder to interpret than one tree.

## Questions
Q: What does bagging mean?
A) Training each model on a bootstrap sample and aggregating the results
B) Removing the weakest feature
C) Training models one after another on the residuals
D) Scaling the features to a bag of ranges
ANSWER: A
WHY: Bagging trains models on bootstrap samples drawn with replacement and combines their predictions.

Q: Why does a random forest restrict each split to a random subset of features?
A) To make training slower
B) To decorrelate the trees so that averaging reduces variance more
C) To avoid needing labels
D) To reduce the number of samples
ANSWER: B
WHY: If every tree used the strongest feature first they would look alike and averaging would help less.

Q: About what fraction of the training set is left out of a bootstrap sample?
A) About 5%
B) About 37%
C) About 63%
D) None of it
ANSWER: B
WHY: The chance an example is never drawn tends to $e^{-1} \approx 0.368$.

Q: What is the out-of-bag (OOB) error?
A) Error on examples a tree did not see during its own training, aggregated over trees
B) Error on the training set only
C) The error of the worst tree
D) The error after pruning
ANSWER: A
WHY: Each example is predicted using only trees that did not train on it, giving a built-in validation estimate.

Q: Which change usually reduces the variance of a forest?
A) Using fewer, deeper trees on identical data
B) Averaging more decorrelated trees
C) Using one tree only
D) Removing bootstrap sampling
ANSWER: B
WHY: Variance of the average falls as the number of trees grows and as their correlation falls.

Q: How does a random forest combine regression trees?
A) It takes the maximum prediction
B) It averages the predictions
C) It takes the first tree's prediction
D) It uses the mode
ANSWER: B
WHY: For regression the forest outputs the mean of its trees' predictions.

Q: Can a random forest extrapolate a linear trend beyond the training range?
A) Yes, perfectly
B) No, its predictions stay within the range of training targets
C) Only with more trees
D) Only with scaling
ANSWER: B
WHY: Leaves predict averages of seen targets, so outputs cannot go beyond what was observed.

Q: What is a downside of a forest compared with a single tree?
A) Higher variance
B) Lower interpretability and slower prediction
C) It cannot handle numeric features
D) It needs labels to be scaled
ANSWER: B
WHY: Hundreds of trees are hard to read and cost more to store and evaluate.

Q: Is adding more trees likely to cause overfitting?
A) Yes, strongly
B) Rarely; it mainly adds compute and the error plateaus
C) Always
D) Only for classification
ANSWER: B
WHY: Averaging more trees stabilises predictions; gains flatten rather than reverse.

Q: Permutation importance of a feature is measured how?
A) By counting how often it appears at the root
B) By the drop in model score when that feature's values are shuffled
C) By its correlation with the label only
D) By the number of leaves it creates
ANSWER: B
WHY: Shuffling breaks the feature's link to the target, and the score drop shows how much the model relied on it.
