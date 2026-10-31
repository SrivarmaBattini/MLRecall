---
id: logistic-regression
title: Logistic Regression
group: Supervised Learning
difficulty: 1
order: 2
summary: Turn a weighted sum into a probability to classify examples into two or more classes.
prerequisites: linear-regression
---

## What it is and why it matters
Logistic regression is a classifier. It outputs the probability that an example belongs to a class, such as spam or not spam. Despite the name it is a classification method, and it is the basic building block of neural-network output layers.

## Intuition
Take the same weighted sum used in linear regression, but squeeze the result through an S-shaped curve so it lands between 0 and 1. Large positive sums mean "almost surely class 1", large negative sums mean "almost surely class 0", and zero means "unsure". The place where the probability equals 0.5 is the decision boundary.

## Key ideas
- **Logit (log-odds)**: the raw score $z = w^\top x + b$.
- **Sigmoid function**: $\sigma(z) = 1/(1+e^{-z})$, which maps any number to $(0,1)$.
- **Probability**: $p = P(y=1 \mid x) = \sigma(z)$.
- **Decision threshold**: classify as 1 when $p$ is above a cut-off, 0.5 by default. Moving it trades precision against recall.
- **Decision boundary**: the set of points where $z = 0$; it is a straight line or hyperplane.
- **Log loss (cross-entropy)**: the loss that punishes confident wrong answers heavily.
- **Maximum likelihood**: training chooses the weights that make the observed labels most probable.
- **Odds ratio**: $e^{w_j}$ is the factor by which the odds multiply when feature $j$ rises by one unit.
- **Softmax and multiclass**: with more than two classes, softmax turns scores into probabilities that sum to 1.
- **Regularisation**: an L2 or L1 penalty to avoid overfitting; in scikit-learn a smaller `C` means stronger regularisation.
- **Evaluation metrics**: accuracy, precision, recall, F1 and ROC-AUC. Accuracy misleads when classes are imbalanced.

## The math
$$p = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}$$

Log loss for one example with label $y \in \{0,1\}$:

$$\ell = -\left[ y \log p + (1-y)\log(1-p) \right]$$

Its gradient is neat: $\nabla_w J = \frac{1}{n} X^\top (p - y)$. For $K$ classes, softmax gives $p_k = e^{z_k} / \sum_j e^{z_j}$. Precision is $TP/(TP+FP)$, recall is $TP/(TP+FN)$ and F1 is their harmonic mean.

## Worked example
Let $w = 1.5$ and $b = -3$ with one feature. For $x = 2$: $z = 0$ and $p = 0.5$, so the example sits on the decision boundary. For $x = 3$: $z = 1.5$ and $p = \sigma(1.5) \approx 0.818$, so the model predicts class 1 with about 82% confidence. The log loss if the true label is 1 is $-\log 0.818 \approx 0.20$; if the true label is 0 it is $-\log 0.182 \approx 1.70$, much larger.

## Code
```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=1000, n_features=6, weights=[0.8, 0.2], random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, stratify=y, random_state=0)

clf = LogisticRegression(C=1.0, max_iter=1000).fit(X_tr, y_tr)
proba = clf.predict_proba(X_te)[:, 1]
print(classification_report(y_te, proba > 0.5))
```

## Common mistakes
- Reporting accuracy on imbalanced data where predicting the majority class already scores high.
- Using mean squared error instead of log loss; with a sigmoid it gives weak gradients.
- Assuming the 0.5 threshold is always right.
- Reading raw weights as probabilities instead of log-odds.
- Skipping feature scaling when using regularisation.

## Summary
- Logistic regression is a linear score passed through a sigmoid to give probabilities.
- Train with log loss; the boundary is linear.
- Choose the threshold by the cost of false positives versus false negatives.
- Use softmax for several classes and precision, recall and F1 when classes are imbalanced.

## Questions
Q: What is the output range of the sigmoid function?
A) $(-\infty, \infty)$
B) $(0, 1)$
C) $(-1, 1)$
D) $[0, \infty)$
ANSWER: B
WHY: The sigmoid squeezes any real number into the open interval between 0 and 1.

Q: What shape is the decision boundary of plain logistic regression?
A) A circle
B) A straight line or hyperplane
C) A tree of rectangles
D) A parabola
ANSWER: B
WHY: The boundary is where $w^\top x + b = 0$, which is linear in the features.

Q: Why does log loss punish a confident wrong prediction so much?
A) Because $-\log p$ becomes very large when $p$ is near 0 for the true class
B) Because probabilities are capped at 0.5
C) Because it ignores correct predictions
D) Because it uses absolute error
ANSWER: A
WHY: If the true class gets probability close to 0, $-\log p$ grows without bound.

Q: Raising the decision threshold from 0.5 to 0.8 will usually do what?
A) Raise recall and lower precision
B) Raise precision and lower recall
C) Leave both unchanged
D) Make the model multiclass
ANSWER: B
WHY: A stricter threshold predicts positive less often, so fewer false positives but more missed positives.

Q: Which function turns multiclass scores into probabilities that sum to 1?
A) ReLU
B) Sigmoid
C) Softmax
D) Tanh
ANSWER: C
WHY: Softmax exponentiates the scores and normalises them across classes.

Q: If a weight $w_j = 0.7$, by what factor do the odds change per one-unit increase in feature $j$?
A) $0.7$
B) $e^{0.7}$
C) $1/0.7$
D) $\log 0.7$
ANSWER: B
WHY: The weight is a log-odds change, so the odds multiply by $e^{w_j}$.

Q: A model on a dataset that is 95% negative predicts "negative" every time. What is its accuracy?
A) 5%
B) 50%
C) 95%
D) 100%
ANSWER: C
WHY: It is right on every negative example, so accuracy is 95% despite being useless, which is why accuracy misleads here.

Q: How is recall defined?
A) $TP/(TP+FP)$
B) $TP/(TP+FN)$
C) $TN/(TN+FP)$
D) $(TP+TN)/\text{total}$
ANSWER: B
WHY: Recall is the share of actual positives that the model finds.

Q: In scikit-learn's LogisticRegression, what does a smaller `C` mean?
A) Weaker regularisation
B) Stronger regularisation
C) More classes
D) A larger learning rate
ANSWER: B
WHY: `C` is the inverse of regularisation strength, so smaller `C` shrinks the weights more.

Q: Why is mean squared error a poor loss for logistic regression?
A) It cannot be computed on probabilities
B) With a sigmoid it gives a non-convex loss and weak gradients when confidently wrong
C) It only works for regression trees
D) It requires multiclass labels
ANSWER: B
WHY: Log loss matches the probabilistic model and gives strong, well-behaved gradients.
