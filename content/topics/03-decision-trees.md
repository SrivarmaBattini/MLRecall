---
id: decision-trees
title: Decision Trees
group: Supervised Learning
difficulty: 1
order: 3
summary: Split the data with a sequence of yes/no questions until each group is mostly one class.
prerequisites:
---

## What it is and why it matters
A decision tree predicts by asking a chain of simple questions about the features, such as "is age above 30?". It can do classification and regression, needs little data preparation and is easy to read. Trees are also the building block of random forests and gradient boosting.

## Intuition
Think of the game Twenty Questions. Each question should split the possibilities as cleanly as possible. A tree learns which question to ask first, then repeats the process on each branch, until the groups are pure enough or a limit is reached.

## Key ideas
- **Nodes**: the root is the first split, internal nodes are later splits, and leaves hold the final prediction.
- **Split**: a rule on one feature, such as $x_j \le t$.
- **Impurity**: how mixed the classes are in a node.
- **Gini impurity**: $1 - \sum_k p_k^2$; zero for a pure node.
- **Entropy**: $-\sum_k p_k \log_2 p_k$; zero for a pure node, 1 bit for a 50/50 two-class node.
- **Information gain**: parent impurity minus the weighted impurity of the children.
- **Greedy recursive splitting**: pick the best split now, never look back.
- **Stopping rules**: maximum depth, minimum samples per split or leaf.
- **Pruning**: cutting back branches that do not help on validation data.
- **Overfitting**: a deep tree memorises the training set; trees have high variance.
- **Regression trees**: split to reduce squared error; a leaf predicts the mean of its targets.
- **Feature importance**: how much a feature reduced impurity across the tree.
- **No scaling needed**: splits depend on order, not size, of values.

## The math
For a node with class proportions $p_k$:

$$\text{Gini} = 1 - \sum_k p_k^2 \qquad H = -\sum_k p_k \log_2 p_k$$

The gain of a split sending $n_L$ and $n_R$ of $n$ samples left and right:

$$\text{Gain} = I(\text{parent}) - \frac{n_L}{n} I(\text{left}) - \frac{n_R}{n} I(\text{right})$$

where $I$ is Gini or entropy. The tree chooses the feature and threshold with the largest gain.

## Worked example
A node holds 10 samples: 6 "yes" and 4 "no". Its Gini is $1 - (0.6^2 + 0.4^2) = 0.48$. A candidate split sends 4 samples (all "yes") left and 6 samples (2 "yes", 4 "no") right. The left Gini is 0. The right Gini is $1 - ((1/3)^2 + (2/3)^2) = 0.444$. The weighted impurity is $0.4 \times 0 + 0.6 \times 0.444 = 0.267$, so the gain is $0.48 - 0.267 \approx 0.213$.

## Code
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0).fit(X_tr, y_tr)
print("test accuracy:", tree.score(X_te, y_te))
print(export_text(tree, feature_names=["sepal_l", "sepal_w", "petal_l", "petal_w"]))
```

## Common mistakes
- Letting the tree grow without limits and then trusting its training accuracy.
- Assuming feature importances are exact causes rather than usage statistics.
- Scaling features, which is unnecessary here.
- Treating a single tree as stable; small data changes can change the whole tree.
- Using the tree to extrapolate beyond the range of the training targets in regression.

## Summary
- A tree is a hierarchy of splits chosen greedily to maximise impurity reduction.
- Gini and entropy measure impurity; information gain scores a split.
- Control depth and leaf size, or prune, to prevent overfitting.
- Single trees are interpretable but unstable, which motivates ensembles.

## Questions
Q: What is the Gini impurity of a node containing only one class?
A) 0
B) 0.5
C) 1
D) Undefined
ANSWER: A
WHY: With $p = 1$ for one class, Gini is $1 - 1 = 0$, which means perfectly pure.

Q: What is the entropy, in bits, of a two-class node that is split 50/50?
A) 0
B) 0.5
C) 1
D) 2
ANSWER: C
WHY: $-0.5\log_2 0.5 - 0.5\log_2 0.5 = 1$ bit, the maximum for two classes.

Q: What does information gain measure?
A) The depth of the tree
B) The reduction in impurity achieved by a split
C) The number of leaves
D) The learning rate
ANSWER: B
WHY: Gain is the parent's impurity minus the weighted impurity of the children.

Q: Why is tree building called greedy?
A) It uses a large learning rate
B) It picks the best split at each node without revisiting earlier choices
C) It always picks the first feature
D) It trains many trees at once
ANSWER: B
WHY: Each split is chosen for its immediate gain, with no look-ahead or backtracking.

Q: Which setting most directly reduces overfitting in a tree?
A) Increasing the maximum depth
B) Decreasing the minimum samples per leaf
C) Limiting the maximum depth or increasing the minimum samples per leaf
D) Scaling the features
ANSWER: C
WHY: Constraining depth and leaf size stops the tree from carving out tiny groups that fit noise.

Q: Do decision trees need features to be scaled?
A) Yes, always
B) No, splits depend only on the order of values
C) Only for regression
D) Only for categorical features
ANSWER: B
WHY: A threshold split gives the same partition whatever the feature's units.

Q: What does a leaf of a regression tree predict?
A) The median class
B) The mean of the training targets in that leaf
C) The largest target
D) Zero
ANSWER: B
WHY: Predicting the mean minimises squared error within the leaf.

Q: Why are single decision trees described as high variance?
A) They have too few parameters
B) Small changes in the training data can produce a very different tree
C) They cannot overfit
D) They need scaling
ANSWER: B
WHY: Greedy splits near the top can change with small data shifts, which changes everything below.

Q: What is pruning?
A) Adding more features
B) Removing branches that do not improve validation performance
C) Scaling the leaves
D) Changing the loss to cross-entropy
ANSWER: B
WHY: Pruning simplifies the tree to improve generalisation.

Q: A node has 8 samples of class A and 2 of class B. Its Gini impurity is closest to what?
A) 0.32
B) 0.5
C) 0.8
D) 0
ANSWER: A
WHY: $1 - (0.8^2 + 0.2^2) = 1 - 0.68 = 0.32$.
