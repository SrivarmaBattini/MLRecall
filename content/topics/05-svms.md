---
id: svms
title: Support Vector Machines
group: Supervised Learning
difficulty: 3
order: 5
summary: Find the boundary that separates classes with the widest possible margin, using kernels for curved boundaries.
prerequisites: logistic-regression
---

## What it is and why it matters
A support vector machine (SVM) classifies by finding the separating boundary with the largest safety gap, the margin, between classes. With kernels it can draw curved boundaries without building features by hand. SVMs were the standard for small and medium datasets before deep learning and remain useful when data is limited.

## Intuition
Picture two groups of points on a table and a stick between them. Many positions separate the groups, but the best one stays as far as possible from both. Only the points nearest the stick matter; they are the support vectors. Move any other point and the stick does not budge.

## Key ideas
- **Hyperplane**: the boundary $w^\top x + b = 0$.
- **Margin**: the distance between the boundary and the closest points; it equals $2/\lVert w \rVert$ between the two margin lines.
- **Support vectors**: the training points lying on or inside the margin; they alone define the boundary.
- **Hard margin**: requires perfect separation; fails with noise or overlap.
- **Soft margin and slack variables**: $\xi_i$ allow some points to violate the margin.
- **Parameter C**: the penalty for violations. Large $C$ fits the training data tightly; small $C$ allows a wider, simpler margin.
- **Hinge loss**: $\max(0, 1 - y f(x))$; zero for points safely beyond the margin.
- **Kernel trick**: compute inner products in a high-dimensional space without building that space.
- **Common kernels**: linear, polynomial and RBF (Gaussian).
- **Gamma ($\gamma$)**: RBF width. Large $\gamma$ gives wiggly, overfit boundaries; small $\gamma$ gives smooth ones.
- **Feature scaling**: essential, because margins depend on distances.
- **Support vector regression (SVR)**: fits a tube of width $\varepsilon$ around the data.
- **Cost**: training scales poorly with the number of samples; probabilities need extra calibration.

## The math
The soft-margin problem is

$$\min_{w,b,\xi}\ \tfrac{1}{2}\lVert w \rVert^2 + C \sum_i \xi_i \quad \text{s.t.}\quad y_i(w^\top x_i + b) \ge 1 - \xi_i,\ \xi_i \ge 0$$

with labels $y_i \in \{-1, +1\}$. The RBF kernel is

$$K(x, x') = \exp\left(-\gamma \lVert x - x' \rVert^2\right)$$

and predictions use $f(x) = \sum_i \alpha_i y_i K(x_i, x) + b$, where only support vectors have $\alpha_i \ne 0$.

## Worked example
In one dimension, class $-1$ sits at $x = -1$ and class $+1$ at $x = +1$. The boundary $w = 1,\ b = 0$ gives $f(-1) = -1$ and $f(+1) = +1$, so $y f(x) = 1$ for both points: they lie exactly on the margin and both are support vectors. The margin width is $2/\lVert w \rVert = 2$, the distance between the two points.

## Code
```python
from sklearn.datasets import make_moons
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

X, y = make_moons(n_samples=400, noise=0.25, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, random_state=0)

svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale")).fit(X_tr, y_tr)
print("test accuracy:", svm.score(X_te, y_te))
print("support vectors:", svm[-1].n_support_)
```

## Common mistakes
- Forgetting to scale features before training.
- Using a very large $\gamma$ or $C$ and fitting the noise.
- Using SVMs on huge datasets where training becomes too slow.
- Treating the raw decision score as a calibrated probability.
- Choosing a kernel before trying a simple linear one.

## Summary
- SVMs maximise the margin; support vectors define the boundary.
- Soft margins with parameter $C$ handle noisy data.
- Kernels, especially RBF with $\gamma$, give non-linear boundaries.
- Scale features and tune $C$ and $\gamma$ by cross-validation.

## Questions
Q: What are support vectors?
A) All training points
B) The training points on or inside the margin that define the boundary
C) Points far from the boundary
D) The test points
ANSWER: B
WHY: Only points on or inside the margin influence the boundary; removing others changes nothing.

Q: If $\lVert w \rVert$ gets smaller, what happens to the margin $2/\lVert w \rVert$?
A) It gets wider
B) It gets narrower
C) It stays the same
D) It becomes zero
ANSWER: A
WHY: The margin is inversely proportional to $\lVert w \rVert$, so a smaller norm means a wider margin.

Q: What does a large value of $C$ do in a soft-margin SVM?
A) Tolerates many margin violations
B) Penalises violations heavily and fits training data more tightly
C) Removes the kernel
D) Makes the margin wider
ANSWER: B
WHY: $C$ weights the slack penalty, so a large $C$ pushes towards fewer violations and a narrower margin.

Q: What is the kernel trick?
A) Adding noise to the labels
B) Computing inner products in a high-dimensional feature space without constructing it
C) Reducing the number of classes
D) Removing support vectors
ANSWER: B
WHY: Kernels compute $K(x, x')$ directly, which equals the inner product in the implicit feature space.

Q: A very large $\gamma$ in the RBF kernel tends to produce what?
A) A smooth, almost linear boundary
B) A very wiggly boundary that overfits
C) An empty set of support vectors
D) A hard margin
ANSWER: B
WHY: Large $\gamma$ makes each point's influence very local, so the boundary hugs individual points.

Q: Why must features be scaled before an RBF SVM?
A) The kernel uses distances, so large-range features would dominate
B) Scaling removes support vectors
C) SVMs cannot handle negative values
D) It reduces the number of classes
ANSWER: A
WHY: The RBF kernel depends on Euclidean distance, which is dominated by features with large numeric ranges.

Q: What is the hinge loss for a point with $y f(x) = 2$?
A) 0
B) 1
C) 2
D) $-1$
ANSWER: A
WHY: $\max(0, 1 - 2) = 0$; points safely beyond the margin cost nothing.

Q: Which is a drawback of kernel SVMs?
A) They cannot make non-linear boundaries
B) Training becomes slow as the number of samples grows large
C) They need no hyperparameters
D) They cannot use labels
ANSWER: B
WHY: Kernel SVM training scales at least quadratically with the number of samples.

Q: What do slack variables $\xi_i$ allow?
A) Points to violate the margin, or even be misclassified, at a cost
B) Extra features
C) Faster kernels
D) Negative margins only
ANSWER: A
WHY: Slack lets the optimisation trade some margin violations for a simpler boundary.

Q: What does support vector regression (SVR) fit?
A) A hyperplane with the widest margin between classes
B) A tube of width $\varepsilon$ in which errors are ignored
C) A decision tree
D) Cluster centres
ANSWER: B
WHY: SVR uses an $\varepsilon$-insensitive loss, ignoring errors smaller than $\varepsilon$.
