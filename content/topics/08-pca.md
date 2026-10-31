---
id: pca
title: Principal Component Analysis
group: Unsupervised Learning
difficulty: 2
order: 8
summary: Compress many correlated features into a few directions that keep most of the variation.
prerequisites: linear-regression
---

## What it is and why it matters
Principal component analysis (PCA) rewrites your data along new axes ordered by how much the data varies along them. Keeping only the first few axes reduces dimensionality, speeds up models, removes noise and lets you plot high-dimensional data in two dimensions.

## Intuition
A cloud of points shaped like a long thin cigar is mostly described by one direction, its length. PCA finds that direction first, then the next most spread-out direction at right angles to it, and so on. If the cigar is thin enough you can drop the thin directions and lose little.

## Key ideas
- **Variance**: the spread of the data; PCA treats high variance as information.
- **Centering and standardising**: subtract the mean, and usually divide by the standard deviation so no feature dominates.
- **Covariance matrix**: records how pairs of features vary together.
- **Principal components**: orthogonal directions of maximum variance; they are the eigenvectors of the covariance matrix.
- **Eigenvalues**: the variance along each component.
- **Explained variance ratio**: each component's eigenvalue divided by the total.
- **Scree plot and cumulative variance**: tools to choose how many components to keep.
- **Projection**: $Z = X V_k$ gives the reduced data.
- **Reconstruction error**: what is lost when you map back from $k$ components.
- **SVD**: in practice PCA is computed with the singular value decomposition.
- **Loadings**: the weights showing how original features form each component.
- **Limits**: linear only; components can be hard to interpret; sensitive to scaling; variance is not always relevance.
- **Curse of dimensionality**: in many dimensions data becomes sparse and distances lose meaning, which is one reason to reduce dimensions.

## The math
For centred data $X$ with $n$ rows, the covariance matrix is

$$C = \frac{1}{n-1} X^\top X$$

Solve $C v_i = \lambda_i v_i$ with eigenvalues sorted $\lambda_1 \ge \lambda_2 \ge \dots$. The first $k$ eigenvectors form $V_k$ and the reduced data is $Z = X V_k$. The explained variance ratio of component $i$ is

$$\frac{\lambda_i}{\sum_j \lambda_j}$$

## Worked example
Suppose two standardised features have covariance matrix $\begin{pmatrix} 3 & 1 \\ 1 & 3 \end{pmatrix}$. Its eigenvalues are $4$ and $2$, with eigenvectors $(1,1)/\sqrt{2}$ and $(1,-1)/\sqrt{2}$. The first component explains $4/(4+2) \approx 66.7\%$ of the variance and the second $33.3\%$. Keeping only the first component compresses two features to one while retaining two thirds of the variation.

## Code
```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)           # 64 features
pca = make_pipeline(StandardScaler(), PCA(n_components=0.95)).fit(X)
Z = pca.transform(X)
print("kept components:", Z.shape[1])
print("explained variance:", pca[-1].explained_variance_ratio_[:5])
```

## Common mistakes
- Skipping standardisation when features have different units.
- Fitting PCA on the full dataset before splitting, which leaks test information.
- Assuming high-variance directions are the most useful for prediction.
- Interpreting components as if each were a single original feature.
- Using PCA when the structure is strongly non-linear.

## Summary
- PCA finds orthogonal directions of maximum variance (eigenvectors of the covariance matrix).
- Eigenvalues give the variance explained; keep enough components to reach a target.
- Standardise first, and fit only on training data.
- It is linear and unsupervised, so it may discard variation that matters for the target.

## Questions
Q: What does the first principal component represent?
A) The direction of maximum variance in the data
B) The direction of minimum variance
C) The mean of the data
D) The label direction
ANSWER: A
WHY: PCA orders directions by the variance of the data along them, largest first.

Q: Principal components are what mathematically?
A) Eigenvectors of the covariance matrix
B) Rows of the raw data
C) Cluster centres
D) Residuals of a regression
ANSWER: A
WHY: Each component is an eigenvector of the covariance matrix, with its eigenvalue as the variance.

Q: What does an eigenvalue of a component tell you?
A) The amount of variance along that component
B) The number of features
C) The label
D) The learning rate
ANSWER: A
WHY: Larger eigenvalues correspond to directions with more spread.

Q: Why standardise features before PCA?
A) Without it, features with large units dominate the variance
B) To add features
C) To increase the number of components
D) To avoid centring
ANSWER: A
WHY: PCA chases variance, which depends on the units of each feature.

Q: Components with eigenvalues 5, 3 and 2: what share of variance does the first explain?
A) 20%
B) 30%
C) 50%
D) 100%
ANSWER: C
WHY: $5/(5+3+2) = 0.5$.

Q: Why fit PCA only on the training set?
A) To avoid leaking information from the test set
B) To make it faster
C) To get more components
D) PCA needs labels
ANSWER: A
WHY: Computing components from test data lets test information influence the preprocessing.

Q: What is a scree plot used for?
A) Choosing how many components to keep by looking at eigenvalues
B) Plotting labels
C) Measuring accuracy
D) Choosing the learning rate
ANSWER: A
WHY: The bend in the eigenvalues, or the cumulative variance curve, suggests where extra components stop adding much.

Q: Which is a limitation of PCA?
A) It captures only linear structure
B) It requires labels
C) It always increases dimensionality
D) It cannot be computed
ANSWER: A
WHY: PCA uses linear projections, so curved manifolds may not be captured well.

Q: Reconstruction error after keeping $k$ components measures what?
A) The information lost by dropping the other components
B) The number of clusters
C) The accuracy of a classifier
D) The learning rate
ANSWER: A
WHY: Mapping back from $k$ components approximates the original data, and the gap is the loss.

Q: How are principal components related to each other?
A) They are orthogonal
B) They are identical
C) They are parallel
D) They are random
ANSWER: A
WHY: Eigenvectors of a symmetric covariance matrix are mutually perpendicular.
