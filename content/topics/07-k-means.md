---
id: k-means
title: K-Means Clustering
group: Unsupervised Learning
difficulty: 1
order: 7
summary: Group unlabelled data into K clusters by repeatedly assigning points to the nearest centre.
prerequisites:
---

## What it is and why it matters
K-means finds groups in data that has no labels, for example customer segments or image colour palettes. It is simple, fast and widely used as a first look at structure in data.

## Intuition
Scatter $K$ flags on a map. Every house walks to the nearest flag. Each flag then moves to the centre of the houses that chose it. Houses re-choose, flags move again, and the process repeats until nothing changes. The flags end up at the centres of natural groups.

## Key ideas
- **Unsupervised learning**: no target labels; we look for structure.
- **Cluster and centroid**: a group of points and its mean position.
- **K**: the number of clusters, chosen by you.
- **Assignment step**: give each point to its nearest centroid, usually by Euclidean distance.
- **Update step**: move each centroid to the mean of its assigned points.
- **Lloyd's algorithm**: alternate the two steps until assignments stop changing.
- **Inertia**: the total squared distance from points to their centroids (within-cluster sum of squares); lower means tighter clusters.
- **Local minima and initialisation**: results depend on the start. k-means++ spreads the starting centroids; `n_init` reruns with several starts.
- **Choosing K**: the elbow method (inertia versus K) and the silhouette score.
- **Feature scaling**: needed, because distance is used.
- **Assumptions and limits**: expects round, similar-sized clusters; sensitive to outliers; gives hard assignments.

## The math
The objective is

$$J = \sum_{k=1}^{K} \sum_{x \in C_k} \lVert x - \mu_k \rVert^2$$

The assignment step sets $c(x) = \arg\min_k \lVert x - \mu_k \rVert$ and the update step sets $\mu_k = \frac{1}{|C_k|}\sum_{x \in C_k} x$. Each step cannot increase $J$, so the algorithm converges, though possibly to a local minimum. The silhouette of a point is $(b - a)/\max(a, b)$, where $a$ is its mean distance to its own cluster and $b$ to the nearest other cluster.

## Worked example
Points on a line: $1, 2, 9, 10$ with $K = 2$ and initial centroids $1$ and $9$. Assign: $\{1, 2\}$ go to the first centroid and $\{9, 10\}$ to the second. Update: the centroids become $1.5$ and $9.5$. Assigning again changes nothing, so the algorithm has converged. Inertia is $4 \times 0.5^2 = 1.0$.

## Code
```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=0)
X = StandardScaler().fit_transform(X)
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    print(k, "inertia:", round(km.inertia_, 1), "silhouette:", round(silhouette_score(X, km.labels_), 3))
```

## Common mistakes
- Not scaling features, so one large-range feature decides the clusters.
- Running once with one random start and trusting the result.
- Picking $K$ by inertia alone; inertia always falls as $K$ rises.
- Expecting k-means to find elongated, ring-shaped or very different-sized clusters.
- Leaving outliers in, which drag centroids away.

## Summary
- K-means alternates between assigning points and updating centroids.
- It minimises within-cluster squared distance and converges to a local minimum.
- Use k-means++ and several restarts; choose $K$ with elbow and silhouette.
- Scale the data, and remember it assumes compact, round clusters.

## Questions
Q: What does the assignment step of k-means do?
A) Moves centroids to the cluster means
B) Gives each point to its nearest centroid
C) Chooses K
D) Removes outliers
ANSWER: B
WHY: Every point joins the cluster whose centroid is closest.

Q: What does the update step do?
A) Moves each centroid to the mean of its assigned points
B) Splits the largest cluster
C) Changes the distance metric
D) Randomises the labels
ANSWER: A
WHY: The mean minimises the sum of squared distances within a cluster.

Q: What is inertia?
A) The number of clusters
B) The sum of squared distances from points to their assigned centroids
C) The number of iterations
D) The largest cluster size
ANSWER: B
WHY: Inertia is the k-means objective, the within-cluster sum of squares.

Q: Why does inertia alone make a poor choice of K?
A) It always decreases as K increases
B) It is always zero
C) It cannot be computed
D) It increases with K
ANSWER: A
WHY: More clusters can always fit the data at least as tightly, down to zero when each point is its own cluster.

Q: What is the purpose of k-means++ initialisation?
A) To choose K automatically
B) To spread out the starting centroids for better, more stable results
C) To remove outliers
D) To scale the data
ANSWER: B
WHY: Well-spread starting centroids reduce the chance of poor local minima.

Q: Which cluster shape suits k-means best?
A) Long thin curves
B) Concentric rings
C) Compact, roughly round clusters of similar size
D) Nested spirals
ANSWER: C
WHY: Using distance to a mean implicitly assumes roundish clusters.

Q: What does a silhouette score near 1 indicate?
A) Points are well matched to their own cluster and far from others
B) Points are in the wrong clusters
C) K is too small
D) Clusters overlap heavily
ANSWER: A
WHY: A value near 1 means a point is much closer to its own cluster than to the nearest other.

Q: Why scale features before k-means?
A) The algorithm uses distances, so large-range features would dominate
B) To make K larger
C) To add labels
D) To speed up plotting
ANSWER: A
WHY: Euclidean distance is dominated by features with large numeric ranges.

Q: Does k-means always converge to the global optimum?
A) Yes
B) No, only to a local minimum that depends on initialisation
C) Only if K is 2
D) Only with scaling
ANSWER: B
WHY: The objective is non-convex, so different starts can end in different solutions.

Q: What is the elbow method?
A) Plotting inertia against K and looking for where the improvement slows sharply
B) Removing the weakest cluster
C) Taking the largest K
D) Measuring the angle between centroids
ANSWER: A
WHY: The bend in the curve suggests a K beyond which extra clusters add little.
