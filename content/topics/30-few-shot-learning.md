---
id: few-shot-learning
title: Few-shot Learning
group: Transfer and Few-shot Learning
difficulty: 3
order: 30
summary: Learn to recognise new classes from only a handful of labelled examples by learning how to learn.
prerequisites: transfer-learning, word-embeddings
---

## What it is and why it matters
Few-shot learning trains models that can pick up a new concept from just a few labelled examples, sometimes one. This matters when labelled data is expensive or rare, such as unusual medical conditions, new product categories or niche languages. People do this easily; standard deep networks need thousands of examples per class.

## Intuition
Show a child one photo of a pangolin and they will recognise another pangolin later, because they already understand animals, shapes and textures. Few-shot learning builds that background ability first, by training across many small tasks, so that a new task only needs a quick adjustment or a comparison with a few examples.

## Key ideas
- **N-way K-shot**: a task with $N$ classes and $K$ labelled examples per class.
- **Support set**: the few labelled examples given for a new task.
- **Query set**: the unlabelled examples to be classified.
- **Meta-learning ("learning to learn")**: train across many tasks so the model adapts quickly to new ones.
- **Episodic training**: each training step simulates a few-shot task, with its own support and query sets.
- **Metric-based methods**: learn an embedding in which same-class examples are close.
- **Siamese networks**: compare pairs of examples.
- **Matching networks**: classify queries by attention over support examples.
- **Prototypical networks**: represent each class by the mean embedding (prototype) of its support examples; classify by the nearest prototype.
- **Optimisation-based methods**: MAML learns initial weights that adapt well after a few gradient steps.
- **Few-shot with pretrained models**: fine-tuning with small data, or in-context examples shown in the prompt of a large language model.
- **Data augmentation**: expands tiny support sets.
- **Risks**: overfitting to the support set, and sensitivity to which examples were chosen.

## The math
A prototypical network embeds examples with a network $f_\phi$. The prototype of class $k$ from its support set $S_k$ is

$$c_k = \frac{1}{|S_k|} \sum_{x_i \in S_k} f_\phi(x_i)$$

and a query $x$ is classified by

$$p(y = k \mid x) = \frac{\exp\left(-d\left(f_\phi(x), c_k\right)\right)}{\sum_{j} \exp\left(-d\left(f_\phi(x), c_j\right)\right)}$$

where $d$ is usually squared Euclidean distance. Training minimises the cross-entropy of query labels over many episodes.

## Worked example
A 2-way 2-shot task in a one-dimensional embedding: class A support embeddings are $\{1, 3\}$ and class B are $\{9, 11\}$. The prototypes are $c_A = 2$ and $c_B = 10$. A query embeds to $4$. The distances are $|4 - 2| = 2$ and $|4 - 10| = 6$. Using plain distance in the softmax, the weights are $e^{-2} = 0.135$ and $e^{-6} = 0.0025$, giving $p(A) \approx 0.982$. The query is classified as A.

## Code
```python
import torch

def prototype_predict(support, support_labels, query, n_classes):
    # support: (n_support, dim), query: (n_query, dim) - outputs of an embedding network
    protos = torch.stack([support[support_labels == k].mean(0) for k in range(n_classes)])
    dists = torch.cdist(query, protos) ** 2          # squared Euclidean distances
    return torch.softmax(-dists, dim=1)              # class probabilities for each query

support = torch.tensor([[1.0], [3.0], [9.0], [11.0]])
labels = torch.tensor([0, 0, 1, 1])
print(prototype_predict(support, labels, torch.tensor([[4.0]]), 2))
```

## Common mistakes
- Training on ordinary batches and expecting the model to do well on few-shot tasks; use episodic training.
- Evaluating on classes that also appear in training.
- Choosing a support set that is unrepresentative of the class.
- Reporting results from a single episode instead of averaging many.
- Ignoring that embedding quality drives everything in metric-based methods.

## Summary
- Few-shot learning classifies new classes from a handful of examples ($N$-way $K$-shot).
- Meta-learning trains across many episodes so adaptation is quick.
- Prototypical networks classify by distance to class mean embeddings; MAML learns an easily adaptable initialisation.
- Pretrained and in-context approaches are practical alternatives with large models.

## Questions
Q: What does "5-way 1-shot" mean?
A) Five classes with one labelled example per class
B) One class with five examples
C) Five shots of training data in total
D) Five layers and one head
ANSWER: A
WHY: $N$ is the number of classes and $K$ the labelled examples per class.

Q: What is the support set?
A) The few labelled examples provided for a new task
B) The test set of unlabelled data
C) The pretrained weights
D) The loss
ANSWER: A
WHY: It defines the new classes for the model.

Q: What is the query set?
A) The examples to be classified using the support set
B) The labelled examples
C) The classes
D) The embedding
ANSWER: A
WHY: Queries are the unlabelled examples whose labels we predict.

Q: What is meta-learning?
A) Training across many tasks so the model can adapt quickly to new ones
B) Training on one big task only
C) Compressing a model
D) Removing labels
ANSWER: A
WHY: It is "learning to learn".

Q: How does a prototypical network represent a class?
A) By the mean embedding of its support examples
B) By its largest example
C) By a random example
D) By a decision tree
ANSWER: A
WHY: The prototype is the average of the class's embedded support points.

Q: How does a prototypical network classify a query?
A) By softmax over negative distances to the class prototypes
B) By majority vote of random trees
C) By clustering the query alone
D) By its pixel count
ANSWER: A
WHY: Closer prototypes receive higher probabilities.

Q: What does MAML learn?
A) An initialisation that adapts well to new tasks after a few gradient steps
B) A distance metric only
C) A tokenizer
D) A kernel
ANSWER: A
WHY: Model-agnostic meta-learning optimises for fast adaptation.

Q: Why use episodic training?
A) It matches the few-shot test conditions during training
B) It increases dataset size
C) It removes the embedding
D) It avoids labels
ANSWER: A
WHY: Each episode mimics a small support-query task, so the model practises the skill it will use.

Q: Support embeddings A: $\{1, 3\}$ and B: $\{9, 11\}$; a query embeds to 4. Which class is predicted?
A) A
B) B
C) Tie
D) Neither
ANSWER: A
WHY: The query is $2$ from prototype A at $2$ and $6$ from prototype B at $10$.

Q: What is a risk of few-shot learning?
A) Overfitting to the few support examples or an unrepresentative support set
B) Having too much data
C) Having no classes
D) Having too many labels
ANSWER: A
WHY: With so little data, a poor choice of examples can mislead the model.
