---
id: neural-networks-intro
title: Neural Networks Intro
group: Deep Learning
difficulty: 2
order: 9
summary: Stack layers of simple units with non-linear activations to learn complex functions from data.
prerequisites: logistic-regression
---

## What it is and why it matters
A neural network is a function built from layers of small units called neurons. Each neuron computes a weighted sum and passes it through a non-linear activation. Stacking layers lets the network learn very complicated patterns, which is why neural networks power image, speech and language systems.

## Intuition
One neuron is like logistic regression: a weighted vote followed by a squashing function. A layer is many such votes in parallel, each noticing a different pattern. The next layer combines those patterns into bigger ones. Early layers might detect edges, later layers shapes, and the last layer decides.

## Key ideas
- **Neuron**: computes $a = f(w^\top x + b)$.
- **Layers**: input layer, one or more hidden layers and an output layer; "deep" means many hidden layers.
- **Weights and biases**: the learnable parameters.
- **Activation functions**: sigmoid, tanh, ReLU ($\max(0, z)$) and softmax for outputs. Without non-linearity, stacked layers collapse into one linear map.
- **Forward pass**: computing the output layer by layer.
- **Loss function**: mean squared error for regression, cross-entropy for classification.
- **Universal approximation**: a wide enough network with a non-linear activation can approximate any reasonable function.
- **Training**: adjust weights to reduce the loss, using gradients from backpropagation.
- **Epoch, batch and mini-batch**: one pass over the data, the examples used per update, and stochastic gradient descent on small batches.
- **Optimisers**: SGD, momentum and Adam.
- **Initialisation**: random starting weights with sensible scale (Xavier, He); all zeros would make neurons identical.
- **Regularisation**: dropout, weight decay (L2), early stopping and data augmentation.
- **Hyperparameters**: learning rate, depth, width, batch size.

## The math
A layer computes $a^{(l)} = f\left(W^{(l)} a^{(l-1)} + b^{(l)}\right)$ starting from $a^{(0)} = x$. Common activations:

$$\text{ReLU}(z) = \max(0, z) \qquad \sigma(z) = \frac{1}{1 + e^{-z}} \qquad \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

Cross-entropy for a class label $y$ with predicted probabilities $p$ is $-\log p_y$. Mini-batch gradient descent updates $\theta \leftarrow \theta - \eta \nabla_\theta L$.

## Worked example
One neuron has $w = [0.5, -1]$, $b = 0.5$ and input $x = [2, 1]$. The weighted sum is $z = 0.5 \times 2 - 1 \times 1 + 0.5 = 0.5$. With ReLU the output is $0.5$. With a sigmoid the output is $\sigma(0.5) \approx 0.622$.

## Code
```python
import torch
from torch import nn

X = torch.randn(512, 10)
y = (X[:, 0] + X[:, 1] > 0).long()

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 2))
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(50):
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()
print("final loss:", loss.item())
```

## Common mistakes
- Using no activation between layers, which keeps the model linear.
- Setting the learning rate too high or too low and blaming the architecture.
- Initialising all weights to the same value.
- Not shuffling data or not normalising inputs.
- Training too long without a validation set and overfitting.

## Summary
- Neurons compute a weighted sum plus a non-linear activation; layers stack them.
- Training minimises a loss with mini-batch gradient descent and optimisers like Adam.
- Regularisation (dropout, weight decay, early stopping) controls overfitting.
- Good initialisation, scaled inputs and a sensible learning rate matter a lot.

## Questions
Q: Why are non-linear activation functions needed?
A) Without them stacked layers reduce to a single linear transformation
B) They make training slower
C) They remove the need for a loss
D) They reduce the number of weights
ANSWER: A
WHY: A composition of linear maps is linear, so depth adds no power without non-linearity.

Q: What is ReLU?
A) $\max(0, z)$
B) $1/(1+e^{-z})$
C) $\tanh(z)$
D) $z^2$
ANSWER: A
WHY: ReLU outputs zero for negative inputs and the input itself for positive inputs.

Q: Which output activation suits multiclass classification?
A) ReLU
B) Softmax
C) Identity
D) Tanh
ANSWER: B
WHY: Softmax turns scores into probabilities that sum to one across classes.

Q: Which loss is standard for classification networks?
A) Cross-entropy
B) Hinge on weights
C) Absolute weights
D) Learning rate
ANSWER: A
WHY: Cross-entropy compares predicted class probabilities with the true class.

Q: What is a mini-batch?
A) A small subset of training examples used for one gradient update
B) The whole dataset
C) A single weight
D) A type of layer
ANSWER: A
WHY: Mini-batch SGD estimates the gradient from a small sample, which is faster and adds helpful noise.

Q: What does dropout do during training?
A) Randomly zeroes some activations to reduce overfitting
B) Drops the learning rate
C) Removes the output layer
D) Shuffles the labels
ANSWER: A
WHY: Randomly dropping units stops neurons from relying on each other too much.

Q: Why not initialise all weights to zero?
A) Neurons in a layer would stay identical and learn the same thing
B) It makes the loss infinite
C) It increases accuracy
D) It removes biases
ANSWER: A
WHY: Identical weights receive identical gradients, so symmetry is never broken.

Q: What is one epoch?
A) One pass through the whole training set
B) One weight update
C) One layer
D) One batch
ANSWER: A
WHY: An epoch means every training example has been used once.

Q: Which optimiser adapts a learning rate per parameter using running averages of gradients?
A) Adam
B) Plain gradient descent
C) Random search
D) Normal equation
ANSWER: A
WHY: Adam keeps moving averages of gradients and squared gradients to scale each update.

Q: Training loss keeps falling but validation loss rises. What is happening?
A) Overfitting
B) Underfitting
C) A perfect fit
D) The learning rate is zero
ANSWER: A
WHY: The network is memorising training data and generalising worse, so early stopping or regularisation helps.
