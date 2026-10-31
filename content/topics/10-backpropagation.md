---
id: backpropagation
title: Backpropagation
group: Deep Learning
difficulty: 3
order: 10
summary: Use the chain rule to compute how the loss changes with every weight, so a network can learn.
prerequisites: neural-networks-intro
---

## What it is and why it matters
Backpropagation is the algorithm that computes the gradient of the loss with respect to every weight in a neural network, efficiently, in one backward sweep. The gradients tell the optimiser how to change each weight. Without it, training deep networks would be impractical.

## Intuition
When a network gets an answer wrong, blame must be shared out. The output layer knows how wrong it was. It tells the layer before it how much each of its outputs contributed to the error, and that layer passes blame further back. Each weight receives a share proportional to its influence. The chain rule of calculus is the bookkeeping for this.

## Key ideas
- **Forward pass**: compute and store every intermediate value from input to loss.
- **Computational graph**: the network as a graph of simple operations.
- **Chain rule**: the derivative of a composition is the product of local derivatives.
- **Local gradient**: the derivative of one operation's output with respect to its input.
- **Backward pass**: multiply gradients backwards through the graph, reusing earlier results.
- **Gradient of a weight**: $\partial L / \partial w$, the sensitivity of the loss to that weight.
- **Weight update**: $w \leftarrow w - \eta \, \partial L / \partial w$.
- **Automatic differentiation**: frameworks like PyTorch build the graph and run backprop for you.
- **Vanishing gradients**: gradients shrink through many layers (for example sigmoid derivatives below 0.25), so early layers learn slowly.
- **Exploding gradients**: gradients grow; fixed with gradient clipping and good initialisation.
- **Fixes**: ReLU activations, careful initialisation, normalisation layers and residual connections.
- **Gradient checking**: comparing backprop with a numerical estimate to catch bugs.

## The math
For a single sigmoid neuron with $z = wx + b$, $\hat{y} = \sigma(z)$ and loss $L = \tfrac{1}{2}(\hat{y} - y)^2$, the chain rule gives

$$\frac{\partial L}{\partial w} = \underbrace{(\hat{y} - y)}_{\partial L / \partial \hat{y}} \cdot \underbrace{\sigma(z)\left(1 - \sigma(z)\right)}_{\partial \hat{y} / \partial z} \cdot \underbrace{x}_{\partial z / \partial w}$$

and $\partial L / \partial b$ is the same without the final factor $x$. In a deep network the gradient at layer $l$ is $\delta^{(l)} = \left(W^{(l+1)\top} \delta^{(l+1)}\right) \odot f'(z^{(l)})$.

## Worked example
Take $x = 1$, $w = 0.5$, $b = 0$ and target $y = 1$. Forward: $z = 0.5$, $\hat{y} = \sigma(0.5) \approx 0.6225$, loss $\approx 0.0713$. Backward: $\partial L / \partial \hat{y} = -0.3775$; $\sigma'(z) = 0.6225 \times 0.3775 \approx 0.2350$; so $\partial L / \partial z \approx -0.0887$. Therefore $\partial L / \partial w \approx -0.0887$ and $\partial L / \partial b \approx -0.0887$. With learning rate $0.1$, the new weight is $0.5 + 0.00887 \approx 0.509$, which moves the output towards the target.

## Code
```python
import torch

x = torch.tensor(1.0)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(1.0)

y_hat = torch.sigmoid(w * x + b)
loss = 0.5 * (y_hat - y) ** 2
loss.backward()                      # backpropagation
print("dL/dw:", w.grad.item(), "dL/db:", b.grad.item())   # about -0.0887 each
```

## Common mistakes
- Forgetting to zero gradients between steps, so they accumulate.
- Using sigmoid or tanh in a very deep network and getting vanishing gradients.
- Confusing backpropagation (computing gradients) with gradient descent (using them).
- Building operations outside the framework graph, so gradients do not flow.
- Ignoring exploding gradients in recurrent networks; clip them.

## Summary
- Backpropagation applies the chain rule backwards through the computational graph.
- It reuses forward-pass values to get every weight's gradient efficiently.
- The optimiser then uses those gradients to update the weights.
- Vanishing and exploding gradients are the main practical problems, tackled with ReLU, normalisation, residual connections and clipping.

## Questions
Q: Which mathematical rule is at the heart of backpropagation?
A) The chain rule
B) Bayes' rule
C) The central limit theorem
D) The triangle inequality
ANSWER: A
WHY: Backprop multiplies local derivatives along the path from the loss to each weight.

Q: What does the backward pass compute?
A) Gradients of the loss with respect to parameters
B) The next input batch
C) The accuracy
D) The validation split
ANSWER: A
WHY: It produces $\partial L / \partial \theta$ for each parameter.

Q: What is the difference between backpropagation and gradient descent?
A) Backprop computes gradients; gradient descent uses them to update weights
B) They are identical
C) Backprop updates weights directly without gradients
D) Gradient descent only works on trees
ANSWER: A
WHY: Backprop is the gradient computation; the optimiser step is separate.

Q: Why does the forward pass store intermediate values?
A) They are needed to compute local gradients during the backward pass
B) To increase accuracy
C) To save the data
D) They are not used
ANSWER: A
WHY: Local derivatives like $\sigma'(z)$ depend on the activations from the forward pass.

Q: What is the vanishing-gradient problem?
A) Gradients become very small in early layers, so they learn slowly
B) Gradients become infinite
C) The loss becomes negative
D) The data vanishes
ANSWER: A
WHY: Multiplying many small derivatives shrinks the signal as it travels backwards.

Q: Which activation helps reduce vanishing gradients compared with sigmoid?
A) ReLU
B) A constant function
C) Step function
D) Square root
ANSWER: A
WHY: ReLU has derivative 1 for positive inputs, so gradients do not shrink there.

Q: What does gradient clipping address?
A) Exploding gradients
B) Overfitting directly
C) Missing labels
D) Slow data loading
ANSWER: A
WHY: Clipping limits the gradient norm so a huge gradient cannot wreck the weights.

Q: In PyTorch, what does `loss.backward()` do?
A) Computes gradients and stores them in `.grad`
B) Updates the weights
C) Resets the model
D) Evaluates the validation set
ANSWER: A
WHY: It runs backprop; updating weights is done by `optimizer.step()`.

Q: Why call `optimizer.zero_grad()` each step?
A) Gradients accumulate by default and would mix across steps
B) To reset the labels
C) To lower the learning rate
D) To save memory only
ANSWER: A
WHY: PyTorch adds new gradients onto the old ones unless they are cleared.

Q: What is gradient checking?
A) Comparing backprop gradients with numerical finite-difference estimates
B) Checking gradients are positive
C) Removing gradients
D) Sorting gradients
ANSWER: A
WHY: A numerical estimate of the derivative gives an independent check for implementation bugs.
