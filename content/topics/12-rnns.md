---
id: rnns
title: Recurrent Neural Networks
group: Deep Learning
difficulty: 2
order: 12
summary: Process sequences step by step while carrying a memory called the hidden state.
prerequisites: backpropagation
---

## What it is and why it matters
A recurrent neural network (RNN) reads a sequence, such as words or sensor readings, one element at a time and keeps a hidden state that summarises what it has seen. This lets it handle inputs of any length and use order and context, which a plain feed-forward network cannot.

## Intuition
Read a sentence word by word with a notepad. After each word you update your notes using the new word and your previous notes. Your notes are the hidden state. The same note-taking habit, the same weights, is applied at every step.

## Key ideas
- **Sequence data**: text, audio, time series, where order matters.
- **Hidden state $h_t$**: the network's memory after step $t$.
- **Recurrence**: the new state depends on the current input and the previous state.
- **Weight sharing across time**: the same weights are used at every step.
- **Unrolling**: drawing the loop as a chain of copies, one per time step.
- **Backpropagation through time (BPTT)**: backprop across the unrolled chain.
- **Truncated BPTT**: backprop over limited windows to save compute.
- **Vanishing and exploding gradients**: repeated multiplication across many steps; RNNs struggle with long-range dependencies.
- **Gradient clipping**: caps the gradient norm to prevent explosions.
- **Input and output patterns**: many-to-one (sentiment), one-to-many (captioning), many-to-many (translation, tagging).
- **Bidirectional RNN**: reads the sequence forwards and backwards.
- **Teacher forcing**: feeding the true previous token during training of a generator.
- **Limits**: sequential processing is slow to parallelise; LSTMs, GRUs and transformers improve on it.

## The math
A basic RNN cell updates its state as

$$h_t = \tanh\left(W_x x_t + W_h h_{t-1} + b\right)$$

and may produce an output $y_t = W_y h_t + c$. Gradients flowing back $T$ steps include a product of $T$ Jacobians $\prod_t \partial h_t / \partial h_{t-1}$, which tends to shrink towards zero or blow up, depending on the size of $W_h$.

## Worked example
Use scalars: $W_x = 1$, $W_h = 0.5$, $b = 0$, $h_0 = 0$ and inputs $x = [1, 2]$. Step 1: $h_1 = \tanh(1 \cdot 1 + 0.5 \cdot 0) = \tanh(1) \approx 0.762$. Step 2: $h_2 = \tanh(1 \cdot 2 + 0.5 \cdot 0.762) = \tanh(2.381) \approx 0.983$. The final state depends on both inputs, and the earlier input's influence is carried through $h_1$.

## Code
```python
import torch
from torch import nn

rnn = nn.RNN(input_size=8, hidden_size=16, batch_first=True)
x = torch.randn(4, 20, 8)            # batch of 4 sequences, 20 steps, 8 features
outputs, h_last = rnn(x)
print(outputs.shape)                  # (4, 20, 16): hidden state at every step
print(h_last.shape)                   # (1, 4, 16): final hidden state
```

## Common mistakes
- Expecting a vanilla RNN to remember information from hundreds of steps ago.
- Forgetting gradient clipping when training on long sequences.
- Mixing up tensor layouts (batch first or sequence first).
- Using the final hidden state for variable-length batches without handling padding.
- Using teacher forcing in training but never checking free-running generation.

## Summary
- An RNN applies the same cell at each step and carries a hidden state forward.
- Training uses backpropagation through time.
- Plain RNNs suffer from vanishing and exploding gradients, so long-range memory is weak.
- Gated variants (LSTM, GRU) and transformers address this limitation.

## Questions
Q: What is the hidden state of an RNN?
A) A summary of the sequence seen so far, passed to the next step
B) The loss value
C) The batch size
D) The output label
ANSWER: A
WHY: The hidden state carries information from previous steps forward.

Q: Which statement about RNN weights is true?
A) The same weights are used at every time step
B) Each step has different weights
C) Weights are not learned
D) Only the output has weights
ANSWER: A
WHY: Weight sharing across time lets the model handle any sequence length.

Q: What is backpropagation through time?
A) Backprop applied across the unrolled sequence of steps
B) Training with time-stamped labels
C) A kind of pooling
D) Predicting the future only
ANSWER: A
WHY: The recurrent loop is unrolled into a chain, and gradients flow back through every step.

Q: Why do vanilla RNNs struggle with long-range dependencies?
A) Gradients vanish or explode across many steps
B) They have no hidden state
C) They cannot take sequences
D) They use too many labels
ANSWER: A
WHY: Repeated multiplication across many steps shrinks or inflates the gradient signal.

Q: What does gradient clipping do?
A) Limits the gradient norm to prevent exploding gradients
B) Adds extra layers
C) Removes the hidden state
D) Shuffles the sequence
ANSWER: A
WHY: It rescales gradients that exceed a threshold so updates stay stable.

Q: A sentiment classifier reading a sentence and outputting one label is which pattern?
A) Many-to-one
B) One-to-many
C) One-to-one
D) Many-to-many with alignment
ANSWER: A
WHY: Many inputs (words) map to a single output (the sentiment).

Q: What does a bidirectional RNN do?
A) Reads the sequence in both directions and combines the states
B) Uses two loss functions
C) Doubles the learning rate
D) Reverses the labels
ANSWER: A
WHY: Combining past and future context helps tasks where the whole sequence is available.

Q: What is teacher forcing?
A) Feeding the ground-truth previous token as input during training
B) Forcing the learning rate down
C) Using a teacher network
D) Skipping validation
ANSWER: A
WHY: It speeds up learning but can mismatch generation, where the model must use its own previous outputs.

Q: Why are RNNs slow to train on long sequences compared with transformers?
A) Each step depends on the previous one, so steps cannot be parallelised
B) They have too few parameters
C) They cannot use GPUs
D) They need no gradients
ANSWER: A
WHY: The sequential dependency prevents processing all positions at once.

Q: With $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$, what does $W_h$ control?
A) How much of the previous hidden state influences the new one
B) The output class
C) The batch size
D) The input length
ANSWER: A
WHY: $W_h$ transforms the previous state before it is combined with the new input.
