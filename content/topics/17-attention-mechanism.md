---
id: attention-mechanism
title: Attention Mechanism
group: Transformers and Language
difficulty: 2
order: 17
summary: Let a model look back at all input positions and focus on the ones that matter for the current output.
prerequisites: rnns, word-embeddings
---

## What it is and why it matters
Attention lets a model decide, at every step, which parts of the input to focus on. It was introduced to fix a flaw in sequence-to-sequence models, which squeezed a whole sentence into one fixed-size vector. With attention the decoder can look back at every input word and weight them by relevance.

## Intuition
When translating a long sentence, you do not memorise the whole thing and then write. You glance back at the part of the source that corresponds to the word you are writing now. Attention automates that glance: it scores every source position, turns the scores into weights that add up to 1, and takes a weighted average of the source information.

## Key ideas
- **Bottleneck problem**: a single fixed vector cannot hold a long sentence.
- **Query**: what we are looking for (the decoder's current state).
- **Keys**: labels describing each source position.
- **Values**: the information stored at each position.
- **Alignment score**: how well a query matches each key.
- **Softmax weights**: scores turned into attention weights that sum to 1.
- **Context vector**: the weighted sum of values, fed to the decoder.
- **Additive (Bahdanau) attention**: scores from a small neural network.
- **Multiplicative (Luong) or dot-product attention**: scores from dot products.
- **Soft versus hard attention**: soft is differentiable and weights everything; hard picks one position.
- **Attention maps**: weights can be visualised to see what the model focused on, though they are not a full explanation.
- **Cross-attention**: queries come from one sequence and keys and values from another.

## The math
For a query $q$ and source states $h_1, \dots, h_T$:

$$e_j = \text{score}(q, h_j) \qquad \alpha_j = \frac{\exp(e_j)}{\sum_{k=1}^{T} \exp(e_k)} \qquad c = \sum_{j=1}^{T} \alpha_j h_j$$

Dot-product scoring uses $e_j = q^\top h_j$. Additive scoring uses $e_j = v^\top \tanh(W_1 q + W_2 h_j)$.

## Worked example
Suppose the scores for three source words are $e = [2, 1, 0]$. Exponentiating gives $[7.389, 2.718, 1]$ with sum $11.107$. The attention weights are $[0.665, 0.245, 0.090]$. The context vector is $0.665\, h_1 + 0.245\, h_2 + 0.090\, h_3$, dominated by the first word. If the three values were $[10, 20, 30]$ the context would be $0.665 \times 10 + 0.245 \times 20 + 0.09 \times 30 \approx 14.25$.

## Code
```python
import numpy as np

def attention(query, keys, values):
    scores = keys @ query                      # one score per source position
    weights = np.exp(scores - scores.max())
    weights = weights / weights.sum()          # softmax
    return weights @ values, weights

keys = values = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
context, w = attention(np.array([2.0, 0.5]), keys, values)
print("weights:", w.round(3), "context:", context.round(3))
```

## Common mistakes
- Treating attention weights as a guaranteed explanation of the model's reasoning.
- Forgetting to mask padding positions, which then receive weight.
- Confusing keys and values; keys are for matching, values are for what is retrieved.
- Using attention over very long sequences without considering its quadratic cost in self-attention.
- Mixing up which sequence supplies queries in cross-attention.

## Summary
- Attention scores every input position against a query and takes a weighted average.
- It removes the fixed-vector bottleneck of early sequence-to-sequence models.
- Queries, keys and values are the three roles; softmax makes the weights sum to 1.
- Dot-product and additive scoring are the two classic choices.

## Questions
Q: What problem did attention originally solve?
A) The bottleneck of compressing a whole sentence into one fixed vector
B) Slow GPUs
C) Missing labels
D) Overfitting of trees
ANSWER: A
WHY: Attention lets the decoder access every encoder state instead of one summary vector.

Q: What do the softmax attention weights satisfy?
A) They are positive and sum to 1
B) They are all equal
C) They can be negative
D) They sum to the sequence length
ANSWER: A
WHY: Softmax turns scores into a probability-like distribution.

Q: What is the context vector?
A) The weighted sum of the values using attention weights
B) The loss
C) The query alone
D) A one-hot vector
ANSWER: A
WHY: It summarises the source in a way tailored to the current query.

Q: What is the role of the query?
A) What the model is looking for at the current step
B) The label of each source position
C) The data batch
D) The optimiser
ANSWER: A
WHY: The query is compared against keys to decide where to focus.

Q: For scores $[2, 1, 0]$, which position gets the highest attention weight?
A) The first
B) The second
C) The third
D) All equal
ANSWER: A
WHY: Softmax preserves order, so the largest score receives the largest weight.

Q: What distinguishes additive attention from dot-product attention?
A) Additive uses a small neural network to compute scores
B) Additive has no softmax
C) Additive uses no keys
D) Additive is for images only
ANSWER: A
WHY: Bahdanau attention scores with $v^\top \tanh(W_1 q + W_2 h)$.

Q: What is cross-attention?
A) Queries come from one sequence and keys and values from another
B) Attention within one sequence only
C) Attention without softmax
D) Attention over labels
ANSWER: A
WHY: It lets, for example, a decoder attend to encoder outputs.

Q: Why mask padding in attention?
A) So padded positions do not receive attention weight
B) To increase the batch size
C) To change the loss
D) To add noise
ANSWER: A
WHY: Padding carries no information and would distort the weights.

Q: Are attention weights a perfect explanation of model decisions?
A) No, they show where the model looked but are not a complete explanation
B) Yes, always
C) They are random
D) They are the labels
ANSWER: A
WHY: Weights are one view of the computation; other parts of the network also affect outputs.

Q: What is soft attention?
A) A differentiable weighted average over all positions
B) Picking exactly one position by sampling only
C) Attention with no weights
D) A kind of pooling over labels
ANSWER: A
WHY: Soft attention is smooth, so it can be trained with ordinary backpropagation.
