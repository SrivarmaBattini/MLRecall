---
id: self-attention
title: Self-Attention
group: Transformers and Language
difficulty: 3
order: 18
summary: Let every token in a sequence look at every other token to build context-aware representations.
prerequisites: attention-mechanism
---

## What it is and why it matters
Self-attention applies attention within a single sequence. Every token compares itself with every other token and builds a new representation as a weighted mix of all of them. It is the core operation of transformers, and it lets a model capture long-range relationships in one step and process all positions in parallel.

## Intuition
In the sentence "The animal didn't cross the street because it was tired", what does "it" refer to? To build a good representation of "it", the model should look at "animal". Self-attention lets each word ask a question of all the others and pull in the most relevant information, so "it" ends up carrying some of the meaning of "animal".

## Key ideas
- **Self-attention**: queries, keys and values all come from the same sequence.
- **Query, key, value projections**: three learned matrices $W_Q$, $W_K$, $W_V$ create $Q$, $K$ and $V$ from the input.
- **Scaled dot-product**: compare queries and keys with dot products, then divide by $\sqrt{d_k}$.
- **Why scale**: large dot products would push softmax into saturated regions with tiny gradients.
- **Softmax across keys**: each token gets a distribution over all tokens.
- **Multi-head attention**: several attention heads run in parallel, each learning a different relationship; outputs are concatenated and mixed.
- **Positional encoding**: self-attention ignores order by itself, so position information must be added.
- **Masking**: padding masks hide padded tokens; causal masks stop a token seeing the future.
- **Complexity**: time and memory grow as $O(n^2)$ with sequence length $n$.
- **Parallelism**: unlike RNNs, all positions are computed at once.
- **Path length**: any two tokens are connected directly, easing long-range learning.

## The math
With input matrix $X$ (one row per token), compute $Q = X W_Q$, $K = X W_K$, $V = X W_V$. Then

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

Multi-head attention runs $h$ heads and combines them:

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W_O$$

## Worked example
A query has dot products $4$ and $0$ with two keys, and $d_k = 4$. Scaling divides by $\sqrt{4} = 2$, giving scores $2$ and $0$. Softmax gives $e^2/(e^2 + 1) \approx 0.881$ and $0.119$. The output is $0.881\, v_1 + 0.119\, v_2$. Without scaling the scores would be $4$ and $0$, giving weights $0.982$ and $0.018$, a sharper and less trainable distribution.

## Code
```python
import torch
import torch.nn.functional as F

def self_attention(x, Wq, Wk, Wv, causal=False):
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5)
    if causal:
        mask = torch.triu(torch.ones_like(scores, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    return F.softmax(scores, dim=-1) @ V

x = torch.randn(5, 16)                       # 5 tokens, model dimension 16
Wq, Wk, Wv = (torch.randn(16, 8) for _ in range(3))
print(self_attention(x, Wq, Wk, Wv).shape)   # (5, 8)
```

## Common mistakes
- Omitting positional information and expecting word order to be understood.
- Forgetting the causal mask when training a left-to-right language model.
- Forgetting the $\sqrt{d_k}$ scaling.
- Ignoring the quadratic memory cost for long inputs.
- Assuming each head is always meaningful; some heads are redundant.

## Summary
- Self-attention builds each token's new representation from all tokens using query, key and value projections.
- Scaled dot-product attention is $\text{softmax}(QK^\top/\sqrt{d_k})V$.
- Multi-head attention captures several relationship types at once.
- It needs positional encodings and costs $O(n^2)$ in sequence length.

## Questions
Q: In self-attention, where do queries, keys and values come from?
A) All from the same input sequence
B) Queries from labels, keys from images
C) All from random noise
D) Only from the output
ANSWER: A
WHY: Each token's vector is projected three ways within the same sequence.

Q: Why divide the dot products by $\sqrt{d_k}$?
A) To keep scores from becoming so large that softmax saturates
B) To make scores negative
C) To remove the keys
D) To speed up the GPU
ANSWER: A
WHY: Large scores make softmax very peaked, which gives tiny gradients.

Q: What does multi-head attention provide?
A) Several parallel attention patterns, each focusing on different relationships
B) A bigger batch
C) A faster optimiser
D) Fewer parameters only
ANSWER: A
WHY: Different heads can specialise, such as syntax or coreference.

Q: Why is positional encoding needed?
A) Self-attention is otherwise insensitive to token order
B) To increase the vocabulary
C) To remove padding
D) To reduce memory
ANSWER: A
WHY: Attention treats the input as a set, so order must be injected.

Q: What does a causal mask do?
A) Prevents a token from attending to later tokens
B) Hides padding only
C) Doubles the heads
D) Removes the softmax
ANSWER: A
WHY: Language models must not see the future words they are predicting.

Q: What is the time and memory cost of self-attention in sequence length $n$?
A) $O(n^2)$
B) $O(1)$
C) $O(\log n)$
D) $O(n)$ always
ANSWER: A
WHY: Every token is compared with every other token, giving an $n \times n$ score matrix.

Q: What advantage does self-attention have over RNNs for training?
A) All positions can be processed in parallel
B) It needs fewer layers always
C) It has no parameters
D) It needs no embeddings
ANSWER: A
WHY: RNNs process step by step; self-attention computes all tokens at once.

Q: What is the shape of the score matrix $QK^\top$ for $n$ tokens?
A) $n \times n$
B) $n \times 1$
C) $1 \times n$
D) $d \times d$
ANSWER: A
WHY: One score for each pair of tokens.

Q: In $\text{softmax}(QK^\top/\sqrt{d_k})V$, what is multiplied by the weights?
A) The value vectors $V$
B) The query vectors
C) The labels
D) The keys again
ANSWER: A
WHY: The output is a weighted combination of the values.

Q: Why concatenate head outputs and multiply by $W_O$?
A) To mix the information from different heads into the model dimension
B) To remove heads
C) To normalise the loss
D) To compute the mask
ANSWER: A
WHY: The output projection combines the heads' separate results into one representation.
