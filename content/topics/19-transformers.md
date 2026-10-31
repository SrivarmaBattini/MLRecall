---
id: transformers
title: Transformers
group: Transformers and Language
difficulty: 3
order: 19
summary: Stack self-attention and feed-forward layers with residual connections to model language at scale.
prerequisites: self-attention
---

## What it is and why it matters
The transformer is an architecture built from attention instead of recurrence. Introduced in 2017 for machine translation, it now underlies BERT, GPT and most modern language, vision and speech models. It trains in parallel, scales to huge datasets and captures long-range context.

## Intuition
Each transformer layer does two things for every token. First, the token gathers information from the other tokens (attention). Second, it thinks about what it gathered on its own (a small feed-forward network). Shortcut connections keep the original information flowing, and normalisation keeps values stable. Stacking many such layers builds deeper understanding.

## Key ideas
- **Encoder**: reads the input and produces contextual representations.
- **Decoder**: generates the output one token at a time, using masked self-attention and cross-attention to the encoder.
- **Token embeddings plus positional encodings**: the input to the first layer.
- **Positional encodings**: sinusoidal or learned vectors giving order information.
- **Multi-head self-attention sublayer**: mixes information across tokens.
- **Position-wise feed-forward network**: two linear layers with a non-linearity, applied to each token separately.
- **Residual connection**: adds the sublayer's input to its output, easing training of deep stacks.
- **Layer normalisation**: stabilises activations.
- **Masked self-attention**: in the decoder, blocks looking at future tokens.
- **Cross-attention**: the decoder queries the encoder's output.
- **Final linear layer and softmax**: turn the decoder output into next-token probabilities.
- **Training tricks**: learning-rate warm-up, dropout, label smoothing, teacher forcing.
- **Three families**: encoder-only (BERT), decoder-only (GPT) and encoder-decoder (T5, original transformer).

## The math
One encoder block, with $x$ the input to the block:

$$x' = \text{LayerNorm}\left(x + \text{MultiHeadAttention}(x)\right)$$

$$\text{out} = \text{LayerNorm}\left(x' + \text{FFN}(x')\right) \qquad \text{FFN}(x) = W_2 \, \text{ReLU}(W_1 x + b_1) + b_2$$

Sinusoidal positional encoding for position $pos$ and dimension index $i$:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \qquad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

## Worked example
The original model used $d_{model} = 512$ with 8 heads, so each head has $d_k = 512/8 = 64$ dimensions. The feed-forward layer expands $512 \to 2048 \to 512$. Self-attention parameters per block: four matrices of size $512 \times 512$ (for $Q$, $K$, $V$ and the output) give about $1.05$ million. The feed-forward layer adds about $2 \times 512 \times 2048 \approx 2.1$ million, so roughly $3.1$ million parameters per block, before biases and norms.

## Code
```python
import torch
from torch import nn

layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512,
                                   dropout=0.1, batch_first=True)
encoder = nn.TransformerEncoder(layer, num_layers=4)

x = torch.randn(2, 20, 128)              # batch of 2 sequences, 20 tokens each
pad_mask = torch.zeros(2, 20, dtype=torch.bool)
print(encoder(x, src_key_padding_mask=pad_mask).shape)   # (2, 20, 128)
```

## Common mistakes
- Forgetting positional encodings, so the model sees a bag of words.
- Training without learning-rate warm-up, causing instability.
- Using too long sequences without considering the quadratic attention cost.
- Mixing up decoder masking and padding masking.
- Assuming bigger is always better without enough data or compute.

## Summary
- A transformer stacks blocks of multi-head self-attention and feed-forward layers with residual connections and layer norm.
- Positional encodings supply order; masking controls what each token may see.
- Encoder-only, decoder-only and encoder-decoder variants serve different tasks.
- Parallel training and scalability explain its dominance.

## Questions
Q: What replaced recurrence in the transformer?
A) Self-attention
B) Convolution only
C) Decision trees
D) K-means
ANSWER: A
WHY: Attention lets every position access every other directly, with no recurrent loop.

Q: What does the feed-forward sublayer do?
A) Applies a small neural network to each token position independently
B) Mixes tokens together
C) Computes the mask
D) Creates embeddings
ANSWER: A
WHY: It transforms each token's representation separately after attention has mixed information.

Q: Why are residual connections used?
A) They make deep stacks easier to train by preserving information and gradients
B) They remove layer norm
C) They reduce the vocabulary
D) They add positions
ANSWER: A
WHY: Adding the input to the output provides a direct path for gradients.

Q: Why does the decoder use masked self-attention?
A) To stop a token from attending to future tokens during generation
B) To remove padding
C) To speed up the encoder
D) To compute embeddings
ANSWER: A
WHY: When predicting the next word the model must not see it.

Q: What does cross-attention connect?
A) The decoder to the encoder's outputs
B) The encoder to itself only
C) The labels to the input
D) The loss to the weights
ANSWER: A
WHY: Decoder queries attend to encoder keys and values.

Q: With $d_{model} = 512$ and 8 heads, what is $d_k$ per head?
A) 8
B) 64
C) 512
D) 4096
ANSWER: B
WHY: $512 / 8 = 64$.

Q: Which model family is BERT?
A) Encoder-only
B) Decoder-only
C) Encoder-decoder
D) Convolutional
ANSWER: A
WHY: BERT uses only the encoder stack and is trained with masked language modelling.

Q: Which model family is GPT?
A) Decoder-only
B) Encoder-only
C) Encoder-decoder
D) Recurrent
ANSWER: A
WHY: GPT models generate text left to right with a causal decoder stack.

Q: What is the purpose of learning-rate warm-up?
A) To avoid unstable updates early in training by starting small
B) To increase batch size
C) To remove dropout
D) To speed up tokenisation
ANSWER: A
WHY: Starting with a small learning rate stabilises early training of deep attention models.

Q: What does layer normalisation do?
A) Normalises activations across the features of each token to stabilise training
B) Normalises across the batch only
C) Removes the residual
D) Adds positions
ANSWER: A
WHY: It keeps activation scales consistent, which helps optimisation.
