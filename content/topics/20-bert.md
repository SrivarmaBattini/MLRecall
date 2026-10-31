---
id: bert
title: BERT
group: Transformers and Language
difficulty: 3
order: 20
summary: A bidirectional transformer encoder pretrained by masked-word prediction and fine-tuned for many language tasks.
prerequisites: transformers
---

## What it is and why it matters
BERT (Bidirectional Encoder Representations from Transformers) is a transformer encoder pretrained on huge amounts of text. It reads a sentence in both directions at once, so every word's representation uses its full context. After pretraining, a small amount of fine-tuning adapts it to tasks such as classification, question answering and tagging, and it set new records across many benchmarks when released.

## Intuition
Earlier language models read left to right, so the word "bank" in "I sat by the river bank" was understood only from words before it. BERT is trained like a fill-in-the-blank exercise: hide some words and predict them from everything around them, left and right. To do this well, it has to learn grammar, facts and context.

## Key ideas
- **Encoder-only transformer**: stacked self-attention layers with no decoder.
- **Bidirectional context**: each token attends to tokens on both sides.
- **Masked language modelling (MLM)**: about 15% of tokens are chosen and the model predicts them.
- **The 80/10/10 rule**: of the chosen tokens, 80% become `[MASK]`, 10% a random token and 10% stay unchanged, which reduces the mismatch with fine-tuning where `[MASK]` never appears.
- **Next sentence prediction (NSP)**: a second pretraining task deciding whether sentence B follows sentence A (later work found it of limited value).
- **Special tokens**: `[CLS]` at the start (its final vector represents the whole input), `[SEP]` between sentences, `[PAD]` for padding.
- **WordPiece tokenisation**: splits rare words into subword pieces.
- **Input embeddings**: token embedding plus segment embedding plus position embedding.
- **Pretraining then fine-tuning**: learn general language first, then adapt with labelled data.
- **Fine-tuning heads**: a classifier on `[CLS]`, start and end predictors for question answering, per-token classifier for tagging.
- **Sizes**: BERT-base has 12 layers, hidden size 768, 12 heads and about 110 million parameters; BERT-large has 24 layers.
- **Contextual embeddings**: the same word gets different vectors in different sentences.
- **Limits**: a 512-token input limit, and it is not built for text generation.

## The math
For masked positions $M$, the MLM loss is the negative log-likelihood of the correct tokens:

$$L_{MLM} = -\sum_{i \in M} \log p_\theta\left(x_i \mid x_{\setminus M}\right)$$

The input vector for a token is the sum $E_{token} + E_{segment} + E_{position}$. For classification, fine-tuning adds a linear layer on the final `[CLS]` vector $h_{CLS}$ and trains with cross-entropy: $p = \text{softmax}(W h_{CLS} + b)$.

## Worked example
Take the input "The cat sat on the mat". Suppose "sat" is selected for masking. With 80% probability the model sees "The cat [MASK] on the mat" and must predict "sat". Each selected token follows the 80/10/10 rule, so in a 100-token passage about 15 tokens are selected, of which roughly 12 become `[MASK]`, 1 or 2 are replaced by random words and 1 or 2 are left alone. The loss is computed only on those 15 positions.

## Code
```python
# pip install transformers torch
from transformers import pipeline

fill = pipeline("fill-mask", model="bert-base-uncased")
for guess in fill("Machine learning models learn from [MASK].")[:3]:
    print(guess["token_str"], round(guess["score"], 3))
```

## Common mistakes
- Using BERT for free-form text generation; it is an encoder.
- Feeding inputs longer than 512 tokens without truncating or chunking.
- Fine-tuning with a learning rate that is too high and destroying pretrained knowledge.
- Forgetting the attention mask for padded batches.
- Assuming the `[CLS]` vector is a good sentence embedding without fine-tuning.

## Summary
- BERT is a bidirectional transformer encoder pretrained with masked language modelling.
- Pretraining learns general language; fine-tuning adapts it with a small task head.
- `[CLS]`, `[SEP]` and WordPiece define its input format.
- It excels at understanding tasks, not generation.

## Questions
Q: What does the "B" in BERT stand for?
A) Bidirectional
B) Binary
C) Bayesian
D) Batch
ANSWER: A
WHY: BERT conditions on context from both the left and the right.

Q: What is masked language modelling?
A) Predicting randomly hidden tokens from their surrounding context
B) Predicting the next word only
C) Translating sentences
D) Clustering words
ANSWER: A
WHY: The model must recover masked tokens using the full bidirectional context.

Q: Roughly what fraction of tokens are selected for masking in BERT pretraining?
A) 15%
B) 50%
C) 1%
D) 90%
ANSWER: A
WHY: The original recipe chooses about 15% of tokens.

Q: Why replace some selected tokens with random tokens or leave them unchanged?
A) To reduce the mismatch between pretraining and fine-tuning, where `[MASK]` never appears
B) To increase the vocabulary
C) To save memory
D) To remove positions
ANSWER: A
WHY: The 80/10/10 rule stops the model from relying on seeing `[MASK]`.

Q: What is the `[CLS]` token used for?
A) Its final vector represents the whole input for classification tasks
B) Padding
C) Separating sentences
D) Marking masked words
ANSWER: A
WHY: Classification heads read the final hidden state of `[CLS]`.

Q: Which part of the transformer does BERT use?
A) The encoder only
B) The decoder only
C) Both encoder and decoder
D) Neither
ANSWER: A
WHY: BERT is a stack of transformer encoder layers.

Q: What does WordPiece tokenisation do?
A) Splits rare words into subword units from a fixed vocabulary
B) Removes stopwords
C) Translates words
D) Sorts words alphabetically
ANSWER: A
WHY: Subword units let a fixed vocabulary cover any word.

Q: What does fine-tuning mean?
A) Continuing training the pretrained model on a labelled task, usually with a small task head
B) Training from random weights
C) Deleting layers
D) Changing the tokeniser only
ANSWER: A
WHY: It adapts general language knowledge to a specific task.

Q: What is the approximate size of BERT-base?
A) 110 million parameters
B) 110 thousand parameters
C) 11 billion parameters
D) 1 million parameters
ANSWER: A
WHY: 12 layers, hidden size 768 and 12 heads give about 110 million parameters.

Q: Why is BERT not ideal for text generation?
A) It is bidirectional and not trained to predict the next token left to right
B) It has no tokens
C) It cannot read text
D) It has no attention
ANSWER: A
WHY: Generation requires an autoregressive decoder, which BERT lacks.
