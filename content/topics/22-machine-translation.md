---
id: machine-translation
title: Machine Translation
group: Transformers and Language
difficulty: 2
order: 22
summary: Translate text between languages with encoder-decoder models, attention and beam search.
prerequisites: transformers, attention-mechanism
---

## What it is and why it matters
Machine translation (MT) automatically converts text from a source language to a target language. Neural machine translation (NMT) replaced older phrase-based systems and is used in everyday tools, from web page translation to customer support. It is also the original home of the transformer and a classic example of sequence-to-sequence learning.

## Intuition
A human translator reads the whole sentence, understands it, then writes it in the other language, word by word, glancing back at the source as needed. NMT does the same: an encoder reads and represents the source sentence, and a decoder writes the target one token at a time, using attention to look back at the relevant source words.

## Key ideas
- **Sequence-to-sequence (seq2seq)**: an encoder maps the source to representations and a decoder generates the output.
- **Attention in translation**: the decoder aligns each output word with relevant source words.
- **Transformer NMT**: the modern standard, using self-attention and cross-attention.
- **Subword tokenisation**: Byte-Pair Encoding (BPE) or SentencePiece handles rare words and shared vocabulary.
- **Autoregressive decoding**: each target token is predicted given the source and the tokens generated so far.
- **Teacher forcing**: during training the true previous tokens are fed in.
- **Exposure bias**: at test time the model sees its own, possibly wrong, previous outputs.
- **Greedy decoding versus beam search**: greedy takes the best token each step; beam search keeps the top few partial translations.
- **BLEU score**: compares n-gram overlap with reference translations and penalises outputs that are too short.
- **Other metrics**: chrF and learned metrics such as COMET correlate better with human judgement.
- **Back-translation**: translating target-language text back to create extra training pairs.
- **Multilingual models and low-resource languages**: one model for many languages helps those with little data.
- **Common errors**: hallucination, dropped content, wrong gender or terminology, literal translation of idioms.

## The math
The model factors the probability of a translation $y$ given source $x$:

$$p(y \mid x) = \prod_{t=1}^{T} p\left(y_t \mid y_{<t}, x\right)$$

Training minimises cross-entropy on the true target tokens. BLEU combines modified n-gram precisions $p_n$ with a brevity penalty:

$$\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right) \qquad BP = \min\left(1, e^{1 - r/c}\right)$$

where $r$ is the reference length and $c$ the candidate length.

## Worked example
The reference has $r = 10$ words and the candidate has $c = 8$ words. The brevity penalty is $e^{1 - 10/8} = e^{-0.25} \approx 0.779$, so even perfect n-gram precision would be scaled down to about 78%. For beam search with width 2, at each step the decoder keeps the two highest-probability partial sentences, rather than just one, and finally returns the one with the best total score.

## Code
```python
# pip install transformers sentencepiece torch
from transformers import pipeline

translator = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
out = translator("Machine learning helps computers learn from data.", num_beams=4)
print(out[0]["translation_text"])
```

## Common mistakes
- Evaluating with BLEU alone and treating small differences as meaningful.
- Using greedy decoding when beam search would improve fluency.
- Ignoring domain mismatch, such as training on news and translating legal text.
- Not handling very long inputs, which can be truncated or dropped.
- Trusting fluent output as correct; models can hallucinate content.

## Summary
- NMT uses an encoder-decoder with attention, today usually a transformer.
- Subword tokenisation, teacher forcing and beam search are standard ingredients.
- BLEU, chrF and COMET evaluate quality; none is perfect.
- Back-translation and multilingual training help low-resource languages.

## Questions
Q: What is the basic structure of a neural machine translation system?
A) An encoder for the source and a decoder for the target
B) A single decision tree
C) A clustering algorithm
D) A convolution only
ANSWER: A
WHY: Seq2seq models encode the source and decode the target.

Q: What does attention add to translation?
A) It lets the decoder focus on relevant source words for each output word
B) It removes the decoder
C) It translates without training
D) It reduces vocabulary to one word
ANSWER: A
WHY: Attention aligns output positions with source positions.

Q: Why use subword units like BPE?
A) They handle rare and unseen words with a fixed vocabulary
B) They remove grammar
C) They shorten the training set
D) They remove the decoder
ANSWER: A
WHY: Rare words are broken into known pieces instead of becoming unknown tokens.

Q: What is teacher forcing?
A) Feeding the true previous target tokens to the decoder during training
B) Forcing the model to skip tokens
C) Using a teacher dataset only
D) Reducing the learning rate
ANSWER: A
WHY: It speeds up training but causes exposure bias at test time.

Q: What is exposure bias?
A) The mismatch between training on true prefixes and testing on the model's own outputs
B) A bias in the label set
C) A kind of tokenisation
D) A learning rate schedule
ANSWER: A
WHY: At inference, early mistakes feed into later steps, which training did not show.

Q: What does beam search do?
A) Keeps several of the best partial translations at each step
B) Picks one word at random
C) Searches the training set
D) Removes attention
ANSWER: A
WHY: It explores multiple hypotheses and returns the best-scoring sentence.

Q: A candidate translation is 8 words and the reference is 10. What does the BLEU brevity penalty do?
A) Reduces the score because the candidate is shorter than the reference
B) Increases the score
C) Does nothing
D) Sets the score to 1
ANSWER: A
WHY: $BP = e^{1 - 10/8} \approx 0.78$ penalises short outputs.

Q: What is back-translation?
A) Translating target-language text into the source language to create extra training pairs
B) Reversing the word order
C) Translating twice with the same model for evaluation only
D) Removing the encoder
ANSWER: A
WHY: Synthetic source sentences from monolingual target text add training data.

Q: Why is BLEU alone not a complete measure of quality?
A) It counts n-gram overlap and misses meaning-preserving paraphrases
B) It uses too many references
C) It cannot be computed
D) It measures speed
ANSWER: A
WHY: A good translation worded differently from the reference can score poorly.

Q: What does autoregressive decoding mean?
A) Each output token depends on the previously generated tokens
B) All tokens are generated at once independently
C) Tokens are sorted
D) The source is ignored
ANSWER: A
WHY: The probability factorises as a product of next-token probabilities given the past.
