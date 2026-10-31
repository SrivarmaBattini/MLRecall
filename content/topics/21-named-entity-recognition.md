---
id: named-entity-recognition
title: Named Entity Recognition
group: Transformers and Language
difficulty: 2
order: 21
summary: Find and label names of people, places, organisations and other entities in text.
prerequisites: bert, rnns
---

## What it is and why it matters
Named entity recognition (NER) scans text and marks spans that refer to real-world things, such as people, organisations, locations, dates and products. It powers search, document analysis, chatbots and information extraction from news, contracts and medical records.

## Intuition
Imagine reading a news article with a highlighter, using different colours for people, places and companies. NER does exactly that, token by token. To decide whether "Washington" is a person or a place, it needs the surrounding words, which is why context-aware models work best.

## Key ideas
- **Sequence labelling**: assign one label to every token.
- **Entity types**: PER (person), ORG (organisation), LOC (location), plus domain types such as DATE or DRUG.
- **BIO (IOB) tagging**: `B-` begins an entity, `I-` continues it, `O` is outside any entity.
- **Tokens and subwords**: transformer tokenisers split words into pieces; labels are usually assigned to the first piece.
- **Classic approaches**: hand-written rules, gazetteers (lists of names) and feature-based classifiers.
- **CRF layer**: a conditional random field scores whole tag sequences so impossible transitions (such as `O` followed by `I-PER`) are discouraged.
- **BiLSTM-CRF**: a bidirectional LSTM with a CRF on top, a strong pre-transformer model.
- **Transformer token classification**: fine-tune BERT with a classifier on every token.
- **Entity-level evaluation**: an entity counts as correct only if both its boundaries and its type match; precision, recall and F1 are computed on entities.
- **Challenges**: ambiguity, rare names, nested entities, new domains with different vocabulary.
- **Annotation**: needs labelled data; agreement between annotators is a quality check.

## The math
Entity-level scores use counts of true positives ($TP$), false positives ($FP$) and false negatives ($FN$):

$$P = \frac{TP}{TP + FP} \qquad R = \frac{TP}{TP + FN} \qquad F_1 = \frac{2PR}{P + R}$$

A CRF models the probability of a tag sequence $y$ given tokens $x$ as $p(y \mid x) \propto \exp\left(\sum_t \text{emit}(y_t, x_t) + \text{trans}(y_{t-1}, y_t)\right)$, where the transition scores learn which tags may follow which.

## Worked example
Sentence: "Ada Lovelace visited London in 1842". Tags: Ada `B-PER`, Lovelace `I-PER`, visited `O`, London `B-LOC`, in `O`, 1842 `B-DATE`. Suppose a model predicts 3 entities, of which 2 are exactly right, while the gold data contains 4 entities. Then $P = 2/3 \approx 0.667$, $R = 2/4 = 0.5$ and $F_1 = 2 \times 0.667 \times 0.5 / (0.667 + 0.5) \approx 0.571$.

## Code
```python
# pip install transformers torch
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
for ent in ner("Ada Lovelace worked with Charles Babbage in London."):
    print(ent["entity_group"], ent["word"], round(float(ent["score"]), 3))
```

## Common mistakes
- Scoring token by token instead of by whole entity.
- Forgetting to align labels with subword tokens.
- Ignoring BIO consistency, producing `I-` tags with no `B-`.
- Expecting a model trained on news to work on medical text without adaptation.
- Overlooking class imbalance; most tokens are `O`.

## Summary
- NER labels each token as the beginning, inside or outside of an entity type.
- CRF layers and transformer encoders both capture context; BERT fine-tuning is the standard today.
- Evaluate on whole entities with precision, recall and F1.
- Domain shift and subword alignment are the main practical issues.

## Questions
Q: What kind of task is NER?
A) Sequence labelling, one label per token
B) Image classification
C) Clustering
D) Regression
ANSWER: A
WHY: Each token gets a tag such as `B-PER`, `I-PER` or `O`.

Q: What does the tag `B-LOC` mean?
A) The beginning of a location entity
B) A location inside an entity
C) Not an entity
D) The end of a person
ANSWER: A
WHY: `B-` marks the first token of an entity span.

Q: What does the tag `O` mean?
A) The token is outside any entity
B) An organisation
C) An object
D) Optional
ANSWER: A
WHY: `O` marks tokens that are not part of any named entity.

Q: What does a CRF layer add on top of per-token predictions?
A) It scores whole tag sequences, discouraging invalid transitions
B) It adds more tokens
C) It removes labels
D) It translates text
ANSWER: A
WHY: Transition scores learn that, for example, `I-PER` rarely follows `O`.

Q: How is an entity counted as correct in entity-level evaluation?
A) Both its span boundaries and type match the gold annotation
B) Only its first token matches
C) Any overlap counts
D) Only its type matches
ANSWER: A
WHY: Strict entity-level scoring requires exact boundaries and the right type.

Q: With 3 predicted entities (2 correct) and 4 gold entities, what is recall?
A) 0.5
B) 0.667
C) 0.75
D) 1.0
ANSWER: A
WHY: Recall is $2/4 = 0.5$.

Q: How is NER typically done with BERT?
A) Fine-tune with a classifier on every token's output
B) Use only the `[CLS]` token
C) Train a decision tree on raw text
D) Use k-means
ANSWER: A
WHY: Token classification gives each subword token a label.

Q: Why is subword alignment a concern for NER?
A) Tokenisers split words into pieces, so labels must be mapped to pieces consistently
B) Words have no labels
C) BERT cannot read subwords
D) It changes the label set
ANSWER: A
WHY: One word may become several tokens, and labels need a rule such as using the first piece.

Q: Which is a typical NER challenge?
A) Domain shift, such as moving from news to medical text
B) Too many labelled data always
C) No ambiguity ever
D) Tokens have no context
ANSWER: A
WHY: Vocabulary and entity styles differ between domains, which hurts models trained elsewhere.

Q: Why is accuracy a poor metric for NER?
A) Most tokens are `O`, so a model predicting all `O` scores high accuracy
B) Accuracy cannot be computed
C) It uses probabilities
D) NER has no labels
ANSWER: A
WHY: Class imbalance makes token accuracy look good even when entities are missed.
