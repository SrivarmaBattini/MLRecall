---
id: word-embeddings
title: Word Embeddings
group: Transformers and Language
difficulty: 2
order: 16
summary: Represent each word as a dense vector so that words with similar meaning sit close together.
prerequisites: neural-networks-intro
---

## What it is and why it matters
A word embedding maps each word to a list of numbers, typically 100 to 300 of them. Words used in similar contexts get similar vectors, so a model can generalise: if it has learned about "doctor", it knows something about "physician". Embeddings are the entry point of nearly every language model.

## Intuition
"You shall know a word by the company it keeps." The word "coffee" appears near "cup", "morning" and "drink"; "tea" appears near the same words. If we place words in space so that words with similar neighbours end up close, meaning becomes geometry. Directions in the space can even capture relations, such as gender or tense.

## Key ideas
- **One-hot vector**: a huge sparse vector with a single 1; it carries no notion of similarity.
- **Dense embedding**: a short learned vector for each word.
- **Distributional hypothesis**: words in similar contexts have similar meanings.
- **Word2Vec**: learns embeddings by predicting words from context.
- **CBOW**: predict the centre word from its surrounding words.
- **Skip-gram**: predict the surrounding words from the centre word; works well for rare words.
- **Negative sampling**: train against a few random wrong words instead of the full softmax.
- **GloVe**: learns embeddings from global word co-occurrence counts.
- **Cosine similarity**: the angle-based measure of how alike two vectors are.
- **Analogies**: $\text{king} - \text{man} + \text{woman} \approx \text{queen}$.
- **Embedding dimension**: a trade-off between capacity and cost.
- **Out-of-vocabulary words**: unseen words have no vector; FastText uses subword pieces to help.
- **Static versus contextual**: Word2Vec gives one vector per word; models like BERT give a different vector depending on the sentence.
- **Embedding layer**: a lookup table inside a neural network, trained along with the model.

## The math
Cosine similarity between vectors $a$ and $b$:

$$\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$$

Skip-gram maximises the probability of a context word $w_o$ given a centre word $w_c$:

$$p(w_o \mid w_c) = \frac{\exp(u_{w_o}^\top v_{w_c})}{\sum_{w \in V} \exp(u_w^\top v_{w_c})}$$

Negative sampling replaces the sum over the whole vocabulary $V$ with a handful of sampled negative words.

## Worked example
Let $a = [1, 0]$ and $b = [1, 1]$. The dot product is $1$, $\lVert a \rVert = 1$ and $\lVert b \rVert = \sqrt{2}$, so $\cos(a, b) = 1/\sqrt{2} \approx 0.707$, an angle of 45 degrees. Vectors pointing the same way give 1, perpendicular vectors give 0 and opposite vectors give $-1$. A one-hot vocabulary of 50,000 words needs 50,000 numbers per word; a 300-dimensional embedding needs 300.

## Code
```python
# pip install gensim
from gensim.models import Word2Vec

sentences = [["machine", "learning", "is", "fun"],
             ["deep", "learning", "uses", "neural", "networks"],
             ["machine", "learning", "uses", "data"]] * 50
model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1, negative=5, epochs=30)
print(model.wv.most_similar("learning", topn=3))
```

## Common mistakes
- Assuming one vector per word captures every sense (bank of a river versus a bank for money).
- Training embeddings on too little text.
- Comparing vectors from different models or runs as if they shared a coordinate system.
- Treating analogy results as perfect facts rather than statistical patterns.
- Ignoring biases present in the training text.

## Summary
- Embeddings are dense vectors where similar words are close.
- Word2Vec (CBOW, skip-gram) and GloVe learn them from context statistics.
- Cosine similarity compares vectors.
- Static embeddings give one vector per word; contextual models give one per usage.

## Questions
Q: What is the main weakness of one-hot word vectors?
A) They contain no information about similarity between words
B) They are too small
C) They cannot be stored
D) They have negative values only
ANSWER: A
WHY: Every pair of one-hot vectors is equally far apart, so meaning is not represented.

Q: What does the distributional hypothesis state?
A) Words in similar contexts have similar meanings
B) Words with the same length are similar
C) Rare words are the most important
D) Words are independent
ANSWER: A
WHY: Context patterns reveal meaning, which is the basis of embedding learning.

Q: In skip-gram, what is predicted?
A) The surrounding context words given the centre word
B) The centre word given its context
C) The next sentence
D) The document label
ANSWER: A
WHY: Skip-gram uses the centre word to predict nearby words.

Q: What does CBOW predict?
A) The centre word from its surrounding context words
B) The next document
C) A sentiment label
D) The part of speech only
ANSWER: A
WHY: Continuous bag of words averages the context to predict the centre word.

Q: Why use negative sampling?
A) To avoid computing a softmax over the whole vocabulary
B) To make vectors longer
C) To remove rare words
D) To add labels
ANSWER: A
WHY: It trains against a few random wrong words, which is much cheaper.

Q: What is the cosine similarity of two vectors pointing in the same direction?
A) 1
B) 0
C) $-1$
D) 0.5
ANSWER: A
WHY: The angle is zero, and $\cos 0 = 1$.

Q: Which method helps with out-of-vocabulary words by using subword pieces?
A) FastText
B) One-hot encoding
C) K-means
D) PCA
ANSWER: A
WHY: FastText builds a word's vector from character n-grams, so unseen words get a sensible vector.

Q: What is a limitation of static embeddings like Word2Vec?
A) One vector per word regardless of the sentence context
B) They cannot be trained
C) They need labels
D) They only work for numbers
ANSWER: A
WHY: A word with several meanings still gets a single vector.

Q: What does the analogy $\text{king} - \text{man} + \text{woman}$ approximately give?
A) queen
B) table
C) kingdom
D) man
ANSWER: A
WHY: The relation "male to female" is encoded as a consistent direction in the space.

Q: What is an embedding layer in a neural network?
A) A trainable lookup table from word index to vector
B) A pooling layer
C) A loss function
D) A normalisation step
ANSWER: A
WHY: It maps each token id to a learned dense vector.
