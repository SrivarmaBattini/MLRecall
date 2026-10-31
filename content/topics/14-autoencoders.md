---
id: autoencoders
title: Autoencoders
group: Deep Learning
difficulty: 2
order: 14
summary: Compress data into a small code and rebuild it, learning useful features without labels.
prerequisites: neural-networks-intro, pca
---

## What it is and why it matters
An autoencoder is a neural network trained to copy its input to its output through a narrow bottleneck. To succeed, it must learn a compact code that keeps the important information. Autoencoders are used for compression, noise removal, anomaly detection and learning features without labels.

## Intuition
Imagine describing a photo to a friend over a text message limited to 20 words, and your friend must redraw the photo from your words. You would learn to describe only what matters. The encoder writes the message, the decoder redraws, and the bottleneck forces the message to be a useful summary.

## Key ideas
- **Encoder**: maps the input $x$ to a code $z$.
- **Bottleneck (latent code)**: the compressed representation $z$.
- **Decoder**: rebuilds $\hat{x}$ from $z$.
- **Reconstruction loss**: how far $\hat{x}$ is from $x$, for example mean squared error.
- **Undercomplete autoencoder**: a code smaller than the input forces compression.
- **Denoising autoencoder**: corrupt the input and train to reconstruct the clean version.
- **Sparse autoencoder**: a penalty keeps most code units near zero.
- **Variational autoencoder (VAE)**: the encoder outputs a distribution; a KL term keeps the latent space smooth, so new samples can be generated.
- **Reparameterisation trick**: $z = \mu + \sigma \odot \epsilon$ lets gradients flow through sampling.
- **Relation to PCA**: a linear autoencoder with squared error learns the same subspace as PCA; non-linear ones go further.
- **Uses**: dimensionality reduction, denoising, pretraining, anomaly detection (large reconstruction error signals something unusual).
- **Pitfall**: with too much capacity the network can just copy the input and learn nothing useful.

## The math
Deterministic autoencoder: $z = f_\phi(x)$, $\hat{x} = g_\theta(z)$, and the loss is

$$L = \lVert x - \hat{x} \rVert^2$$

A VAE maximises the evidence lower bound, written here as a loss:

$$L = \underbrace{\lVert x - \hat{x} \rVert^2}_{\text{reconstruction}} + \underbrace{D_{KL}\left(q(z \mid x) \,\Vert\, \mathcal{N}(0, I)\right)}_{\text{regularises the latent space}}$$

## Worked example
Input $x = [1, 0, 1, 0]$ is compressed to a 2-number code and reconstructed as $\hat{x} = [0.9, 0.1, 0.8, 0.1]$. The squared errors are $0.01, 0.01, 0.04, 0.01$, so the mean squared error is $0.07/4 = 0.0175$. For anomaly detection, a normal example might reconstruct with an error near that, while an unusual one reconstructs badly with a much larger error and is flagged.

## Code
```python
import torch
from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, d_in=784, d_code=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, d_code))
        self.dec = nn.Sequential(nn.Linear(d_code, 128), nn.ReLU(), nn.Linear(128, d_in))
    def forward(self, x):
        return self.dec(self.enc(x))

model, loss_fn = AutoEncoder(), nn.MSELoss()
x = torch.rand(64, 784)
loss = loss_fn(model(x), x)       # reconstruction loss
loss.backward()
```

## Common mistakes
- Making the code as large as the input, so the network learns the identity.
- Treating reconstruction quality as proof the code is useful for another task.
- Forgetting the KL term weight in a VAE, which gives blurry samples or collapsed codes.
- Using anomaly detection with a threshold chosen without validation data.
- Expecting plain autoencoders to generate good new samples; VAEs are built for that.

## Summary
- An autoencoder is encoder, bottleneck and decoder trained to reconstruct its input.
- The bottleneck forces useful, compact features without labels.
- Variants include denoising, sparse and variational autoencoders.
- Large reconstruction error is a practical anomaly signal.

## Questions
Q: What does the encoder produce?
A) A compressed latent code of the input
B) The final label
C) The loss
D) The optimiser
ANSWER: A
WHY: The encoder maps the input to a smaller code $z$.

Q: What is the training target of a standard autoencoder?
A) The input itself
B) A class label
C) The next frame
D) The code
ANSWER: A
WHY: The network is trained to reconstruct its own input.

Q: Why is a bottleneck useful?
A) It forces the network to keep only the most important information
B) It increases the data size
C) It removes the decoder
D) It makes training unnecessary
ANSWER: A
WHY: With limited capacity the code cannot store everything, so it must summarise.

Q: What does a denoising autoencoder do?
A) Reconstructs the clean input from a corrupted version
B) Adds noise to the output
C) Removes the encoder
D) Classifies images
ANSWER: A
WHY: Learning to undo corruption encourages robust features.

Q: What does the KL term in a VAE do?
A) Pushes the latent distribution towards a simple prior, keeping the space smooth
B) Measures classification accuracy
C) Replaces the decoder
D) Sets the learning rate
ANSWER: A
WHY: It regularises the latent space so that sampling from the prior yields sensible outputs.

Q: What is the reparameterisation trick for?
A) Letting gradients flow through the random sampling step
B) Compressing images
C) Choosing the batch size
D) Pruning layers
ANSWER: A
WHY: Writing $z = \mu + \sigma \epsilon$ moves randomness outside the learnable path.

Q: A linear autoencoder with squared error is closely related to which method?
A) PCA
B) K-means
C) Decision trees
D) Q-learning
ANSWER: A
WHY: It learns the same principal subspace as PCA.

Q: How can an autoencoder flag anomalies?
A) Unusually large reconstruction error
B) Unusually small loss rate
C) A larger code
D) More layers
ANSWER: A
WHY: A model trained on normal data reconstructs unusual inputs poorly.

Q: What happens if the code size equals the input size and there is no regularisation?
A) The network may learn the identity and nothing useful
B) It will always learn perfect features
C) Training is impossible
D) It becomes a classifier
ANSWER: A
WHY: Without a constraint, copying is a trivial solution.

Q: Which autoencoder can generate new samples by sampling the latent space?
A) Variational autoencoder
B) Plain undercomplete autoencoder
C) Sparse autoencoder with no prior
D) Linear PCA
ANSWER: A
WHY: A VAE's latent space is shaped to follow a prior that can be sampled.
