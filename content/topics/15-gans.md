---
id: gans
title: Generative Adversarial Networks
group: Deep Learning
difficulty: 3
order: 15
summary: Train a generator and a discriminator against each other so the generator learns to create realistic data.
prerequisites: neural-networks-intro, cnns
---

## What it is and why it matters
A generative adversarial network (GAN) has two networks that compete. The generator makes fake samples, such as images, and the discriminator tries to tell fake from real. As each improves, the generator's fakes become more realistic. GANs produced the first photorealistic synthetic faces and are used in image synthesis, super-resolution and data augmentation.

## Intuition
A forger and a detective. The forger paints fake banknotes; the detective inspects notes and says real or fake. Every time the detective catches a fake, the forger learns what gave it away. Every time the forger fools the detective, the detective learns to look harder. Over time the forgeries become nearly perfect.

## Key ideas
- **Generator $G$**: turns random noise $z$ into a fake sample $G(z)$.
- **Discriminator $D$**: outputs the probability that a sample is real.
- **Latent vector $z$**: random noise that is the generator's input.
- **Adversarial game**: $D$ tries to maximise its accuracy, $G$ tries to fool $D$.
- **Minimax objective**: the formal two-player loss.
- **Alternating training**: update $D$ for a step, then $G$ for a step.
- **Non-saturating generator loss**: $G$ maximises $\log D(G(z))$, which gives stronger gradients early on.
- **Nash equilibrium**: ideally $G$ matches the real distribution and $D$ outputs 0.5 everywhere.
- **Mode collapse**: the generator produces only a few kinds of output.
- **Training instability**: balance is delicate; if $D$ wins too easily $G$ gets no useful gradient.
- **Variants**: DCGAN (convolutional), conditional GAN (generate a chosen class), Wasserstein GAN (more stable loss), StyleGAN.
- **Evaluation**: Fréchet Inception Distance (FID) and Inception Score compare generated and real data statistics.

## The math
The original GAN objective is a minimax game:

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{data}}\left[\log D(x)\right] + \mathbb{E}_{z \sim p_z}\left[\log\left(1 - D(G(z))\right)\right]$$

For a fixed $G$, the best discriminator is $D^*(x) = \dfrac{p_{data}(x)}{p_{data}(x) + p_g(x)}$. When the generator's distribution $p_g$ equals $p_{data}$, $D^*(x) = 1/2$ everywhere.

## Worked example
At the ideal equilibrium the discriminator outputs $0.5$ on every input. The value of the objective is then $\log 0.5 + \log 0.5 = -2 \ln 2 \approx -1.386$. If a discriminator were perfect (outputs 1 on real, 0 on fake), the objective would be $0$ and $G$ would receive almost no gradient from the saturating form, which is why the non-saturating generator loss is used in practice.

## Code
```python
import torch
from torch import nn

G = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 2))      # noise -> 2-D sample
D = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(0.2), nn.Linear(64, 1), nn.Sigmoid())
opt_g, opt_d = torch.optim.Adam(G.parameters(), 2e-4), torch.optim.Adam(D.parameters(), 2e-4)
bce = nn.BCELoss()

real = torch.randn(128, 2) * 0.5 + 2.0
fake = G(torch.randn(128, 16))

# Discriminator step: real -> 1, fake -> 0
loss_d = bce(D(real), torch.ones(128, 1)) + bce(D(fake.detach()), torch.zeros(128, 1))
opt_d.zero_grad(); loss_d.backward(); opt_d.step()
# Generator step: make D say "real"
loss_g = bce(D(fake), torch.ones(128, 1))
opt_g.zero_grad(); loss_g.backward(); opt_g.step()
```

## Common mistakes
- Letting the discriminator become far stronger than the generator.
- Forgetting `detach()` on fake samples in the discriminator step.
- Judging quality by the loss values, which do not track sample quality well.
- Ignoring mode collapse by looking at a few good samples only.
- Using the same learning rate and architecture without monitoring balance.

## Summary
- A GAN pits a generator against a discriminator in a minimax game.
- The generator learns to produce samples the discriminator cannot distinguish from real ones.
- Training is unstable; mode collapse and vanishing gradients are the main problems.
- Variants such as DCGAN, conditional GAN and Wasserstein GAN improve stability and control.

## Questions
Q: What does the generator take as input?
A) Random noise
B) A class label only
C) A real image
D) The discriminator's weights
ANSWER: A
WHY: The generator maps a random latent vector $z$ to a sample.

Q: What does the discriminator output?
A) The probability that a sample is real
B) A new image
C) The noise vector
D) The learning rate
ANSWER: A
WHY: $D$ is a classifier distinguishing real from generated data.

Q: What is mode collapse?
A) The generator produces only a few kinds of samples
B) The discriminator is deleted
C) The loss becomes negative
D) The batch size becomes zero
ANSWER: A
WHY: The generator finds a few outputs that fool $D$ and stops covering the data diversity.

Q: At the ideal equilibrium, what does the discriminator output?
A) 0.5 for every input
B) 1 for everything
C) 0 for everything
D) Random classes
ANSWER: A
WHY: If fakes are indistinguishable from real data, $D$ can only guess.

Q: Why use `detach()` on fake samples in the discriminator update?
A) So gradients do not flow into the generator during that step
B) To increase image size
C) To speed up the GPU
D) To change labels
ANSWER: A
WHY: The discriminator step should update only $D$.

Q: Why is the non-saturating generator loss used?
A) It gives stronger gradients when the discriminator easily rejects fakes
B) It removes the discriminator
C) It halves the data
D) It improves FID automatically
ANSWER: A
WHY: Maximising $\log D(G(z))$ avoids the vanishing gradient of minimising $\log(1 - D(G(z)))$ early on.

Q: What does a conditional GAN allow?
A) Generating samples of a chosen class or condition
B) Training without a discriminator
C) Removing noise input
D) Classifying only
ANSWER: A
WHY: Feeding a label to both networks lets you control what is generated.

Q: Which metric is commonly used to evaluate GAN image quality?
A) Fréchet Inception Distance (FID)
B) Mean squared error to labels
C) R-squared
D) Silhouette score
ANSWER: A
WHY: FID compares feature statistics of real and generated images.

Q: What happens if the discriminator becomes perfect early in training?
A) The generator gets little useful gradient and training stalls
B) The generator improves quickly
C) Mode collapse is fixed
D) The loss is zero and training is done
ANSWER: A
WHY: A perfectly confident discriminator offers no informative signal for improving fakes.

Q: What is the role of the Wasserstein GAN?
A) Provides a more stable training objective based on a distance between distributions
B) Replaces the generator with a tree
C) Removes the noise vector
D) Makes images smaller
ANSWER: A
WHY: The Wasserstein distance gives smoother gradients and a loss that correlates better with quality.
