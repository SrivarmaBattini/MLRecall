---
id: ppo
title: Proximal Policy Optimization (PPO)
group: Reinforcement Learning
difficulty: 3
order: 28
summary: Improve a policy in safe, clipped steps so that learning stays stable and simple to tune.
prerequisites: actor-critic
---

## What it is and why it matters
Proximal policy optimisation (PPO) is a policy-gradient method that stops each update from changing the policy too much. It keeps the simplicity of first-order methods while achieving much of the stability of trust-region methods. PPO is a default choice in robotics, games and the reinforcement learning step used to tune large language models from human feedback.

## Intuition
A big step in policy space can wreck performance, because the data you collected came from the old policy and may not describe the new one. PPO lets the policy move, but not too far: once an action's probability has changed by a certain amount in the good direction, there is no extra reward for pushing further. The "proximal" in the name means staying near the old policy.

## Key ideas
- **Policy-gradient instability**: large updates can collapse performance.
- **Trust-region idea (TRPO)**: limit how much the policy changes, measured by KL divergence; effective but complex.
- **Probability ratio**: $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_{old}}(a_t \mid s_t)$.
- **Clipped surrogate objective**: caps the benefit of moving the ratio outside $[1 - \epsilon, 1 + \epsilon]$.
- **Clip range $\epsilon$**: commonly 0.2.
- **Advantage $A_t$**: usually from GAE; positive means the action was better than average.
- **Multiple epochs on one batch**: because of clipping, the same data can be reused for several gradient steps.
- **Value loss and entropy bonus**: the full loss adds a critic term and an exploration term.
- **Advantage normalisation**: standardising advantages per batch stabilises training.
- **On-policy, batch-based**: collect rollouts, update, then discard.
- **Parallel environments**: many actors collect data at once.
- **Use in RLHF**: a PPO variant fine-tunes language models against a reward model, often with a KL penalty to a reference model.
- **Practical tuning**: learning rate, clip range, number of epochs, batch size and GAE $\lambda$.

## The math
The clipped objective, to be maximised:

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[ \min\left( r_t(\theta) A_t, \; \text{clip}\left(r_t(\theta), 1 - \epsilon, 1 + \epsilon\right) A_t \right) \right]$$

The combined loss to minimise is $-L^{CLIP} + c_1 L_{value} - c_2 \, \mathcal{H}[\pi_\theta]$, where $\mathcal{H}$ is the policy entropy.

## Worked example
Let $\epsilon = 0.2$ and $A_t = 2$ (a good action). If the ratio is $1.5$, the unclipped term is $1.5 \times 2 = 3$ and the clipped term is $1.2 \times 2 = 2.4$. The minimum is $2.4$, so pushing the ratio beyond $1.2$ brings no extra reward. If instead $A_t = -2$ (a bad action) and the ratio is $1.5$, the unclipped term is $-3$ and the clipped term is $-2.4$. The minimum is $-3$, so the objective still penalises the policy for having raised the probability of a bad action; clipping does not hide that mistake.

## Code
```python
import torch

def ppo_clip_loss(new_logp, old_logp, advantages, eps=0.2):
    ratio = torch.exp(new_logp - old_logp)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - eps, 1 + eps) * adv
    return -torch.min(unclipped, clipped).mean()      # minimise the negative objective

new = torch.tensor([-0.5, -1.0, -0.2])
old = torch.tensor([-0.7, -0.9, -0.2])
print(ppo_clip_loss(new, old, torch.tensor([2.0, -1.0, 0.5])))
```

## Common mistakes
- Running too many epochs on the same batch, so the policy drifts far from the data's policy.
- Forgetting to store the old log-probabilities before updating.
- Not normalising advantages.
- Treating PPO as off-policy and reusing old rollouts for many iterations.
- Setting the clip range far too large and losing the stabilising effect.

## Summary
- PPO maximises a clipped surrogate objective based on the probability ratio.
- Clipping removes the incentive to change the policy too much per update.
- It reuses each batch for several epochs and is simple and robust.
- It is widely used, including for RLHF of language models.

## Questions
Q: What does the PPO ratio $r_t(\theta)$ compare?
A) The new policy's probability of an action to the old policy's probability
B) Reward to cost
C) Value to return
D) Actor to critic
ANSWER: A
WHY: It measures how much the action's probability has changed.

Q: What does clipping the ratio achieve?
A) It removes the incentive to move the policy too far from the old one
B) It makes the policy deterministic
C) It speeds up the environment
D) It removes the critic
ANSWER: A
WHY: Beyond $1 \pm \epsilon$ extra change brings no extra objective gain.

Q: What is a common value for the clip range $\epsilon$?
A) 0.2
B) 5
C) 0.0001
D) 100
ANSWER: A
WHY: 0.2 is the standard default from the original paper.

Q: Why can PPO reuse a batch for several epochs?
A) Clipping keeps updates close to the data-collecting policy
B) It is off-policy by design
C) It uses a replay buffer
D) It has no gradient
ANSWER: A
WHY: The objective discourages drifting far, so the old data remain roughly valid.

Q: With $\epsilon = 0.2$, $A = 2$ and ratio $1.5$, what is the objective term $\min(r A, \text{clip}(r) A)$?
A) 2.4
B) 3.0
C) 1.5
D) 0.8
ANSWER: A
WHY: Clipped term $1.2 \times 2 = 2.4$ is smaller than $3.0$.

Q: With $A = -2$ and ratio $1.5$, what does the min choose?
A) $-3$, the unclipped term
B) $-2.4$
C) 0
D) 2
ANSWER: A
WHY: The unclipped term $-3$ is lower, so the penalty for raising a bad action's probability is kept.

Q: What is the advantage typically estimated with in PPO?
A) Generalised advantage estimation
B) K-means
C) PCA
D) Attention
ANSWER: A
WHY: GAE gives a good bias-variance balance.

Q: What does the entropy bonus do?
A) Encourages exploration by rewarding a more random policy
B) Makes the policy deterministic
C) Adds noise to rewards
D) Changes the clip range
ANSWER: A
WHY: Higher entropy keeps the agent trying different actions.

Q: PPO is a simpler alternative to which method?
A) TRPO, trust region policy optimisation
B) K-means
C) SVMs
D) Random forests
ANSWER: A
WHY: PPO approximates the trust-region constraint with clipping instead of a KL-constrained optimisation.

Q: How is PPO used in RLHF for language models?
A) To fine-tune the model to maximise a reward model's score, often with a KL penalty
B) To tokenise text
C) To cluster prompts
D) To label images
ANSWER: A
WHY: The reward from human preference data guides policy updates while staying close to a reference model.
