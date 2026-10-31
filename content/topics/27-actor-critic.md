---
id: actor-critic
title: Actor-Critic Methods
group: Reinforcement Learning
difficulty: 3
order: 27
summary: Combine a policy (actor) with a learned value function (critic) for lower-variance, faster learning.
prerequisites: policy-gradients, q-learning
---

## What it is and why it matters
Actor-critic methods use two components. The actor is a policy that chooses actions. The critic is a value function that estimates how good the situation is and tells the actor how much better or worse its action turned out. This combines the strengths of policy gradients (direct policy learning, continuous actions) and value learning (low-variance, step-by-step updates).

## Intuition
A singer (the actor) performs while a coach (the critic) listens. The coach does not wait until the end of the concert to give feedback. After each phrase the coach says "better than I expected" or "worse than I expected", and the singer adjusts. Meanwhile the coach also improves their own judgement by comparing predictions with what actually happened.

## Key ideas
- **Actor**: the policy $\pi_\theta(a \mid s)$.
- **Critic**: the value function $V_w(s)$ (or $Q_w(s, a)$).
- **TD error as advantage**: $\delta = r + \gamma V(s') - V(s)$ tells whether things went better than predicted.
- **Bias-variance trade-off**: Monte Carlo returns have high variance and no bias; TD estimates have lower variance but some bias from imperfect critics.
- **Online, step-by-step updates**: no need to wait for episode end.
- **Two updates**: the actor ascends $\delta \nabla \log \pi$; the critic reduces $\delta^2$.
- **A2C and A3C**: advantage actor-critic, synchronous or asynchronous with many parallel workers.
- **n-step returns**: use several steps of real reward before bootstrapping.
- **Generalised advantage estimation (GAE)**: a weighted mix of n-step advantages controlled by $\lambda$.
- **Shared network**: actor and critic can share early layers with two output heads.
- **Entropy regularisation**: keeps the policy exploratory.
- **Stability**: bad critics mislead the actor, so learning rates and normalisation matter.

## The math
The TD error serves as an advantage estimate:

$$\delta_t = r_{t+1} + \gamma V_w(s_{t+1}) - V_w(s_t)$$

Updates for one step:

$$\theta \leftarrow \theta + \alpha_\theta \, \delta_t \, \nabla_\theta \log \pi_\theta(a_t \mid s_t) \qquad w \leftarrow w + \alpha_w \, \delta_t \, \nabla_w V_w(s_t)$$

GAE combines TD errors: $\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$.

## Worked example
At state $s$ the critic predicts $V(s) = 2$. The agent acts, receives $r = 1$ and lands in $s'$ where $V(s') = 3$. With $\gamma = 0.9$, $\delta = 1 + 0.9 \times 3 - 2 = 1.7$. The positive error says things went better than the critic expected, so the actor makes that action more likely and the critic raises $V(s)$ towards $3.7$. If $V(s')$ had been $0$ the error would be $1 + 0 - 2 = -1$, and the action would be discouraged.

## Code
```python
import torch
from torch import nn

actor = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
critic = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))
opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), 3e-3)
gamma = 0.99

def ac_step(s, a, r, s2, done):
    v, v2 = critic(s).squeeze(-1), critic(s2).squeeze(-1).detach()
    td_target = r + gamma * (1 - done) * v2
    delta = td_target - v
    logp = torch.log_softmax(actor(s), -1).gather(1, a.unsqueeze(1)).squeeze(1)
    actor_loss = -(logp * delta.detach()).mean()
    critic_loss = delta.pow(2).mean()
    opt.zero_grad(); (actor_loss + 0.5 * critic_loss).backward(); opt.step()
```

## Common mistakes
- Letting the actor's gradient flow into the critic through the advantage, or the reverse; detach where needed.
- Using a critic learning rate that is too low, so the advantage estimates stay poor.
- Not normalising advantages.
- Forgetting terminal handling in the TD target.
- Dropping the entropy bonus and losing exploration too early.

## Summary
- The actor chooses actions; the critic estimates values and supplies a low-variance learning signal.
- The TD error acts as the advantage and drives both updates.
- A2C, A3C and GAE are common practical forms.
- A poor critic can mislead the actor, so both must learn together carefully.

## Questions
Q: What is the role of the critic?
A) To estimate the value of states so it can tell the actor how good its action was
B) To choose actions
C) To store transitions
D) To compute the discount
ANSWER: A
WHY: The critic's value estimates give a learning signal for the actor.

Q: What is the role of the actor?
A) The policy that selects actions
B) The value function
C) The reward
D) The environment
ANSWER: A
WHY: The actor is $\pi_\theta(a \mid s)$.

Q: What is the TD error $\delta$?
A) $r + \gamma V(s') - V(s)$
B) $V(s) - r$
C) $\gamma V(s)$
D) $r$ only
ANSWER: A
WHY: It compares the one-step bootstrapped target with the current estimate.

Q: With $V(s) = 2$, $r = 1$, $\gamma = 0.9$ and $V(s') = 3$, what is $\delta$?
A) 1.7
B) 2.0
C) 0.7
D) 3.7
ANSWER: A
WHY: $1 + 0.9 \times 3 - 2 = 1.7$.

Q: Why does actor-critic usually have lower variance than REINFORCE?
A) It uses a learned value baseline and TD estimates instead of full noisy returns
B) It ignores rewards
C) It uses no randomness
D) It has no policy
ANSWER: A
WHY: Bootstrapped estimates avoid summing many noisy future rewards.

Q: What is the cost of using the critic's estimates?
A) Some bias, because the critic is imperfect
B) No cost
C) Higher variance only
D) Loss of exploration
ANSWER: A
WHY: Bootstrapping from an inexact critic introduces bias.

Q: What does A2C stand for?
A) Advantage Actor-Critic
B) Adaptive Action Control
C) Actor-to-Critic
D) Approximate Q-Control
ANSWER: A
WHY: A2C uses advantage estimates to update the actor.

Q: What does GAE ($\lambda$) do?
A) Blends n-step advantage estimates to trade off bias and variance
B) Changes the environment
C) Sets the discount only
D) Removes the critic
ANSWER: A
WHY: $\lambda$ controls how many steps of real reward are trusted over the critic.

Q: Why detach the advantage when computing the actor loss?
A) So the actor's gradient does not flow into the critic through the advantage
B) To save memory only
C) To change the sign
D) To add noise
ANSWER: A
WHY: The actor should treat the advantage as a fixed weight.

Q: What does a positive TD error tell the actor?
A) The action did better than expected, so make it more likely
B) The action did worse than expected
C) Stop training
D) The state is terminal
ANSWER: A
WHY: Positive advantage reinforces the chosen action.
