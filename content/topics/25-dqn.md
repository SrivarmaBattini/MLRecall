---
id: dqn
title: Deep Q-Networks (DQN)
group: Reinforcement Learning
difficulty: 3
order: 25
summary: Replace the Q-table with a neural network, stabilised by experience replay and a target network.
prerequisites: q-learning, neural-networks-intro
---

## What it is and why it matters
A deep Q-network (DQN) uses a neural network to approximate $Q(s, a)$, so Q-learning can handle huge or continuous state spaces such as the pixels of a video game. DeepMind's DQN learned to play many Atari games from raw pixels, showing that deep learning and reinforcement learning could be combined.

## Intuition
A table cannot store a value for every possible screen image, so a network learns to estimate values and generalise to similar screens. But naive training is unstable: the data is correlated (consecutive frames look alike) and the target keeps moving as the network changes. DQN adds two stabilisers: a memory of past experience to sample from, and a slowly updated copy of the network to provide steady targets.

## Key ideas
- **Function approximation**: a network $Q_\theta(s, a)$ with parameters $\theta$.
- **Input and output**: the network takes the state and outputs one value per action.
- **Experience replay buffer**: stores past transitions $(s, a, r, s', done)$.
- **Random mini-batches**: sampling from the buffer breaks correlation between consecutive samples and reuses data.
- **Target network**: a delayed copy $Q_{\theta^-}$ used to compute targets, updated every few thousand steps.
- **Loss**: squared error or Huber loss between $Q_\theta(s, a)$ and the target.
- **Epsilon-greedy exploration** with a decaying schedule.
- **Deadly triad**: function approximation, bootstrapping and off-policy learning together can diverge.
- **Frame stacking**: stacking recent frames so the state shows motion.
- **Reward clipping and gradient clipping**: keep learning stable across games.
- **Double DQN**: reduces overestimation by choosing the action with one network and evaluating it with the other.
- **Dueling networks**: separate streams for state value and action advantages.
- **Prioritised replay**: sample transitions with large errors more often.
- **Limitations**: discrete actions only, and sample-hungry.

## The math
For a sampled transition, the target uses the target network:

$$y = r + \gamma \max_{a'} Q_{\theta^-}(s', a') \quad \text{(or } y = r \text{ if terminal)}$$

and the network minimises

$$L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \left( y - Q_\theta(s, a) \right)^2 \right]$$

Double DQN replaces the target with $y = r + \gamma \, Q_{\theta^-}\left(s', \arg\max_{a'} Q_\theta(s', a')\right)$.

## Worked example
A transition has $r = 1$, $\gamma = 0.99$ and the target network says the best next action has value $10$. The target is $y = 1 + 0.99 \times 10 = 10.9$. If the online network currently predicts $Q_\theta(s, a) = 9$, the squared error is $(10.9 - 9)^2 = 3.61$, and a gradient step increases that prediction. If the transition ended the episode, the target would simply be $y = 1$.

## Code
```python
import torch
from torch import nn

q_net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
target_net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2))
target_net.load_state_dict(q_net.state_dict())
opt, gamma = torch.optim.Adam(q_net.parameters(), 1e-3), 0.99

def dqn_step(s, a, r, s2, done):                  # a mini-batch sampled from the replay buffer
    with torch.no_grad():
        target = r + gamma * (1 - done) * target_net(s2).max(dim=1).values
    pred = q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
    loss = nn.functional.smooth_l1_loss(pred, target)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
```

## Common mistakes
- Training on consecutive transitions without a replay buffer.
- Forgetting to treat terminal transitions with no bootstrapped future value.
- Updating the target network every step, which removes its stabilising effect.
- Letting gradients flow through the target computation.
- Using DQN for continuous actions, where it does not apply directly.

## Summary
- DQN approximates $Q(s, a)$ with a neural network and trains it with Q-learning targets.
- Experience replay and a target network stabilise training.
- Double, dueling and prioritised variants improve it further.
- It suits discrete action spaces; continuous control needs other methods.

## Questions
Q: Why use a neural network instead of a Q-table?
A) To handle very large or continuous state spaces by generalising across states
B) To avoid using rewards
C) To make the problem episodic
D) To remove exploration
ANSWER: A
WHY: A network can estimate values for states it has never seen exactly.

Q: What does the replay buffer do?
A) Stores past transitions to sample from randomly, breaking correlation
B) Stores the policy
C) Stores the loss
D) Stores the weights
ANSWER: A
WHY: Random mini-batches from past experience decorrelate data and reuse it.

Q: What is the purpose of the target network?
A) To provide stable targets that change slowly
B) To select actions
C) To store the replay buffer
D) To compute exploration
ANSWER: A
WHY: A moving target makes training unstable; a delayed copy steadies it.

Q: What is the DQN target for a non-terminal transition?
A) $r + \gamma \max_{a'} Q_{\theta^-}(s', a')$
B) $r$
C) $Q_\theta(s, a)$
D) $\gamma r$
ANSWER: A
WHY: It uses the target network's best next-state value.

Q: What is the target for a transition that ends the episode?
A) $r$
B) $r + \gamma \max Q$
C) 0
D) $\gamma$
ANSWER: A
WHY: There is no next state to bootstrap from.

Q: What problem does Double DQN address?
A) Overestimation of action values from taking a max over noisy estimates
B) Slow data loading
C) Lack of exploration
D) Continuous actions
ANSWER: A
WHY: Using one network to choose and another to evaluate reduces upward bias.

Q: What does frame stacking provide?
A) Information about motion by including several recent frames in the state
B) More colour channels
C) Larger rewards
D) Faster GPU
ANSWER: A
WHY: One frame cannot show velocity; several frames can.

Q: Which action spaces suit DQN directly?
A) Discrete
B) Continuous
C) Neither
D) Only infinite
ANSWER: A
WHY: DQN takes a max over a finite set of actions.

Q: What is the deadly triad?
A) Function approximation, bootstrapping and off-policy learning together
B) Rewards, states and actions
C) Exploration, exploitation and discounting
D) Actor, critic and policy
ANSWER: A
WHY: This combination can make value estimates diverge, motivating stabilisers like replay and target networks.

Q: What does prioritised experience replay do?
A) Samples transitions with larger TD errors more often
B) Removes old transitions
C) Doubles the buffer
D) Skips terminal states
ANSWER: A
WHY: Surprising transitions offer more to learn from.
