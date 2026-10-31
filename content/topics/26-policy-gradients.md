---
id: policy-gradients
title: Policy Gradients
group: Reinforcement Learning
difficulty: 3
order: 26
summary: Learn the policy directly by nudging its parameters to make high-return actions more likely.
prerequisites: mdps, backpropagation
---

## What it is and why it matters
Policy-gradient methods learn the policy itself, a function that outputs action probabilities, instead of learning values and acting greedily. They handle continuous actions naturally, can represent stochastic policies and form the basis of actor-critic methods and PPO.

## Intuition
Imagine a player who chooses moves with some randomness. After a game, if it went well, make every move that was played a little more likely next time; if it went badly, make them a little less likely. Over many games, good behaviour becomes more probable. The policy gradient is the mathematical version of "do more of what worked".

## Key ideas
- **Parameterised policy $\pi_\theta(a \mid s)$**: a network whose weights are $\theta$.
- **Stochastic policy**: outputs a distribution over actions; softmax for discrete actions, Gaussian for continuous ones.
- **Objective $J(\theta)$**: the expected return under the policy.
- **Policy-gradient theorem**: gives $\nabla_\theta J$ without differentiating the environment.
- **Log-derivative trick**: $\nabla \pi = \pi \nabla \log \pi$.
- **REINFORCE**: a Monte Carlo method using full-episode returns.
- **Return $G_t$**: the discounted reward from step $t$ onwards.
- **High variance**: returns are noisy, which makes learning slow and erratic.
- **Baseline**: subtract a value (often $V(s)$) from the return to cut variance without adding bias.
- **Advantage $A(s, a) = Q(s, a) - V(s)$**: how much better an action is than average.
- **On-policy**: data must come from the current policy.
- **Entropy bonus**: rewards randomness to maintain exploration.
- **Strengths and weaknesses**: works with continuous actions and gives smooth policy changes, but needs many samples.

## The math
The policy-gradient theorem gives

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \, G_t \right]$$

With a baseline $b(s_t)$ the estimate becomes $\nabla_\theta \log \pi_\theta(a_t \mid s_t) \left(G_t - b(s_t)\right)$, which has the same expected value but lower variance. The gradient-ascent update is $\theta \leftarrow \theta + \alpha \, \nabla_\theta J$.

## Worked example
A softmax policy picks action 1 with probability $0.7$. The agent takes action 1 and the episode return is $G = 10$. Without a baseline the update direction is $\nabla \log \pi(1) \times 10$. With a baseline of $6$, the factor is $10 - 6 = 4$, so the push is smaller but still positive; the action did better than average. If the return had been $3$, the factor would be $3 - 6 = -3$, and the action's probability would be reduced.

## Code
```python
import torch
from torch import nn

policy = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 2))
opt, gamma = torch.optim.Adam(policy.parameters(), 1e-2), 0.99

def reinforce_update(states, actions, rewards):
    G, returns = 0.0, []
    for r in reversed(rewards):                         # discounted returns
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)   # simple baseline
    logp = torch.log_softmax(policy(torch.stack(states)), dim=-1)
    chosen = logp[range(len(actions)), actions]
    loss = -(chosen * returns).sum()                    # ascend expected return
    opt.zero_grad(); loss.backward(); opt.step()
```

## Common mistakes
- Forgetting the minus sign, so the code minimises return instead of maximising it.
- Not using a baseline and wondering why learning is noisy.
- Reusing old data from a different policy with a plain on-policy method.
- Letting the policy become deterministic too early, losing exploration.
- Computing returns forwards instead of backwards through the episode.

## Summary
- Policy-gradient methods optimise the policy parameters directly using $\nabla \log \pi \times$ return.
- REINFORCE is the simple Monte Carlo version; baselines and advantages cut variance.
- They support continuous and stochastic actions but are sample-hungry and on-policy.
- Actor-critic methods build on this idea with a learned value function.

## Questions
Q: What does a policy-gradient method learn directly?
A) The policy's parameters
B) A Q-table
C) The transition model
D) The reward function
ANSWER: A
WHY: It adjusts $\theta$ in $\pi_\theta(a \mid s)$ to increase expected return.

Q: What does the log-derivative trick give?
A) $\nabla \pi = \pi \nabla \log \pi$, allowing gradient estimation by sampling
B) A new reward
C) A faster environment
D) A discount factor
ANSWER: A
WHY: It turns the gradient of an expectation into an expectation of a gradient of log-probabilities.

Q: In REINFORCE, what multiplies $\nabla \log \pi(a_t \mid s_t)$?
A) The return $G_t$
B) The learning rate only
C) The state
D) The next action
ANSWER: A
WHY: Actions followed by high returns get their probability increased more.

Q: Why subtract a baseline from the return?
A) To reduce variance without changing the expected gradient
B) To make rewards negative
C) To increase bias
D) To remove the policy
ANSWER: A
WHY: A baseline that does not depend on the action leaves the gradient unbiased.

Q: What is the advantage function?
A) $Q(s, a) - V(s)$, how much better an action is than average in that state
B) The reward
C) The loss
D) The policy entropy
ANSWER: A
WHY: It measures relative benefit of an action.

Q: What is a major drawback of REINFORCE?
A) High variance of gradient estimates
B) It cannot use neural networks
C) It needs a model of the environment
D) It only handles tiny action spaces
ANSWER: A
WHY: Monte Carlo returns are noisy, so learning is slow and unstable.

Q: Which output distribution suits continuous actions?
A) Gaussian with a learned mean and spread
B) Softmax over a fixed list
C) One-hot
D) Uniform over states
ANSWER: A
WHY: A Gaussian policy samples real-valued actions.

Q: What does on-policy mean here?
A) Training data must be collected with the current policy
B) The policy is fixed
C) The data are labelled
D) The model is offline
ANSWER: A
WHY: The gradient estimate assumes actions came from $\pi_\theta$.

Q: What does an entropy bonus do?
A) Encourages exploration by rewarding more random policies
B) Reduces the learning rate
C) Increases the discount
D) Removes the baseline
ANSWER: A
WHY: It prevents the policy from collapsing to a single action too early.

Q: Why compute discounted returns backwards through an episode?
A) $G_t = r_{t+1} + \gamma G_{t+1}$ lets each return reuse the next one
B) To reverse the rewards
C) To skip steps
D) To change the policy
ANSWER: A
WHY: The recursion makes the computation simple and efficient.
