---
id: q-learning
title: Q-Learning
group: Reinforcement Learning
difficulty: 2
order: 24
summary: Learn the value of every action in every state from experience, without a model of the environment.
prerequisites: mdps
---

## What it is and why it matters
Q-learning is a reinforcement learning algorithm that learns the action-value function $Q(s, a)$ directly from interaction. It needs no model of how the environment works, and it learns the optimal policy even while the agent behaves exploratively. It is the foundation of deep Q-networks and many later methods.

## Intuition
Keep a table with a row for each state and a column for each action, holding your current estimate of how good each action is. After every step, compare what you expected with what you actually got (the reward plus your estimate of the best thing you can do next). Nudge the table entry a little towards that better estimate. Over many steps, the table converges to the truth.

## Key ideas
- **Model-free**: no knowledge of transition probabilities is needed.
- **Q-table**: a value for every state-action pair, suitable for small discrete problems.
- **Temporal-difference (TD) learning**: update from one step of experience instead of waiting for the episode to end.
- **Bootstrapping**: updating an estimate using another estimate.
- **TD target**: $r + \gamma \max_{a'} Q(s', a')$.
- **TD error**: target minus current estimate.
- **Learning rate $\alpha$**: how far to move towards the target.
- **Discount factor $\gamma$**: weight on future value.
- **Epsilon-greedy exploration**: with probability $\epsilon$ act randomly, otherwise take the best-known action; $\epsilon$ is often decayed over time.
- **Off-policy**: it learns about the greedy policy while following a different, exploring behaviour policy.
- **SARSA**: the on-policy cousin that uses the action actually taken next.
- **Convergence**: guaranteed in the tabular case if all state-action pairs are visited often and the learning rate is reduced appropriately.
- **Limitation**: tables do not scale to huge or continuous state spaces.

## The math
The Q-learning update after observing $(s, a, r, s')$:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

If $s'$ is terminal, the target is just $r$. Epsilon-greedy picks $a = \arg\max_a Q(s, a)$ with probability $1 - \epsilon$ and a random action otherwise.

## Worked example
Suppose $Q(s, a) = 0.5$, $\alpha = 0.1$, the agent gets $r = 1$, $\gamma = 0.9$ and the best next value is $\max_{a'} Q(s', a') = 2$. The TD target is $1 + 0.9 \times 2 = 2.8$. The TD error is $2.8 - 0.5 = 2.3$. The updated value is $0.5 + 0.1 \times 2.3 = 0.73$. The estimate moved a tenth of the way towards the target.

## Code
```python
# pip install gymnasium
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
Q = np.zeros((env.observation_space.n, env.action_space.n))
alpha, gamma, eps = 0.1, 0.95, 1.0
rng = np.random.default_rng(0)

for episode in range(3000):
    s, _ = env.reset()
    done = False
    while not done:
        a = rng.integers(4) if rng.random() < eps else int(Q[s].argmax())
        s2, r, terminated, truncated, _ = env.step(a)
        target = r + (0 if terminated else gamma * Q[s2].max())
        Q[s, a] += alpha * (target - Q[s, a])
        s, done = s2, terminated or truncated
    eps = max(0.05, eps * 0.999)
print("greedy policy:", Q.argmax(axis=1))
```

## Common mistakes
- Never decaying exploration, or decaying it too fast.
- Forgetting that terminal states have no future value.
- Using a table for a large state space instead of function approximation.
- Setting the learning rate so high that estimates oscillate.
- Confusing Q-learning (off-policy) with SARSA (on-policy).

## Summary
- Q-learning learns $Q(s, a)$ from experience using the TD update with a max over next actions.
- It is model-free and off-policy, so it can learn the optimal policy while exploring.
- Epsilon-greedy balances exploration and exploitation.
- Tables limit it to small problems, which motivates deep Q-networks.

## Questions
Q: What does Q-learning estimate?
A) The value of each action in each state
B) The transition probabilities
C) The reward function
D) The number of states
ANSWER: A
WHY: It learns $Q(s, a)$, the expected return of taking $a$ in $s$ and acting well afterwards.

Q: What does model-free mean?
A) No knowledge of the transition probabilities is required
B) No reward is used
C) No states are used
D) No policy exists
ANSWER: A
WHY: The agent learns from sampled experience rather than a known model.

Q: What is the TD target in Q-learning?
A) $r + \gamma \max_{a'} Q(s', a')$
B) $r$ alone always
C) $Q(s, a)$
D) $\gamma Q(s, a)$
ANSWER: A
WHY: It combines the observed reward with the best estimated future value.

Q: With $Q = 0.5$, $\alpha = 0.1$, $r = 1$, $\gamma = 0.9$ and next max $Q = 2$, what is the updated $Q$?
A) 0.73
B) 2.8
C) 0.5
D) 2.3
ANSWER: A
WHY: Target $2.8$, error $2.3$, update $0.5 + 0.1 \times 2.3 = 0.73$.

Q: What does epsilon-greedy do?
A) Acts randomly with probability $\epsilon$ and greedily otherwise
B) Always acts randomly
C) Always acts greedily
D) Chooses the worst action
ANSWER: A
WHY: It balances exploration and exploitation.

Q: Why is Q-learning called off-policy?
A) It learns the greedy policy's value while following a different behaviour policy
B) It uses no policy
C) It requires a policy gradient
D) It only works offline
ANSWER: A
WHY: The max in the target evaluates the greedy policy regardless of the action actually taken.

Q: How does SARSA differ from Q-learning?
A) SARSA uses the next action actually taken instead of the max
B) SARSA has no reward
C) SARSA uses no learning rate
D) SARSA is model-based
ANSWER: A
WHY: SARSA is on-policy; its target uses $Q(s', a')$ for the chosen $a'$.

Q: What is the main limitation of tabular Q-learning?
A) It does not scale to large or continuous state spaces
B) It cannot learn
C) It needs a model
D) It has no hyperparameters
ANSWER: A
WHY: A table needs an entry per state-action pair.

Q: What is the TD error?
A) The difference between the TD target and the current estimate
B) The reward only
C) The policy's entropy
D) The learning rate
ANSWER: A
WHY: It measures how wrong the current estimate was.

Q: What is the target at a terminal next state?
A) Just the reward $r$
B) $r + \gamma \max Q$
C) Zero always
D) Infinity
ANSWER: A
WHY: There is no future after a terminal state, so there is nothing to bootstrap.
