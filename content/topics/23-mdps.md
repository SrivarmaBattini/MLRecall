---
id: mdps
title: Markov Decision Processes
group: Reinforcement Learning
difficulty: 2
order: 23
summary: The mathematical frame for reinforcement learning, with states, actions, rewards and a goal of maximising long-term return.
prerequisites:
---

## What it is and why it matters
A Markov decision process (MDP) is the standard model of sequential decision making. An agent repeatedly observes a state, picks an action, receives a reward and moves to a new state. Reinforcement learning is the problem of finding good behaviour in an MDP. Understanding MDPs gives you the vocabulary for every RL method.

## Intuition
Think of a robot in a maze. At each moment it knows where it is (state), chooses a move (action) and gets feedback (reward), such as a penalty for each step and a prize at the exit. The moves it makes now change where it is later, so a good decision must weigh the long-term consequences, not just the immediate reward.

## Key ideas
- **Agent and environment**: the decision maker and everything it interacts with.
- **State $s$**: a description of the situation.
- **Action $a$**: a choice the agent can make.
- **Reward $r$**: a number telling the agent how good the last step was.
- **Transition probability $P(s' \mid s, a)$**: the chance of moving to $s'$ after taking $a$ in $s$.
- **Markov property**: the future depends only on the current state and action, not on the earlier history.
- **Policy $\pi(a \mid s)$**: the agent's rule for choosing actions.
- **Return $G_t$**: the discounted sum of future rewards.
- **Discount factor $\gamma$**: between 0 and 1; small values favour near-term rewards.
- **Episodic versus continuing tasks**: tasks that end versus tasks that go on forever.
- **State-value $V^\pi(s)$**: expected return from $s$ following $\pi$.
- **Action-value $Q^\pi(s, a)$**: expected return after taking $a$ in $s$, then following $\pi$.
- **Bellman equations**: recursive relations that link a value to the values of next states.
- **Optimal policy and value**: $\pi^*$ and $V^*$ achieve the highest value in every state.
- **Dynamic programming**: value iteration and policy iteration solve small MDPs when the model is known.
- **Exploration versus exploitation**: trying new actions versus using what is known to work.

## The math
The return from time $t$ is

$$G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$$

The Bellman expectation equation for a policy $\pi$:

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \left[ r(s, a, s') + \gamma V^\pi(s') \right]$$

and the Bellman optimality equation:

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a)\left[ r + \gamma V^*(s') \right]$$

## Worked example
An agent receives rewards $1, 1, 1$ on three consecutive steps and then the episode ends. With $\gamma = 0.9$, the return is $1 + 0.9 + 0.81 = 2.71$. With $\gamma = 0.5$ it would be $1 + 0.5 + 0.25 = 1.75$, showing how a smaller discount cares less about the future. For a state with two actions leading to values $5$ and $7$ with certainty and immediate rewards $1$ and $0$, using $\gamma = 0.9$ the action values are $1 + 4.5 = 5.5$ and $0 + 6.3 = 6.3$, so the better action is the second.

## Code
```python
import numpy as np

# 2 states, 2 actions: P[a][s, s'] and R[a][s]
P = np.array([[[0.9, 0.1], [0.2, 0.8]],     # action 0
              [[0.5, 0.5], [0.1, 0.9]]])    # action 1
R = np.array([[0.0, 1.0], [0.5, 2.0]])
gamma, V = 0.9, np.zeros(2)

for _ in range(200):                          # value iteration
    Q = R + gamma * (P @ V)                   # shape (actions, states)
    V = Q.max(axis=0)
print("optimal values:", V.round(3), "policy:", Q.argmax(axis=0))
```

## Common mistakes
- Using $\gamma = 1$ in a continuing task, which can make returns infinite.
- Confusing the reward (immediate) with the return (long-term total).
- Defining states that omit information needed to predict the future, breaking the Markov property.
- Confusing $V$ with $Q$.
- Designing rewards that the agent can exploit without achieving the real goal.

## Summary
- An MDP has states, actions, transitions, rewards and a discount factor.
- A policy maps states to actions; the goal is to maximise expected discounted return.
- Value functions and Bellman equations organise the problem.
- Value iteration and policy iteration solve MDPs with known dynamics.

## Questions
Q: What does the Markov property say?
A) The future depends only on the current state and action
B) The future depends on the entire history
C) Rewards are always positive
D) Actions are random
ANSWER: A
WHY: The present state summarises everything relevant from the past.

Q: What is a policy?
A) A rule that maps states to actions or action probabilities
B) The reward function
C) The transition matrix
D) The discount factor
ANSWER: A
WHY: A policy tells the agent what to do in each state.

Q: What is the return $G_t$?
A) The discounted sum of future rewards from time $t$
B) The immediate reward only
C) The number of states
D) The value of the policy
ANSWER: A
WHY: It weights later rewards by powers of $\gamma$.

Q: What does the discount factor $\gamma$ control?
A) How much future rewards matter compared with immediate ones
B) The learning rate
C) The number of actions
D) The size of the state space
ANSWER: A
WHY: A small $\gamma$ makes the agent short-sighted.

Q: With rewards $1, 1, 1$ and $\gamma = 0.9$, what is the return?
A) 2.71
B) 3
C) 1.9
D) 2.0
ANSWER: A
WHY: $1 + 0.9 + 0.81 = 2.71$.

Q: What does $V^\pi(s)$ represent?
A) The expected return starting from $s$ and following $\pi$
B) The reward at $s$
C) The probability of $s$
D) The number of visits
ANSWER: A
WHY: It is the value of being in state $s$ under policy $\pi$.

Q: How does $Q^\pi(s, a)$ differ from $V^\pi(s)$?
A) $Q$ is the value of taking a specific action first, then following $\pi$
B) They are identical
C) $Q$ ignores rewards
D) $V$ depends on actions only
ANSWER: A
WHY: $Q$ fixes the first action, while $V$ averages over the policy's action choice.

Q: What do the Bellman equations express?
A) A recursive relationship between a state's value and the values of its successor states
B) The loss function
C) The gradient
D) The exploration rate
ANSWER: A
WHY: Value equals immediate reward plus discounted value of what comes next.

Q: What does value iteration compute?
A) The optimal value function by repeatedly applying the Bellman optimality update
B) A random policy
C) A clustering
D) The gradient of the policy
ANSWER: A
WHY: It iterates $V \leftarrow \max_a [\ldots]$ until convergence.

Q: What is the exploration-exploitation trade-off?
A) Balancing trying new actions against using actions known to work well
B) Choosing between two loss functions
C) Choosing the batch size
D) Choosing the number of layers
ANSWER: A
WHY: Too little exploration can miss better actions; too much wastes reward.
