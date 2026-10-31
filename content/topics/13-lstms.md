---
id: lstms
title: Long Short-Term Memory (LSTM)
group: Deep Learning
difficulty: 3
order: 13
summary: Use gates to decide what to remember, forget and output, so sequences can be modelled over long spans.
prerequisites: rnns
---

## What it is and why it matters
A long short-term memory network (LSTM) is a recurrent network with a protected memory cell and gates that control it. It was designed to fix the vanishing-gradient problem of simple RNNs, so it can carry information across many time steps. LSTMs powered speech recognition, translation and text models before transformers.

## Intuition
Think of a conveyor belt running alongside the main process, the cell state. Information can ride along unchanged for a long time. At each step three gatekeepers decide what to do: one decides what to throw off the belt, one decides what new information to put on it, and one decides what to read off it as output. Because the belt is mostly added to rather than rewritten, learning signals travel back along it without fading.

## Key ideas
- **Cell state $c_t$**: the long-term memory running through time.
- **Hidden state $h_t$**: the short-term output passed on and used for predictions.
- **Forget gate $f_t$**: values between 0 and 1 that decide how much of the old cell state to keep.
- **Input gate $i_t$**: decides how much of the new candidate to write.
- **Candidate memory $\tilde{c}_t$**: the proposed new content, using $\tanh$.
- **Output gate $o_t$**: decides how much of the cell state to expose as $h_t$.
- **Gates use sigmoids**: outputs in $(0, 1)$ act like soft switches.
- **Additive update**: the cell state is updated by addition, which keeps gradients healthier over time.
- **Long-term dependencies**: facts such as a subject at the start of a sentence can be remembered much later.
- **GRU**: a simpler variant with two gates (update and reset) and no separate cell state.
- **Stacked and bidirectional LSTMs**: more depth and context.
- **Practical tips**: gradient clipping, dropout between layers, and initialising the forget-gate bias to a positive value.

## The math
With input $x_t$, previous hidden state $h_{t-1}$ and previous cell state $c_{t-1}$:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f) \qquad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i) \qquad o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c) \qquad c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \qquad h_t = o_t \odot \tanh(c_t)$$

where $\odot$ is element-wise multiplication.

## Worked example
Consider one memory unit with $c_{t-1} = 2$. The forget gate is $f_t = 0.9$, the input gate is $i_t = 0.5$ and the candidate is $\tilde{c}_t = 0.4$. The new cell state is $c_t = 0.9 \times 2 + 0.5 \times 0.4 = 1.8 + 0.2 = 2.0$. The memory was mostly kept (forget gate near 1) with a small addition. If $f_t$ were 0, the old value would be erased completely.

## Code
```python
import torch
from torch import nn

lstm = nn.LSTM(input_size=8, hidden_size=32, num_layers=2,
               batch_first=True, dropout=0.2, bidirectional=False)
x = torch.randn(4, 50, 8)                 # 4 sequences, 50 steps
out, (h_n, c_n) = lstm(x)
print(out.shape, h_n.shape, c_n.shape)     # (4,50,32) (2,4,32) (2,4,32)
```

## Common mistakes
- Thinking an LSTM has no vanishing-gradient problem at all; it is reduced, not removed.
- Confusing the hidden state with the cell state.
- Forgetting gradient clipping on long sequences.
- Using an LSTM when a transformer or a simple model would be faster and just as good.
- Ignoring padding in variable-length batches.

## Summary
- An LSTM keeps a cell state controlled by forget, input and output gates.
- The additive cell update lets information and gradients persist over many steps.
- The hidden state is the gated, read-out version of the cell state.
- GRUs are a lighter alternative; transformers have largely replaced both for long text.

## Questions
Q: What is the job of the forget gate?
A) Decide how much of the previous cell state to keep
B) Decide the output class
C) Set the learning rate
D) Add new input
ANSWER: A
WHY: It outputs values between 0 and 1 that scale the old cell state.

Q: What does the input gate control?
A) How much of the new candidate memory is written to the cell
B) How much old memory is erased
C) The batch size
D) The loss
ANSWER: A
WHY: The input gate scales the candidate before it is added to the cell state.

Q: What does the output gate control?
A) How much of the cell state is exposed as the hidden state
B) How much is forgotten
C) The weight decay
D) The sequence length
ANSWER: A
WHY: $h_t = o_t \odot \tanh(c_t)$, so the output gate filters the cell's content.

Q: Which equation updates the cell state?
A) $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
B) $c_t = \tanh(h_t)$
C) $c_t = x_t$
D) $c_t = o_t + h_t$
ANSWER: A
WHY: The cell state combines kept old memory and gated new memory by addition.

Q: Why does the additive cell update help gradients?
A) Gradients can flow back through the cell state without repeated squashing
B) It removes the loss
C) It doubles the data
D) It stops training
ANSWER: A
WHY: Addition with a gate near 1 gives a nearly direct path, so gradients decay much less.

Q: What activation do the gates use?
A) Sigmoid
B) ReLU
C) Softmax
D) Identity
ANSWER: A
WHY: Sigmoid outputs in $(0, 1)$ act as soft on/off switches.

Q: With $c_{t-1} = 2$, $f_t = 0.9$, $i_t = 0.5$, $\tilde{c}_t = 0.4$, what is $c_t$?
A) 1.8
B) 2.0
C) 2.2
D) 0.4
ANSWER: B
WHY: $0.9 \times 2 + 0.5 \times 0.4 = 1.8 + 0.2 = 2.0$.

Q: How does a GRU differ from an LSTM?
A) It uses fewer gates and no separate cell state
B) It has more gates
C) It has no gates
D) It cannot handle sequences
ANSWER: A
WHY: A GRU merges the cell and hidden states and uses update and reset gates.

Q: What does a forget gate output of exactly 0 do?
A) Erases the old cell content for that unit
B) Keeps everything
C) Doubles the memory
D) Stops the network
ANSWER: A
WHY: Multiplying the old cell state by 0 discards it.

Q: A common practical trick for LSTMs is to do what?
A) Clip gradients
B) Remove gates
C) Use zero input
D) Disable backpropagation
ANSWER: A
WHY: Clipping prevents exploding gradients, which can still occur in LSTMs.
