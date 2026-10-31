---
id: cnns
title: Convolutional Neural Networks
group: Deep Learning
difficulty: 2
order: 11
summary: Slide small shared filters across an image to detect patterns, from edges to whole objects.
prerequisites: neural-networks-intro
---

## What it is and why it matters
A convolutional neural network (CNN) is built for grid-like data such as images. Instead of connecting every pixel to every neuron, it slides small filters across the image. This uses far fewer parameters and recognises a pattern wherever it appears, which made CNNs the foundation of modern computer vision.

## Intuition
Imagine a small magnifying window that looks for one pattern, say a vertical edge. You slide it over the whole picture and mark where the pattern is strong. Many windows, each looking for a different pattern, give a stack of maps. Deeper layers look for combinations: edges form corners, corners form eyes, eyes form faces.

## Key ideas
- **Convolution**: sliding a small filter over the input and computing a weighted sum at each position.
- **Filter (kernel)**: a small grid of learnable weights, such as 3 by 3.
- **Feature map**: the output of one filter across the image.
- **Channels**: the depth of an input (RGB has 3) or the number of feature maps.
- **Stride**: how many pixels the filter moves each step.
- **Padding**: adding border pixels; "same" keeps the size, "valid" uses none.
- **Local connectivity and receptive field**: each output sees only a small region, which grows with depth.
- **Parameter sharing**: one filter is reused at every position, cutting parameters and giving translation equivariance.
- **Pooling**: max or average pooling reduces size and adds small-shift tolerance.
- **Typical stack**: convolution, ReLU, pooling, repeated, then fully connected layers.
- **Batch normalisation**: stabilises and speeds up training.
- **Classic architectures**: LeNet, AlexNet, VGG and ResNet (residual skip connections enable very deep networks).
- **Data augmentation**: flips, crops and colour shifts to reduce overfitting.

## The math
For input size $n$, filter size $f$, padding $p$ and stride $s$, the output size is

$$\left\lfloor \frac{n + 2p - f}{s} \right\rfloor + 1$$

A convolution layer with $C_{in}$ input channels, $C_{out}$ filters of size $f \times f$ has $(f \cdot f \cdot C_{in} + 1) \cdot C_{out}$ parameters, where the $+1$ is the bias per filter.

## Worked example
A $32 \times 32$ image goes through a $5 \times 5$ filter with no padding and stride 1: the output is $32 - 5 + 1 = 28$, so $28 \times 28$. A $2 \times 2$ max pool with stride 2 then halves it to $14 \times 14$. A layer taking 3 channels to 16 filters of size $3 \times 3$ has $(3 \cdot 3 \cdot 3 + 1) \times 16 = 448$ parameters, regardless of image size.

## Code
```python
import torch
from torch import nn

cnn = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(), nn.Linear(32 * 8 * 8, 10),
)
x = torch.randn(4, 3, 32, 32)       # batch of four 32x32 RGB images
print(cnn(x).shape)                  # torch.Size([4, 10])
```

## Common mistakes
- Forgetting that the filter weights are shared, and miscounting parameters.
- Using fully connected layers on raw large images, which explodes the parameter count.
- Pooling too aggressively and losing fine detail.
- Skipping data augmentation on small image datasets.
- Not normalising pixel values.

## Summary
- CNNs use small shared filters, local connectivity and pooling.
- Early layers find edges; later layers combine them into complex patterns.
- Stride, padding and filter size set the output size.
- Residual connections and batch norm make deep CNNs trainable.

## Questions
Q: What is a convolutional filter?
A) A small grid of learnable weights slid across the input
B) A fully connected layer
C) A pooling operation
D) The loss function
ANSWER: A
WHY: A filter computes a weighted sum over a local patch at each position.

Q: What is parameter sharing in a CNN?
A) The same filter weights are reused at every position
B) Each pixel has its own weights
C) Layers share the loss
D) Batches share labels
ANSWER: A
WHY: Reusing one filter across the image cuts parameters and detects a pattern anywhere.

Q: A $32 \times 32$ input, $5 \times 5$ filter, stride 1, no padding gives what output size?
A) $28 \times 28$
B) $32 \times 32$
C) $27 \times 27$
D) $36 \times 36$
ANSWER: A
WHY: $(32 + 0 - 5)/1 + 1 = 28$.

Q: What does max pooling do?
A) Keeps the largest value in each window, shrinking the map
B) Adds padding
C) Increases the number of channels
D) Normalises the data
ANSWER: A
WHY: It downsamples while keeping the strongest responses.

Q: How many parameters does a conv layer with 3 input channels, 16 filters and $3 \times 3$ kernels have, including biases?
A) 432
B) 448
C) 144
D) 16
ANSWER: B
WHY: $(3 \cdot 3 \cdot 3 + 1) \times 16 = 448$.

Q: What does padding with "same" aim to do?
A) Keep the output the same spatial size as the input (for stride 1)
B) Remove edges
C) Double the image
D) Reduce channels
ANSWER: A
WHY: Border padding lets the filter centre on edge pixels so the size is preserved.

Q: What is the receptive field of a neuron?
A) The region of the input that can affect it
B) The learning rate
C) The batch size
D) The number of filters
ANSWER: A
WHY: It is the part of the input the neuron "sees", and it grows with depth.

Q: Why did residual (skip) connections matter for deep CNNs?
A) They let gradients flow and make very deep networks trainable
B) They remove convolution
C) They shrink the image
D) They replace the loss
ANSWER: A
WHY: Identity shortcuts ease optimisation and reduce vanishing gradients.

Q: Which technique reduces overfitting on small image datasets?
A) Data augmentation such as flips and crops
B) Removing pooling
C) Using larger images only
D) Dropping labels
ANSWER: A
WHY: Augmentation creates varied training examples from the same data.

Q: Convolution gives which useful property?
A) Translation equivariance: a shifted input gives a shifted feature map
B) Rotation invariance for free
C) No parameters
D) Perfect accuracy
ANSWER: A
WHY: Because the same filter slides everywhere, a pattern's response moves with the pattern.
