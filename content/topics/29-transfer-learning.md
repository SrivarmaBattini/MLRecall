---
id: transfer-learning
title: Transfer Learning
group: Transfer and Few-shot Learning
difficulty: 2
order: 29
summary: Reuse a model trained on a large task as the starting point for a new task with less data.
prerequisites: cnns, neural-networks-intro
---

## What it is and why it matters
Transfer learning takes knowledge gained on one task and applies it to another. Instead of training from scratch, you start from a model pretrained on a large dataset, such as ImageNet for images or a large text corpus for language, and adapt it. It saves data, time and compute, and often gives better accuracy than training alone.

## Intuition
Someone who has learned to play the piano finds the guitar easier than a complete beginner does, because rhythm, reading music and finger coordination transfer. A network that has learned to see edges, textures and shapes in millions of photos does not need to relearn those basics to recognise a rare plant disease.

## Key ideas
- **Source and target task**: where the knowledge comes from and where it is applied.
- **Pretrained model**: a model already trained on a large dataset.
- **Feature extraction**: freeze the pretrained layers and train only a new head.
- **Fine-tuning**: continue training some or all pretrained layers on the new data, with a small learning rate.
- **Layer-wise learning rates and gradual unfreezing**: early layers (general features) change less than late layers (task-specific).
- **New output head**: replace the final layer to match the new classes.
- **Domain shift**: the target data differs from the source data, which reduces the benefit.
- **Negative transfer**: when the source knowledge hurts the target task.
- **Catastrophic forgetting**: fine-tuning can erase useful source knowledge.
- **Data size guidance**: small and similar target data suits feature extraction; larger or different data suits more fine-tuning.
- **Parameter-efficient fine-tuning**: adapters and LoRA train a small number of extra parameters.
- **Domain adaptation**: methods that align source and target distributions.
- **Typical uses**: ImageNet-pretrained CNNs, pretrained BERT or GPT, speech models.

## The math
Fine-tuning minimises the target loss starting from pretrained weights $\theta_0$:

$$\theta^* = \arg\min_\theta \; L_{target}(\theta) \quad \text{initialised at } \theta_0 \text{ from the source task}$$

Low-rank adaptation (LoRA) keeps $W$ frozen and learns a small update $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$ and rank $r \ll \min(d, k)$, so the layer computes $(W + BA)x$.

## Worked example
A ResNet-50 produces a 2,048-dimensional feature vector. Replacing its final layer with a new head for 10 classes adds $2048 \times 10 + 10 = 20{,}490$ trainable parameters. If only that head is trained, the 20 thousand parameters are learned from your small dataset while about 23 million are reused from ImageNet. LoRA on a $4096 \times 4096$ weight with rank $r = 8$ trains $2 \times 4096 \times 8 = 65{,}536$ parameters instead of $16.8$ million.

## Code
```python
import torch
from torch import nn
from torchvision import models

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model.parameters():
    p.requires_grad = False                       # freeze the pretrained backbone
model.fc = nn.Linear(model.fc.in_features, 5)     # new head for 5 classes

trainable = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.Adam(trainable, lr=1e-3)
print("trainable parameters:", sum(p.numel() for p in trainable))
```

## Common mistakes
- Fine-tuning everything with a large learning rate and destroying the pretrained features.
- Forgetting to match the input preprocessing used during pretraining.
- Transferring from a source task that is unrelated to the target.
- Ignoring batch-normalisation behaviour when layers are frozen.
- Evaluating on data that overlaps with the pretraining set.

## Summary
- Transfer learning reuses a pretrained model to learn new tasks with less data.
- Feature extraction freezes the backbone; fine-tuning updates some or all layers carefully.
- Domain shift can limit gains, and negative transfer or forgetting are risks.
- Adapters and LoRA fine-tune large models cheaply.

## Questions
Q: What is feature extraction in transfer learning?
A) Freezing the pretrained layers and training only a new head
B) Training from random weights
C) Removing the head only
D) Collecting more labels
ANSWER: A
WHY: The pretrained backbone supplies fixed features for the new classifier.

Q: What is fine-tuning?
A) Continuing to train some or all pretrained layers on the new task
B) Freezing everything
C) Deleting layers
D) Changing the dataset labels
ANSWER: A
WHY: Weights are adapted to the target data, usually with a small learning rate.

Q: Why use a smaller learning rate when fine-tuning?
A) To avoid destroying the useful pretrained features
B) To make training slower on purpose
C) To remove the head
D) Because gradients are zero
ANSWER: A
WHY: Large updates can erase what the model already knows.

Q: Which layers usually capture the most general features?
A) Early layers
B) The final layer only
C) None
D) The loss
ANSWER: A
WHY: Early layers learn edges and textures that transfer across tasks.

Q: What is negative transfer?
A) Source knowledge that hurts performance on the target task
B) Transfer with no data
C) A faster training speed
D) A negative learning rate
ANSWER: A
WHY: If the tasks are too different, pretrained features can mislead.

Q: What is catastrophic forgetting?
A) Loss of previously learned knowledge when training on new data
B) Forgetting labels in the dataset
C) Deleting the model
D) A bug in the loader
ANSWER: A
WHY: Updating weights for a new task can overwrite earlier knowledge.

Q: Which approach is best with a very small target dataset similar to the source?
A) Feature extraction with a new head
B) Training from scratch
C) Fine-tuning everything with a large rate
D) No pretrained model
ANSWER: A
WHY: Fewer trainable parameters reduces overfitting when data is scarce.

Q: What does LoRA do?
A) Trains small low-rank update matrices while keeping the original weights frozen
B) Compresses images
C) Clusters data
D) Removes layers
ANSWER: A
WHY: It adapts large models with few trainable parameters.

Q: A new head for 10 classes on a 2,048-dimensional feature has how many parameters including biases?
A) 20,490
B) 2,048
C) 20,480
D) 10
ANSWER: A
WHY: $2048 \times 10 + 10 = 20{,}490$.

Q: What is domain shift?
A) The target data distribution differs from the source data distribution
B) A change of optimiser
C) A change of batch size
D) A change of the loss only
ANSWER: A
WHY: Differences between source and target can reduce how well features transfer.
