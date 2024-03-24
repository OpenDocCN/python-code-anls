<img src="./scattering.png" width="600px"></img>

## Scattering Compositional Learner

Implementation of <a href="https://arxiv.org/abs/2007.04212">Scattering Compositional Learner</a>, which reached superhuman levels on Raven's Progressive Matrices, a type of IQ test for analogical reasoning.

This repository is meant to be exploratory, so it may not follow the exact architecture of the paper down to the T. It is meant to find the underlying inductive bias that could be exported for use in attention networks. The paper suggests this to be the 'Scattering Transform', which is basically  grouped convolutions but where each group is tranformed by one shared neural network.

If you would like the exact architecture used in the paper, the <a href="https://github.com/dhh1995/SCL">official repository is here</a>.

## Install

```bash
$ pip install scattering-transform
```

## Use

Complete Scattering Compositional Learner network

```python
import torch
import torch.nn.functional as F
from scattering_transform import SCL, SCLTrainingWrapper

# data - (batch, number of choices, channel dimension, image height, image width)

questions = torch.randn(1, 8, 1, 160, 160)
answers   = torch.randn(1, 8, 1, 160, 160)
labels    = torch.tensor([2])

# instantiate model

model = SCL(
    image_size = 160,                           # size of image
    set_size = 9,                               # number of questions + 1 answer
    conv_channels = [1, 16, 16, 32, 32, 32],    # convolutional channel progression, 1 for greyscale, 3 for rgb
    conv_output_dim = 80,                       # model dimension, the output dimension of the vision net
    attr_heads = 10,                            # number of attribute heads
    attr_net_hidden_dims = [128],               # attribute scatter transform MLP hidden dimension(s)
    rel_heads = 80,                             # number of relationship heads
    rel_net_hidden_dims = [64, 23, 5]           # MLP for relationship net
)

model = SCLTrainingWrapper(model)
logits = model(questions, answers) # (1, 8) - the logits of each answer being the correct match

# train

loss = F.cross_entropy(logits, labels)
loss.backward()
```

Scattering Transform, which is basically one MLP that acts over groups of the dimension

```python
import torch
from scattering_transform import ScatteringTransform

# for potential use in a Transformer

mlp = ScatteringTransform(
    dims = [1024, 4096, 1024],    # MLP - dimension in -> hidden sizes -> dimension out
    heads = 16,                   # number of groups (heads)
    activation = nn.LeakyReLU     # activation to use in the MLP
)

x = torch.randn(1, 512, 1024)
mlp(x) # (1, 512, 1024)
```

## Citation

```bibtex
@misc{wu2020scattering,
    title={The Scattering Compositional Learner: Discovering Objects, Attributes, Relationships in Analogical Reasoning},
    author={Yuhuai Wu and Honghua Dong and Roger Grosse and Jimmy Ba},
    year={2020},
    eprint={2007.04212},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
