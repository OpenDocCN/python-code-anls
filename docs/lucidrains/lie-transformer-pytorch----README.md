<img src="./diagram.png" width="700px"></img>

<img src="./diagram-2.png" width="700px"></img>

## Lie Transformer - Pytorch

Implementation of <a href="https://arxiv.org/abs/2012.10885">Lie Transformer</a>, Equivariant Self-Attention, in Pytorch. Only the SE3 version will be present in this repository, as it may be needed for Alphafold2 replication.

## Install

```bash
$ pip install lie-transformer-pytorch
```

## Usage

```python
import torch
from lie_transformer_pytorch import LieTransformer

model = LieTransformer(
    dim = 512,
    depth = 2,
    heads = 8,
    dim_head = 64,
    liftsamples = 4
)

coors = torch.randn(1, 64, 3)
features = torch.randn(1, 64, 512)
mask = torch.ones(1, 64).bool()

out = model(features, coors, mask = mask) # (1, 256, 512) <- 256 = (seq len * liftsamples)
```

Allowing Lie Transformer take care of embedding the features, just specify the number of unique tokens (node types).

```python
import torch
from lie_transformer_pytorch import LieTransformer

model = LieTransformer(
    num_tokens = 28,           # say 28 different types of atoms
    dim = 512,
    depth = 2,
    heads = 8,
    dim_head = 64,
    liftsamples = 4
)

atoms = torch.randint(0, 28, (1, 64))
coors = torch.randn(1, 64, 3)
mask = torch.ones(1, 64).bool()

out = model(atoms, coors, mask = mask) # (1, 256, 512) <- 256 = (seq len * liftsamples)
```

Although it was not in the paper, I decided to allow for passing in edge information as well (bond types). The edge information will be embedded by the dimension specified, concatted with the location, and passed through the MLP before summed with the attention matrix.

Simply set two more keyword arguments on initialization of the transformer, and then pass in the specific bond types as shape `b x seq x seq`.

```python
import torch
from lie_transformer_pytorch import LieTransformer

model = LieTransformer(
    num_tokens = 28,           # say 28 different types of atoms
    num_edge_types = 4,        # number of different edge types
    edge_dim = 16,             # dimension of edges
    dim = 512,
    depth = 2,
    heads = 8,
    dim_head = 64,
    liftsamples = 4
)

atoms = torch.randint(0, 28, (1, 64))
bonds = torch.randint(0, 4, (1, 64, 64))
coors = torch.randn(1, 64, 3)
mask = torch.ones(1, 64).bool()

out = model(atoms, coors, edges = bonds, mask = mask) # (1, 256, 512) <- 256 = (seq len * liftsamples)
```

## Credit

This repository is largely adapted from <a href="https://github.com/mfinzi/LieConv">LieConv</a>, cited below

## Citations

```bibtex
@misc{hutchinson2020lietransformer,
    title       = {LieTransformer: Equivariant self-attention for Lie Groups}, 
    author      = {Michael Hutchinson and Charline Le Lan and Sheheryar Zaidi and Emilien Dupont and Yee Whye Teh and Hyunjik Kim},
    year        = {2020},
    eprint      = {2012.10885},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@misc{finzi2020generalizing,
    title   = {Generalizing Convolutional Neural Networks for Equivariance to Lie Groups on Arbitrary Continuous Data}, 
    author  = {Marc Finzi and Samuel Stanton and Pavel Izmailov and Andrew Gordon Wilson},
    year    = {2020},
    eprint  = {2002.12880},
    archivePrefix = {arXiv},
    primaryClass = {stat.ML}
}
```
