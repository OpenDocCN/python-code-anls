## LogAvgExp - Pytorch

Implementation of <a href='https://arxiv.org/abs/2111.01742'>LogAvgExp</a> for Pytorch

## Install

```bash
$ pip install logavgexp-pytorch
```

## Usage

```python
import torch
from logavgexp_pytorch import logavgexp

# basically it is an improved logsumexp (differentiable max)
# normalized for length

x = torch.arange(1000)
y = logavgexp(x, dim = 0, temp = 0.01) # ~998.8

# more than 1 dimension

x = torch.randn(1, 2048, 5)
y = logavgexp(x, dim = 1, temp = 0.2) # (1, 5)

# keep dimension

x = torch.randn(1, 2048, 5)
y = logavgexp(x, dim = 1, temp = 0.2, keepdim = True) # (1, 1, 5)

# masking (False for mask out with large negative value)

x = torch.randn(1, 2048, 5)
m = torch.randint(0, 2, (1, 2048, 1)).bool()

y = logavgexp(x, mask = m, dim = 1, temp = 0.2, keepdim = True) # (1, 1, 5)

```

With learned temperature

```python
# learned temperature
import torch
from torch import nn
from logavgexp_pytorch import logavgexp

learned_temp = nn.Parameter(torch.ones(1) * -5).exp().clamp(min = 1e-8) # make sure temperature can't hit 0

x = torch.randn(1, 2048, 5)
y = logavgexp(x, temp = learned_temp, dim = 1) # (1, 5)
```

Or you can use the `LogAvgExp` class to handle the learned temperature parameter

```python
import torch
from logavgexp_pytorch import LogAvgExp

logavgexp = LogAvgExp(
    temp = 0.01,
    dim = 1,
    learned_temp = True
)

x = torch.randn(1, 2048, 5)
y = logavgexp(x) # (1, 5)
```

## LogAvgExp2D

```python
import torch
from logavgexp_pytorch import LogAvgExp2D

logavgexp_pool = LogAvgExp2D((2, 2), stride = 2) # (2 x 2) pooling

img = torch.randn(1, 16, 64, 64)
out = logavgexp_pool(img) # (1, 16, 32, 32)
```

## Todo

- [ ] look into a https://github.com/openai/triton Triton optimized version

## Citations

```bibtex
@misc{lowe2021logavgexp,
    title   = {LogAvgExp Provides a Principled and Performant Global Pooling Operator}, 
    author  = {Scott C. Lowe and Thomas Trappenberg and Sageev Oore},
    year    = {2021},
    eprint  = {2111.01742},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
