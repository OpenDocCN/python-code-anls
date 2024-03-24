## jax2torch

Use Jax functions in Pytorch with DLPack, as outlined <a href="https://gist.github.com/mattjj/e8b51074fed081d765d2f3ff90edf0e9">in a gist</a> by <a href="https://github.com/mattjj">@mattjj</a>. The repository was made for the purposes of making this <a href="https://github.com/spetti/SMURF">differentiable alignment work</a> interoperable with Pytorch projects.

## Install

```bash
$ pip install jax2torch
```

## Memory management

By default, Jax pre-allocates 90% of VRAM, which leaves Pytorch with very little left over.  To prevent this behavior, set the `XLA_PYTHON_CLIENT_PREALLOCATE` environmental variable to false before running any Jax code:

```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
```

## Usage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GBEEnpuCvLS1bhb_xGCO5Y40rFiQrh6G?usp=sharing) Quick test

```python
import jax
import torch
from jax2torch import jax2torch
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# Jax function

@jax.jit
def jax_pow(x, y = 2):
  return x ** y

# convert to Torch function

torch_pow = jax2torch(jax_pow)

# run it on Torch data!

x = torch.tensor([1., 2., 3.])
y = torch_pow(x, y = 3)
print(y)  # tensor([1., 8., 27.])

# And differentiate!

x = torch.tensor([2., 3.], requires_grad = True)
y = torch.sum(torch_pow(x, y = 3))
y.backward()
print(x.grad) # tensor([12., 27.])
```
