## Local Attention - Flax

Autoregressive Local Attention - Flax module for Jax

## Install

```bash
$ pip install local-attention-flax
```

## Usage

```python
from jax import random
from local_attention_flax import LocalAttention

attn = LocalAttention(
    dim = 256,
    dim_head = 64,
    heads = 8,
    window_size = 128
)

key = random.PRNGKey(0)
x = random.normal(key, (512, 256))

params = attn.init(key, x)
out = attn.apply(params, x)  # (512, 256)
```
