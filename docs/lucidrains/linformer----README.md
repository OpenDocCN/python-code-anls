## Linformer for Pytorch

An implementation of Linformer in Pytorch. Linformer comes with two deficiencies. (1) It does not work for the auto-regressive case. (2) Assumes a fixed sequence length. However, if benchmarks show it to perform well enough, it will be added to <a href="https://github.com/lucidrains/linear-attention-transformer">this repository</a> as a self-attention layer to be used in the encoder.

Linformer has been <a href="https://ai.facebook.com/blog/how-facebook-uses-super-efficient-ai-models-to-detect-hate-speech/">put into production</a> by Facebook!

## Install

```bash
$ pip install linformer
```

## Usage

Linformer language model

```python
import torch
from linformer import LinformerLM

model = LinformerLM(
    num_tokens = 20000,
    dim = 512,
    seq_len = 4096,
    depth = 12,
    heads = 8,
    dim_head = 128,        # be able to set the dimension of each head in multi-head attention
    k = 256,               # this is the k that the key/values are projected to along the sequence dimension
    one_kv_head = True,    # share one key/value head across all heads
    share_kv = False,      # share the same projection for keys and values
    reversible = True      # make network reversible, like Reformer
)

x = torch.randint(0, 20000, (1, 4096))
model(x) # (1, 4096, 20000)
```

Linformer

```python
import torch
from linformer import Linformer

model = Linformer(
    dim = 512,
    seq_len = 4096,
    depth = 12,
    heads = 8,
    k = 256,
    one_kv_head = True,
    share_kv = True
)

x = torch.randn(1, 4096, 512)
model(x) # (1, 4096, 512)
```

Single Self-Attention layer

```python
import torch
from linformer import LinformerSelfAttention

attn = LinformerSelfAttention(
    dim = 512,
    seq_len = 4096,
    heads = 8,
    k = 256,
    one_kv_head = True,
    share_kv = True
)

x = torch.randn(1, 4096, 512)
attn(x) # (1, 4096, 512)
```

Self-Attention layer above receiving contextual keys. The sequence length is validated on the length of the contextual keys instead of the source sequence.

```python
import torch
from linformer import LinformerSelfAttention

attn = LinformerSelfAttention(
    dim = 512,
    seq_len = 8192,
    heads = 8,
    k = 256,
    one_kv_head = True,
    share_kv = True
)

x = torch.randn(1, 2048, 512)
context = torch.randn(1, 8192, 512)
attn(x, context) # (1, 2048, 512)
```

## Citations

```bibtex
@misc{wang2020linformer,
    title={Linformer: Self-Attention with Linear Complexity},
    author={Sinong Wang and Belinda Z. Li and Madian Khabsa and Han Fang and Hao Ma},
    year={2020},
    eprint={2006.04768},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

```bibtex
@inproceedings{kitaev2020reformer,
    title       = {Reformer: The Efficient Transformer},
    author      = {Nikita Kitaev and Lukasz Kaiser and Anselm Levskaya},
    booktitle   = {International Conference on Learning Representations},
    year        = {2020},
    url         = {https://openreview.net/forum?id=rkgNKkHtvB}
}
```
