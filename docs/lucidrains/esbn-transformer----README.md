## ESBN Transformer (wip)

An attempt to merge <a href="https://openreview.net/forum?id=LSFCEb3GYU7">ESBN</a> with Transformers, to endow Transformers with the ability to emergently bind symbols and improve extrapolation. The resulting architecture will be benchmarked with the Give-N task as outlined in <a href="https://arxiv.org/abs/2105.10577">this paper</a>, commonly used to assess whether a child has acquired an understanding of counting.

## Usage

```py
import torch
from esbn_transformer import EsbnTransformer

model = EsbnTransformer(
    num_tokens = 256,
    dim = 512,
    depth = 4,
    max_seq_len = 512
)

x = torch.randint(0, 256, (1, 512))
out = model(x) # (1, 512, 256)
```

## Citations

```py
@misc{webb2020emergent,
    title   = {Emergent Symbols through Binding in External Memory}, 
    author  = {Taylor W. Webb and Ishan Sinha and Jonathan D. Cohen},
    year    = {2020},
    eprint  = {2012.14601},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI}
}
```

```py
@misc{dulberg2021modelling,
    title   = {Modelling the development of counting with memory-augmented neural networks}, 
    author  = {Zack Dulberg and Taylor Webb and Jonathan Cohen},
    year    = {2021},
    eprint  = {2105.10577},
    archivePrefix = {arXiv},
    primaryClass = {cs.AI}
}
```
