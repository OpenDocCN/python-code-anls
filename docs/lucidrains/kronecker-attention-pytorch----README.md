<img src="./kronecker.png" width="650px"></img>

## Kronecker Attention Pytorch

Implementation of <a href="https://arxiv.org/abs/2007.08442">Kronecker Attention</a> in Pytorch. Results look less than stellar, but if someone found some context where this architecture works well, please post in the issues and let everyone know.

## Install

```bash
$ pip install kronecker_attention_pytorch
```

## Usage

```python
import torch
from kronecker_attention_pytorch import KroneckerSelfAttention

attn = KroneckerSelfAttention(
    chan = 32,
    heads = 8,
    dim_heads = 64
)

x = torch.randn(1, 32, 256, 512)
attn(x) # (1, 32, 256, 512)
```

## Citations

```bibtex
@article{Gao_2020,
   title={Kronecker Attention Networks},
   url={http://dx.doi.org/10.1145/3394486.3403065},
   journal={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
   publisher={ACM},
   author={Gao, Hongyang and Wang, Zhengyang and Ji, Shuiwang},
   year={2020},
   month={Aug}
}
```
