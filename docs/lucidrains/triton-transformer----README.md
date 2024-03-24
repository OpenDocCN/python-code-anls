## Transformer in Triton (wip)

Implementation of a Transformer, but completely in <a href="https://triton-lang.org/">Triton</a>. I'm completely new to lower-level neural net code, so this repository will mostly be a learning experience, with the end-goal being a vanilla transformer that is faster and more efficient to train.

## Results

Layernorm forward

<img src="./images/layernorm.png" width="400px"></img>

Layernorm forwards and backwards

<img src="./images/layernorm-forward-backward.png" width="400px"></img>

Softmax forwards and backwards

<img src="./images/softmax.png" width="400px"></img>

## Install

```bash
$ pip install triton-transformer
```

## Usage

```python
import torch
from triton_transformer import Transformer

model = Transformer(
    num_tokens = 256,       # vocab size
    max_seq_len = 1024,     # maximum sequence length
    dim = 512,              # dimension
    depth = 6,              # depth
    heads = 8,              # number of heads
    dim_head = 64,          # dimension per head
    causal = True,          # autoregressive or not
    attn_dropout = 0.1,     # attention dropout
    ff_dropout = 0.1,       # feedforward dropout
    use_triton = True       # use this to turn on / off triton
).cuda()

x = torch.randint(0, 256, (1, 1024)).cuda()
logits = model(x) # (1, 1024, 256)
```

To train, just pass in the labels with the keyword `labels` on forward, and the cross entropy loss will be returned for backprop.

ex. BERT

```python
import torch
from triton_transformer import Transformer

model = Transformer(
    num_tokens = 20000,
    max_seq_len = 512,
    dim = 512,
    depth = 12,
    heads = 8,
    dim_head = 64,
    use_triton = True
).cuda()

x = torch.randint(0, 20000, (1, 512)).cuda()
labels = torch.randint(0, 20000, (1, 512)).cuda()
mask = torch.ones(1, 512).bool().cuda()

loss = model(x, mask = mask, labels = labels)
loss.backward()
```

## Test - GPT training

```bash
$ python train.py
```

## Todo

- [x] softmax
- [x] cross-entropy (using triton ops)
- [x] layernorm forward
- [x] layernorm backwards
- [x] batch matrix multiply + fused act forwards
- [x] optimize layernorm backwards (figure out how much to store vs recompute)
- [x] use memory efficient dropout from Triton tutorials
- [ ] batch matrix multiply + fused act backwards
- [ ] fused attention (expand on softmax)
- [ ] use triton matmul for other projections
- [ ] benchmark and optimize
- [ ] kernels conditional on inference vs training
- [ ] efficient triangular matmul kernel for causal attention

## Citations

```bibtex
@article{Tillet2019TritonAI,
    title   = {Triton: an intermediate language and compiler for tiled neural network computations},
    author  = {Philippe Tillet and H. Kung and D. Cox},
    journal = {Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages},
    year    = {2019}
}
```

```bibtex
@misc{vaswani2017attention,
    title   = {Attention Is All You Need}, 
    author  = {Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
    year    = {2017},
    eprint  = {1706.03762},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL}
}
```

```bibtex
@misc{so2021primer,
    title   = {Primer: Searching for Efficient Transformers for Language Modeling},
    author  = {David R. So and Wojciech Mańke and Hanxiao Liu and Zihang Dai and Noam Shazeer and Quoc V. Le},
    year    = {2021},
    eprint  = {2109.08668},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@article{chowdhery2022PaLM,
  title   = {PaLM: Scaling Language Modeling with Pathways},
  author  = {Chowdhery, Aakanksha et al},
  year    = {2022}
}
```
