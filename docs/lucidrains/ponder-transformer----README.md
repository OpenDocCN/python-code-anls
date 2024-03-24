## Ponder(ing) Transformer

Implementation of a Transformer that learns to adapt the number of computational steps it takes depending on the difficulty of the input sequence, using the scheme from the <a href="https://arxiv.org/abs/2107.05407">PonderNet</a> paper. Will also try to abstract out a pondering module that can be used with any block that returns an output with the halting probability.

This repository would not have been possible without repeated viewings of <a href="https://www.youtube.com/watch?v=nQDZmf2Yb9k">Yannic's educational video</a>

## Install

```bash
$ pip install ponder-transformer
```

## Usage

```python
import torch
from ponder_transformer import PonderTransformer

model = PonderTransformer(
    num_tokens = 20000,
    dim = 512,
    max_seq_len = 512
)

mask = torch.ones(1, 512).bool()

x = torch.randint(0, 20000, (1, 512))
y = torch.randint(0, 20000, (1, 512))

loss = model(x, labels = y, mask = mask)
loss.backward()
```

Now you can set the model to `.eval()` mode and it will terminate early when all samples of the batch have emitted a halting signal

```python
import torch
from ponder_transformer import PonderTransformer

model = PonderTransformer(
    num_tokens = 20000,
    dim = 512,
    max_seq_len = 512,
    causal = True
)

x = torch.randint(0, 20000, (2, 512))
mask = torch.ones(2, 512).bool()

model.eval() # setting to eval makes it return the logits as well as the halting indices

logits, layer_indices = model(x,  mask = mask) # (2, 512, 20000), (2)

# layer indices will contain, for each batch element, which layer they exited
```

## Citations

```bibtex
@misc{banino2021pondernet,
    title   = {PonderNet: Learning to Ponder}, 
    author  = {Andrea Banino and Jan Balaguer and Charles Blundell},
    year    = {2021},
    eprint  = {2107.05407},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```
