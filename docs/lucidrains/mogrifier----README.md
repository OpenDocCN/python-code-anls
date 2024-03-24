<img src="./mogrifier.png" width="600px"></img>

[![PyPI version](https://badge.fury.io/py/mogrifier.svg)](https://badge.fury.io/py/mogrifier)

## Mogrifier

A complete implementation of <a href="https://openreview.net/forum?id=SJe5P6EYvS">Mogrifier</a>, a circuit for enhancing LSTMs and potentially other networks. It allows two vectors to modulate each other by having each gate the other in an interleaved, iterative fashion.

## Install

```bash
$ pip install mogrifier
```

## Usage

```python
import torch
from mogrifier import Mogrifier

m = Mogrifier(
    dim = 512,
    iters = 5,          # number of iterations, defaults to 5 as paper recommended for LSTM
    factorize_k = 16    # factorize weight matrices into (dim x k) and (k x dim), if specified
)

x = torch.randn(1, 16, 512)
h = torch.randn(1, 16, 512)

x_out, h_out = m(x, h) # (1, 16, 512), (1, 16, 512)
```

## Citation

```bibtex
@inproceedings{Melis2020Mogrifier,
    title={Mogrifier LSTM},
    author={Gábor Melis and Tomáš Kočiský and Phil Blunsom},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=SJe5P6EYvS}
}
```

<img src="./transmogrifier.jpg" width="700px"></img>