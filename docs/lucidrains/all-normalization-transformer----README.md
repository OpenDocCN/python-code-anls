## Transformer with Normalized Attention

A Transformer that consists of only normalization as its sole non-linearity, as proposed in the paper <a href="https://arxiv.org/abs/2005.09561">Normalized Attention Without Probability Cage</a>. This repository will build on the paper's contributions and attempt to make it work for the auto-regressive case.

Update - It works. You can have an entire language model built on only matrix multiplies and normalization.

## Pre-requisites

```bash
$ pip install -r requirements.txt
```

## Train

```python
$ python train_enwik8.py
```

## Citations

```bibtex
@misc{richter2020normalized,
    title={Normalized Attention Without Probability Cage},
    author={Oliver Richter and Roger Wattenhofer},
    year={2020},
    eprint={2005.09561},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
