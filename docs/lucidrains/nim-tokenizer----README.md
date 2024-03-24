## Nim Tokenizer (wip)

Implementation of a simple BPE tokenizer, but in <a href="https://nim-lang.org/">Nim</a>. May contain <a href="https://arxiv.org/abs/1910.13267">BPE Dropout</a> too

## Todo

- [ ] figure out the special treatment of whitespaces as done in <a href="https://arxiv.org/abs/2305.06161">starcoder</a> and make sure it is supported

## Citations

```bibtex
@inproceedings{Wang2019NeuralMT,
    title   = {Neural Machine Translation with Byte-Level Subwords},
    author  = {Changhan Wang and Kyunghyun Cho and Jiatao Gu},
    booktitle = {AAAI Conference on Artificial Intelligence},
    year    = {2019}
}
```

```bibtex
@inproceedings{provilkov-etal-2020-bpe,
    title   = "{BPE}-Dropout: Simple and Effective Subword Regularization",
    author  = "Provilkov, Ivan  and Emelianenko, Dmitrii  and Voita, Elena",
    booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
    month   = jul,
    year    = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url     = "https://aclanthology.org/2020.acl-main.170",
    doi     = "10.18653/v1/2020.acl-main.170",
    pages   = "1882--1892",
}
```
