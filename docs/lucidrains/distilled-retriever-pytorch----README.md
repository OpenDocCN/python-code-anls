## Distilling Knowledge from Reader to Retriever

Implementation of the retriever distillation procedure as outlined in the paper <a href="https://arxiv.org/abs/2012.04584">Distilling Knowledge from Reader to Retriever</a> in Pytorch. They propose to train the retriever using the cross attention scores as pseudo-labels. SOTA on QA.

Update: The BM25 gains actually do not look as impressive as the BERT gains. Also, it seems like distilling with BERT as the starting point never gets to the same level as BM25.

I am thinking whether it makes more sense to modify Marge (https://github.com/lucidrains/marge-pytorch) so one minimizes a loss between an extra prediction head on top of the retriever to the cross-attention scores, during training.

## Citations

```py
@misc{izacard2020distilling,
    title={Distilling Knowledge from Reader to Retriever for Question Answering}, 
    author={Gautier Izacard and Edouard Grave},
    year={2020},
    eprint={2012.04584},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
