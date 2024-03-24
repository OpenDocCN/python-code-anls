## Enformer TPU training script (wip)

The full training script for Enformer (Tensorflow Sonnet) on TPU clusters, in an effort to migrate the model to <a href="https://github.com/lucidrains/enformer-pytorch">pytorch</a>.

This was pieced together from the <a href="https://github.com/deepmind/deepmind-research/tree/master/enformer">Deepmind Enformer repository</a>, the <a href="https://colab.research.google.com/github/deepmind/deepmind_research/blob/master/enformer/enformer-training.ipynb">colab training notebook</a>, as well as <a href="https://github.com/calico/basenji/blob/84c681a4b02f592a3de90799cee7f17d96f81ef8/basenji/archive/augmentation.py">Basenji sequence augmentation code</a>

It accounts for:

1. distributed TPU training
2. distributed datasets
3. distributed validation
4. gradient clipping
5. cross replica batchnorms
6. dataset augmentation

Training takes about 3 days on v3-64

## Downloading sequence data for extending context length to 196,608

```py
$ gsutil cp gs://basenji_barnyard/hg38.ml.fa.gz ./ && gunzip hg38.ml.fa.gz
$ gsutil cp gs://basenji_barnyard/mm10.ml.fa.gz ./ && gunzip mm10.ml.fa.gz
$ gsutil cp gs://basenji_barnyard/data/human/sequences.bed ./human-sequences.bed
$ gsutil cp gs://basenji_barnyard/data/mouse/sequences.bed ./mouse-sequences.bed
```

## Todo

- [x] fix script for differences in sequence length in basenji training data, which is ~130k vs ~190k bp as in paper - Training in progress

## Citations

```py
@article {Avsec2021.04.07.438649,
    author  = {Avsec, {\v Z}iga and Agarwal, Vikram and Visentin, Daniel and Ledsam, Joseph R. and Grabska-Barwinska, Agnieszka and Taylor, Kyle R. and Assael, Yannis and Jumper, John and Kohli, Pushmeet and Kelley, David R.},
    title   = {Effective gene expression prediction from sequence by integrating long-range interactions},
    elocation-id = {2021.04.07.438649},
    year    = {2021},
    doi     = {10.1101/2021.04.07.438649},
    publisher = {Cold Spring Harbor Laboratory},
    URL     = {https://www.biorxiv.org/content/early/2021/04/08/2021.04.07.438649},
    eprint  = {https://www.biorxiv.org/content/early/2021/04/08/2021.04.07.438649.full.pdf},
    journal = {bioRxiv}
}
```
