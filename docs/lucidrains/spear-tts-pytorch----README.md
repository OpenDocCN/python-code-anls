<img src="./spear-tts.png" width="450px"></img>

## Spear-TTS - Pytorch

Implementation of <a href="https://arxiv.org/abs/2302.03540">Spear-TTS</a> - multi-speaker text-to-speech attention network, in Pytorch

The text-to-semantic module built here will be used for <a href="https://github.com/lucidrains/soundstorm-pytorch">SoundStorm</a> for conditioning.

## Appreciation

- <a href="https://stability.ai/">Stability</a> for their generous sponsorships to work on and open source cutting edge artificial intelligence research

- <a href="https://github.com/lucasnewman">Lucas Newman</a> for completing the <a href="https://github.com/lucidrains/spear-tts-pytorch/pull/4">backtranslation</a> portion, as well as beam search decoding!

- <a href="https://github.com/lucasnewman">Lucas Newman</a> for completing the final text to semantic transformer training code!

## Install

```bash
$ pip install spear-tts-pytorch
```

## Usage

```python
import torch

from audiolm_pytorch import HubertWithKmeans

from spear_tts_pytorch import (
    TextToSemantic,
    SemanticToTextDatasetGenerator,
    GeneratedAudioTextDataset,
    MockDataset
)

wav2vec = HubertWithKmeans(
    checkpoint_path = './hubert_base_ls960.pt',
    kmeans_path = './hubert_base_ls960_L9_km500.bin'
)

model = TextToSemantic(
    wav2vec = wav2vec,
    dim = 512,
    num_text_token_ids = 256,
    heads = 8,
    target_kv_heads = 2, # grouped query attention, for memory efficient decoding
    source_depth = 1,
    target_depth = 1
)

ds = MockDataset(10)

dataset_generator = SemanticToTextDatasetGenerator(
    model = model,
    dataset = ds,
    folder = './output_folder'
)

dataset_generator(max_length = 2)

generated_dataset = GeneratedAudioTextDataset(
    folder = './output_folder'
)

assert len(generated_dataset) == 10
```

## Todo

- [x] add eos logic + generate, and hook up end-to-end generation in soundstorm
- [x] add first pretraining speech-to-speech with the reconstruction of 60% deleted tokens
- [x] add dropouts for this project, as low-resource
- [x] add total flexiblity of which layers of encoder / decoder to freeze during training
- [x] add step for training on small speech -> text corpus and generating pseudo-labelled dataset + finetuning (thanks to @lucasnewman)
- [x] add final step of finetuning on text -> speech + pseudolabelled dataset
- [x] figure out the best way to store and manage the pseudo-labelled generated dataset
- [x] batched beam search decoding
- [x] allow for using rotary positions in decoder + flash attention, give Tri another citation
- [x] integrate speculative decoding with some improvisation - done in same model using early exit strategy

- [ ] add cached key / values for starter + single / grouped key values, make sure flash attention can support specialized causal mask before flash attention 2 is in pytorch core
- [ ] polish the audio-text generation workflow
- [ ] concatting the real audio-text dataset with the generated one -> or being able to convert real audio-text dataset to generated

## Citations

```bibtex
@misc{kharitonov2023speak,
    title   = {Speak, Read and Prompt: High-Fidelity Text-to-Speech with Minimal Supervision}, 
    author  = {Eugene Kharitonov and Damien Vincent and Zalán Borsos and Raphaël Marinier and Sertan Girgin and Olivier Pietquin and Matt Sharifi and Marco Tagliasacchi and Neil Zeghidour},
    year    = {2023},
    eprint  = {2302.03540},
    archivePrefix = {arXiv},
    primaryClass = {cs.SD}
}
```

```bibtex
@inproceedings{dao2022flashattention,
    title   = {Flash{A}ttention: Fast and Memory-Efficient Exact Attention with {IO}-Awareness},
    author  = {Dao, Tri and Fu, Daniel Y. and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
    booktitle = {Advances in Neural Information Processing Systems},
    year    = {2022}
}
```

```bibtex
@misc{shi2023enhance,
    title   = {Enhance audio generation controllability through representation similarity regularization}, 
    author  = {Yangyang Shi and Gael Le Lan and Varun Nagaraja and Zhaoheng Ni and Xinhao Mei and Ernie Chang and Forrest Iandola and Yang Liu and Vikas Chandra},
    year    = {2023},
    eprint  = {2309.08773},
    archivePrefix = {arXiv},
    primaryClass = {cs.SD}
}
```

```bibtex
@article{Ainslie2023GQATG,
    title   = {GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints},
    author  = {Joshua Ainslie and James Lee-Thorp and Michiel de Jong and Yury Zemlyanskiy and Federico Lebr'on and Sumit K. Sanghai},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2305.13245},
    url     = {https://api.semanticscholar.org/CorpusID:258833177}
}
```

```bibtex
@inproceedings{Leviathan2022FastIF,
    title   = {Fast Inference from Transformers via Speculative Decoding},
    author  = {Yaniv Leviathan and Matan Kalman and Y. Matias},
    booktitle = {International Conference on Machine Learning},
    year    = {2022},
    url     = {https://api.semanticscholar.org/CorpusID:254096365}
}
```

