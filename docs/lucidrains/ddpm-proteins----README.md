## Denoising Diffusion Probabilistic Model for Proteins

Implementation of <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Model</a> in Pytorch. It is a new approach to generative modeling that may <a href="https://ajolicoeur.wordpress.com/the-new-contender-to-gans-score-matching-with-langevin-sampling/">have the potential</a> to rival GANs. It uses denoising score matching to estimate the gradient of the data distribution, followed by Langevin sampling to sample from the true distribution. This implementation was transcribed from the official Tensorflow version <a href="https://github.com/hojonathanho/diffusion">here</a>.

This specific repository will be using a heavily modifying version of the U-net for learning on protein structure, with eventual conditioning from MSA Transformers attention heads.

<img src="./sample.png" width="400px"></img>

** at around 40k iterations **

## Install

```bash
$ pip install ddpm-proteins
```

## Training

We are using <a href="https://wandb.ai">weights & biases</a> for experimental tracking

First you need to login

```bash
$ wandb login
```

Then you will need to cache all the MSA attention embeddings by first running. For some reason, the below needs to be done multiple times to cache all the proteins correctly (it does work though). I'll get around to fixing this.

```bash
$ python cache.py
```

Finally, you can begin training by invoking

```bash
$ python train.py
```

If you would like to clear or recompute the cache (ie after changing the fetch MSA function), just run

```bash
$ rm -rf ~/.cache.ddpm-proteins
```

## Todo

- [x] condition on mask
- [x] condition on MSA transformers (with caching of tensors in specified directory by protein id)
- [x] all-attention network with uformer https://arxiv.org/abs/2106.03106 (with 1d + 2d conv kernels)
- [ ] reach for size 384
- [ ] add all improvements from https://arxiv.org/abs/2105.05233 and https://cascaded-diffusion.github.io/

## Usage

```python
import torch
from ddpm_proteins import Unet, GaussianDiffusion

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
)

training_images = torch.randn(8, 3, 128, 128)
loss = diffusion(training_images)
loss.backward()
# after a lot of training

sampled_images = diffusion.sample(batch_size = 4)
sampled_images.shape # (4, 3, 128, 128)
```

Or, if you simply want to pass in a folder name and the desired image dimensions, you can use the `Trainer` class to easily train a model.

```python
from ddpm_proteins import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    'path/to/your/images',
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = True                       # turn on mixed precision training with apex
)

trainer.train()
```

Samples and model checkpoints will be logged to `./results` periodically

## Citations

```bibtex
@misc{ho2020denoising,
    title   = {Denoising Diffusion Probabilistic Models},
    author  = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year    = {2020},
    eprint  = {2006.11239},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@inproceedings{anonymous2021improved,
    title   = {Improved Denoising Diffusion Probabilistic Models},
    author  = {Anonymous},
    booktitle = {Submitted to International Conference on Learning Representations},
    year    = {2021},
    url     = {https://openreview.net/forum?id=-NEXDKk8gZ},
    note    = {under review}
}
```

```bibtex
@article{Rao2021.02.12.430858,
    author  = {Rao, Roshan and Liu, Jason and Verkuil, Robert and Meier, Joshua and Canny, John F. and Abbeel, Pieter and Sercu, Tom and Rives, Alexander},
    title   = {MSA Transformer},
    year    = {2021},
    publisher = {Cold Spring Harbor Laboratory},
    URL     = {https://www.biorxiv.org/content/early/2021/02/13/2021.02.12.430858},
    journal = {bioRxiv}
}
```
