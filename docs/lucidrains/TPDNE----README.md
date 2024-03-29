## TPDNE (wip)

<a href="https://twitter.com/kasiababis/status/1379370542986330112">Thispersondoesnotexist</a> went down, so this time, while building it back up, I am going to open source all of it. I'll try to make it modular enough so anyone can deploy their own ever-dreaming GAN (or soon to be 1-2 step DDPM) to be public facing

I may also take some time to do something I've always wanted. To <a href="https://arxiv.org/abs/2305.01644">'Perfuse'</a> my dog into the machine and have it dream her up forever to the public.

## Explained

The site is hosted on Hetzner on a 100$ / month GPU server. Images are generated live, so people, try as they might, cannot exhaust the amount of faces they experience. Through this, they gain an intuition for how vast the latent space of these neural networks are. It also allowed me to explain it to laypeople as having an 'artificial intelligence endlessly dreaming', without it having to be an exaggeration.

How was this feasible without scaling issues? Well, the site is actually a magic trick. Each user, when refreshing the page, actually sees the same image at any point in time. Images are replaced every 250ms, below the human reaction time. By the time the user studies the face and refreshes, the next face will be there, but it is the same face that everyone experiences around the world at the same time.

The model itself was trained by <a href="https://research.nvidia.com/person/tero-karras">Tero Karras</a> under the name <a href="https://arxiv.org/abs/1912.04958">StyleGAN 2</a>.

## Install

```py
$ pip install TPDNE-utils
```

## Usage

```py
from TPDNE_utils import sample_image_and_save_repeatedly

# some function that returns a sampled image in the form of a 3 dimensional ndarray

def generate_image():
    import numpy as np
    return np.random.randn(1024, 1024, 3)

# saves a new sampled image every 250ms as out/sampled.jpeg

sample_image_and_save_repeatedly(generate_image, 'out/sampled')

# use nginx to serve out/sampled.jpeg
# optionally put behind cloudflare
```

## Todo

- [x] take care of an nginx template
- [x] auto-handle various types of tensor outputs. auto-detect channel dimension and move it to last
- [x] handle un-normalization of image tensors into 0-255 uint8 by autodetecting range
- [ ] handle ssl in nginx

## Citations

```py
@inproceedings{Karras2020ada,
    title     = {Training Generative Adversarial Networks with Limited Data},
    author    = {Tero Karras and Miika Aittala and Janne Hellsten and Samuli Laine and Jaakko Lehtinen and Timo Aila},
    booktitle = {Proc. NeurIPS},
    year      = {2020}
}
```
