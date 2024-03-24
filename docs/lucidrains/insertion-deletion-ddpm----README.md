## Insertion Deletion Denoising Diffusion Probabilistic Models (wip)

Implementation of <a href="https://arxiv.org/abs/2107.07675">Insertion Deletion Denoising Diffusion Probabilistic Models</a>. This scheme basically allows for DDPM to work beyond just in-place corruption along the sequence. They try to apply this to text generation with lukewarm results. I think it holds promise for protein design, as it would be able to infill certain regions without being constrained to a fixed number of amino acids.

## Citations

```bibtex
@article{Johnson2021BeyondIC,
    title   = {Beyond In-Place Corruption: Insertion and Deletion In Denoising Probabilistic Models},
    author  = {Daniel D. Johnson and Jacob Austin and Rianne van den Berg and Daniel Tarlow},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2107.07675}
}
```
