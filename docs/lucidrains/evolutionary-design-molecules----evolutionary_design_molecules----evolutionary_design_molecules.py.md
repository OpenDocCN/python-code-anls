# `.\lucidrains\evolutionary-design-molecules\evolutionary_design_molecules\evolutionary_design_molecules.py`

```py
import torch
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList

from beartype import beartype
from einops import rearrange

from vector_quantize_pytorch import LFQ

# helper functions

# 检查变量是否存在
def exists(v):
    return v is not None

# genetic algorithm

# 进化函数，根据给定的初始种群、计算适应度函数，进行遗传算法进化
@beartype
def evolve(
    init_pool: Tensor,
    calc_fitness: Callable,
    generations = 1e5,
    population = 100,
    mutation_rate = 0.04,
    frac_survive_fittest = 0.25,
    frac_tournament = 0.25,
    frac_elite = 0.05,
):

    keep_fittest_len = int(population * frac_survive_fittest)
    num_elite = int(frac_elite * population)
    num_repro_and_mutate = keep_fittest_len - num_elite
    num_tournament_contenders = int(num_repro_and_mutate * FRAC_TOURNAMENT)
    num_children = population - keep_fittest_len
    num_mutate = mutation_rate * gene_length

    assert num_tournament_contenders >= 2

    # genetic algorithm

    generation = 1

    pool = init_pool

    for generation in generations:
        print(f"\n\ngeneration {generation}\n")

        # sort population by fitness

        fitnesses = calc_fitness(pool)

        indices = fitnesses.sort(descending = True).indices
        pool, fitnesses = pool[indices], fitnesses[indices]

        # keep the fittest

        pool, fitnesses = pool[:keep_fittest_len], fitnesses[:keep_fittest_len]

        # display every generation

        for gene, fitness in zip(pool, fitnesses):
            print(f"{decode(gene)} ({fitness.item():.3f})")

        # solved if any fitness is inf

        if (fitnesses == float('inf')).any():
            break

        # elites can pass directly to next generation

        elites, pool = pool[:num_elite], pool[num_elite:]
        elites_fitnesses, fitnesses = fitnesses[:num_elite], fitnesses[num_elite:]

        # deterministic tournament selection - let top 2 winners become parents

        contender_ids = torch.randn((num_children, num_repro_and_mutate)).argsort(dim = -1)[..., :num_tournament_contenders]
        participants, tournaments = pool[contender_ids], fitnesses[contender_ids]
        top2_winners = tournaments.topk(2, dim = -1, largest = True, sorted = False).indices
        top2_winners = repeat(top2_winners, '... -> ... g', g = gene_length)
        parents = participants.gather(1, top2_winners)

        # cross over recombination of parents

        parent1, parent2 = parents.unbind(dim = 1)
        children = torch.cat((parent1[:, :gene_midpoint], parent2[:, gene_midpoint:]), dim = -1)

        pool = torch.cat((pool, children))

        # mutate genes in population

        mutate_mask = torch.randn(pool.shape).argsort(dim = -1) < num_mutate
        noise = torch.randint(0, 2, pool.shape) * 2 - 1
        pool = torch.where(mutate_mask, pool + noise, pool)
        pool.clamp_(0, 255)

        # add back the elites

        pool = torch.cat((elites, pool))

        generation += 1

    return pool

# autoencoder

# 分子自编码器类
class MolecularAutoencoder(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

# main class

# 主类，用于设计分子
class EvolveDesignMoleculesInSilico(Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError
```