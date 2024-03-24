# `.\lucidrains\genetic-algorithm-pytorch\bega.py`

```
"""
Queen-bee evolution for genetic algorithms - Jung 2003

Inspired by evolution of bees, the fittest solution is designated the "queen", and the rest of the population contends to mate with it. The strong exploitation is balanced by a higher than normal mutation rate.
For some problems, the paper claims convergence at 2-3 orders of magnitude faster

https://www.researchgate.net/publication/3385719_Queen-bee_evolution_for_genetic_algorithms
"""

import torch
from einops import repeat
from einx import get_at

# constants

GOAL = 'Attention is all you need'  # 目标字符串

POP_SIZE = 100  # 种群大小
MUTATION_PROB = 0.04  # 突变概率
STRONG_MUTATION_RATE = 0.1  # 强突变率
STRONG_MUTATION_PROB = 0.25  # 强突变概率
NUM_TOURNAMENT_PARTICIPANTS = 25  # 锦标赛参与者数量

# encode and decode functions

def encode(s):
    return torch.tensor([ord(c) for c in s])  # 将字符串编码为张量

def decode(t):
    return ''.join([chr(i) for i in t.tolist()])  # 将张量解码为字符串

# derived constants

gene_length = len(GOAL)  # 目标字符串长度
gene_midpoint = gene_length // 2  # 目标字符串中点位置
target_gene = encode(GOAL)  # 目标字符串编码

strong_mutate_pool_size = STRONG_MUTATION_RATE * POP_SIZE  # 强突变池大小
num_code_mutate = MUTATION_PROB * gene_length  # 码位突变数量
strong_num_code_mutate = STRONG_MUTATION_PROB * gene_length  # 强码位突变数量

# queen bee genetic algorithm

generation = 1  # 代数

pool = torch.randint(0, 255, (POP_SIZE, gene_length))  # 随机初始化种群

queen = queen_fitness = None  # 初始化皇后和皇后适应度

while True:
    print(f"\n\ngeneration {generation}\n")  # 打印当前代数

    # sort population by fitness

    fitnesses = 1. / torch.square(pool - target_gene).sum(dim = -1)  # 计算适应度

    indices = fitnesses.sort(descending = True).indices  # 根据适应度排序种群
    pool, fitnesses = pool[indices], fitnesses[indices]

    # display every generation

    if queen is not None:
        print("queen:")
        print(f"{decode(queen)} ({queen_fitness.item():.3f})\n")  # 打印皇后及其适应度

    for gene, fitness in zip(pool, fitnesses):
        print(f"{decode(gene)} ({fitness.item():.3f})")  # 打印每个基因及其适应度
    
    # if one of the children has a better fitness than queen, that child becomes the new queen
    # and the queen replaces the worst bee in the population, kept around for at least one generation more

    if queen is not None and queen_fitness < fitnesses[0]:
        pool = torch.cat((pool, queen[None, :]), dim = 0)  # 将皇后加入种群
        fitnesses = torch.cat((fitnesses, queen_fitness[None]), dim = 0)
        queen = queen_fitness = None

    # separate the queen bee from the rest of the population

    if queen is None:
        queen, pool = pool[0], pool[1:]  # 分离皇后和种群
        queen_fitness, fitnesses = fitnesses[0], fitnesses[1:]

    # solved if any queen fitness is inf

    if (queen_fitness == float('inf')).any():  # 如果皇后适应度为无穷大，则问题已解决
        break

    # deterministic tournament selection - let top winner become parent with queen

    contender_ids = torch.randn((POP_SIZE - 1, POP_SIZE - 1)).argsort(dim = -1)[..., :NUM_TOURNAMENT_PARTICIPANTS]  # 锦标赛选择参与者
    participants, tournaments = pool[contender_ids], fitnesses[contender_ids]
    top_winner = tournaments.topk(1, dim = -1, largest = True, sorted = False).indices  # 选择最优参与者
    parents = get_at('p [t] g, p 1 -> p g', participants, top_winner)  # 获取父母基因

    # cross over all chosen drones with the queen

    queen_parents = repeat(queen, '... -> p ...', p = POP_SIZE - 1)  # 重复皇后基因
    queen_and_parents = torch.stack((queen_parents, parents), dim = 1)  # 合并皇后和父母基因

    rand_crossover_order = torch.randn(queen_and_parents.shape[:2]).argsort(dim = -1)  # 随机交叉排序

    batch_arange = torch.arange(POP_SIZE - 1)[..., None]
    queen_and_parents = queen_and_parents[batch_arange, rand_crossover_order]
    queen_parents, parents = queen_and_parents.unbind(dim = 1)

    pool = torch.cat((queen_parents[:, :gene_midpoint], parents[:, gene_midpoint:]), dim = -1)  # 交叉生成新种群

    # mutate genes in population

    mutate_mask = torch.randn(pool.shape).argsort(dim = -1) < num_code_mutate  # 生成突变掩码
    noise = torch.randint(0, 2, pool.shape) * 2 - 1
    mutated_pool = torch.where(mutate_mask, pool + noise, pool)  # 码位突变

    strong_mutate_mask = torch.randn(pool.shape).argsort(dim = -1) < strong_num_code_mutate  # 生成强突变掩码
    noise = torch.randint(0, 2, pool.shape) * 2 - 1
    strong_mutated_pool = torch.where(strong_mutate_mask, pool + noise, pool)  # 强码位突变
    # 生成一个布尔掩码，用于选择强变异池中的个体
    strong_mutate_pool_mask = torch.randn(POP_SIZE - 1).argsort(dim=-1) < strong_mutate_pool_size

    # 根据强变异池掩码，选择强变异池中的个体或者普通变异池中的个体，组成新的池
    pool = torch.where(strong_mutate_pool_mask[:, None], strong_mutated_pool, mutated_pool)
    # 将池中的值限制在0到255之间
    pool.clamp_(0, 255)

    # 增加一代
    generation += 1
```