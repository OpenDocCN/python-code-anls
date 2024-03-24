# `.\lucidrains\genetic-algorithm-pytorch\ga.py`

```py
"""
Genetic Algorithm - formalized by John H. Holland in 1992, but has been talked about since 1960-70s

https://www.researchgate.net/figure/Hollands-canonical-genetic-algorithm-Holland-1992_fig4_221174380
"""

import torch
from einx import get_at

# constants

GOAL = 'Attention is all you need'  # 目标字符串

POP_SIZE = 100  # 种群大小
MUTATION_RATE = 0.04  # 变异率
FRAC_FITTEST_SURVIVE = 0.25  # 最适应个体存活比例
FRAC_TOURNAMENT = 0.25  # 锦标赛选择比例
ELITE_FRAC = 0.05  # 精英比例

# encode and decode functions

def encode(s):
    return torch.tensor([ord(c) for c in s])  # 将字符串编码为张量

def decode(t):
    return ''.join([chr(i) for i in t.tolist()])  # 将张量解码为字符串

# derived constants

gene_length = len(GOAL)  # 目标字符串长度
gene_midpoint = gene_length // 2  # 目标字符串中点位置
target_gene = encode(GOAL)  # 目标字符串编码

keep_fittest_len = int(POP_SIZE * FRAC_FITTEST_SURVIVE)  # 保留最适应个体数量
num_elite = int(ELITE_FRAC * POP_SIZE)  # 精英数量
num_repro_and_mutate = keep_fittest_len - num_elite  # 繁殖和变异数量
num_tournament_contenders = int(num_repro_and_mutate * FRAC_TOURNAMENT)  # 锦标赛参与者数量
num_children = POP_SIZE - keep_fittest_len  # 子代数量
num_mutate = MUTATION_RATE * gene_length  # 变异基因数量

assert num_tournament_contenders >= 2  # 断言确保锦标赛参与者数量大于等于2

# genetic algorithm

generation = 1  # 代数计数器

pool = torch.randint(0, 255, (POP_SIZE, gene_length))  # 初始化种群，随机生成基因

while True:
    print(f"\n\ngeneration {generation}\n")  # 打印当前代数

    # sort population by fitness

    fitnesses = 1. / torch.square(pool - target_gene).sum(dim = -1)  # 计算适应度

    indices = fitnesses.sort(descending = True).indices  # 根据适应度对种群排序
    pool, fitnesses = pool[indices], fitnesses[indices]

    # keep the fittest

    pool, fitnesses = pool[:keep_fittest_len], fitnesses[:keep_fittest_len]  # 保留最适应个体

    # display every generation

    for gene, fitness in zip(pool, fitnesses):
        print(f"{decode(gene)} ({fitness.item():.3f})")  # 打印每个个体的基因和适应度

    # solved if any fitness is inf

    if (fitnesses == float('inf')).any():  # 如果有个体的适应度为无穷大，则问题已解决
        break

    # elites can pass directly to next generation

    elites, pool = pool[:num_elite], pool[num_elite:]  # 精英直接传递到下一代
    elites_fitnesses, fitnesses = fitnesses[:num_elite], fitnesses[num_elite:]

    # deterministic tournament selection - let top 2 winners become parents

    contender_ids = torch.randn((num_children, num_repro_and_mutate)).argsort(dim = -1)[..., :num_tournament_contenders]  # 锦标赛选择参与者
    participants, tournaments = pool[contender_ids], fitnesses[contender_ids]
    top2_winners = tournaments.topk(2, dim = -1, largest = True, sorted = False).indices  # 选择前两名作为父母
    parents = get_at('p [t] g, p w -> p w g', participants, top2_winners)  # 获取父母

    # cross over recombination of parents

    parent1, parent2 = parents.unbind(dim = 1)  # 拆分父母
    children = torch.cat((parent1[:, :gene_midpoint], parent2[:, gene_midpoint:]), dim = -1)  # 交叉重组父母基因

    pool = torch.cat((pool, children))  # 将子代加入种群

    # mutate genes in population

    mutate_mask = torch.randn(pool.shape).argsort(dim = -1) < num_mutate  # 生成变异掩码
    noise = torch.randint(0, 2, pool.shape) * 2 - 1  # 生成变异噪声
    pool = torch.where(mutate_mask, pool + noise, pool)  # 变异
    pool.clamp_(0, 255)  # 限制基因值范围在0-255之间

    # add back the elites

    pool = torch.cat((elites, pool))  # 将精英加回种群

    generation += 1  # 代数加一
```