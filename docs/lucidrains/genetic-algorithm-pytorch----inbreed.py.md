# `.\lucidrains\genetic-algorithm-pytorch\inbreed.py`

```
"""
Genetic Algorithm

but without first generation inbreeding
"""

import torch
import einx
from einx import get_at, rearrange

# constants

GOAL = 'Attention is all you need'  # 目标字符串

POP_SIZE = 100  # 种群大小
MUTATION_RATE = 0.04  # 变异率
FRAC_FITTEST_SURVIVE = 0.25  # 存活最适应个体的比例
FRAC_TOURNAMENT = 0.25  # 锦标赛选择的比例
ELITE_FRAC = 0.05  # 精英个体的比例

# encode and decode functions

def encode(s):
    return torch.tensor([ord(c) for c in s])  # 将字符串编码为张量

def decode(t):
    return ''.join([chr(i) for i in t.tolist()])  # 将张量解码为字符串

# derived constants

gene_length = len(GOAL)  # 目标字符串的长度
gene_midpoint = gene_length // 2  # 目标字符串的中点位置
target_gene = encode(GOAL)  # 目标字符串的编码

keep_fittest_len = int(POP_SIZE * FRAC_FITTEST_SURVIVE)  # 保留最适应个体的数量
num_elite = int(ELITE_FRAC * POP_SIZE)  # 精英个体的数量
num_repro_and_mutate = keep_fittest_len - num_elite  # 繁殖和变异的个体数量
num_tournament_contenders = int(num_repro_and_mutate * FRAC_TOURNAMENT)  # 锦标赛的参与者数量
num_children = POP_SIZE - keep_fittest_len  # 子代个体数量
num_mutate = MUTATION_RATE * gene_length  # 变异的基因数量

assert num_tournament_contenders >= 2  # 断言确保锦标赛的参与者数量大于等于2

# genetic algorithm

generation = 1  # 代数

parent_ids = torch.full((POP_SIZE, 2), -1, dtype=torch.long)  # 父母的ID
pool = torch.randint(0, 255, (POP_SIZE, gene_length))  # 种群中的个体

while True:
    print(f"\n\ngeneration {generation}\n")  # 打印当前代数

    # sort population by fitness

    fitnesses = 1. / torch.square(pool - target_gene).sum(dim=-1)  # 计算适应度

    indices = fitnesses.sort(descending=True).indices  # 根据适应度对种群进行排序
    pool, parent_ids, fitnesses = pool[indices], parent_ids[indices], fitnesses[indices]

    # keep the fittest

    pool, parent_ids, fitnesses = pool[:keep_fittest_len], parent_ids[:keep_fittest_len], fitnesses[:keep_fittest_len]  # 保留最适应的个体

    # display every generation

    for gene, fitness in zip(pool, fitnesses):
        print(f"{decode(gene)} ({fitness.item():.3f})")  # 打印每个个体的基因和适应度

    # solved if any fitness is inf

    if (fitnesses == float('inf')).any():  # 如果任何适应度为无穷大，则问题已解决
        break

    # elites can pass directly to next generation

    elites, pool = pool[:num_elite], pool[num_elite:]  # 精英个体直接传递到下一代
    elites_fitnesses, fitnesses = fitnesses[:num_elite], fitnesses[num_elite:]
    elites_parent_ids, parent_ids = parent_ids[:num_elite], parent_ids[num_elite:]

    elites_parent_ids.fill_(-1)  # 将精英个体的父母ID填充为-1

    # deterministic tournament selection
    # 2 tournaments - the second tournament removes all contestants with shared parents with 1st winner

    first_contender_ids = torch.randn((num_children, num_repro_and_mutate)).argsort(dim=-1)[..., :num_tournament_contenders]  # 第一轮锦标赛的参与者ID
    first_participants, participants_parent_ids, tournaments = pool[first_contender_ids], parent_ids[first_contender_ids], fitnesses[first_contender_ids]

    first_winner = tournaments.topk(1, dim=-1, largest=True, sorted=False).indices  # 第一轮锦标赛的获胜者
    first_winner = rearrange('p 1 -> p', first_winner)

    first_parent_ids = get_at('p [t] i, p -> p i', participants_parent_ids, first_winner)  # 第一轮锦标赛的获胜者的父母ID

    # second tournament, masking out any siblings to first winners

    contender_scores = torch.randn((num_children, num_repro_and_mutate))  # 参与者得分
    self_mask = rearrange('i -> i 1', first_winner) == torch.arange(num_repro_and_mutate)  # 自身掩码
    contender_scores = torch.where(self_mask, 1e6, contender_scores)

    sibling_mask = (rearrange('p i -> p 1 i 1', first_parent_ids) == rearrange('c j -> 1 c 1 j', parent_ids))  # 兄弟掩码
    valid_parent_mask = (rearrange('p i -> p 1 i 1', first_parent_ids) != -1) & (rearrange('c j -> 1 c 1 j', parent_ids) != -1)  # 有效父母掩码
    num_shared_parents = (sibling_mask & valid_parent_mask).float().sum(dim=(-1, -2))  # 共享父母的数量
    contender_scores += num_shared_parents * 1e3

    second_contender_ids = contender_scores.argsort(dim=-1)[..., :num_tournament_contenders]  # 第二轮锦标赛的参与者ID
    second_participants, second_tournaments = pool[second_contender_ids], fitnesses[second_contender_ids]
    second_winner = second_tournaments.topk(1, dim=-1, largest=True, sorted=False).indices  # 第二轮锦标赛的获胜者
    second_winner = rearrange('p 1 -> p', second_winner)

    # get parents

    first_ids = get_at('p [t], p -> p', first_contender_ids, first_winner)  # 第一轮锦标赛的获胜者的ID
    second_ids = get_at('p [t], p -> p', second_contender_ids, second_winner)  # 第二轮锦标赛的获胜者的ID

    new_parent_ids = torch.stack((first_ids, second_ids), dim=-1)  # 新的父母ID对
    # 从第一组参与者和第一组获胜者中获取父母1
    parent1 = get_at('p [t] g, p -> p g', first_participants, first_winner)
    # 从第二组参与者和第二组获胜者中获取父母2
    parent2 = get_at('p [t] g, p -> p g', second_participants, second_winner)

    # 交叉重组父母的基因

    # 将父母1的前半部分和父母2的后半部分连接起来形成子代
    children = torch.cat((parent1[:, :gene_midpoint], parent2[:, gene_midpoint:]), dim=-1)

    # 将子代添加到种群中
    pool = torch.cat((pool, children))

    # 重置父母ID数组并将新的父母ID添加到其中
    parent_ids.fill_(-1)
    parent_ids = torch.cat((parent_ids, new_parent_ids))

    # 在种群中突变基因

    # 生成一个用于确定哪些基因需要突变的掩码
    mutate_mask = torch.randn(pool.shape).argsort(dim=-1) < num_mutate
    # 生成一个随机噪声数组，用于基因突变
    noise = torch.randint(0, 2, pool.shape) * 2 - 1
    # 根据掩码决定是否对基因进行突变，并添加随机噪声
    pool = torch.where(mutate_mask, pool + noise, pool)
    # 将基因值限制在0到255之间
    pool.clamp_(0, 255)

    # 将精英个体重新添加到种群中

    # 将精英个体添加回种群中
    pool = torch.cat((elites, pool))
    # 将精英个体的父母ID添加回父母ID数组中
    parent_ids = torch.cat((elites_parent_ids, parent_ids))

    # 递增代数计数器
    generation += 1
```