# `.\lucidrains\genetic-algorithm-pytorch\bcga.py`

```py
"""
Bee Colonies Genetic Algorithm

Here we simulate different colonies to maintain diversity. At each generation, one allow a small subset of bees from each colony to immigrate to another
"""

import torch
import einx

# constants

GOAL = 'Attention is all you need'

COLONIES = 10
POP_SIZE = 250
MUTATION_PROB = 0.05
STRONG_MUTATION_PROB = 0.15
NUM_TOURNAMENT_PARTICIPANTS = 25
MIGRATE_EVERY = 5
FRAC_BEES_MIGRANTS = 0.1

# encode and decode functions

# 检查变量是否存在
def exists(v):
    return v is not None

# 将字符串编码为张量
def encode(s):
    return torch.tensor([ord(c) for c in s])

# 将张量解码为字符串
def decode(t):
    return ''.join([chr(i) for i in t.tolist()])

# 计算适应度函数
def calc_fitness(genes, target):
    return 1. / (genes - target).square().sum(dim = -1)

# derived constants

# 目标字符串长度
gene_length = len(GOAL)
# 目标基因
target_gene = encode(GOAL)

# 计算基因突变数量
num_code_mutate = MUTATION_PROB * gene_length
strong_num_code_mutate = STRONG_MUTATION_PROB * gene_length

# 计算迁移的蜜蜂数量
num_bees_migrate = int((POP_SIZE - 1) * FRAC_BEES_MIGRANTS)

# queen bee genetic algorithm

generation = 1

# 初始化种群
colonies = torch.randint(0, 255, (COLONIES, POP_SIZE - 1, gene_length))
colonies_arange = torch.arange(COLONIES)[..., None]

# 初始化皇后蜜蜂
queens = torch.randint(0, 255, (COLONIES, gene_length))
queen_fitnesses = calc_fitness(queens, target_gene)

while True:
    print(f"\n\ngeneration {generation}\n")

    # sort population by fitness

    # 计算种群适应度
    colony_fitnesses = calc_fitness(colonies, target_gene)

    # 按适应度降序排序种群
    indices = colony_fitnesses.sort(descending = True).indices
    colonies, colony_fitnesses = colonies[colonies_arange, indices], colony_fitnesses[colonies_arange, indices]

    # display every generation

    for i, (pool, fitnesses) in enumerate(zip(colonies[:, :10], colony_fitnesses[:, :10])):
        print(f'\ncolony {i + 1}:\n')

        if exists(queens):
            queen, queen_fitness = queens[i], queen_fitnesses[i]
            print(f"Q: {decode(queen)} ({queen_fitness.item():.3f})\n")

        for gene, fitness in zip(pool, fitnesses):
            print(f"{decode(gene)} ({fitness.item():.3f})")
    
    # if one of the children has a better fitness than queen, that child becomes the new queen
    # and the queen replaces the worst bee in the population, kept around for at least one generation more

    has_new_queen = colony_fitnesses[:, 0] > queen_fitnesses

    pop_arange = torch.arange(POP_SIZE)
    pop_arange_with_offset = pop_arange + has_new_queen[:, None]

    colonies = torch.cat((
        queens[:, None, :],
        colonies,
        queens[:, None, :]
    ), dim = -2)

    colony_fitnesses = torch.cat((
        queen_fitnesses[:, None],
        colony_fitnesses,
        queen_fitnesses[:, None]
    ), dim = -1)

    colonies = colonies[colonies_arange, pop_arange_with_offset]
    colony_fitnesses = colony_fitnesses[colonies_arange, pop_arange_with_offset]

    queens, colonies = colonies[:, 0], colonies[:, 1:]
    queen_fitnesses, colony_fitnesses = colony_fitnesses[:, 0], colony_fitnesses[:, 1:]

    # solved if any fitness is inf

    if (queen_fitnesses == float('inf')).any():
        print(f'\nsolved at generation {generation}')
        break

    # deterministic tournament selection - let top winner become parent with queen

    colonies_arange_ = colonies_arange[..., None]
    contender_ids = torch.randn((COLONIES, POP_SIZE - 1, POP_SIZE - 1)).argsort(dim = -1)[..., :NUM_TOURNAMENT_PARTICIPANTS]
    participants, tournaments = colonies[colonies_arange_, contender_ids], colony_fitnesses[colonies_arange_, contender_ids]
    top_winner = tournaments.topk(1, dim = -1, largest = True, sorted = False).indices
    parents = einx.get_at('... [t] g, ... 1 -> ... g', participants, top_winner)

    # potential parents with queen is strongly mutated ("Mutant Bee")

    strong_mutate_mask = torch.randn(parents.shape).argsort(dim = -1) < strong_num_code_mutate
    noise = torch.randint(0, 2, parents.shape) * 2 - 1
    mutated_parents = torch.where(strong_mutate_mask, parents + noise, parents)
    mutated_parents.clamp_(0, 255)
    # 随机进行50%的基因代码混合，而不是在中点处进行连续的交叉

    # 生成一个随机的掩码，用于确定哪些基因需要进行混合
    rand_mix_mask = torch.randn(mutated_parents.shape).argsort(dim=-1) < (gene_length // 2)

    # 根据随机混合的掩码，将皇后和变异后的父代进行基因混合
    colonies = einx.where('c p g, c g, c p g', rand_mix_mask, queens, mutated_parents)

    # 对种群中的基因进行突变

    # 生成一个用于确定哪些基因需要突变的掩码
    mutate_mask = torch.randn(colonies.shape).argsort(dim=-1) < num_code_mutate
    # 生成一个随机的噪声，用于基因突变
    noise = torch.randint(0, 2, colonies.shape) * 2 - 1

    # 根据突变掩码，对种群中的基因进行突变
    colonies = torch.where(mutate_mask, colonies + noise, colonies)
    # 将基因值限制在0到255之间
    colonies.clamp_(0, 255)

    # 允许一部分蜜蜂迁移到相邻的群落

    # 如果当前代数是迁移周期的倍数，并且有蜜蜂需要迁移
    if not (generation % MIGRATE_EVERY) and num_bees_migrate > 0:
        # 将一部分蜜蜂迁移到相邻的群落
        colonies, migrant_colonies = colonies[:, :-num_bees_migrate], colonies[:, -num_bees_migrate:]
        # 将迁移的蜜蜂群落向右滚动一个位置
        migrant_colonies = torch.roll(migrant_colonies, 1, dims=0)
        # 将迁移后的蜜蜂群落合并回原始种群
        colonies = torch.cat((colonies, migrant_colonies), dim=1)

    # 增加代数计数

    generation += 1
```