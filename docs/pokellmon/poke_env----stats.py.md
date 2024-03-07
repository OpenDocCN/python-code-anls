# `.\PokeLLMon\poke_env\stats.py`

```
# 该模块包含与统计相关的实用函数和对象

import math
from typing import List

from poke_env.data import GenData

# 定义将统计名称映射到索引的字典
STATS_TO_IDX = {
    "hp": 0,
    "atk": 1,
    "def": 2,
    "spa": 3,
    "spd": 4,
    "spe": 5,
    "satk": 3,
    "sdef": 4,
}

# 计算原始统计值的函数
def _raw_stat(base: int, ev: int, iv: int, level: int, nature_multiplier: float) -> int:
    """Converts to raw stat
    :param base: the base stat
    :param ev: Stat Effort Value (EV)
    :param iv: Stat Individual Values (IV)
    :param level: pokemon level
    :param nature_multiplier: stat multiplier of the nature (either 0.9, 1 or 1.1)
    :return: the raw stat
    """
    s = math.floor(
        (5 + math.floor((math.floor(ev / 4) + iv + 2 * base) * level / 100))
        * nature_multiplier
    )
    return int(s)

# 计算原始 HP 值的函数
def _raw_hp(base: int, ev: int, iv: int, level: int) -> int:
    """Converts to raw hp
    :param base: the base stat
    :param ev: HP Effort Value (EV)
    :param iv: HP Individual Value (IV)
    :param level: pokemon level
    :return: the raw hp
    """
    s = math.floor((math.floor(ev / 4) + iv + 2 * base) * level / 100) + level + 10
    return int(s)

# 计算原始统计值的函数
def compute_raw_stats(
    species: str, evs: List[int], ivs: List[int], level: int, nature: str, data: GenData
) -> List[int]:
    """Converts to raw stats
    :param species: pokemon species
    :param evs: list of pokemon's EVs (size 6)
    :param ivs: list of pokemon's IVs (size 6)
    :param level: pokemon level
    :param nature: pokemon nature
    :return: the raw stats in order [hp, atk, def, spa, spd, spe]
    """

    assert len(evs) == 6
    assert len(ivs) == 6

    base_stats = [0] * 6
    # 从数据中获取种类的基础统计值
    for stat, value in data.pokedex[species]["baseStats"].items():
        base_stats[STATS_TO_IDX[stat]] = value

    nature_multiplier = [1.0] * 6
    # 从数据中获取自然属性的统计值倍增器
    for stat, multiplier in data.natures[nature].items():
        if stat != "num":
            nature_multiplier[STATS_TO_IDX[stat]] = multiplier

    raw_stats = [0] * 6
    # 如果精灵种类是"shedinja"，则将生命值设为1
    if species == "shedinja":
        raw_stats[0] = 1
    # 否则，根据基础状态值、努力值、个体值和等级计算生命值
    else:
        raw_stats[0] = _raw_hp(base_stats[0], evs[0], ivs[0], level)

    # 遍历除生命值外的其他五项状态值
    for i in range(1, 6):
        # 根据基础状态值、努力值、个体值、等级和性格系数计算状态值
        raw_stats[i] = _raw_stat(
            base_stats[i], evs[i], ivs[i], level, nature_multiplier[i]
        )

    # 返回计算后的状态值列表
    return raw_stats
```