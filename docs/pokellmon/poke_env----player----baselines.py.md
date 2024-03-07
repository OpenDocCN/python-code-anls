# `.\PokeLLMon\poke_env\player\baselines.py`

```py
# 导入必要的模块
from typing import List
import json
import os

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.player.player import Player
from poke_env.data.gen_data import GenData

# 从文件中加载招式效果数据
with open("./poke_env/data/static/moves/moves_effect.json", "r") as f:
    move_effect = json.load(f)

# 计算招式类型的伤害倍率
def calculate_move_type_damage_multipier(type_1, type_2, type_chart, constraint_type_list):
    # 定义所有可能的宝可梦类型
    TYPE_list = 'BUG,DARK,DRAGON,ELECTRIC,FAIRY,FIGHTING,FIRE,FLYING,GHOST,GRASS,GROUND,ICE,NORMAL,POISON,PSYCHIC,ROCK,STEEL,WATER'.split(",")

    move_type_damage_multiplier_list = []

    # 如果存在第二个类型
    if type_2:
        # 计算每种类型对应的伤害倍率
        for type in TYPE_list:
            move_type_damage_multiplier_list.append(type_chart[type_1][type] * type_chart[type_2][type])
        move_type_damage_multiplier_dict = dict(zip(TYPE_list, move_type_damage_multiplier_list))
    else:
        move_type_damage_multiplier_dict = type_chart[type_1]

    effective_type_list = []
    extreme_type_list = []
    resistant_type_list = []
    extreme_resistant_type_list = []
    immune_type_list = []
    # 根据伤害倍率将类型分为不同的类别
    for type, value in move_type_damage_multiplier_dict.items():
        if value == 2:
            effective_type_list.append(type)
        elif value == 4:
            extreme_type_list.append(type)
        elif value == 1 / 2:
            resistant_type_list.append(type)
        elif value == 1 / 4:
            extreme_resistant_type_list.append(type)
        elif value == 0:
            immune_type_list.append(type)
        else:  # value == 1
            continue
    # 如果约束类型列表不为空
    if constraint_type_list:
        # 更新极端类型列表，取交集
        extreme_type_list = list(set(extreme_type_list).intersection(set(constraint_type_list)))
        # 更新有效类型列表，取交集
        effective_type_list = list(set(effective_type_list).intersection(set(constraint_type_list)))
        # 更新抗性类型列表，取交集
        resistant_type_list = list(set(resistant_type_list).intersection(set(constraint_type_list)))
        # 更新极端抗性类型列表，取交集
        extreme_resistant_type_list = list(set(extreme_resistant_type_list).intersection(set(constraint_type_list)))
        # 更新免疫类型列表，取交集
        immune_type_list = list(set(immune_type_list).intersection(set(constraint_type_list)))

    # 返回更新后的各类型列表
    return extreme_type_list, effective_type_list, resistant_type_list, extreme_resistant_type_list, immune_type_list
# 定义一个函数，根据给定的参数计算并返回对应的移动类型伤害提示
def move_type_damage_wraper(pokemon_name, type_1, type_2, type_chart, constraint_type_list=None):

    # 初始化移动类型伤害提示字符串
    move_type_damage_prompt = ""
    
    # 调用函数计算移动类型伤害倍数，得到各种类型的列表
    extreme_effective_type_list, effective_type_list, resistant_type_list, extreme_resistant_type_list, immune_type_list = calculate_move_type_damage_multipier(
        type_1, type_2, type_chart, constraint_type_list)

    # 如果存在有效的、抵抗的或免疫的类型列表
    if effective_type_list or resistant_type_list or immune_type_list:

        # 构建移动类型伤害提示字符串
        move_type_damage_prompt = f"{pokemon_name}"
        if extreme_effective_type_list:
            move_type_damage_prompt = move_type_damage_prompt + " can be super-effectively attacked by " + ", ".join(
                extreme_effective_type_list) + " moves"
        if effective_type_list:
            move_type_damage_prompt = move_type_damage_prompt + ", can be effectively attacked by " + ", ".join(
                effective_type_list) + " moves"
        if resistant_type_list:
            move_type_damage_prompt = move_type_damage_prompt + ", is resistant to " + ", ".join(
                resistant_type_list) + " moves"
        if extreme_resistant_type_list:
            move_type_damage_prompt = move_type_damage_prompt + ", is super-resistant to " + ", ".join(
                extreme_resistant_type_list) + " moves"
        if immune_type_list:
            move_type_damage_prompt = move_type_damage_prompt + ", is immuned to " + ", ".join(
                immune_type_list) + " moves"

    # 返回移动类型伤害提示字符串
    return move_type_damage_prompt


# 定义一个类，继承自Player类，实现最大基础伤害玩家
class MaxBasePowerPlayer(Player):
    
    # 重写choose_move方法
    def choose_move(self, battle: AbstractBattle):
        # 如果存在可用的移动
        if battle.available_moves:
            # 选择基础伤害最大的移动
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        # 如果没有可用的移动，则随机选择一个移动
        return self.choose_random_move(battle)

# 定义一个类，继承自Player类，实现简单启发式玩家
class SimpleHeuristicsPlayer(Player):
    # 定义了各种入场危害效果，将字符串映射到对应的SideCondition枚举值
    ENTRY_HAZARDS = {
        "spikes": SideCondition.SPIKES,
        "stealhrock": SideCondition.STEALTH_ROCK,
        "stickyweb": SideCondition.STICKY_WEB,
        "toxicspikes": SideCondition.TOXIC_SPIKES,
    }

    # 定义了反危害招式，使用集合存储
    ANTI_HAZARDS_MOVES = {"rapidspin", "defog"}

    # 定义了速度等级系数
    SPEED_TIER_COEFICIENT = 0.1
    # 定义了生命值分数系数
    HP_FRACTION_COEFICIENT = 0.4
    # 定义了交换出场匹配阈值
    SWITCH_OUT_MATCHUP_THRESHOLD = -2

    # 估算对战情况，返回得分
    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon):
        # 计算对手对我方造成的伤害倍率的最大值
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        # 减去我方对对手造成的伤害倍率的最大值
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        # 根据速度等级差异调整得分
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        # 根据生命值分数调整得分
        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    # 判断是否应该使用极巨化
    def _should_dynamax(self, battle: AbstractBattle, n_remaining_mons: int):
        if battle.can_dynamax and self._dynamax_disable is False:
            # 最后一个满血的精灵
            if (
                len([m for m in battle.team.values() if m.current_hp_fraction == 1])
                == 1
                and battle.active_pokemon.current_hp_fraction == 1
            ):
                return True
            # 有优势且双方都是满血
            if (
                self._estimate_matchup(
                    battle.active_pokemon, battle.opponent_active_pokemon
                )
                > 0
                and battle.active_pokemon.current_hp_fraction == 1
                and battle.opponent_active_pokemon.current_hp_fraction == 1
            ):
                return True
            # 只剩下一个精灵
            if n_remaining_mons == 1:
                return True
        return False
    # 判断是否应该替换出当前精灵
    def _should_switch_out(self, battle: AbstractBattle):
        # 获取当前精灵和对手精灵
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        # 如果有一个适合替换的精灵...
        if [
            m
            for m in battle.available_switches
            if self._estimate_matchup(m, opponent) > 0
        ]:
            # ...并且有一个“好”的理由替换出去
            if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
                return True
            if (
                active.boosts["atk"] <= -3
                and active.stats["atk"] >= active.stats["spa"]
            ):
                return True
            if (
                active.boosts["spa"] <= -3
                and active.stats["atk"] <= active.stats["spa"]
            ):
                return True
            if (
                self._estimate_matchup(active, opponent)
                < self.SWITCH_OUT_MATCHUP_THRESHOLD
            ):
                return True
        return False

    # 估算精灵的状态
    def _stat_estimation(self, mon: Pokemon, stat: str):
        # 计算状态提升值
        if mon.boosts[stat] > 1:
            boost = (2 + mon.boosts[stat]) / 2
        else:
            boost = 2 / (2 - mon.boosts[stat])
        return ((2 * mon.base_stats[stat] + 31) + 5) * boost

    # 计算奖励值
    def calc_reward(
            self, current_battle: AbstractBattle
    ) -> float:
        # 计算奖励值
        return self.reward_computing_helper(
            current_battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )
    # 根据状态和等级返回加成倍数
    def boost_multiplier(self, state, level):
        # 如果状态是准确度
        if state == "accuracy":
            # 根据等级返回对应的加成倍数
            if level == 0:
                return 1.0
            if level == 1:
                return 1.33
            if level == 2:
                return 1.66
            if level == 3:
                return 2.0
            if level == 4:
                return 2.5
            if level == 5:
                return 2.66
            if level == 6:
                return 3.0
            if level == -1:
                return 0.75
            if level == -2:
                return 0.6
            if level == -3:
                return 0.5
            if level == -4:
                return 0.43
            if level == -5:
                return 0.36
            if level == -6:
                return 0.33
        # 如果状态不是准确度
        else:
            # 根据等级返回对应的加成倍数
            if level == 0:
                return 1.0
            if level == 1:
                return 1.5
            if level == 2:
                return 2.0
            if level == 3:
                return 2.5
            if level == 4:
                return 3.0
            if level == 5:
                return 3.5
            if level == 6:
                return 4.0
            if level == -1:
                return 0.67
            if level == -2:
                return 0.5
            if level == -3:
                return 0.4
            if level == -4:
                return 0.33
            if level == -5:
                return 0.29
            if level == -6:
                return 0.25
    # 检查给定状态的值，并返回相应的状态字符串
    def check_status(self, status):
        # 如果状态存在
        if status:
            # 根据状态值返回相应的状态字符串
            if status.value == 1:
                return "burnt"
            elif status.value == 2:
                return "fainted"
            elif status.value == 3:
                return "frozen"
            elif status.value == 4:
                return "paralyzed"
            elif status.value == 5:
                return "poisoned"
            elif status.value == 7:
                return "toxic"
            elif status.value == 6:
                return "sleeping"
        # 如果状态不存在，则返回"healthy"
        else:
            return "healthy"
```