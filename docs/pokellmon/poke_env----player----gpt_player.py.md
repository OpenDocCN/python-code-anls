# `.\PokeLLMon\poke_env\player\gpt_player.py`

```
import json  # 导入 json 模块
import os  # 导入 os 模块
import random  # 导入 random 模块
from typing import List  # 导入 List 类型提示
from poke_env.environment.abstract_battle import AbstractBattle  # 导入 AbstractBattle 类
from poke_env.environment.double_battle import DoubleBattle  # 导入 DoubleBattle 类
from poke_env.environment.move_category import MoveCategory  # 导入 MoveCategory 类
from poke_env.environment.pokemon import Pokemon  # 导入 Pokemon 类
from poke_env.environment.side_condition import SideCondition  # 导入 SideCondition 类
from poke_env.player.player import Player, BattleOrder  # 导入 Player 和 BattleOrder 类
from typing import Dict, List, Optional, Union  # 导入 Dict, List, Optional, Union 类型提示
from poke_env.environment.move import Move  # 导入 Move 类
import time  # 导入 time 模块
import json  # 再次导入 json 模块（重复导入）
from openai import OpenAI  # 导入 OpenAI 类
from poke_env.data.gen_data import GenData  # 导入 GenData 类

def calculate_move_type_damage_multipier(type_1, type_2, type_chart, constraint_type_list):
    TYPE_list = 'BUG,DARK,DRAGON,ELECTRIC,FAIRY,FIGHTING,FIRE,FLYING,GHOST,GRASS,GROUND,ICE,NORMAL,POISON,PSYCHIC,ROCK,STEEL,WATER'.split(",")

    move_type_damage_multiplier_list = []  # 初始化一个空列表，用于存储每种类型的伤害倍率

    if type_2:  # 如果存在第二种类型
        for type in TYPE_list:  # 遍历每种类型
            move_type_damage_multiplier_list.append(type_chart[type_1][type] * type_chart[type_2][type])  # 计算两种类型之间的伤害倍率并添加到列表中
        move_type_damage_multiplier_dict = dict(zip(TYPE_list, move_type_damage_multiplier_list))  # 将类型和对应的伤害倍率组成字典
    else:  # 如果只有一种类型
        move_type_damage_multiplier_dict = type_chart[type_1]  # 直接使用第一种类型的伤害倍率字典

    effective_type_list = []  # 初始化有效类型列表
    extreme_type_list = []  # 初始化极效类型列表
    resistant_type_list = []  # 初始化抵抗类型列表
    extreme_resistant_type_list = []  # 初始化极度抵抗类型列表
    immune_type_list = []  # 初始化免疫类型列表
    for type, value in move_type_damage_multiplier_dict.items():  # 遍历每种类型及其对应的伤害倍率
        if value == 2:  # 如果伤害倍率为 2
            effective_type_list.append(type)  # 添加到有效类型列表
        elif value == 4:  # 如果伤害倍率为 4
            extreme_type_list.append(type)  # 添加到极效类型列表
        elif value == 1 / 2:  # 如果伤害倍率为 1/2
            resistant_type_list.append(type)  # 添加到抵抗类型列表
        elif value == 1 / 4:  # 如果伤害倍率为 1/4
            extreme_resistant_type_list.append(type)  # 添加到极度抵抗类型列表
        elif value == 0:  # 如果伤害倍率为 0
            immune_type_list.append(type)  # 添加到免疫类型列表
        else:  # 如果伤害倍率为 1
            continue  # 继续循环
    # 如果约束类型列表不为空
    if constraint_type_list:
        # 将极端类型列表与约束类型列表的交集作为新的极端类型列表
        extreme_type_list = list(set(extreme_type_list).intersection(set(constraint_type_list)))
        # 将有效类型列表与约束类型列表的交集作为新的有效类型列表
        effective_type_list = list(set(effective_type_list).intersection(set(constraint_type_list)))
        # 将抗性类型列表与约束类型列表的交集作为新的抗性类型列表
        resistant_type_list = list(set(resistant_type_list).intersection(set(constraint_type_list)))
        # 将极端抗性类型列表与约束类型列表的交集作为新的极端抗性类型列表
        extreme_resistant_type_list = list(set(extreme_resistant_type_list).intersection(set(constraint_type_list)))
        # 将免疫类型列表与约束类型列表的交集作为新的免疫类型列表
        immune_type_list = list(set(immune_type_list).intersection(set(constraint_type_list)))

    # 返回各类型列表的首字母大写形式
    return (list(map(lambda x: x.capitalize(), extreme_type_list)),
           list(map(lambda x: x.capitalize(), effective_type_list)),
           list(map(lambda x: x.capitalize(), resistant_type_list)),
           list(map(lambda x: x.capitalize(), extreme_resistant_type_list)),
           list(map(lambda x: x.capitalize(), immune_type_list)))
# 定义一个函数，用于计算给定精灵对应的移动类型伤害提示
def move_type_damage_wraper(pokemon, type_chart, constraint_type_list=None):

    # 初始化变量，用于存储精灵的两种类型
    type_1 = None
    type_2 = None
    # 如果精灵有第一种类型
    if pokemon.type_1:
        # 获取第一种类型的名称
        type_1 = pokemon.type_1.name
        # 如果精灵有第二种类型
        if pokemon.type_2:
            # 获取第二种类型的名称
            type_2 = pokemon.type_2.name

    # 初始化移动类型伤害提示字符串
    move_type_damage_prompt = ""
    # 调用函数计算移动类型伤害倍数，得到不同类型的列表
    extreme_effective_type_list, effective_type_list, resistant_type_list, extreme_resistant_type_list, immune_type_list = calculate_move_type_damage_multipier(
        type_1, type_2, type_chart, constraint_type_list)

    # 根据不同类型的列表生成移动类型伤害提示
    if extreme_effective_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(extreme_effective_type_list) +
                                   f"-type attack is extremely-effective (4x damage) to {pokemon.species}.")

    if effective_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(effective_type_list) +
                                   f"-type attack is super-effective (2x damage) to {pokemon.species}.")

    if resistant_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(resistant_type_list) +
                                   f"-type attack is ineffective (0.5x damage) to {pokemon.species}.")

    if extreme_resistant_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(extreme_resistant_type_list) +
                                   f"-type attack is highly ineffective (0.25x damage) to {pokemon.species}.")

    if immune_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(immune_type_list) +
                                   f"-type attack is zero effect (0x damage) to {pokemon.species}.")

    # 返回移动类型伤害提示字符串
    return move_type_damage_prompt


# 定义一个类，继承自Player类
class LLMPlayer(Player):
    # 使用 OpenAI API 进行对话生成，返回生成的文本
    def chatgpt(self, system_prompt, user_prompt, model, temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=200) -> str:
        # 创建 OpenAI 客户端对象
        client = OpenAI(api_key=self.api_key)
        # 如果需要返回 JSON 格式的响应
        if json_format:
            # 调用 API 完成对话生成，返回 JSON 格式的响应
            response = client.chat.completions.create(
                response_format={"type": "json_object"},
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                stream=False,
                # seed=seed,
                stop=stop,
                max_tokens=max_tokens
            )
        else:
            # 调用 API 完成对话生成
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                stream=False,
                # seed=seed,
                max_tokens=max_tokens,
                stop=stop
            )
        # 获取生成的文本内容
        outputs = response.choices[0].message.content
        # 记录完成的 token 数量
        self.completion_tokens += response.usage.completion_tokens
        # 记录 prompt 的 token 数量
        self.prompt_tokens += response.usage.prompt_tokens

        # 返回生成的文本
        return outputs
    # 估算两只精灵之间的对战得分
    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon):
        # 计算对手对该精灵造成的伤害加成中的最大值
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        # 计算该精灵对对手造成的伤害加成中的最大值
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        # 根据速度判断得分
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        # 根据当前生命值比例调整得分
        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    # 判断是否应该使用极巨化
    def _should_dynamax(self, battle: AbstractBattle):
        # 统计队伍中剩余未倒下的精灵数量
        n_remaining_mons = len(
            [m for m in battle.team.values() if m.fainted is False]
        )
        if battle.can_dynamax and self._dynamax_disable is False:
            # 如果只剩下一只全血的精灵
            if (
                len([m for m in battle.team.values() if m.current_hp_fraction == 1])
                == 1
                and battle.active_pokemon.current_hp_fraction == 1
            ):
                return True
            # 如果有对战优势且双方都是全血状态
            if (
                self._estimate_matchup(
                    battle.active_pokemon, battle.opponent_active_pokemon
                )
                > 0
                and battle.active_pokemon.current_hp_fraction == 1
                and battle.opponent_active_pokemon.current_hp_fraction == 1
            ):
                return True
            # 如果只剩下一只精灵
            if n_remaining_mons == 1:
                return True
        return False
    # 解析LLM输出，找到JSON内容的起始位置
    json_start = llm_output.find('{')
    # 找到JSON内容的结束位置，从后往前找第一个}
    json_end = llm_output.rfind('}') + 1
    # 提取JSON内容
    json_content = llm_output[json_start:json_end]
    # 将JSON内容加载为Python对象
    llm_action_json = json.loads(json_content)
    # 初始化下一个动作为None
    next_action = None
    
    # 如果JSON中包含"move"字段
    if "move" in llm_action_json.keys():
        # 获取LLM中的移动ID并处理格式
        llm_move_id = llm_action_json["move"]
        llm_move_id = llm_move_id.replace(" ","").replace("-", "")
        # 遍历可用的移动列表，匹配LLM中的移动ID
        for i, move in enumerate(battle.available_moves):
            if move.id.lower() == llm_move_id.lower():
                # 创建相应的移动指令
                next_action = self.create_order(move, dynamax=self._should_dynamax(battle))

    # 如果JSON中包含"switch"字段
    elif "switch" in llm_action_json.keys():
        # 获取LLM中的交换精灵种类并匹配可用的交换精灵列表
        llm_switch_species = llm_action_json["switch"]
        for i, pokemon in enumerate(battle.available_switches):
            if pokemon.species.lower() == llm_switch_species.lower():
                # 创建相应的交换指令
                next_action = self.create_order(pokemon)

    # 如果下一个动作仍为None，则抛出数值错误异常
    if next_action is None:
        raise ValueError("Value Error")
    # 返回下一个动作
    return next_action
    # 解析LLM输出，找到JSON内容的起始位置
    json_start = llm_output.find('{')
    # 找到JSON内容的结束位置，从后往前找第一个}
    json_end = llm_output.rfind('}') + 1
    # 提取JSON内容
    json_content = llm_output[json_start:json_end]
    # 将JSON内容转换为Python对象
    llm_action_json = json.loads(json_content)
    next_action = None
    # 获取动作和目标
    action = llm_action_json["decision"]["action"]
    target = llm_action_json["decision"]["target"]
    # 处理目标字符串，去除空格和下划线
    target = target.replace(" ", "").replace("_", "")
    # 如果动作是移动
    if action.lower() == "move":
        # 遍历可用的移动
        for i, move in enumerate(battle.available_moves):
            # 如果移动ID匹配目标
            if move.id.lower() == target.lower():
                # 创建移动指令
                next_action = self.create_order(move, dynamax=self._should_dynamax(battle))

    # 如果动作是交换
    elif action.lower() == "switch":
        # 遍历可用的交换精灵
        for i, pokemon in enumerate(battle.available_switches):
            # 如果精灵种类匹配目标
            if pokemon.species.lower() == target.lower():
                # 创建交换指令
                next_action = self.create_order(pokemon)

    # 如果没有找到下一步动作，抛出数值错误
    if next_action is None:
        raise ValueError("Value Error")

    # 返回下一步动作
    return next_action

    # 检查状态并返回对应的字符串
    def check_status(self, status):
        if status:
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
        else:
            return ""
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
    # 返回战斗摘要信息，包括击败得分、剩余得分、胜利列表和标签列表
    def battle_summary(self):
        
        # 初始化空列表用于存储击败得分、剩余得分、胜利列表和标签列表
        beat_list = []
        remain_list = []
        win_list = []
        tag_list = []
        
        # 遍历每场战斗，计算击败得分、剩余得分、是否胜利以及标签
        for tag, battle in self.battles.items():
            beat_score = 0
            # 计算对手队伍的击败得分
            for mon in battle.opponent_team.values():
                beat_score += (1-mon.current_hp_fraction)

            beat_list.append(beat_score)

            remain_score = 0
            # 计算己方队伍的剩余得分
            for mon in battle.team.values():
                remain_score += mon.current_hp_fraction

            remain_list.append(remain_score)
            # 如果战斗胜利，则在胜利列表中添加1
            if battle.won:
                win_list.append(1)

            tag_list.append(tag)

        # 返回击败得分列表、剩余得分列表、胜利列表和标签列表
        return beat_list, remain_list, win_list, tag_list

    # 辅助计算奖励值的函数
    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
    ) -> float:
        """A helper function to compute rewards."""

        # 如果战斗不在奖励缓冲区中，则将其添加，并设置初始值
        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        # 遍历我方队伍中的每只精灵
        for mon in battle.team.values():
            # 根据当前生命值比例计算当前值
            current_value += mon.current_hp_fraction * hp_value
            # 如果精灵已经倒下，则减去倒下值
            if mon.fainted:
                current_value -= fainted_value
            # 如果精灵有异常状态，则减去异常状态值
            elif mon.status is not None:
                current_value -= status_value

        # 根据己方队伍中精灵数量与总精灵数量的差值计算当前值
        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        # 遍历对方队伍中的每只精灵
        for mon in battle.opponent_team.values():
            # 根据当前生命值比例计算当前值
            current_value -= mon.current_hp_fraction * hp_value
            # 如果精灵已经倒下，则加上倒下值
            if mon.fainted:
                current_value += fainted_value
            # 如果精灵有异常状态，则加上异常状态值
            elif mon.status is not None:
                current_value += status_value

        # 根据对方队伍中精灵数量与总精灵数量的差值计算当前值
        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        # 如果战斗胜利，则加上胜利值
        if battle.won:
            current_value += victory_value
        # 如果战斗失败，则减去胜利值
        elif battle.lost:
            current_value -= victory_value

        # 计算当前值与奖励缓冲区中的值的差值作为返回值
        to_return = current_value - self._reward_buffer[battle] # the return value is the delta
        self._reward_buffer[battle] = current_value

        return to_return

    def choose_max_damage_move(self, battle: AbstractBattle):
        # 如果有可用的招式，则选择基础威力最大的招式
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        # 如果没有可用的招式，则随机选择一个招式
        return self.choose_random_move(battle)
```