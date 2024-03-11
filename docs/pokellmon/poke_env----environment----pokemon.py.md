# `.\PokeLLMon\poke_env\environment\pokemon.py`

```py
from __future__ import annotations
# 导入未来版本的注解特性

from typing import Any, Dict, List, Optional, Tuple, Union
# 导入类型提示相关的模块

from poke_env.data import GenData, to_id_str
# 从 poke_env.data 模块导入 GenData 和 to_id_str 函数

from poke_env.environment.effect import Effect
from poke_env.environment.move import SPECIAL_MOVES, Move
from poke_env.environment.pokemon_gender import PokemonGender
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.status import Status
from poke_env.environment.z_crystal import Z_CRYSTAL
# 导入与 Pokemon 类相关的环境和效果模块

import math
# 导入数学模块

class Pokemon:
    __slots__ = (
        "_ability",
        "_active",
        "_active",
        "_base_stats",
        "_boosts",
        "_current_hp",
        "_data",
        "_effects",
        "_first_turn",
        "_gender",
        "_heightm",
        "_item",
        "_last_details",
        "_last_request",
        "_level",
        "_max_hp",
        "_moves",
        "_must_recharge",
        "_possible_abilities",
        "_preparing_move",
        "_preparing_target",
        "_protect_counter",
        "_shiny",
        "_revealed",
        "_species",
        "_status",
        "_status_counter",
        "_terastallized",
        "_terastallized_type",
        "_type_1",
        "_type_2",
        "_weightkg",
    )
    # 定义 Pokemon 类的属性列表

    def __init__(
        self,
        gen: int,
        *,
        species: Optional[str] = None,
        request_pokemon: Optional[Dict[str, Any]] = None,
        details: Optional[str] = None,
    ):
        # Base data
        self._data = GenData.from_gen(gen)

        # Species related attributes
        self._base_stats: Dict[str, int]
        self._heightm: int
        self._possible_abilities: List[str]
        self._species: str = ""
        self._type_1: PokemonType
        self._type_2: Optional[PokemonType] = None
        self._weightkg: int

        # Individual related attributes
        self._ability: Optional[str] = None
        self._active: bool
        self._gender: Optional[PokemonGender] = None
        self._level: int = 100
        self._max_hp: Optional[int] = 0
        self._moves: Dict[str, Move] = {}
        self._shiny: Optional[bool] = False

        # Battle related attributes

        self._active: bool = False
        self._boosts: Dict[str, int] = {
            "accuracy": 0,
            "atk": 0,
            "def": 0,
            "evasion": 0,
            "spa": 0,
            "spd": 0,
            "spe": 0,
        }
        self._current_hp: Optional[int] = 0
        self._effects: Dict[Effect, int] = {}
        self._first_turn: bool = False
        self._terastallized: bool = False
        self._terastallized_type: Optional[PokemonType] = None
        self._item: Optional[str] = self._data.UNKNOWN_ITEM
        self._last_request: Optional[Dict[str, Any]] = {}
        self._last_details: str = ""
        self._must_recharge: bool = False
        self._preparing_move: Optional[Move] = None
        self._preparing_target = None
        self._protect_counter: int = 0
        self._revealed: bool = False
        self._status: Optional[Status] = None
        self._status_counter: int = 0

        # 根据不同情况更新属性
        if request_pokemon:
            self.update_from_request(request_pokemon)
        elif details:
            self._update_from_details(details)
        elif species:
            self._update_from_pokedex(species)

    # 返回对象的字符串表示形式
    def __repr__(self) -> str:
        return self.__str__()
    def __str__(self) -> str:
        # 如果状态为空，则状态表示为 None
        if self._status is None:
            status_repr = None
        else:
            status_repr = self._status.name

        # 返回包含种类、活跃状态和状态的字符串表示
        return (
            f"{self._species} (pokemon object) "
            f"[Active: {self._active}, Status: {status_repr}]"
        )

    def _add_move(self, move_id: str, use: bool = False) -> Optional[Move]:
        """Store the move if applicable."""
        # 获取移动的 ID
        id_ = Move.retrieve_id(move_id)

        # 如果不应该存储该移动，则返回
        if not Move.should_be_stored(id_, self._data.gen):
            return

        # 如果移动不在已有移动列表中，则创建新的移动对象并存储
        if id_ not in self._moves:
            move = Move(move_id=id_, raw_id=move_id, gen=self._data.gen)
            self._moves[id_] = move
        # 如果需要使用该移动，则调用移动对象的 use 方法
        if use:
            self._moves[id_].use()

        return self._moves[id_]

    def boost(self, stat: str, amount: int):
        # 增加指定属性的增益值，并确保在 -6 到 6 之间
        self._boosts[stat] += amount
        if self._boosts[stat] > 6:
            self._boosts[stat] = 6
        elif self._boosts[stat] < -6:
            self._boosts[stat] = -6

    def cant_move(self):
        # 重置首回合标志和保护计数器，如果状态为睡眠则增加状态计数器
        self._first_turn = False
        self._protect_counter = 0

        if self._status == Status.SLP:
            self._status_counter += 1

    def clear_active(self):
        # 清除活跃状态
        self._active = False

    def clear_boosts(self):
        # 将所有属性的增益值清零
        for stat in self._boosts:
            self._boosts[stat] = 0

    def _clear_effects(self):
        # 清除所有效果
        self._effects = {}

    def clear_negative_boosts(self):
        # 将所有负增益值清零
        for stat, value in self._boosts.items():
            if value < 0:
                self._boosts[stat] = 0

    def clear_positive_boosts(self):
        # 将所有正增益值清零
        for stat, value in self._boosts.items():
            if value > 0:
                self._boosts[stat] = 0

    def copy_boosts(self, mon: Pokemon):
        # 复制另一个 Pokemon 对象的增益值
        self._boosts = dict(mon._boosts.items())
    # 更新精灵的状态，可以传入一个状态字符串，默认为 None
    def cure_status(self, status: Optional[str] = None):
        # 如果传入了状态并且状态与当前状态相同，则将当前状态置为 None，并重置状态计数器
        if status and Status[status.upper()] == self._status:
            self._status = None
            self._status_counter = 0
        # 如果状态为 None 并且精灵没有失去意识，则将当前状态置为 None
        elif status is None and not self.fainted:
            self._status = None

    # 造成伤害的方法，设置精灵的生命状态
    def damage(self, hp_status: str):
        self.set_hp_status(hp_status)

    # 结束效果的方法，根据效果字符串结束对应效果
    def end_effect(self, effect_str: str):
        effect = Effect.from_showdown_message(effect_str)
        # 如果效果存在于精灵的效果列表中，则移除该效果
        if effect in self._effects:
            self._effects.pop(effect)

    # 结束物品效果的方法，将精灵的物品置为 None
    def end_item(self, item: str):
        self._item = None
        # 如果物品为 "powerherb"，则重置准备中的招式和目标
        if item == "powerherb":
            self._preparing_move = None
            self._preparing_target = False

    # 结束回合的方法，处理中毒状态计数和效果的回合计数
    def end_turn(self):
        # 如果精灵处于中毒状态，则中毒计数加一
        if self._status == Status.TOX:
            self._status_counter += 1
        # 遍历精灵的效果列表，如果效果需要计数回合，则计数加一
        for effect in self.effects:
            if effect.is_turn_countable:
                self.effects[effect] += 1

    # 精灵失去意识的方法，将当前生命值置为 0，状态置为 FNT
    def faint(self):
        self._current_hp = 0
        self._status = Status.FNT

    # 改变形态的方法，根据传入的种类字符串更新精灵的形态
    def forme_change(self, species: str):
        species = species.split(",")[0]
        self._update_from_pokedex(species, store_species=False)

    # 恢复生命值的方法，设置精灵的生命状态
    def heal(self, hp_status: str):
        self.set_hp_status(hp_status)

    # 反转增益的方法，将精灵的增益值取反
    def invert_boosts(self):
        self._boosts = {k: -v for k, v in self._boosts.items()}

    # 超级进化的方法，根据传入的进化石字符串进行超级进化
    def mega_evolve(self, stone: str):
        species_id_str = to_id_str(self.species)
        # 根据当前种类判断是否需要超级进化
        mega_species = (
            species_id_str + "mega"
            if not species_id_str.endswith("mega")
            else species_id_str
        )
        # 如果超级进化后的种类存在于资料库中，则更新精灵的种类
        if mega_species in self._data.pokedex:
            self._update_from_pokedex(mega_species, store_species=False)
        # 如果进化石的最后一个字符为 "XYxy" 中的一个，则进行特殊形态的超级进化
        elif stone[-1] in "XYxy":
            mega_species = mega_species + stone[-1].lower()
            self._update_from_pokedex(mega_species, store_species=False)
    # 根据移动 ID 进行移动操作，可以设置失败标志和使用标志
    def moved(self, move_id: str, failed: bool = False, use: bool = True):
        # 重置必须充能标志和准备移动和目标
        self._must_recharge = False
        self._preparing_move = None
        self._preparing_target = None
        # 添加移动到精灵的移动列表中
        move = self._add_move(move_id, use=use)

        # 如果移动是保护反击且未失败，则增加保护反击计数
        if move and move.is_protect_counter and not failed:
            self._protect_counter += 1
        else:
            self._protect_counter = 0

        # 如果精灵处于睡眠状态，则增加状态计数
        if self._status == Status.SLP:
            self._status_counter += 1

        # 如果移动列表超过4个，则保留当前移动并替换其他移动
        if len(self._moves) > 4:
            new_moves = {}

            # 保留当前移动
            if move and move in self._moves.values():
                new_moves = {
                    move_id: m for move_id, m in self._moves.items() if m is move
                }

            for move in self._moves:
                if len(new_moves) == 4:
                    break
                elif move not in new_moves:
                    new_moves[move] = self._moves[move]

            self._moves = new_moves

    # 准备移动，设置准备移动和目标
    def prepare(self, move_id: str, target: Optional[Pokemon]):
        self.moved(move_id, use=False)

        move_id = Move.retrieve_id(move_id)
        move = self.moves[move_id]

        self._preparing_move = move
        self._preparing_target = target

    # 激怒状态，根据精灵种类更新属性
    def primal(self):
        species_id_str = to_id_str(self._species)
        primal_species = (
            species_id_str + "primal"
            if not species_id_str.endswith("primal")
            else species_id_str
        )
        self._update_from_pokedex(primal_species, store_species=False)

    # 设置属性提升，确保提升值在 -6 到 6 之间
    def set_boost(self, stat: str, amount: int):
        assert (
            abs(amount) <= 6
        ), f"{stat} of mon {self._species} is not <= 6. Got {amount}"
        self._boosts[stat] = int(amount)

    # 设置生命值，调用设置生命状态的方法
    def set_hp(self, hp_status: str):
        self.set_hp_status(hp_status)
    # 设置精灵的生命值状态
    def set_hp_status(self, hp_status: str):
        # 如果生命值状态为 "0 fnt"，则将精灵设为失去意识状态并返回
        if hp_status == "0 fnt":
            self.faint()
            return

        # 如果生命值状态中包含空格，则分割出生命值和状态
        if " " in hp_status:
            hp, status = hp_status.split(" ")
            # 将状态转换为大写形式并存储在精灵的状态属性中
            self._status = Status[status.upper()]
        else:
            hp = hp_status

        # 提取生命值中的数字部分并分割出当前生命值和最大生命值
        hp = "".join([c for c in hp if c in "0123456789/"]).split("/")
        self._current_hp = int(hp[0])
        self._max_hp = int(hp[1])

    # 启动效果
    def start_effect(self, effect_str: str):
        # 从效果字符串中创建效果对象
        effect = Effect.from_showdown_message(effect_str)
        # 如果效果不在精灵的效果列表中，则添加到列表中；如果效果可以计数，则增加计数
        if effect not in self._effects:
            self._effects[effect] = 0
        elif effect.is_action_countable:
            self._effects[effect] += 1

        # 如果效果会打破保护状态，则将保护计数器设为0
        if effect.breaks_protect:
            self._protect_counter = 0

    # 交换攻击和特攻提升
    def _swap_boosts(self):
        self._boosts["atk"], self._boosts["spa"] = (
            self._boosts["spa"],
            self._boosts["atk"],
        )

    # 精灵交换上场
    def switch_in(self, details: Optional[str] = None):
        self._active = True

        # 如果有详细信息，则更新精灵状态
        if details:
            self._update_from_details(details)

        self._first_turn = True
        self._revealed = True

    # 精灵交换下场
    def switch_out(self):
        self._active = False
        self.clear_boosts()
        self._clear_effects()
        self._first_turn = False
        self._must_recharge = False
        self._preparing_move = None
        self._preparing_target = None
        self._protect_counter = 0

        # 如果精灵处于中毒状态，则将中毒计数器设为0
        if self._status == Status.TOX:
            self._status_counter = 0

    # 转变为特殊状态
    def terastallize(self, type_: str):
        # 将精灵的特殊状态类型设为给定类型，并标记为特殊状态
        self._terastallized_type = PokemonType.from_name(type_)
        self._terastallized = True

    # 转变为另一个精灵
    def transform(self, into: Pokemon):
        current_hp = self.current_hp
        # 更新精灵的信息为另一个精灵的信息，但不存储物种信息
        self._update_from_pokedex(into.species, store_species=False)
        self._current_hp = int(current_hp)
        self._boosts = into.boosts.copy()
    # 从宝可梦图鉴更新信息，包括种类、基础属性、类型、能力、身高和体重
    def _update_from_pokedex(self, species: str, store_species: bool = True):
        # 将种类名称转换为小写字符串
        species = to_id_str(species)
        # 获取宝可梦图鉴中该种类的信息
        dex_entry = self._data.pokedex[species]
        # 如果需要存储种类信息，则更新当前实例的种类属性
        if store_species:
            self._species = species
        # 更新当前实例的基础属性
        self._base_stats = dex_entry["baseStats"]

        # 更新当前实例的第一个类型
        self._type_1 = PokemonType.from_name(dex_entry["types"][0])
        # 如果宝可梦只有一个类型，则第二个类型为空
        if len(dex_entry["types"]) == 1:
            self._type_2 = None
        else:
            self._type_2 = PokemonType.from_name(dex_entry["types"][1])

        # 将宝可梦可能的能力转换为小写字符串列表
        self._possible_abilities = [
            to_id_str(ability) for ability in dex_entry["abilities"].values()
        ]

        # 如果宝可梦只有一种可能的能力，则将其设置为当前实例的能力
        if len(self._possible_abilities) == 1:
            self.ability = self._possible_abilities[0]

        # 更新当前实例的身高和体重属性
        self._heightm = dex_entry["heightm"]
        self._weightkg = dex_entry["weightkg"]
    # 从给定的详情字符串更新属性
    def _update_from_details(self, details: str):
        # 如果详情字符串与上次相同，则直接返回
        if details == self._last_details:
            return
        else:
            # 更新上次的详情字符串
            self._last_details = details

        # 如果详情字符串中包含", shiny"，则设置为闪光属性并移除该部分
        if ", shiny" in details:
            self._shiny = True
            details = details.replace(", shiny", "")
        else:
            self._shiny = False

        # 将详情字符串按", "分割
        split_details = details.split(", ")

        gender = None
        level = None

        # 遍历分割后的详情字符串
        for split_detail in split_details:
            # 如果以"tera:"开头，则设置特拉斯化类型并移除该部分
            if split_detail.startswith("tera:"):
                self._terastallized_type = PokemonType.from_name(split_detail[5:])

                split_details.remove(split_detail)
                break

        # 根据分割后的长度确定物种、等级和性别
        if len(split_details) == 3:
            species, level, gender = split_details
        elif len(split_details) == 2:
            if split_details[1].startswith("L"):
                species, level = split_details
            else:
                species, gender = split_details
        else:
            species = to_id_str(split_details[0])

        # 根据性别设置性别属性
        if gender:
            self._gender = PokemonGender.from_request_details(gender)
        else:
            self._gender = PokemonGender.NEUTRAL

        # 根据等级设置等级属性
        if level:
            self._level = int(level[1:])
        else:
            self._level = 100

        # 如果物种不同于当前物种，则更新属性
        if species != self._species:
            self._update_from_pokedex(species)
    # 从请求中更新精灵信息
    def update_from_request(self, request_pokemon: Dict[str, Any]):
        # 更新精灵的活跃状态
        self._active = request_pokemon["active"]

        # 如果请求的精灵信息与上次请求相同，则直接返回
        if request_pokemon == self._last_request:
            return

        # 根据请求中是否包含能力信息来更新精灵的能力
        if "ability" in request_pokemon:
            self.ability = request_pokemon["ability"]
        elif "baseAbility" in request_pokemon:
            self.ability = request_pokemon["baseAbility"]

        # 更新上次请求的精灵信息
        self._last_request = request_pokemon

        # 根据请求中的状态信息更新精灵的血量状态
        condition = request_pokemon["condition"]
        self.set_hp_status(condition)

        # 更新精灵的持有物品
        self._item = request_pokemon["item"]

        # 更新精灵的详细信息
        details = request_pokemon["details"]
        self._update_from_details(details)

        # 添加请求中的招式到精灵的招式列表中
        for move in request_pokemon["moves"]:
            self._add_move(move)

        # 如果招式数量超过4个，则只保留最新的4个招式
        if len(self._moves) > 4:
            moves_to_keep = {
                Move.retrieve_id(move_id) for move_id in request_pokemon["moves"]
            }
            self._moves = {
                move_id: move
                for move_id, move in self._moves.items()
                if move_id in moves_to_keep
            }

    # 使用 Z 招式
    def used_z_move(self):
        # 将精灵的持有物品设为 None
        self._item = None

    # 判断是否被幻觉术影响
    def was_illusioned(self):
        # 重置精灵的血量、最大血量和状态
        self._current_hp = None
        self._max_hp = None
        self._status = None

        # 保存上次请求的精灵信息
        last_request = self._last_request
        self._last_request = None

        # 如果存在上次请求的信息，则更新精灵信息并切换出场
        if last_request:
            self.update_from_request(last_request)

        # 切换出场
        self.switch_out()
    # 根据请求中的信息获取可用的移动列表
    def available_moves_from_request(self, request: Dict[str, Any]) -> List[Move]:
        # 初始化一个空的移动列表
        moves: List[Move] = []

        # 从请求中提取可用的移动列表
        request_moves: List[str] = [
            move["id"] for move in request["moves"] if not move.get("disabled", False)
        ]
        # 遍历请求中的每个移动
        for move in request_moves:
            # 如果移动在当前角色的移动列表中
            if move in self.moves:
                # 如果当前角色处于极巨化状态，则添加极巨化后的移动
                if self.is_dynamaxed:
                    moves.append(self.moves[move].dynamaxed)
                else:
                    moves.append(self.moves[move])
            # 如果移动是特殊移动
            elif move in SPECIAL_MOVES:
                moves.append(Move(move, gen=self._data.gen))
            # 如果移动是"hiddenpower"且只有一个隐藏能力
            elif (
                move == "hiddenpower"
                and len([m for m in self.moves if m.startswith("hiddenpower")]) == 1
            ):
                moves.append(
                    [v for m, v in self.moves.items() if m.startswith("hiddenpower")][0]
                )
            else:
                # 断言当前角色的移动列表中包含特定的移动
                assert {
                    "copycat",
                    "metronome",
                    "mefirst",
                    "mirrormove",
                    "assist",
                    "transform",
                    "mimic",
                }.intersection(self.moves), (
                    f"Error with move {move}. Expected self.moves to contain copycat, "
                    "metronome, mefirst, mirrormove, assist, transform or mimic. Got"
                    f" {self.moves}"
                )
                moves.append(Move(move, gen=self._data.gen))
        # 返回最终的移动列表
        return moves
    def damage_multiplier(self, type_or_move: Union[PokemonType, Move]) -> float:
        """
        Returns the damage multiplier associated with a given type or move on this
        pokemon.

        This method is a shortcut for PokemonType.damage_multiplier with relevant types.

        :param type_or_move: The type or move of interest.
        :type type_or_move: PokemonType or Move
        :return: The damage multiplier associated with given type on the pokemon.
        :rtype: float
        """
        # 如果输入的是 Move 对象，则获取其类型
        if isinstance(type_or_move, Move):
            type_or_move = type_or_move.type
        # 调用 PokemonType.damage_multiplier 方法计算伤害倍率
        return type_or_move.damage_multiplier(
            self._type_1, self._type_2, type_chart=self._data.type_chart
        )

    @property
    def ability(self) -> Optional[str]:
        """
        :return: The pokemon's ability. None if unknown.
        :rtype: str, optional
        """
        # 返回宝可梦的能力
        return self._ability

    @ability.setter
    def ability(self, ability: Optional[str]):
        # 设置宝可梦的能力，如果输入为 None，则设为 None，否则转换为小写字符串
        if ability is None:
            self._ability = None
        else:
            self._ability = to_id_str(ability)

    @property
    def active(self) -> Optional[bool]:
        """
        :return: Boolean indicating whether the pokemon is active.
        :rtype: bool
        """
        # 返回一个布尔值，指示宝可梦是否处于活跃状态
        return self._active

    @active.setter
    def active(self, value: Optional[bool]):
        # 设置宝可梦的活跃状态
        self.active = value

    @property
    def available_z_moves(self) -> List[Move]:
        """
        Caution: this property is not properly tested yet.

        :return: The set of moves that pokemon can use as z-moves.
        :rtype: List[Move]
        """
        # 检查是否持有 Z 水晶道具，如果是则返回可以使用的 Z 招式列表
        if isinstance(self.item, str) and self.item.endswith("iumz"):
            type_, move = Z_CRYSTAL[self.item]
            if type_:
                return [
                    move
                    for move in self._moves.values()
                    if move.type == type_ and move.can_z_move
                ]
            elif move in self._moves:
                return [self._moves[move]]
        return []

    @property
    def base_species(self) -> str:
        """
        :return: The pokemon's base species.
        :rtype: str
        """
        # 返回精灵的基础种类
        dex_entry = self._data.pokedex[self._species]
        if "baseSpecies" in dex_entry:
            return to_id_str(dex_entry["baseSpecies"])
        return self._species

    @property
    def base_stats(self) -> Dict[str, int]:
        """
        :return: The pokemon's base stats.
        :rtype: Dict[str, int]
        """
        # 返回精灵的基础属性
        return self._base_stats

    @property
    def boosts(self) -> Dict[str, int]:
        """
        :return: The pokemon's boosts.
        :rtype: Dict[str, int]
        """
        # 返回精灵的属性提升
        return self._boosts

    @boosts.setter
    def boosts(self, value: Dict[str, int]):
        self._boosts = value

    @property
    def current_hp(self) -> int:
        """
        :return: The pokemon's current hp. For your pokemons, this is the actual value.
            For opponent's pokemon, this value depends on showdown information: it can
            be on a scale from 0 to 100 or on a pixel scale.
        :rtype: int
        """
        # 返回精灵的当前生命值，如果为对手的精灵，则根据信息返回不同的值
        return self._current_hp or 0

    @property
    def current_hp_fraction(self) -> float:
        """
        :return: The pokemon's current remaining hp fraction.
        :rtype: float
        """
        # 计算当前剩余 HP 占总 HP 的比例
        if self.current_hp:
            return self.current_hp / self.max_hp
        return 0

    @property
    def effects(self) -> Dict[Effect, int]:
        """
        :return: A Dict mapping the effects currently affecting the pokemon and the
            associated counter.
        :rtype: Dict[Effect, int]
        """
        # 返回当前影响宝可梦的效果及其计数的字典
        return self._effects

    @property
    def fainted(self) -> bool:
        """
        :return: Wheter the pokemon has fainted.
        :rtype: bool
        """
        # 返回宝可梦是否已经倒下
        return Status.FNT == self._status

    @property
    def first_turn(self) -> bool:
        """
        :return: Wheter this is this pokemon's first action since its last switch in.
        :rtype: bool
        """
        # 返回这只宝可梦自上次交换以来是否第一次行动
        return self._first_turn

    @property
    def gender(self) -> Optional[PokemonGender]:
        """
        :return: The pokemon's gender.
        :rtype: PokemonGender, optional
        """
        # 返回宝可梦的性别
        return self._gender

    @property
    def height(self) -> float:
        """
        :return: The pokemon's height, in meters.
        :rtype: float
        """
        # 返回宝可梦的身高，单位为米
        return self._heightm

    @property
    def is_dynamaxed(self) -> bool:
        """
        :return: Whether the pokemon is currently dynamaxed
        :rtype: bool
        """
        # 返回宝可梦当前是否处于极巨化状态
        return Effect.DYNAMAX in self.effects

    @property
    def item(self) -> Optional[str]:
        """
        :return: The pokemon's item.
        :rtype: str | None
        """
        # 返回宝可梦携带的物品
        return self._item

    @item.setter
    def item(self, item: Optional[str]):
        # 设置宝可梦携带的物品
        self._item = item

    @property
    def level(self) -> int:
        """
        :return: The pokemon's level.
        :rtype: int
        """
        # 返回宝可梦的等级
        return self._level

    @property
    def max_hp(self) -> int:
        """
        :return: The pokemon's max hp. For your pokemons, this is the actual value.
            For opponent's pokemon, this value depends on showdown information: it can
            be on a scale from 0 to 100 or on a pixel scale.
        :rtype: int
        """
        # 返回精灵的最大生命值，如果没有则返回0
        return self._max_hp or 0

    @property
    def moves(self) -> Dict[str, Move]:
        """
        :return: A dictionary of the pokemon's known moves.
        :rtype: Dict[str, Move]
        """
        # 返回精灵已知招式的字典
        return self._moves

    @property
    def must_recharge(self) -> bool:
        """
        :return: A boolean indicating whether the pokemon must recharge.
        :rtype: bool
        """
        # 返回一个布尔值，指示精灵是否需要充能
        return self._must_recharge

    @must_recharge.setter
    def must_recharge(self, value: bool):
        # 设置精灵是否需要充能
        self._must_recharge = value

    @property
    def pokeball(self) -> Optional[str]:
        """
        :return: The pokeball in which is the pokemon.
        :rtype: str | None
        """
        # 返回精灵所在的精灵球
        if self._last_request is not None:
            return self._last_request.get("pokeball", None)

    @property
    def possible_abilities(self) -> List[str]:
        """
        :return: The list of possible abilities for this pokemon.
        :rtype: List[str]
        """
        # 返回该精灵可能的能力列表
        return self._possible_abilities

    @property
    def preparing(self) -> bool:
        """
        :return: Whether this pokemon is preparing a multi-turn move.
        :rtype: bool
        """
        # 返回一个布尔值，指示该精灵是否正在准备多回合招式
        return bool(self._preparing_target) or bool(self._preparing_move)

    @property
    def preparing_target(self) -> Optional[Union[bool, Pokemon]]:
        """
        :return: The moves target - optional.
        :rtype: Any
        """
        # 返回准备的招式目标，可选
        return self._preparing_target

    @property
    def preparing_move(self) -> Optional[Move]:
        """
        :return: The move being prepared - optional.
        :rtype: Move, optional
        """
        # 返回正在准备的招式，可选
        return self._preparing_move

    @property
    def protect_counter(self) -> int:
        """
        :return: How many protect-like moves where used in a row by this pokemon.
        :rtype: int
        """
        # 返回该精灵连续使用保护类招式的次数
        return self._protect_counter

    @property
    def revealed(self) -> bool:
        """
        :return: Whether this pokemon has appeared in the current battle.
        :rtype: bool
        """
        # 返回该精灵是否已经在当前战斗中出现过
        return self._revealed

    @property
    def shiny(self) -> bool:
        """
        :return: Whether this pokemon is shiny.
        :rtype: bool
        """
        # 返回该精灵是否为闪光精灵
        return bool(self._shiny)

    @property
    def species(self) -> str:
        """
        :return: The pokemon's species.
        :rtype: str | None
        """
        # 返回该精灵的种类
        return self._species

    @property
    def stats(self) -> Optional[Dict[str, Optional[int]]]:
        """
        :return: The pokemon's stats, as a dictionary.
        :rtype: Dict[str, int | None]
        """
        # 如果上一个请求存在，则返回上一个请求中的stats，否则返回默认值
        if self._last_request is not None:
            return self._last_request.get(
                "stats",
                {"atk": None, "def": None, "spa": None, "spd": None, "spe": None},
            )

    @property
    def status(self) -> Optional[Status]:
        """
        :return: The pokemon's status.
        :rtype: Optional[Status]
        """
        # 返回该精灵的状态
        return self._status
    # 计算精灵的属性值，包括IV和EV
    def calculate_stats(self, ivs=(31,) * 6, evs=(85,) * 6):
        # 定义一个内部函数，用于计算精灵的属性值
        def common_pkmn_stat_calc(stat: int, iv: int, ev: int, level: int):
            return math.floor(((2 * stat + iv + math.floor(ev / 4)) * level) / 100)

        # 创建一个新的属性字典
        new_stats = dict()
        # 计算并更新HP属性值
        new_stats['hp'] = common_pkmn_stat_calc(
                            self._base_stats['hp'],
                            ivs[0],
                            evs[0],
                            self._level
                        ) + self._level + 10

        # 计算并更新攻击属性值
        new_stats['atk'] = common_pkmn_stat_calc(
                            self._base_stats['atk'],
                            ivs[1],
                            evs[1],
                            self._level
                        ) + 5

        # 计算并更新防御属性值
        new_stats['def'] = common_pkmn_stat_calc(
                            self._base_stats['def'],
                            ivs[2],
                            evs[2],
                            self._level
                        ) + 5

        # 计算并更新特攻属性值
        new_stats['spa'] = common_pkmn_stat_calc(
                            self._base_stats['spa'],
                            ivs[3],
                            evs[3],
                            self._level
                        ) + 5

        # 计算并更新特防属性值
        new_stats['spd'] = common_pkmn_stat_calc(
                            self._base_stats['spd'],
                            ivs[4],
                            evs[4],
                            self._level
                        ) + 5

        # 计算并更新速度属性值
        new_stats['spe'] = common_pkmn_stat_calc(
                            self._base_stats['spe'],
                            ivs[5],
                            evs[5],
                            self._level
                        ) + 5

        # 将属性值转换为整数
        new_stats = {k: int(v) for k, v in new_stats.items()}
        # 返回新的属性字典
        return new_stats

    # 属性装饰器
    @property
    # 返回宝可梦的状态回合计数。只计算中毒和睡眠状态。
    def status_counter(self) -> int:
        """
        :return: The pokemon's status turn count. Only counts TOXIC and SLEEP statuses.
        :rtype: int
        """
        return self._status_counter

    # 设置宝可梦的状态
    @status.setter
    def status(self, status: Optional[Union[Status, str]]):
        self._status = Status[status.upper()] if isinstance(status, str) else status

    # 返回宝可梦的STAB倍数
    @property
    def stab_multiplier(self) -> float:
        """
        :return: The pokemon's STAB multiplier.
        :rtype: float
        """
        if self._terastallized and self._terastallized_type in (
            self._type_1,
            self._type_2,
        ):
            return 2
        return 1

    # 返回宝可梦当前是否处于terastallized状态
    @property
    def terastallized(self) -> bool:
        """
        :return: Whether the pokemon is currently terastallized
        :rtype: bool
        """
        return self._terastallized

    # 返回宝可梦的第一个类型
    @property
    def type_1(self) -> PokemonType:
        """
        :return: The pokemon's first type.
        :rtype: PokemonType
        """
        if self._terastallized and self._terastallized_type is not None:
            return self._terastallized_type
        return self._type_1

    # 返回宝可梦的第二个类型
    @property
    def type_2(self) -> Optional[PokemonType]:
        """
        :return: The pokemon's second type.
        :rtype: Optional[PokemonType]
        """
        if self._terastallized:
            return None
        return self._type_2

    # 返回宝可梦的类型，以元组形式返回
    @property
    def types(self) -> Tuple[PokemonType, Optional[PokemonType]]:
        """
        :return: The pokemon's types, as a tuple.
        :rtype: Tuple[PokemonType, Optional[PokemonType]]
        """
        return self.type_1, self._type_2

    # 返回宝可梦的重量，单位为千克
    @property
    def weight(self) -> float:
        """
        :return: The pokemon's weight, in kilograms.
        :rtype: float
        """
        return self._weightkg
```