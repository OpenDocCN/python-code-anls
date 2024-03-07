# `.\PokeLLMon\poke_env\teambuilder\teambuilder_pokemon.py`

```
"""This module defines the TeambuilderPokemon class, which is used as an intermediate
format to specify pokemon builds in teambuilders custom classes.
"""
# 导入必要的模块
from typing import List, Optional

from poke_env.data import to_id_str

# 定义 TeambuilderPokemon 类
class TeambuilderPokemon:
    # 不同属性对应的 IV 值
    HP_TO_IVS = {
        "bug": [31, 31, 31, 30, 31, 30],
        "dark": [31, 31, 31, 31, 31, 31],
        "dragon": [30, 31, 31, 31, 31, 31],
        "electric": [31, 31, 31, 31, 30, 31],
        "fighting": [31, 31, 30, 30, 30, 30],
        "fire": [31, 30, 31, 30, 31, 30],
        "flying": [31, 31, 31, 30, 30, 30],
        "ghost": [31, 30, 31, 31, 31, 30],
        "grass": [30, 31, 31, 31, 30, 31],
        "ground": [31, 31, 31, 31, 30, 30],
        "ice": [31, 30, 30, 31, 31, 31],
        "poison": [31, 31, 30, 31, 30, 30],
        "psychic": [30, 31, 31, 30, 31, 31],
        "rock": [31, 31, 30, 30, 31, 30],
        "steel": [31, 31, 31, 31, 31, 30],
        "water": [31, 31, 31, 30, 30, 31],
    }
    # 初始化属性
    evs: List[int]
    ivs: List[int]
    moves: List[str]

    # 初始化方法
    def __init__(
        self,
        nickname: Optional[str] = None,
        species: Optional[str] = None,
        item: Optional[str] = None,
        ability: Optional[str] = None,
        moves: Optional[List[str]] = None,
        nature: Optional[str] = None,
        evs: Optional[List[int]] = None,
        gender: Optional[str] = None,
        ivs: Optional[List[int]] = None,
        shiny: Optional[bool] = None,
        level: Optional[int] = None,
        happiness: Optional[int] = None,
        hiddenpowertype: Optional[str] = None,
        gmax: Optional[bool] = None,
        tera_type: Optional[str] = None,
    # 定义 Pokemon 类，包含各种属性和方法
    ):
        # 初始化 Pokemon 对象的属性
        self.nickname = nickname
        self.species = species
        self.item = item
        self.ability = ability
        self.nature = nature
        self.gender = gender
        self.shiny = shiny
        self.level = level
        self.happiness = happiness
        self.hiddenpowertype = hiddenpowertype
        self.gmax = gmax
        self.tera_type = tera_type
        # 如果 EVs 为 None，则初始化为全 0
        self.evs = evs if evs is not None else [0] * 6
        # 如果 IVs 为 None，则初始化为全 31
        self.ivs = ivs if ivs is not None else [31] * 6

        # 如果 moves 为 None，则初始化为空列表
        if moves is None:
            self.moves = []
        else:
            self.moves = moves

    # 定义 __repr__ 方法，返回格式化后的字符串表示
    def __repr__(self) -> str:
        return self.formatted

    # 定义 __str__ 方法，返回格式化后的字符串表示
    def __str__(self) -> str:
        return self.formatted

    # 定义 formatted_evs 属性，返回格式化后的 EVs 字符串
    @property
    def formatted_evs(self) -> str:
        f_evs = ",".join([str(el) if el != 0 else "" for el in self.evs])
        if f_evs == "," * 5:
            return ""
        return f_evs

    # 定义 formatted_ivs 属性，返回格式化后的 IVs 字符串
    @property
    def formatted_ivs(self) -> str:
        f_ivs = ",".join([str(el) if el != 31 else "" for el in self.ivs])
        if f_ivs == "," * 5:
            return ""
        return f_ivs

    # 定义 formatted_moves 属性，返回格式化后的 moves 字符串
    @property
    def formatted_moves(self) -> str:
        return ",".join([to_id_str(move) for move in self.moves])

    # 定义 formatted_endstring 属性，返回格式化后的结束字符串
    @property
    def formatted_endstring(self) -> str:
        f_str = f",{self.hiddenpowertype or ''},"

        # 根据条件返回不同的字符串
        if self.gmax:
            return f_str + ",G"
        elif self.tera_type:
            return f_str + f",,,{self.tera_type}"

        if self.hiddenpowertype:
            return f_str

        return ""

    # 定义 formatted 属性
    @property
    # 返回格式化后的字符串表示
    def formatted(self) -> str:
        # 准备数据以便格式化
        self._prepare_for_formatting()
        # 格式化输出字符串
        return "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s%s" % (
            self.nickname or "",
            to_id_str(self.species) if self.species else "",
            to_id_str(self.item) if self.item else "",
            to_id_str(self.ability) if self.ability else "",
            self.formatted_moves or "",
            self.nature or "",
            self.formatted_evs or "",
            self.gender or "",
            self.formatted_ivs or "",
            "S" if self.shiny else "",
            self.level or "",
            self.happiness or "",
            self.formatted_endstring,
        )

    # 准备数据以便格式化
    def _prepare_for_formatting(self):
        # 遍历每个招式
        for move in self.moves:
            # 将招式转换为标识符形式
            move = to_id_str(move)
            # 如果招式以"hiddenpower"开头且长度大于11且所有IV值都为31
            if (
                move.startswith("hiddenpower")
                and len(move) > 11
                and all([iv == 31 for iv in self.ivs])
            ):
                # 将IV值替换为对应的隐藏力类型的IV值
                self.ivs = list(self.HP_TO_IVS[move[11:]])
```