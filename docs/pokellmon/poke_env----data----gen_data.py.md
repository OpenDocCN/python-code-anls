# `.\PokeLLMon\poke_env\data\gen_data.py`

```py
# 导入必要的模块和函数
from __future__ import annotations
import os
from functools import lru_cache
from typing import Any, Dict, Optional, Union
import orjson
from poke_env.data.normalize import to_id_str

# 定义一个类 GenData
class GenData:
    # 限制实例的属性，只能包含在 __slots__ 中指定的属性
    __slots__ = ("gen", "moves", "natures", "pokedex", "type_chart", "learnset")
    
    # 定义一个类变量 UNKNOWN_ITEM
    UNKNOWN_ITEM = "unknown_item"
    
    # 定义一个类变量 _gen_data_per_gen，用于存储不同世代的 GenData 实例
    _gen_data_per_gen: Dict[int, GenData] = {}
    
    # 初始化方法，接受一个 gen 参数
    def __init__(self, gen: int):
        # 如果该世代的 GenData 已经初始化过，则抛出异常
        if gen in self._gen_data_per_gen:
            raise ValueError(f"GenData for gen {gen} already initialized.")
        
        # 初始化实例属性
        self.gen = gen
        self.moves = self.load_moves(gen)
        self.natures = self.load_natures()
        self.pokedex = self.load_pokedex(gen)
        self.type_chart = self.load_type_chart(gen)
        self.learnset = self.load_learnset()
    
    # 定义深拷贝方法，返回当前实例本身
    def __deepcopy__(self, memodict: Optional[Dict[int, Any]] = None) -> GenData:
        return self
    
    # 加载指定世代的招式数据
    def load_moves(self, gen: int) -> Dict[str, Any]:
        with open(
            os.path.join(self._static_files_root, "moves", f"gen{gen}moves.json")
        ) as f:
            return orjson.loads(f.read())
    
    # 加载自然性格数据
    def load_natures(self) -> Dict[str, Dict[str, Union[int, float]]]:
        with open(os.path.join(self._static_files_root, "natures.json")) as f:
            return orjson.loads(f.read())
    
    # 加载学会招式数据
    def load_learnset(self) -> Dict[str, Dict[str, Union[int, float]]]:
        with open(os.path.join(self._static_files_root, "learnset.json")) as f:
            return orjson.loads(f.read())
    # 加载宝可梦图鉴数据，根据给定的世代号
    def load_pokedex(self, gen: int) -> Dict[str, Any]:
        # 打开对应世代号的宝可梦图鉴 JSON 文件
        with open(
            os.path.join(self._static_files_root, "pokedex", f"gen{gen}pokedex.json")
        ) as f:
            # 使用 orjson 库加载 JSON 文件内容
            dex = orjson.loads(f.read())

        # 创建一个空字典用于存储其他形态的宝可梦数据
        other_forms_dex: Dict[str, Any] = {}
        # 遍历宝可梦图鉴数据
        for value in dex.values():
            # 如果存在"cosmeticFormes"字段
            if "cosmeticFormes" in value:
                # 遍历所有的其他形态
                for other_form in value["cosmeticFormes"]:
                    # 将其他形态的数据存入字典中
                    other_forms_dex[to_id_str(other_form)] = value

        # 处理皮卡丘的特殊形态
        for name, value in dex.items():
            # 如果名称以"pikachu"开头且不是"pikachu"或"pikachugmax"
            if name.startswith("pikachu") and name not in {"pikachu", "pikachugmax"}:
                # 添加对应的"gmax"形态数据
                other_forms_dex[name + "gmax"] = dex["pikachugmax"]

        # 将其他形态数据合并到原始数据中
        dex.update(other_forms_dex)

        # 更新宝可梦数据中的"species"字段
        for name, value in dex.items():
            # 如果存在"baseSpecies"字段
            if "baseSpecies" in value:
                # 将"species"字段设置为"baseSpecies"字段的值
                value["species"] = value["baseSpecies"]
            else:
                # 否则将"baseSpecies"字段设置为名称的标准化形式
                value["baseSpecies"] = to_id_str(name)

        # 返回更新后的宝可梦图鉴数据
        return dex
    # 加载指定世代的类型相克表
    def load_type_chart(self, gen: int) -> Dict[str, Dict[str, float]]:
        # 打开对应世代的类型相克表 JSON 文件
        with open(
            os.path.join(
                self._static_files_root, "typechart", f"gen{gen}typechart.json"
            )
        ) as chart:
            # 将 JSON 文件内容加载为字典
            json_chart = orjson.loads(chart.read())

        # 获取所有类型并转换为大写
        types = [str(type_).upper() for type_ in json_chart]
        # 初始化类型相克表字典
        type_chart = {type_1: {type_2: 1.0 for type_2 in types} for type_1 in types}

        # 遍历类型相克表数据
        for type_, data in json_chart.items():
            type_ = type_.upper()

            # 遍历每个类型对应的伤害倍数
            for other_type, damage_taken in data["damageTaken"].items():
                if other_type.upper() not in types:
                    continue

                # 确保伤害倍数在合法范围内
                assert damage_taken in {0, 1, 2, 3}, (data["damageTaken"], type_)

                # 根据伤害倍数设置相应的伤害值
                if damage_taken == 0:
                    type_chart[type_][other_type.upper()] = 1
                elif damage_taken == 1:
                    type_chart[type_][other_type.upper()] = 2
                elif damage_taken == 2:
                    type_chart[type_][other_type.upper()] = 0.5
                elif damage_taken == 3:
                    type_chart[type_][other_type.upper()] = 0

            # 确保所有类型都在类型相克表中
            assert set(types).issubset(set(type_chart))

        # 确保类型相克表的长度与类型列表长度相同
        assert len(type_chart) == len(types)

        # 确保每个类型的相克效果字典长度与类型列表长度相同
        for effectiveness in type_chart.values():
            assert len(effectiveness) == len(types)

        # 返回类型相克表
        return type_chart

    # 返回静态文件根目录路径
    @property
    def _static_files_root(self) -> str:
        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")

    # 根据世代创建 GenData 实例
    @classmethod
    @lru_cache(None)
    def from_gen(cls, gen: int) -> GenData:
        # 创建指定世代的 GenData 实例
        gen_data = GenData(gen)
        # 将 GenData 实例存储到类属性中
        cls._gen_data_per_gen[gen] = gen_data

        return gen_data

    # 根据格式创建 GenData 实例
    @classmethod
    @lru_cache(None)
    def from_format(cls, format: str) -> GenData:
        # 解析出世代号
        gen = int(format[3])  # Update when Gen 10 comes
        # 根据世代号创建 GenData 实例
        return cls.from_gen(gen)
```