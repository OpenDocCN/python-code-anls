# `.\PokeLLMon\poke_env\environment\__init__.py`

```py
# 从 poke_env.environment 模块中导入各种类和常量
from poke_env.environment import (
    abstract_battle,
    battle,
    double_battle,
    effect,
    field,
    move,
    move_category,
    pokemon,
    pokemon_gender,
    pokemon_type,
    side_condition,
    status,
    weather,
    z_crystal,
)
# 从 poke_env.environment.abstract_battle 模块中导入 AbstractBattle 类
from poke_env.environment.abstract_battle import AbstractBattle
# 从 poke_env.environment.battle 模块中导入 Battle 类
from poke_env.environment.battle import Battle
# 从 poke_env.environment.double_battle 模块中导入 DoubleBattle 类
from poke_env.environment.double_battle import DoubleBattle
# 从 poke_env.environment.effect 模块中导入 Effect 类
from poke_env.environment.effect import Effect
# 从 poke_env.environment.field 模块中导入 Field 类
from poke_env.environment.field import Field
# 从 poke_env.environment.move 模块中导入 SPECIAL_MOVES, EmptyMove, Move 类
from poke_env.environment.move import SPECIAL_MOVES, EmptyMove, Move
# 从 poke_env.environment.move_category 模块中导入 MoveCategory 类
from poke_env.environment.move_category import MoveCategory
# 从 poke_env.environment.pokemon 模块中导入 Pokemon 类
from poke_env.environment.pokemon import Pokemon
# 从 poke_env.environment.pokemon_gender 模块中导入 PokemonGender 类
from poke_env.environment.pokemon_gender import PokemonGender
# 从 poke_env.environment.pokemon_type 模块中导入 PokemonType 类
from poke_env.environment.pokemon_type import PokemonType
# 从 poke_env.environment.side_condition 模块中导入 STACKABLE_CONDITIONS, SideCondition 类
from poke_env.environment.side_condition import STACKABLE_CONDITIONS, SideCondition
# 从 poke_env.environment.status 模块中导入 Status 类
from poke_env.environment.status import Status
# 从 poke_env.environment.weather 模块中导入 Weather 类
from poke_env.environment.weather import Weather
# 从 poke_env.environment.z_crystal 模块中导入 Z_CRYSTAL 类
from poke_env.environment.z_crystal import Z_CRYSTAL

# 定义 __all__ 列表，包含所有导入的类和常量的名称
__all__ = [
    "AbstractBattle",
    "Battle",
    "DoubleBattle",
    "Effect",
    "EmptyMove",
    "Field",
    "Move",
    "MoveCategory",
    "Pokemon",
    "PokemonGender",
    "PokemonType",
    "SPECIAL_MOVES",
    "STACKABLE_CONDITIONS",
    "SideCondition",
    "Status",
    "Weather",
    "Z_CRYSTAL",
    "abstract_battle",
    "battle",
    "double_battle",
    "effect",
    "field",
    "move",
    "move_category",
    "pokemon",
    "pokemon_gender",
    "pokemon_type",
    "side_condition",
    "status",
    "weather",
    "z_crystal",
]
```