# `.\PokeLLMon\poke_env\environment\z_crystal.py`

```py
"""This module contains objects related ot z-crystal management. It should not be used
directly.
"""
# 导入必要的类型
from typing import Dict, Optional, Tuple

# 定义一个字典，存储 Z 晶石的信息，键为晶石名称，值为元组，元组包含两个元素：对应的宝可梦属性和招式
Z_CRYSTAL: Dict[str, Tuple[Optional[PokemonType], Optional[str]]] = {
    "buginiumz": (PokemonType.BUG, None),
    "darkiniumz": (PokemonType.DARK, None),
    "dragoniumz": (PokemonType.DRAGON, None),
    "electriumz": (PokemonType.ELECTRIC, None),
    "fairiumz": (PokemonType.FAIRY, None),
    "fightiniumz": (PokemonType.FIGHTING, None),
    "firiumz": (PokemonType.FIRE, None),
    "flyiniumz": (PokemonType.FLYING, None),
    "ghostiumz": (PokemonType.GHOST, None),
    "grassiumz": (PokemonType.GRASS, None),
    "groundiumz": (PokemonType.GROUND, None),
    "iciumz": (PokemonType.ICE, None),
    "normaliumz": (PokemonType.NORMAL, None),
    "poisoniumz": (PokemonType.POISON, None),
    "psychiumz": (PokemonType.PSYCHIC, None),
    "rockiumz": (PokemonType.ROCK, None),
    "steeliumz": (PokemonType.STEEL, None),
    "wateriumz": (PokemonType.WATER, None),
    "aloraichiumz": (None, "thunderbolt"),
    "decidiumz": (None, "spiritshackle"),
    "eeviumz": (None, "lastresort"),
    "inciniumz": (None, "darkestlariat"),
    "kommoniumz": (None, "clangingscales"),
    "lunaliumz": (None, "moongeistbeam"),
    "lycaniumz": (None, "stoneedge"),
    "marshadiumz": (None, "spectralthief"),
    "mewniumz": (None, "psychic"),
    "mimikiumz": (None, "playrough"),
    "pikaniumz": (None, "volttackle"),
    "pikashuniumz": (None, "thunderbolt"),
    "primariumz": (None, "sparklingaria"),
    "snorliumz": (None, "gigaimpact"),
    "solganiumz": (None, "sunsteelstrike"),
    "tapuniumz": (None, "naturesmadness"),
    "ultranecroziumz": (None, "photongeyser"),
}
```