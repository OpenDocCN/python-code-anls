# `.\PokeLLMon\poke_env\environment\status.py`

```py
"""
This module defines the Status class, which represents statuses a pokemon can be afflicted with.
"""

# 导入必要的模块
from enum import Enum, auto, unique

# 定义 Status 类，表示宝可梦可能受到的状态
@unique
class Status(Enum):
    """Enumeration, represent a status a pokemon can be afflicted with."""

    # 定义不同的状态
    BRN = auto()  # 烧伤状态
    FNT = auto()  # 濒死状态
    FRZ = auto()  # 冰冻状态
    PAR = auto()  # 麻痹状态
    PSN = auto()  # 中毒状态
    SLP = auto()  # 睡眠状态
    TOX = auto()  # 中毒状态（剧毒）

    # 定义 __str__ 方法，返回状态对象的字符串表示
    def __str__(self) -> str:
        return f"{self.name} (status) object"
```