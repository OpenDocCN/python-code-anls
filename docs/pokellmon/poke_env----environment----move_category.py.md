# `.\PokeLLMon\poke_env\environment\move_category.py`

```
# 导入需要的模块
from enum import Enum, auto, unique

# 定义一个枚举类 MoveCategory，表示一个移动类别
@unique
class MoveCategory(Enum):
    """Enumeration, represent a move category."""

    # 定义枚举值 PHYSICAL，表示物理攻击类别
    PHYSICAL = auto()
    # 定义枚举值 SPECIAL，表示特殊攻击类别
    SPECIAL = auto()
    # 定义枚举值 STATUS，表示状态类别
    STATUS = auto()

    # 定义 __str__ 方法，返回枚举值的名称和描述信息
    def __str__(self) -> str:
        return f"{self.name} (move category) object"
```