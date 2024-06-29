# `D:\src\scipysrc\matplotlib\lib\matplotlib\_enums.pyi`

```py
# 导入枚举类 Enum
from enum import Enum

# 定义一个私有类 _AutoStringNameEnum，继承自 Enum 类
class _AutoStringNameEnum(Enum):
    # 定义 __hash__ 方法，返回值为整数类型
    def __hash__(self) -> int:
        # 省略实际的实现，本注释中未给出具体内容

# 定义 JoinStyle 枚举类，继承自 str 和 _AutoStringNameEnum
class JoinStyle(str, _AutoStringNameEnum):
    # 定义枚举成员 miter，并设置其值为字符串 'miter'
    miter: str
    # 定义枚举成员 round，并设置其值为字符串 'round'
    round: str
    # 定义枚举成员 bevel，并设置其值为字符串 'bevel'
    bevel: str
    
    # 静态方法 demo，无返回值
    @staticmethod
    def demo() -> None:
        # 省略方法体内容，本注释中未给出具体内容

# 定义 CapStyle 枚举类，继承自 str 和 _AutoStringNameEnum
class CapStyle(str, _AutoStringNameEnum):
    # 定义枚举成员 butt，并设置其值为字符串 'butt'
    butt: str
    # 定义枚举成员 projecting，并设置其值为字符串 'projecting'
    projecting: str
    # 定义枚举成员 round，并设置其值为字符串 'round'
    round: str
    
    # 静态方法 demo，无返回值
    @staticmethod
    def demo() -> None:
        # 省略方法体内容，本注释中未给出具体内容
```