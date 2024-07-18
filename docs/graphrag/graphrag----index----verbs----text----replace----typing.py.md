# `.\graphrag\graphrag\index\verbs\text\replace\typing.py`

```py
# 版权声明和许可证信息，声明版权归 Microsoft Corporation 所有，并遵循 MIT 许可证
# 导入 dataclass 模块，用于支持数据类的定义
from dataclasses import dataclass

# @dataclass 装饰器，用于定义一个数据类 Replacement
@dataclass
class Replacement:
    """Replacement class definition."""
    # 数据类 Replacement 的属性：模式字符串，用于匹配
    pattern: str
    # 数据类 Replacement 的属性：替换字符串，用于替换匹配到的模式
    replacement: str
```