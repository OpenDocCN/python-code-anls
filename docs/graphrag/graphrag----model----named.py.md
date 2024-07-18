# `.\graphrag\graphrag\model\named.py`

```py
# 版权声明和许可信息，表示代码版权归Microsoft Corporation所有，使用MIT许可证
# 导入dataclass模块，用于创建数据类
from dataclasses import dataclass

# 从当前目录下的identified模块中导入Identified类
from .identified import Identified

# 使用dataclass装饰器，定义一个数据类Named，继承自Identified类
@dataclass
class Named(Identified):
    """表示一个带有名称/标题的项目的协议。"""

    title: str
    """项目的名称/标题。"""
```