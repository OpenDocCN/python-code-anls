# `.\graphrag\graphrag\model\identified.py`

```py
# 版权声明和许可信息，指明版权和使用许可为 MIT 许可
# 2024 年版权所有 Microsoft Corporation.
# Licensed under the MIT License

"""包含 'Identified' 协议的模块。"""

# 从 dataclasses 模块导入 dataclass 装饰器
from dataclasses import dataclass

# 使用 dataclass 装饰器定义一个类 Identified，表示一个带有 ID 的协议
@dataclass
class Identified:
    """表示带有 ID 的项目的协议。"""

    id: str
    """项目的 ID。"""

    short_id: str | None
    """用于在用户界面或报告文本中引用该社区的可选人类可读 ID。"""
```