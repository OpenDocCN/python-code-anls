# `.\graphrag\graphrag\index\verbs\entities\summarize\strategies\graph_intelligence\__init__.py`

```py
# 版权声明，标明版权归 Microsoft Corporation 所有，使用 MIT 许可证授权
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 包的顶层模块说明，表示这是实体解析图智能包的根目录
"""The Entity Resolution graph intelligence package root."""

# 从当前包中导入 run_graph_intelligence 模块中的 run 函数，供外部调用
from .run_graph_intelligence import run

# 模块的公开接口列表，只有 run 函数是公开的
__all__ = ["run"]
```