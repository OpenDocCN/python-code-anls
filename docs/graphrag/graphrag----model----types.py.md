# `.\graphrag\graphrag\model\types.py`

```py
# 版权声明和许可声明，指出代码版权归属于 2024 年的 Microsoft 公司，使用 MIT 许可证授权
# 导入需要的库中的 Callable 类型
"""Common types for the GraphRAG knowledge model."""

# 定义 TextEmbedder 类型别名，表示接受一个字符串参数并返回一个浮点数列表的可调用对象
from collections.abc import Callable

TextEmbedder = Callable[[str], list[float]]
```