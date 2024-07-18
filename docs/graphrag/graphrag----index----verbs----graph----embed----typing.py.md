# `.\graphrag\graphrag\index\verbs\graph\embed\typing.py`

```py
# 版权声明，声明此代码版权归 Microsoft Corporation 所有，使用 MIT 许可证授权

"""一个包含不同列表和字典的模块。"""

# 暂时使用这个而不是包装器
# 导入 Any 类型用于类型提示
from typing import Any

# NodeList 类型别名，表示由字符串组成的列表
NodeList = list[str]

# EmbeddingList 类型别名，表示由任意类型组成的列表
EmbeddingList = list[Any]

# NodeEmbeddings 类型别名，表示由字符串键和浮点数列表值组成的字典
NodeEmbeddings = dict[str, list[float]]
"""标签 -> 嵌入"""
```