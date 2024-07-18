# `.\graphrag\graphrag\index\verbs\text\chunk\strategies\typing.py`

```py
# 版权声明和许可声明，指出版权和许可条款
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

# 导入必要的模块和类
# 导入 collections.abc 中的 Callable 和 Iterable 类型
from collections.abc import Callable, Iterable
# 导入 typing 模块中的 Any 类型
from typing import Any

# 从 datashaper 模块中导入 ProgressTicker 类
from datashaper import ProgressTicker

# 从 graphrag.index.verbs.text.chunk.typing 模块中导入 TextChunk 类
from graphrag.index.verbs.text.chunk.typing import TextChunk

# 定义 ChunkStrategy 类型别名，它是一个可调用对象（Callable），接受三个参数：文档文本列表，字典以及 ProgressTicker 对象，
# 返回一个可迭代的 TextChunk 对象列表
ChunkStrategy = Callable[
    [list[str], dict[str, Any], ProgressTicker], Iterable[TextChunk]
]
```