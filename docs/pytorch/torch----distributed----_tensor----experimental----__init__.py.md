# `.\pytorch\torch\distributed\_tensor\experimental\__init__.py`

```
# 声明脚本的类型检查设置，允许未标记类型的函数
# Copyright (c) Meta Platforms, Inc. and affiliates
从 contextlib 模块中导入上下文管理器
from contextlib import contextmanager

从 torch.distributed._tensor.api 中导入 DTensor 类
from torch.distributed._tensor.api import DTensor

从 torch.distributed._tensor.experimental.local_map 中导入 local_map 函数
from torch.distributed._tensor.experimental.local_map import local_map

声明将被导出的符号列表
__all__ = ["local_map", "implicit_replication"]

定义一个上下文管理器函数 implicit_replication
@contextmanager
def implicit_replication():
    尝试执行以下操作：
    DTensor._op_dispatcher._allow_implicit_replication 设置为 True，允许隐式复制
    yield  # 执行上下文管理器的主体部分

    最终执行：
    DTensor._op_dispatcher._allow_implicit_replication 设置为 False，禁止隐式复制
```