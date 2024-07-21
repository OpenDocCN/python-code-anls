# `.\pytorch\torch\_C\_lazy_ts_backend.pyi`

```py
# 导入需要的模块和类
from typing import Any  # 从 typing 模块导入 Any 类型

from torch import Tensor  # 从 torch 模块导入 Tensor 类型

# 初始化函数的声明，暂未实现具体功能
def _init(): ...

# 获取张量列表中张量的设备数据和节点信息的函数声明
# 参数:
# - tensors: 张量列表，类型为 list[Tensor]
# 返回值:
# - 包含设备数据和节点信息的元组，类型为 tuple[list[int], list[Any]]
def _get_tensors_ts_device_data_node(
    tensors: list[Tensor],
) -> tuple[list[int], list[Any]]: ...

# 运行缓存图的函数声明，根据哈希字符串和图输入获取张量列表
# 参数:
# - hash_str: 哈希字符串，类型为 str
# - graph_inputs: 图输入列表，类型为 list[Any]
# 返回值:
# - 包含输出张量列表，类型为 list[Tensor]
def _run_cached_graph(hash_str: str, graph_inputs: list[Any]) -> list[Tensor]: ...
```