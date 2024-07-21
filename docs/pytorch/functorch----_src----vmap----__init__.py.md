# `.\pytorch\functorch\_src\vmap\__init__.py`

```
# 从 torch._functorch.vmap 模块中导入以下函数和类：
# _add_batch_dim: 添加批次维度的函数
# _broadcast_to_and_flatten: 广播和展平的函数
# _create_batched_inputs: 创建批次输入的函数
# _get_name: 获取名称的函数
# _process_batched_inputs: 处理批次输入的函数
# _remove_batch_dim: 移除批次维度的函数
# _unwrap_batched: 解包批次数据的函数
# _validate_and_get_batch_size: 验证并获取批次大小的函数
# Tensor: PyTorch 中的张量类
# tree_flatten: 将树形结构展平的函数
# tree_unflatten: 将展平的结构重新转换为树形结构的函数
from torch._functorch.vmap import (
    _add_batch_dim,
    _broadcast_to_and_flatten,
    _create_batched_inputs,
    _get_name,
    _process_batched_inputs,
    _remove_batch_dim,
    _unwrap_batched,
    _validate_and_get_batch_size,
    Tensor,
    tree_flatten,
    tree_unflatten,
)
```