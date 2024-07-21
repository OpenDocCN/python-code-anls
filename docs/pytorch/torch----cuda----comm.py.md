# `.\pytorch\torch\cuda\comm.py`

```
# 从 torch.nn.parallel.comm 中导入以下函数，它们已被移至该模块中
from torch.nn.parallel.comm import (
    broadcast,                   # 导入 broadcast 函数
    broadcast_coalesced,         # 导入 broadcast_coalesced 函数
    gather,                      # 导入 gather 函数
    reduce_add,                  # 导入 reduce_add 函数
    reduce_add_coalesced,        # 导入 reduce_add_coalesced 函数
    scatter,                     # 导入 scatter 函数
)

# __all__ 列表定义了在导入该模块时的公开接口，包含了上述导入的函数名
__all__ = [
    "broadcast",                 # 将 broadcast 函数添加到公开接口
    "broadcast_coalesced",       # 将 broadcast_coalesced 函数添加到公开接口
    "reduce_add",                # 将 reduce_add 函数添加到公开接口
    "reduce_add_coalesced",      # 将 reduce_add_coalesced 函数添加到公开接口
    "scatter",                   # 将 scatter 函数添加到公开接口
    "gather",                    # 将 gather 函数添加到公开接口
]
```