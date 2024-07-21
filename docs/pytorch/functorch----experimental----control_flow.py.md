# `.\pytorch\functorch\experimental\control_flow.py`

```
# 导入 torch 模块中的 cond 函数，防止 Flake8 检查 F401（未使用导入）警告
from torch import cond  # noqa: F401

# 导入 torch._higher_order_ops.cond 模块中的 UnsupportedAliasMutationException 异常类，防止 Flake8 检查 F401 警告
from torch._higher_order_ops.cond import UnsupportedAliasMutationException  # noqa: F401

# 导入 torch._higher_order_ops.map 模块中的以下函数，防止 Flake8 检查 F401 警告
from torch._higher_order_ops.map import (
    _stack_pytree,   # 导入 _stack_pytree 函数
    _unstack_pytree, # 导入 _unstack_pytree 函数
    map,             # 导入 map 函数
)
```