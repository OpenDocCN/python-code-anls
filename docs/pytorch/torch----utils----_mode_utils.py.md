# `.\pytorch\torch\utils\_mode_utils.py`

```py
# 引入 torch 库
import torch
# 引入 TypeVar 用于类型变量
from typing import TypeVar

# 定义一个类型变量 T，可以代表任意类型
T = TypeVar('T')

# 定义函数 all_same_mode，用于判断给定的 modes 序列中所有元素是否都相同
def all_same_mode(modes):
    # 使用 all 函数检查 modes 中所有元素是否与 modes[0] 相等
    return all(tuple(mode == modes[0] for mode in modes))

# 禁用 Torch 的调度功能，将 no_dispatch 设置为 torch._C._DisableTorchDispatch
no_dispatch = torch._C._DisableTorchDispatch
```