# `.\pytorch\torch\_inductor\hooks.py`

```
# mypy: allow-untyped-defs
# 引入上下文管理工具模块
import contextlib
# 引入类型提示模块
from typing import Callable, List, TYPE_CHECKING

# 如果类型检查开启，则导入 torch 模块
if TYPE_CHECKING:
    import torch

# 全局变量，用于存储中间钩子函数的列表
INTERMEDIATE_HOOKS: List[Callable[[str, "torch.Tensor"], None]] = []

# 定义上下文管理器函数，用于注册中间钩子函数
@contextlib.contextmanager
def intermediate_hook(fn):
    # 将新的钩子函数添加到全局钩子函数列表中
    INTERMEDIATE_HOOKS.append(fn)
    try:
        # 执行 yield 语句之前的代码块
        yield
    finally:
        # 在结束时，移除最后一个添加的钩子函数
        INTERMEDIATE_HOOKS.pop()

# 执行中间钩子函数的函数，接收名称和值作为参数
def run_intermediate_hooks(name, val):
    # 声明全局变量 INTERMEDIATE_HOOKS
    global INTERMEDIATE_HOOKS
    # 将当前的钩子函数列表存储到局部变量 hooks 中
    hooks = INTERMEDIATE_HOOKS
    # 清空全局钩子函数列表
    INTERMEDIATE_HOOKS = []
    try:
        # 遍历每个钩子函数并执行，传入名称和值作为参数
        for hook in hooks:
            hook(name, val)
    finally:
        # 恢复全局钩子函数列表为之前的状态
        INTERMEDIATE_HOOKS = hooks
```