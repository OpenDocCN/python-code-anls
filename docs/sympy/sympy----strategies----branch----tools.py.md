# `D:\src\scipysrc\sympy\sympy\strategies\branch\tools.py`

```
# 从核心模块中导入 exhaust 和 multiplex 函数
from .core import exhaust, multiplex
# 从遍历模块中导入 top_down 函数
from .traverse import top_down

# 定义名为 canon 的函数，用于规范化策略
def canon(*rules):
    """ Strategy for canonicalization

    Apply each branching rule in a top-down fashion through the tree.
    Multiplex through all branching rule traversals
    Keep doing this until there is no change.
    """
    # 将 rules 中的每个规则按顶向下的方式应用于树结构，并通过 multiplex 函数进行多路复用
    return exhaust(multiplex(*map(top_down, rules)))
```