# `D:\src\scipysrc\sympy\sympy\strategies\traverse.py`

```
# 导入 Sympy 库中的工具函数和策略模块
"""Strategies to Traverse a Tree."""
from sympy.strategies.util import basic_fns
from sympy.strategies.core import chain, do_one

# 从树的顶部向下应用规则，首先处理顶部节点
def top_down(rule, fns=basic_fns):
    """Apply a rule down a tree running it on the top nodes first."""
    return chain(rule, lambda expr: sall(top_down(rule, fns), fns)(expr))

# 从树的底部向上应用规则，首先处理底部节点
def bottom_up(rule, fns=basic_fns):
    """Apply a rule down a tree running it on the bottom nodes first."""
    return chain(lambda expr: sall(bottom_up(rule, fns), fns)(expr), rule)

# 从树的顶部向下应用一次规则，成功时停止
def top_down_once(rule, fns=basic_fns):
    """Apply a rule down a tree - stop on success."""
    return do_one(rule, lambda expr: sall(top_down(rule, fns), fns)(expr))

# 从树的底部向上应用一次规则，成功时停止
def bottom_up_once(rule, fns=basic_fns):
    """Apply a rule up a tree - stop on success."""
    return do_one(lambda expr: sall(bottom_up(rule, fns), fns)(expr), rule)

# 定义一个函数，将规则应用于参数
def sall(rule, fns=basic_fns):
    """Strategic all - apply rule to args."""
    # 从 fns 中获取操作、创建新表达式、获取子节点和判断是否为叶子节点的函数
    op, new, children, leaf = map(fns.get, ('op', 'new', 'children', 'leaf'))

    # 定义递归函数 all_rl，应用规则到表达式的所有子节点
    def all_rl(expr):
        if leaf(expr):
            return expr  # 如果是叶子节点，直接返回
        else:
            # 对表达式的每个子节点应用规则
            args = map(rule, children(expr))
            # 创建一个新的表达式，将操作应用于表达式及其子节点的规则结果
            return new(op(expr), *args)

    return all_rl  # 返回定义好的递归函数 all_rl
```