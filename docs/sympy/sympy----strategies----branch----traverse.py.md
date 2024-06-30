# `D:\src\scipysrc\sympy\sympy\strategies\branch\traverse.py`

```
""" Branching Strategies to Traverse a Tree """
# 导入 itertools 中的 product 函数，用于生成迭代器的笛卡尔积
from itertools import product
# 从 sympy.strategies.util 中导入 basic_fns 函数
from sympy.strategies.util import basic_fns
# 从当前模块的 core 中导入 chain, identity, do_one 函数
from .core import chain, identity, do_one


def top_down(brule, fns=basic_fns):
    """ Apply a rule down a tree running it on the top nodes first """
    # 应用一个规则在树中向下执行，首先在顶部节点上运行
    return chain(do_one(brule, identity),
                 lambda expr: sall(top_down(brule, fns), fns)(expr))


def sall(brule, fns=basic_fns):
    """ Strategic all - apply rule to args """
    # 获取 fns 中 op, new, children, leaf 函数
    op, new, children, leaf = map(fns.get, ('op', 'new', 'children', 'leaf'))

    def all_rl(expr):
        # 如果是叶子节点，直接产生该节点
        if leaf(expr):
            yield expr
        else:
            # 否则，获取当前节点的操作符
            myop = op(expr)
            # 对节点的子节点应用 brule 规则，生成所有可能的参数组合
            argss = product(*map(brule, children(expr)))
            # 遍历所有参数组合，应用 new 函数生成新节点
            for args in argss:
                yield new(myop, *args)
    return all_rl
```