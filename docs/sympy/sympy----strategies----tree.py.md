# `D:\src\scipysrc\sympy\sympy\strategies\tree.py`

```
from functools import partial  # 导入 functools 模块中的 partial 函数，用于部分应用函数
from sympy.strategies import chain, minimize  # 导入 sympy.strategies 模块中的 chain 和 minimize 函数
from sympy.strategies.core import identity  # 导入 sympy.strategies.core 模块中的 identity 函数
import sympy.strategies.branch as branch  # 导入 sympy.strategies.branch 模块，并使用别名 branch
from sympy.strategies.branch import yieldify  # 从 sympy.strategies.branch 模块中导入 yieldify 函数


def treeapply(tree, join, leaf=identity):
    """ Apply functions onto recursive containers (tree).

    Explanation
    ===========

    join - a dictionary mapping container types to functions
      e.g. ``{list: minimize, tuple: chain}``

    Keys are containers/iterables.  Values are functions [a] -> a.

    Examples
    ========

    >>> from sympy.strategies.tree import treeapply
    >>> tree = [(3, 2), (4, 1)]
    >>> treeapply(tree, {list: max, tuple: min})
    2

    >>> add = lambda *args: sum(args)
    >>> def mul(*args):
    ...     total = 1
    ...     for arg in args:
    ...         total *= arg
    ...     return total
    >>> treeapply(tree, {list: mul, tuple: add})
    25
    """
    for typ in join:
        if isinstance(tree, typ):  # 如果 tree 是 typ 类型的实例
            return join[typ](*map(partial(treeapply, join=join, leaf=leaf),
                                  tree))
    return leaf(tree)  # 否则应用 leaf 函数到 tree 上


def greedy(tree, objective=identity, **kwargs):
    """ Execute a strategic tree.  Select alternatives greedily

    Trees
    -----

    Nodes in a tree can be either

    function - a leaf
    list     - a selection among operations
    tuple    - a sequence of chained operations

    Textual examples
    ----------------

    Text: Run f, then run g, e.g. ``lambda x: g(f(x))``
    Code: ``(f, g)``

    Text: Run either f or g, whichever minimizes the objective
    Code: ``[f, g]``

    Textx: Run either f or g, whichever is better, then run h
    Code: ``([f, g], h)``

    Text: Either expand then simplify or try factor then foosimp. Finally print
    Code: ``([(expand, simplify), (factor, foosimp)], print)``

    Objective
    ---------

    "Better" is determined by the objective keyword.  This function makes
    choices to minimize the objective.  It defaults to the identity.

    Examples
    ========

    >>> from sympy.strategies.tree import greedy
    >>> inc    = lambda x: x + 1
    >>> dec    = lambda x: x - 1
    >>> double = lambda x: 2*x

    >>> tree = [inc, (dec, double)] # either inc or dec-then-double
    >>> fn = greedy(tree)
    >>> fn(4)  # lowest value comes from the inc
    5
    >>> fn(1)  # lowest value comes from dec then double
    0

    This function selects between options in a tuple.  The result is chosen
    that minimizes the objective function.

    >>> fn = greedy(tree, objective=lambda x: -x)  # maximize
    >>> fn(4)  # highest value comes from the dec then double
    6
    >>> fn(1)  # highest value comes from the inc
    2

    Greediness
    ----------

    This is a greedy algorithm.  In the example:

        ([a, b], c)  # do either a or b, then do c

    the choice between running ``a`` or ``b`` is made without foresight to c
    """
    optimize = partial(minimize, objective=objective)  # 使用 minimize 函数和指定的 objective 创建一个部分应用函数
    # 对树形数据结构应用优化和链式操作的函数
    return treeapply(tree, {list: optimize, tuple: chain}, **kwargs)
# 定义一个函数 `allresults`，用于执行给定的树形结构，返回所有可能的结果的惰性迭代器。

def allresults(tree, leaf=yieldify):
    """ Execute a strategic tree.  Return all possibilities.

    Returns a lazy iterator of all possible results

    Exhaustiveness
    --------------

    This is an exhaustive algorithm.  In the example

        ([a, b], [c, d])

    All of the results from

        (a, c), (b, c), (a, d), (b, d)

    are returned.  This can lead to combinatorial blowup.

    See sympy.strategies.greedy for details on input
    """
    # 调用 `treeapply` 函数对给定的树形结构应用策略，并使用指定的叶子函数 `leaf`
    return treeapply(tree, {list: branch.multiplex, tuple: branch.chain},
                     leaf=leaf)


def brute(tree, objective=identity, **kwargs):
    # 定义一个函数 `brute`，用于在给定树形结构中寻找最小值，通过指定的 `objective` 函数进行评估。
    return lambda expr: min(tuple(allresults(tree, **kwargs)(expr)),
                            key=objective)
```