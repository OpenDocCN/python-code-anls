# `D:\src\scipysrc\sympy\sympy\multipledispatch\utils.py`

```
# 导入 OrderedDict 类，用于保持添加顺序的字典
from collections import OrderedDict


def expand_tuples(L):
    """
    >>> from sympy.multipledispatch.utils import expand_tuples
    >>> expand_tuples([1, (2, 3)])
    [(1, 2), (1, 3)]

    >>> expand_tuples([1, 2])
    [(1, 2)]
    """
    # 如果列表 L 是空的，则返回包含一个空元组的列表
    if not L:
        return [()]
    # 如果列表 L 的第一个元素不是元组，则递归地将其余部分扩展成元组，并将第一个元素与每个扩展后的元组组合起来
    elif not isinstance(L[0], tuple):
        rest = expand_tuples(L[1:])
        return [(L[0],) + t for t in rest]
    # 如果列表 L 的第一个元素是元组，则对其进行扩展，并将每个扩展后的元组与元组的第一个元素逐个组合起来
    else:
        rest = expand_tuples(L[1:])
        return [(item,) + t for t in rest for item in L[0]]


# 从 theano/theano/gof/sched.py 中获取的函数，避免许可问题，该函数由 Matthew Rocklin 编写
def _toposort(edges):
    """ Topological sort algorithm by Kahn [1] - O(nodes + vertices)

    inputs:
        edges - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of edges

    >>> from sympy.multipledispatch.utils import _toposort
    >>> _toposort({1: (2, 3), 2: (3, )})
    [1, 2, 3]

    Closely follows the wikipedia page [2]

    [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    Communications of the ACM
    [2] https://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    # 将 edges 的依赖关系反转为 incoming_edges 字典
    incoming_edges = reverse_dict(edges)
    # 将 incoming_edges 的值转换为集合
    incoming_edges = {k: set(val) for k, val in incoming_edges.items()}
    # 使用 OrderedDict 来维护节点的顺序
    S = OrderedDict.fromkeys(v for v in edges if v not in incoming_edges)
    # 用于存储拓扑排序结果的列表
    L = []

    while S:
        # 从 S 中弹出一个节点 n，并添加到拓扑排序结果列表 L 中
        n, _ = S.popitem()
        L.append(n)
        # 遍历 n 的后继节点 m
        for m in edges.get(n, ()):
            # 确保 m 的 incoming_edges 中包含 n，并将 n 从 m 的 incoming_edges 中移除
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            # 如果 m 的 incoming_edges 为空，则将 m 加入到 S 中
            if not incoming_edges[m]:
                S[m] = None
    # 如果 edges 中还有节点存在 incoming_edges，则表示存在环路
    if any(incoming_edges.get(v, None) for v in edges):
        raise ValueError("Input has cycles")
    # 返回拓扑排序的结果列表 L
    return L


def reverse_dict(d):
    """Reverses direction of dependence dict

    >>> d = {'a': (1, 2), 'b': (2, 3), 'c':()}
    >>> reverse_dict(d)  # doctest: +SKIP
    {1: ('a',), 2: ('a', 'b'), 3: ('b',)}

    :note: dict order are not deterministic. As we iterate on the
        input dict, it make the output of this function depend on the
        dict order. So this function output order should be considered
        as undeterministic.

    """
    # 用于存储反转后依赖关系的结果字典
    result = {}
    # 遍历输入字典 d 的键值对
    for key in d:
        # 遍历 d[key] 中的每个值 val，并将 key 添加到 result[val] 的元组中
        for val in d[key]:
            result[val] = result.get(val, ()) + (key, )
    # 返回反转后的依赖关系字典 result
    return result


# 从 toolz 中获取的函数，避免许可问题，此版本由 Matthew Rocklin 编写
def groupby(func, seq):
    """ Group a collection by a key function

    >>> from sympy.multipledispatch.utils import groupby
    >>> names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    >>> groupby(len, names)  # doctest: +SKIP
    {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}

    >>> iseven = lambda x: x % 2 == 0
    >>> groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])  # doctest: +SKIP
    {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}

    See Also:
        ``countby``
    """
    # 创建一个空字典用于存储按照 func 分组后的结果
    d = {}
    # 遍历序列中的每个元素
    for item in seq:
        # 使用给定的函数对当前元素进行处理，得到一个键
        key = func(item)
        # 如果键不在字典 d 中，则将该键初始化为空列表
        if key not in d:
            d[key] = []
        # 将当前元素添加到对应键的列表中
        d[key].append(item)
    # 返回整理好的字典 d，其中每个键对应一个列表，包含所有映射到该键的元素
    return d
```