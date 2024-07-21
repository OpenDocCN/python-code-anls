# `.\pytorch\torch\fx\experimental\unification\multipledispatch\utils.py`

```
# mypy: allow-untyped-defs
# 引入OrderedDict，用于确保字典的有序性
from collections import OrderedDict

# 定义模块导出的公共接口
__all__ = ["raises", "expand_tuples", "reverse_dict", "groupby", "typename"]

# 函数用于检查给定的 lambda 函数是否会引发特定的异常
def raises(err, lamda):
    try:
        lamda()
        return False
    except err:
        return True


# 函数用于将包含元组的列表扩展成所有可能的元组组合
def expand_tuples(L):
    """
    >>> expand_tuples([1, (2, 3)])
    [(1, 2), (1, 3)]
    >>> expand_tuples([1, 2])
    [(1, 2)]
    """
    if not L:
        return [()]
    elif not isinstance(L[0], tuple):
        rest = expand_tuples(L[1:])
        return [(L[0],) + t for t in rest]
    else:
        rest = expand_tuples(L[1:])
        return [(item,) + t for t in rest for item in L[0]]


# 使用 Kahn 算法进行拓扑排序，避免许可问题，此代码由 Matthew Rocklin 编写
def _toposort(edges):
    """ Topological sort algorithm by Kahn [1] - O(nodes + vertices)
    inputs:
        edges - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of edges
    >>> _toposort({1: (2, 3), 2: (3, )})
    [1, 2, 3]
    >>> # Closely follows the wikipedia page [2]
    >>> # [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    >>> # Communications of the ACM
    >>> # [2] http://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    # 反转依赖关系字典并转换为有序字典
    incoming_edges = reverse_dict(edges)
    incoming_edges = OrderedDict((k, set(val))
                                 for k, val in incoming_edges.items())
    # 初始化一个有序字典 S，包含不依赖其他节点的节点
    S = OrderedDict.fromkeys(v for v in edges if v not in incoming_edges)
    L = []

    # 执行 Kahn 算法的主循环
    while S:
        n, _ = S.popitem()
        L.append(n)
        for m in edges.get(n, ()):
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                S[m] = None
    # 检查是否存在循环依赖
    if any(incoming_edges.get(v, None) for v in edges):
        raise ValueError("Input has cycles")
    return L


# 函数用于反转依赖关系字典
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
    result = OrderedDict()  # type: ignore[var-annotated]
    for key in d:
        for val in d[key]:
            result[val] = result.get(val, tuple()) + (key, )
    return result


# 从 toolz 中获取的函数，用于按照指定函数对序列进行分组
def groupby(func, seq):
    """ Group a collection by a key function
    >>> names = ['Alice', 'Bob', 'Charlie', 'Dan', 'Edith', 'Frank']
    >>> groupby(len, names)  # doctest: +SKIP
    {3: ['Bob', 'Dan'], 5: ['Alice', 'Edith', 'Frank'], 7: ['Charlie']}
    >>> iseven = lambda x: x % 2 == 0
    >>> groupby(iseven, [1, 2, 3, 4, 5, 6, 7, 8])  # doctest: +SKIP
    """
    {False: [1, 3, 5, 7], True: [2, 4, 6, 8]}
    See Also:
        ``countby``
    """

这部分是一个多行字符串（docstring），描述了一个字典，其中键为布尔值 False 和 True，对应的值是列表。


    d = OrderedDict()  # type: ignore[var-annotated]

创建一个有序字典 `d`，用于存储按照特定条件分组后的元素，类型注释中忽略对变量类型的检查。


    for item in seq:

遍历输入的序列 `seq` 中的每个元素。


        key = func(item)

调用函数 `func` 对当前元素 `item` 进行处理，生成一个键 `key`。


        if key not in d:
            d[key] = list()

如果 `key` 不在字典 `d` 的键中，就将其初始化为一个空列表。


        d[key].append(item)

将当前元素 `item` 添加到对应 `key` 的列表中。


    return d

返回最终分组后的有序字典 `d`，其中每个键对应一个列表，包含符合该键条件的所有元素。
# 定义函数 `typename`，用于获取类型 `type` 的名称或包含在 `type` 中类型的名称元组
def typename(type):
    """Get the name of `type`.
    Parameters
    ----------
    type : Union[Type, Tuple[Type]]
        要获取名称的类型，可以是单个类型或类型的元组
    Returns
    -------
    str
        返回 `type` 的名称或包含在 `type` 中类型名称的元组
    Examples
    --------
    >>> typename(int)
    'int'
    >>> typename((int, float))
    '(int, float)'
    """
    # 尝试获取类型 `type` 的名称
    try:
        return type.__name__
    # 如果类型没有 `__name__` 属性，则处理 AttributeError
    except AttributeError:
        # 如果 `type` 是单个类型，则递归调用 `typename` 函数返回其名称
        if len(type) == 1:
            return typename(*type)
        # 否则，构建包含 `type` 中每个类型名称的元组并返回
        return f"({', '.join(map(typename, type))})"
```