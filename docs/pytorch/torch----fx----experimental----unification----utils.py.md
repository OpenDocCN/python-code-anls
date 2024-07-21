# `.\pytorch\torch\fx\experimental\unification\utils.py`

```py
# mypy: allow-untyped-defs
__all__ = ["hashable", "transitive_get", "raises", "reverse_dict", "xfail", "freeze"]
# 定义了模块的公共接口，这些函数可以被模块外部导入和使用

def hashable(x):
    try:
        hash(x)
        return True
    except TypeError:
        return False
# 判断对象 x 是否可哈希，返回 True 或 False

def transitive_get(key, d):
    """ Transitive dict.get
    >>> d = {1: 2, 2: 3, 3: 4}
    >>> d.get(1)
    2
    >>> transitive_get(1, d)
    4
    """
    while hashable(key) and key in d:
        key = d[key]
    return key
# 从字典 d 中获取 key 的值，若 key 可哈希且存在于 d 中则继续向下获取直到找到最终的值

def raises(err, lamda):
    try:
        lamda()
        return False
    except err:
        return True
# 执行 lambda 函数 lamda，捕获是否抛出指定的异常 err，返回 True 或 False

# Taken from theano/theano/gof/sched.py
# Avoids licensing issues because this was written by Matthew Rocklin
def _toposort(edges):
    """ Topological sort algorithm by Kahn [1] - O(nodes + vertices)
    inputs:
        edges - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of edges
    >>> # xdoctest: +SKIP
    >>> _toposort({1: (2, 3), 2: (3, )})
    [1, 2, 3]
    Closely follows the wikipedia page [2]
    [1] Kahn, Arthur B. (1962), "Topological sorting of large networks",
    Communications of the ACM
    [2] http://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_edges = reverse_dict(edges)
    incoming_edges = {k: set(val) for k, val in incoming_edges.items()}
    S = ({v for v in edges if v not in incoming_edges})
    L = []

    while S:
        n = S.pop()
        L.append(n)
        for m in edges.get(n, ()):
            assert n in incoming_edges[m]
            incoming_edges[m].remove(n)
            if not incoming_edges[m]:
                S.add(m)
    if any(incoming_edges.get(v, None) for v in edges):
        raise ValueError("Input has cycles")
    return L
# 对有向无环图 (DAG) 进行拓扑排序，以解析节点之间的依赖关系，返回排序后的节点列表

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
    result = {}  # type: ignore[var-annotated]
    for key in d:
        for val in d[key]:
            result[val] = result.get(val, tuple()) + (key, )
    return result
# 反转依赖字典 d 的方向，返回一个新的字典，其中键为原字典中的值，值为原字典中的键的集合

def xfail(func):
    try:
        func()
        raise Exception("XFailed test passed")  # pragma:nocover  # noqa: TRY002
    except Exception:
        pass
# 执行 func 函数，如果 func 执行不通过则捕获异常，不抛出异常

def freeze(d):
    """ Freeze container to hashable form
    >>> freeze(1)
    1
    >>> freeze([1, 2])
    (1, 2)
    >>> freeze({1: 2}) # doctest: +SKIP
    frozenset([(1, 2)])
    """
    if isinstance(d, dict):
        return frozenset(map(freeze, d.items()))
    if isinstance(d, set):
        return frozenset(map(freeze, d))
    if isinstance(d, (tuple, list)):
        return tuple(map(freeze, d))
    return d
# 将容器 d 冻结成可哈希的形式，支持字典、集合、元组和列表的转换
```