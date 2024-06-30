# `D:\src\scipysrc\sympy\sympy\multipledispatch\conflict.py`

```
# 从自定义模块中导入 _toposort 和 groupby 函数
from .utils import _toposort, groupby

# 定义一个自定义的警告类 AmbiguityWarning，继承自内置的 Warning 类
class AmbiguityWarning(Warning):
    pass

# 定义函数 supercedes，用于判断 a 是否比 b 更具体和一致
def supercedes(a, b):
    """ A is consistent and strictly more specific than B """
    return len(a) == len(b) and all(map(issubclass, a, b))

# 定义函数 consistent，用于判断参数列表 a 和 b 是否有可能同时满足条件
def consistent(a, b):
    """ It is possible for an argument list to satisfy both A and B """
    return (len(a) == len(b) and
            all(issubclass(aa, bb) or issubclass(bb, aa)
                           for aa, bb in zip(a, b)))

# 定义函数 ambiguous，用于判断 a 和 b 是否一致但不具备严格的更具体关系
def ambiguous(a, b):
    """ A is consistent with B but neither is strictly more specific """
    return consistent(a, b) and not (supercedes(a, b) or supercedes(b, a))

# 定义函数 ambiguities，找出所有使得其中一个签名与另一个签名模糊不清的签名对
def ambiguities(signatures):
    """ All signature pairs such that A is ambiguous with B """
    # 将签名列表转换为元组列表
    signatures = list(map(tuple, signatures))
    # 使用集合推导式找出所有满足条件的签名对
    return {(a, b) for a in signatures for b in signatures
                       if hash(a) < hash(b)
                       and ambiguous(a, b)
                       and not any(supercedes(c, a) and supercedes(c, b)
                                    for c in signatures)}

# 定义函数 super_signature，找出可以解决模糊的签名
def super_signature(signatures):
    """ A signature that would break ambiguities """
    # 获取签名中参数列表的长度 n
    n = len(signatures[0])
    # 确保所有签名的参数列表长度相同
    assert all(len(s) == n for s in signatures)
    
    # 对于每个参数位置 i，选择具有最长 mro 的类型作为超级签名的对应参数类型
    return [max([type.mro(sig[i]) for sig in signatures], key=len)[0]
               for i in range(n)]

# 定义函数 edge，确定在检查参数 a 和 b 时应该先检查哪一个
def edge(a, b, tie_breaker=hash):
    """ A should be checked before B

    Tie broken by tie_breaker, defaults to ``hash``
    """
    # 如果 a 比 b 更具体，返回 True；如果 b 比 a 更具体，返回 False
    if supercedes(a, b):
        if supercedes(b, a):
            # 如果 a 和 b 互相比较没有明显的优劣，则根据 tie_breaker 函数进行决定
            return tie_breaker(a) > tie_breaker(b)
        else:
            return True
    return False

# 定义函数 ordering，返回一个合理的签名顺序，以便从头到尾进行检查
def ordering(signatures):
    """ A sane ordering of signatures to check, first to last

    Topoological sort of edges as given by ``edge`` and ``supercedes``
    """
    # 将签名列表转换为元组列表
    signatures = list(map(tuple, signatures))
    # 找出所有有向边，其中 a 应该在 b 之前检查
    edges = [(a, b) for a in signatures for b in signatures if edge(a, b)]
    # 按照第一个元素分组 edges
    edges = groupby(lambda x: x[0], edges)
    # 对于不在 edges 中的每个签名，将其 edges 值初始化为空列表
    for s in signatures:
        if s not in edges:
            edges[s] = []
    # 将 edges 转换为字典，其中键是签名，值是一个列表，表示需要在该签名之前检查的签名列表
    edges = {k: [b for a, b in v] for k, v in edges.items()}
    # 使用 _toposort 函数对 edges 进行拓扑排序，以确定签名的合理顺序
    return _toposort(edges)
```