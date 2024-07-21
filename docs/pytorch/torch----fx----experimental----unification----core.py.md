# `.\pytorch\torch\fx\experimental\unification\core.py`

```
# mypy: allow-untyped-defs
# 导入 Iterator 类型，忽略类型检查
from collections.abc import Iterator  # type: ignore[import]
# 导入 partial 函数
from functools import partial

# 导入自定义模块中的函数和类，忽略类型检查
from .unification_tools import assoc  # type: ignore[import]
from .utils import transitive_get as walk
from .variable import isvar
from .dispatch import dispatch

# 定义公开接口列表
__all__ = ["reify", "unify"]

###############
# Reification #
###############

# 使用 dispatch 装饰器定义 _reify 函数，处理 Iterator 类型和字典类型的输入
@dispatch(Iterator, dict)
def _reify(t, s):
    return map(partial(reify, s=s), t)
    # 等效的生成器表达式：return (reify(arg, s) for arg in t)
_reify

# 使用 dispatch 装饰器定义 _reify 函数，处理 tuple 类型和字典类型的输入
@dispatch(tuple, dict)  # type: ignore[no-redef]
def _reify(t, s):
    return tuple(reify(iter(t), s))
_reify

# 使用 dispatch 装饰器定义 _reify 函数，处理 list 类型和字典类型的输入
@dispatch(list, dict)  # type: ignore[no-redef]
def _reify(t, s):
    return list(reify(iter(t), s))
_reify

# 使用 dispatch 装饰器定义 _reify 函数，处理 dict 类型和字典类型的输入
@dispatch(dict, dict)  # type: ignore[no-redef]
def _reify(d, s):
    return {k: reify(v, s) for k, v in d.items()}
_reify

# 使用 dispatch 装饰器定义 _reify 函数，处理 object 类型和字典类型的输入
@dispatch(object, dict)  # type: ignore[no-redef]
def _reify(o, s):
    return o  # 捕获所有情况，直接返回对象

# 定义 reify 函数，将表达式中的变量替换为给定的替换值
def reify(e, s):
    """ Replace variables of expression with substitution
    >>> # xdoctest: +SKIP
    >>> x, y = var(), var()
    >>> e = (1, x, (3, y))
    >>> s = {x: 2, y: 4}
    >>> reify(e, s)
    (1, 2, (3, 4))
    >>> e = {1: x, 3: (y, 5)}
    >>> reify(e, s)
    {1: 2, 3: (4, 5)}
    """
    # 如果表达式是变量，尝试使用替换字典 s 进行替换
    if isvar(e):
        return reify(s[e], s) if e in s else e
    # 否则，调用 _reify 函数处理表达式 e
    return _reify(e, s)

###############
# Unification #
###############

# 定义序列类型 seq
seq = tuple, list, Iterator

# 使用 dispatch 装饰器定义 _unify 函数，处理两个序列类型及替换字典的输入
@dispatch(seq, seq, dict)
def _unify(u, v, s):
    # 如果两个序列长度不同，返回 False
    if len(u) != len(v):
        return False
    # 遍历两个序列的对应元素，递归调用 unify 函数进行统一替换
    for uu, vv in zip(u, v):  # 避免递归
        s = unify(uu, vv, s)
        if s is False:
            return False
    return s

# 定义 unify 函数，找到使得 u == v 并满足替换字典 s 的替换
def unify(u, v, s):  # 目前不进行检查
    """ Find substitution so that u == v while satisfying s
    >>> x = var('x')
    >>> unify((1, x), (1, 2), {})
    {~x: 2}
    """
    # 使用替换字典 s 替换 u 和 v
    u = walk(u, s)
    v = walk(v, s)
    # 如果 u 等于 v，则直接返回替换字典 s
    if u == v:
        return s
    # 如果 u 是变量，则将其替换为 v
    if isvar(u):
        return assoc(s, u, v)
    # 如果 v 是变量，则将其替换为 u
    if isvar(v):
        return assoc(s, v, u)
    # 否则，调用 _unify 函数进行进一步统一替换
    return _unify(u, v, s)

# 使用 dispatch 装饰器定义 unify 函数，处理两个对象的输入
@dispatch(object, object)  # type: ignore[no-redef]
def unify(u, v):
    return unify(u, v, {})  # 使用空的替换字典进行替换
```