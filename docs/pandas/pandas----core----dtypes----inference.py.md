# `D:\src\scipysrc\pandas\pandas\core\dtypes\inference.py`

```
"""basic inference routines"""

from __future__ import annotations  # 允许使用类型注释中的'annotations'作为类型

from collections import abc  # 导入abc模块中的集合类抽象基类
from numbers import Number  # 导入Number类型用于数字类型检查
import re  # 导入正则表达式模块
from re import Pattern  # 导入Pattern类型用于正则表达式类型检查
from typing import TYPE_CHECKING  # 导入TYPE_CHECKING用于条件导入类型提示

import numpy as np  # 导入NumPy库并使用np别名

from pandas._libs import lib  # 导入pandas底层库中的lib模块

if TYPE_CHECKING:
    from collections.abc import Hashable  # 条件导入Hashable类型
    from pandas._typing import TypeGuard  # 条件导入TypeGuard类型

is_bool = lib.is_bool  # 从lib模块导入is_bool函数
is_integer = lib.is_integer  # 从lib模块导入is_integer函数
is_float = lib.is_float  # 从lib模块导入is_float函数
is_complex = lib.is_complex  # 从lib模块导入is_complex函数
is_scalar = lib.is_scalar  # 从lib模块导入is_scalar函数
is_decimal = lib.is_decimal  # 从lib模块导入is_decimal函数
is_list_like = lib.is_list_like  # 从lib模块导入is_list_like函数
is_iterator = lib.is_iterator  # 从lib模块导入is_iterator函数


def is_number(obj: object) -> TypeGuard[Number | np.number]:
    """
    Check if the object is a number.

    Returns True when the object is a number, and False if is not.

    Parameters
    ----------
    obj : any type
        The object to check if is a number.

    Returns
    -------
    bool
        Whether `obj` is a number or not.

    See Also
    --------
    api.types.is_integer: Checks a subgroup of numbers.

    Examples
    --------
    >>> from pandas.api.types import is_number
    >>> is_number(1)
    True
    >>> is_number(7.15)
    True

    Booleans are valid because they are int subclass.

    >>> is_number(False)
    True

    >>> is_number("foo")
    False
    >>> is_number("5")
    False
    """
    return isinstance(obj, (Number, np.number))


def iterable_not_string(obj: object) -> bool:
    """
    Check if the object is an iterable but not a string.

    Parameters
    ----------
    obj : The object to check.

    Returns
    -------
    is_iter_not_string : bool
        Whether `obj` is a non-string iterable.

    Examples
    --------
    >>> iterable_not_string([1, 2, 3])
    True
    >>> iterable_not_string("foo")
    False
    >>> iterable_not_string(1)
    False
    """
    return isinstance(obj, abc.Iterable) and not isinstance(obj, str)


def is_file_like(obj: object) -> bool:
    """
    Check if the object is a file-like object.

    For objects to be considered file-like, they must
    be an iterator AND have either a `read` and/or `write`
    method as an attribute.

    Note: file-like objects must be iterable, but
    iterable objects need not be file-like.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    bool
        Whether `obj` has file-like properties.

    Examples
    --------
    >>> import io
    >>> from pandas.api.types import is_file_like
    >>> buffer = io.StringIO("data")
    >>> is_file_like(buffer)
    True
    >>> is_file_like([1, 2, 3])
    False
    """
    if not (hasattr(obj, "read") or hasattr(obj, "write")):
        return False

    return bool(hasattr(obj, "__iter__"))


def is_re(obj: object) -> TypeGuard[Pattern]:
    """
    Check if the object is a regex pattern instance.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    bool
        Whether `obj` is a regex pattern.

    Examples
    --------
    """
    return isinstance(obj, Pattern)
    # 导入 pandas 库中的 is_re 函数，用于检查对象是否为正则表达式对象
    >>> from pandas.api.types import is_re
    # 导入 re 库，用于处理正则表达式相关操作
    >>> import re
    # 调用 is_re 函数，传入一个编译后的正则表达式对象，返回 True 表示是正则表达式对象
    >>> is_re(re.compile(".*"))
    True
    # 调用 is_re 函数，传入一个字符串对象，返回 False 表示不是正则表达式对象
    >>> is_re("foo")
    False
    """
    # 判断给定对象是否是正则表达式对象，并返回结果
    return isinstance(obj, Pattern)
# 检查对象是否可以编译为正则表达式模式实例
def is_re_compilable(obj: object) -> bool:
    try:
        re.compile(obj)  # 尝试编译正则表达式对象
    except TypeError:
        return False  # 如果出现 TypeError 则返回 False
    else:
        return True  # 否则返回 True，表示对象可以成功编译为正则表达式模式


# 检查对象是否类似于数组
def is_array_like(obj: object) -> bool:
    return is_list_like(obj) and hasattr(obj, "dtype")
    # 返回对象是否类似于数组，即对象是否类似列表并且具有 "dtype" 属性


# 检查对象是否类似于嵌套列表
def is_nested_list_like(obj: object) -> bool:
    return (
        is_list_like(obj)
        and hasattr(obj, "__len__")
        and len(obj) > 0
        and all(is_list_like(item) for item in obj)
    )
    # 返回对象是否类似于嵌套列表，即对象是否类似列表并且具有 __len__ 方法，并且包含非空子列表


# 检查对象是否类似于字典
def is_dict_like(obj: object) -> bool:
    dict_like_attrs = ("__getitem__", "keys", "__contains__")
    # 定义字典特性元组，用于检查对象是否类似于字典
    # 返回一个布尔值，判断对象是否拥有所有字典类属性，且不是类本身
    return (
        all(hasattr(obj, attr) for attr in dict_like_attrs)
        # 在GitHub Issue 25196中提到，排除类本身
        and not isinstance(obj, type)
    )
# 检查对象是否为命名元组
def is_named_tuple(obj: object) -> bool:
    """
    Check if the object is a named tuple.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    bool
        Whether `obj` is a named tuple.

    Examples
    --------
    >>> from collections import namedtuple
    >>> from pandas.api.types import is_named_tuple
    >>> Point = namedtuple("Point", ["x", "y"])
    >>> p = Point(1, 2)
    >>>
    >>> is_named_tuple(p)
    True
    >>> is_named_tuple((1, 2))
    False
    """
    return isinstance(obj, abc.Sequence) and hasattr(obj, "_fields")


# 检查对象是否可哈希
def is_hashable(obj: object) -> TypeGuard[Hashable]:
    """
    Return True if hash(obj) will succeed, False otherwise.

    Some types will pass a test against collections.abc.Hashable but fail when
    they are actually hashed with hash().

    Distinguish between these and other types by trying the call to hash() and
    seeing if they raise TypeError.

    Returns
    -------
    bool

    Examples
    --------
    >>> import collections
    >>> from pandas.api.types import is_hashable
    >>> a = ([],)
    >>> isinstance(a, collections.abc.Hashable)
    True
    >>> is_hashable(a)
    False
    """
    # 由于某些类型（如 numpy 标量）无法通过 isinstance(obj, collections.abc.Hashable) 测试，所以我们尝试调用 hash() 来区分可哈希类型。
    try:
        hash(obj)
    except TypeError:
        return False
    else:
        return True


# 检查对象是否为序列
def is_sequence(obj: object) -> bool:
    """
    Check if the object is a sequence of objects.
    String types are not included as sequences here.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    is_sequence : bool
        Whether `obj` is a sequence of objects.

    Examples
    --------
    >>> l = [1, 2, 3]
    >>>
    >>> is_sequence(l)
    True
    >>> is_sequence(iter(l))
    False
    """
    try:
        # 可以迭代
        iter(obj)  # type: ignore[call-overload]
        # 有长度属性
        len(obj)  # type: ignore[arg-type]
        return not isinstance(obj, (str, bytes))
    except (TypeError, AttributeError):
        return False


# 检查对象是否为数据类实例
def is_dataclass(item: object) -> bool:
    """
    Checks if the object is a data-class instance

    Parameters
    ----------
    item : object

    Returns
    --------
    is_dataclass : bool
        True if the item is an instance of a data-class,
        will return false if you pass the data class itself

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Point:
    ...     x: int
    ...     y: int

    >>> is_dataclass(Point)
    False
    >>> is_dataclass(Point(0, 2))
    True

    """
    try:
        import dataclasses

        # 检查对象是否是数据类的实例，且不是数据类本身
        return dataclasses.is_dataclass(item) and not isinstance(item, type)
    # 如果出现 ImportError 异常，则返回 False
    except ImportError:
        return False
```