# `D:\src\scipysrc\pandas\pandas\core\indexes\extension.py`

```
"""
Shared methods for Index subclasses backed by ExtensionArray.
"""

# 引入必要的模块和类型定义
from __future__ import annotations
from inspect import signature
from typing import (
    TYPE_CHECKING,
    TypeVar,
)

# 从 pandas.util._decorators 中导入 cache_readonly 装饰器
from pandas.util._decorators import cache_readonly

# 从 pandas.core.dtypes.generic 中导入 ABCDataFrame 类
from pandas.core.dtypes.generic import ABCDataFrame

# 从 pandas.core.indexes.base 中导入 Index 类
from pandas.core.indexes.base import Index

# 如果是类型检查阶段，导入必要的类型
if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    # 从 pandas._typing 中导入 ArrayLike 和 npt 类型
    from pandas._typing import (
        ArrayLike,
        npt,
    )

    # 从 pandas.core.arrays 中导入 IntervalArray 类
    from pandas.core.arrays import IntervalArray
    # 从 pandas.core.arrays._mixins 中导入 NDArrayBackedExtensionArray 类

# 定义类型变量 _ExtensionIndexT
_ExtensionIndexT = TypeVar("_ExtensionIndexT", bound="ExtensionIndex")


# 定义一个私有函数 _inherit_from_data
def _inherit_from_data(
    name: str, delegate: type, cache: bool = False, wrap: bool = False
):
    """
    Make an alias for a method of the underlying ExtensionArray.

    Parameters
    ----------
    name : str
        Name of an attribute the class should inherit from its EA parent.
    delegate : class
    cache : bool, default False
        Whether to convert wrapped properties into cache_readonly
    wrap : bool, default False
        Whether to wrap the inherited result in an Index.

    Returns
    -------
    attribute, method, property, or cache_readonly
    """

    # 获取 delegate 类中名为 name 的属性
    attr = getattr(delegate, name)

    # 如果属性是 property 或者是 getset_descriptor 类型（例如在 Cython 类中定义的属性）
    if isinstance(attr, property) or type(attr).__name__ == "getset_descriptor":
        # 如果 cache 参数为 True
        if cache:

            # 定义一个缓存的方法 cached
            def cached(self):
                return getattr(self._data, name)

            cached.__name__ = name
            cached.__doc__ = attr.__doc__
            # 使用 cache_readonly 装饰器包装 cached 方法
            method = cache_readonly(cached)

        else:
            # 定义一个获取属性值的方法 fget
            def fget(self):
                result = getattr(self._data, name)
                # 如果 wrap 参数为 True，则根据结果类型选择不同的封装方式
                if wrap:
                    if isinstance(result, type(self._data)):
                        return type(self)._simple_new(result, name=self.name)
                    elif isinstance(result, ABCDataFrame):
                        return result.set_index(self)
                    return Index(result, name=self.name)
                return result

            # 定义一个设置属性值的方法 fset
            def fset(self, value) -> None:
                setattr(self._data, name, value)

            fget.__name__ = name
            fget.__doc__ = attr.__doc__

            # 创建一个 property 属性，并传入 fget 和 fset 方法
            method = property(fget, fset)

    elif not callable(attr):
        # 如果属性不可调用，直接赋值给 method
        method = attr
    else:
        # 如果不是第一种情况，则定义一个方法函数
        def method(self, *args, **kwargs):  # type: ignore[misc]
            # 如果参数 kwargs 中包含 "inplace" 键，抛出 ValueError 异常
            if "inplace" in kwargs:
                raise ValueError(f"cannot use inplace with {type(self).__name__}")
            # 调用 attr 函数处理 self._data 对象，并返回结果
            result = attr(self._data, *args, **kwargs)
            # 如果 wrap 为真
            if wrap:
                # 如果 result 是 self._data 的实例，则返回类型为 type(self)._simple_new(result, name=self.name) 的新对象
                if isinstance(result, type(self._data)):
                    return type(self)._simple_new(result, name=self.name)
                # 如果 result 是 ABCDataFrame 的实例，则返回 result.set_index(self) 的结果
                elif isinstance(result, ABCDataFrame):
                    return result.set_index(self)
                # 否则返回以 result 为数据和 self.name 为名称的 Index 对象
                return Index(result, name=self.name)
            # 如果 wrap 不为真，直接返回 result
            return result

        # 设置方法函数的名称为 name
        method.__name__ = name  # type: ignore[attr-defined]
        # 设置方法函数的文档字符串为 attr 函数的文档字符串
        method.__doc__ = attr.__doc__
        # 设置方法函数的签名为 attr 函数的签名
        method.__signature__ = signature(attr)  # type: ignore[attr-defined]
    # 返回定义好的 method 方法函数
    return method
# 定义一个装饰器函数，用于将 ExtensionArray 的属性固定到 Index 的子类中
def inherit_names(
    names: list[str], delegate: type, cache: bool = False, wrap: bool = False
) -> Callable[[type[_ExtensionIndexT]], type[_ExtensionIndexT]]:
    """
    Class decorator to pin attributes from an ExtensionArray to a Index subclass.

    Parameters
    ----------
    names : List[str]
        属性名列表，需要从 ExtensionArray 中固定到 Index 子类中的属性名集合
    delegate : class
        ExtensionArray 类型，从中继承属性的源
    cache : bool, default False
        是否缓存结果
    wrap : bool, default False
        是否将继承的结果包装在 Index 中

    Returns
    -------
    Callable[[type[_ExtensionIndexT]], type[_ExtensionIndexT]]
        装饰器函数，接受一个 Index 子类并返回它
    """

    def wrapper(cls: type[_ExtensionIndexT]) -> type[_ExtensionIndexT]:
        # 遍历给定的属性名列表，从 delegate 中继承属性并将其设置到 cls 上
        for name in names:
            meth = _inherit_from_data(name, delegate, cache=cache, wrap=wrap)
            setattr(cls, name, meth)

        return cls

    return wrapper


class ExtensionIndex(Index):
    """
    Index subclass for indexes backed by ExtensionArray.
    """

    # 基类已经通过到 _data：
    #  size, __len__, dtype

    _data: IntervalArray | NDArrayBackedExtensionArray

    # ---------------------------------------------------------------------

    def _validate_fill_value(self, value):
        """
        Convert value to be insertable to underlying array.
        """
        # 将 value 转换为可以插入到底层数组的格式
        return self._data._validate_setitem_value(value)

    @cache_readonly
    def _isnan(self) -> npt.NDArray[np.bool_]:
        # error: Incompatible return value type (got "ExtensionArray", expected
        # "ndarray")
        # 返回是否为 NaN 的布尔数组
        return self._data.isna()  # type: ignore[return-value]


class NDArrayBackedExtensionIndex(ExtensionIndex):
    """
    Index subclass for indexes backed by NDArrayBackedExtensionArray.
    """

    _data: NDArrayBackedExtensionArray

    def _get_engine_target(self) -> np.ndarray:
        # 返回底层数据的 ndarray 表示
        return self._data._ndarray

    def _from_join_target(self, result: np.ndarray) -> ArrayLike:
        # 确保结果的 dtype 与底层数据的一致，然后将其转换为 ArrayLike 类型返回
        assert result.dtype == self._data._ndarray.dtype
        return self._data._from_backing_data(result)
```