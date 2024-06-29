# `.\numpy\numpy\typing\tests\test_runtime.py`

```py
"""Test the runtime usage of `numpy.typing`."""

# 导入必要的模块和函数
from __future__ import annotations

from typing import (
    get_type_hints,
    Union,
    NamedTuple,
    get_args,
    get_origin,
    Any,
)

import pytest  # 导入 pytest 模块
import numpy as np  # 导入 NumPy 库
import numpy.typing as npt  # 导入 NumPy 的类型提示模块
import numpy._typing as _npt  # 导入 NumPy 的私有类型提示模块


# 定义一个命名元组来存储类型相关信息
class TypeTup(NamedTuple):
    typ: type
    args: tuple[type, ...]
    origin: None | type


# 定义 NumPy 数组类型的命名元组示例
NDArrayTup = TypeTup(npt.NDArray, npt.NDArray.__args__, np.ndarray)

# 定义不同类型的字典
TYPES = {
    "ArrayLike": TypeTup(npt.ArrayLike, npt.ArrayLike.__args__, Union),
    "DTypeLike": TypeTup(npt.DTypeLike, npt.DTypeLike.__args__, Union),
    "NBitBase": TypeTup(npt.NBitBase, (), None),
    "NDArray": NDArrayTup,
}


# 参数化测试函数，测试 typing.get_args 函数
@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_args(name: type, tup: TypeTup) -> None:
    """Test `typing.get_args`."""
    typ, ref = tup.typ, tup.args
    out = get_args(typ)
    assert out == ref


# 参数化测试函数，测试 typing.get_origin 函数
@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_origin(name: type, tup: TypeTup) -> None:
    """Test `typing.get_origin`."""
    typ, ref = tup.typ, tup.origin
    out = get_origin(typ)
    assert out == ref


# 参数化测试函数，测试 typing.get_type_hints 函数
@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_type_hints(name: type, tup: TypeTup) -> None:
    """Test `typing.get_type_hints`."""
    typ = tup.typ

    # 显式设置 `__annotations__` 来规避 `from __future__ import annotations` 的字符串化
    def func(a): pass
    func.__annotations__ = {"a": typ, "return": None}

    out = get_type_hints(func)
    ref = {"a": typ, "return": type(None)}
    assert out == ref


# 参数化测试函数，测试 typing.get_type_hints 函数使用类型字符串表示
@pytest.mark.parametrize("name,tup", TYPES.items(), ids=TYPES.keys())
def test_get_type_hints_str(name: type, tup: TypeTup) -> None:
    """Test `typing.get_type_hints` with string-representation of types."""
    typ_str, typ = f"npt.{name}", tup.typ

    # 显式设置 `__annotations__` 来规避 `from __future__ import annotations` 的字符串化
    def func(a): pass
    func.__annotations__ = {"a": typ_str, "return": None}

    out = get_type_hints(func)
    ref = {"a": typ, "return": type(None)}
    assert out == ref


# 测试函数，验证 TYPES.keys() 和 numpy.typing.__all__ 是否同步
def test_keys() -> None:
    """Test that ``TYPES.keys()`` and ``numpy.typing.__all__`` are synced."""
    keys = TYPES.keys()
    ref = set(npt.__all__)
    assert keys == ref


# 定义一个字典，存储不同协议类型和对应的示例对象
PROTOCOLS: dict[str, tuple[type[Any], object]] = {
    "_SupportsDType": (_npt._SupportsDType, np.int64(1)),
    "_SupportsArray": (_npt._SupportsArray, np.arange(10)),
    "_SupportsArrayFunc": (_npt._SupportsArrayFunc, np.arange(10)),
    "_NestedSequence": (_npt._NestedSequence, [1]),
}


# 参数化测试类，测试 isinstance 的运行时协议
@pytest.mark.parametrize("cls,obj", PROTOCOLS.values(), ids=PROTOCOLS.keys())
class TestRuntimeProtocol:
    def test_isinstance(self, cls: type[Any], obj: object) -> None:
        """Test isinstance with runtime protocol."""
        assert isinstance(obj, cls)
        assert not isinstance(None, cls)
    # 定义一个测试方法，用于验证给定对象是否是指定类的子类或者其元类
    def test_issubclass(self, cls: type[Any], obj: object) -> None:
        # 如果给定的类是 _npt._SupportsDType 类型，则标记该测试为预期失败
        if cls is _npt._SupportsDType:
            pytest.xfail(
                "Protocols with non-method members don't support issubclass()"
            )
        # 断言给定对象的类型是指定类 cls 的子类或其元类
        assert issubclass(type(obj), cls)
        # 断言 None 的类型不是指定类 cls 的子类或其元类
        assert not issubclass(type(None), cls)
```