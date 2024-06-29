# `.\numpy\numpy\_core\tests\test_arraymethod.py`

```py
"""
This file tests the generic aspects of ArrayMethod.  At the time of writing
this is private API, but when added, public API may be added here.
"""

# 导入必要的模块和库
import sys
import types
from typing import Any

# 导入测试框架 pytest
import pytest

# 导入 NumPy 库，并获取私有函数 _get_castingimpl 别名为 get_castingimpl
import numpy as np
from numpy._core._multiarray_umath import _get_castingimpl as get_castingimpl


class TestResolveDescriptors:
    # 测试 resolve_descriptors 函数的错误路径，注意 `casting_unittests` 测试覆盖了非错误路径

    # 获取 casting 实现，主要/唯一的当前用户
    method = get_castingimpl(type(np.dtype("d")), type(np.dtype("f")))

    @pytest.mark.parametrize("args", [
        (True,),  # 不是元组
        ((None,)),  # 元素太少
        ((None, None, None),),  # 元素太多
        ((None, None),),  # 输入 dtype 是 None，无效
        ((np.dtype("d"), True),),  # 输出 dtype 不是 dtype 类型
        ((np.dtype("f"), None),),  # 输入 dtype 与方法不匹配
    ])
    def test_invalid_arguments(self, args):
        # 断言期望抛出 TypeError 异常
        with pytest.raises(TypeError):
            self.method._resolve_descriptors(*args)


class TestSimpleStridedCall:
    # 测试 resolve_descriptors 函数的错误路径，注意 `casting_unittests` 测试覆盖了非错误路径

    # 获取 casting 实现，主要/唯一的当前用户
    method = get_castingimpl(type(np.dtype("d")), type(np.dtype("f")))

    @pytest.mark.parametrize(["args", "error"], [
        ((True,), TypeError),  # 不是元组
        (((None,),), TypeError),  # 元素太少
        ((None, None), TypeError),  # 输入不是数组
        (((None, None, None),), TypeError),  # 元素太多
        (((np.arange(3), np.arange(3)),), TypeError),  # 不正确的 dtype
        (((np.ones(3, dtype=">d"), np.ones(3, dtype="<f")),), TypeError),  # 不支持字节交换
        (((np.ones((2, 2), dtype="d"), np.ones((2, 2), dtype="f")),), ValueError),  # 不是 1-D 数组
        (((np.ones(3, dtype="d"), np.ones(4, dtype="f")),), ValueError),  # 长度不同
        (((np.frombuffer(b"\0x00"*3*2, dtype="d"), np.frombuffer(b"\0x00"*3, dtype="f")),), ValueError),  # 输出不可写
    ])
    def test_invalid_arguments(self, args, error):
        # 断言期望抛出指定异常
        with pytest.raises(error):
            self.method._simple_strided_call(*args)


@pytest.mark.parametrize(
    "cls", [
        np.ndarray, np.recarray, np.char.chararray, np.matrix, np.memmap
    ]
)
class TestClassGetItem:
    def test_class_getitem(self, cls: type[np.ndarray]) -> None:
        """Test `ndarray.__class_getitem__`."""
        # 测试 ndarray.__class_getitem__ 方法
        alias = cls[Any, Any]
        # 断言 alias 是一个泛型别名类型
        assert isinstance(alias, types.GenericAlias)
        # 断言 alias 的原始类型是 cls
        assert alias.__origin__ is cls

    @pytest.mark.parametrize("arg_len", range(4))
    # 定义测试方法 test_subscript_tup，用于测试给定类型 cls 的元组访问
    # cls 参数指定要测试的 np.ndarray 类型
    # arg_len 参数指定元组的长度
    def test_subscript_tup(self, cls: type[np.ndarray], arg_len: int) -> None:
        # 创建一个由 Any 类型组成的元组 arg_tup，长度为 arg_len
        arg_tup = (Any,) * arg_len
        # 如果 arg_len 为 1 或 2，则断言 cls[arg_tup] 是有效的访问方式
        if arg_len in (1, 2):
            assert cls[arg_tup]
        else:
            # 否则，生成匹配消息，说明参数数量过少或过多
            match = f"Too {'few' if arg_len == 0 else 'many'} arguments"
            # 使用 pytest 来断言会抛出 TypeError 异常，并匹配指定的 match 消息
            with pytest.raises(TypeError, match=match):
                cls[arg_tup]
```