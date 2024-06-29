# `.\numpy\numpy\_core\tests\test_argparse.py`

```py
"""
Tests for the private NumPy argument parsing functionality.
They mainly exists to ensure good test coverage without having to try the
weirder cases on actual numpy functions but test them in one place.

The test function is defined in C to be equivalent to (errors may not always
match exactly, and could be adjusted):

    def func(arg1, /, arg2, *, arg3):
        i = integer(arg1)  # reproducing the 'i' parsing in Python.
        return None
"""

# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 NumPy 库
import numpy as np
# 从 NumPy 的内部模块中导入名为 argparse_example_function 的函数别名 func
from numpy._core._multiarray_tests import argparse_example_function as func


# 定义测试函数 test_invalid_integers，测试无效整数参数的情况
def test_invalid_integers():
    # 使用 pytest 的断言检查是否会抛出 TypeError 异常，且异常信息匹配特定字符串
    with pytest.raises(TypeError,
            match="integer argument expected, got float"):
        func(1.)
    # 使用 pytest 的断言检查是否会抛出 OverflowError 异常
    with pytest.raises(OverflowError):
        func(2**100)


# 定义测试函数 test_missing_arguments，测试缺少参数的情况
def test_missing_arguments():
    # 使用 pytest 的断言检查是否会抛出 TypeError 异常，且异常信息匹配特定字符串
    with pytest.raises(TypeError,
            match="missing required positional argument 0"):
        func()
    # 使用 pytest 的断言检查是否会抛出 TypeError 异常，且异常信息匹配特定字符串
    with pytest.raises(TypeError,
            match="missing required positional argument 0"):
        func(arg2=1, arg3=4)
    # 使用 pytest 的断言检查是否会抛出 TypeError 异常，且异常信息匹配特定字符串
    with pytest.raises(TypeError,
            match=r"missing required argument 'arg2' \(pos 1\)"):
        func(1, arg3=5)


# 定义测试函数 test_too_many_positional，测试传递过多位置参数的情况
def test_too_many_positional():
    # 使用 pytest 的断言检查是否会抛出 TypeError 异常，且异常信息匹配特定字符串
    with pytest.raises(TypeError,
            match="takes from 2 to 3 positional arguments but 4 were given"):
        func(1, 2, 3, 4)


# 定义测试函数 test_multiple_values，测试传递多个值的情况
def test_multiple_values():
    # 使用 pytest 的断言检查是否会抛出 TypeError 异常，且异常信息匹配特定字符串
    with pytest.raises(TypeError,
            match=r"given by name \('arg2'\) and position \(position 1\)"):
        func(1, 2, arg2=3)


# 定义测试函数 test_string_fallbacks，测试字符串参数的回退情况
def test_string_fallbacks():
    # 使用 numpy 的字符串对象测试慢速回退情况
    arg2 = np.str_("arg2")
    missing_arg = np.str_("missing_arg")
    func(1, **{arg2: 3})
    # 使用 pytest 的断言检查是否会抛出 TypeError 异常，且异常信息匹配特定字符串
    with pytest.raises(TypeError,
            match="got an unexpected keyword argument 'missing_arg'"):
        func(2, **{missing_arg: 3})


# 定义测试函数 test_too_many_arguments_method_forwarding，测试方法转发时参数过多的情况
def test_too_many_arguments_method_forwarding():
    # 不直接关联标准参数解析，但有时我们会将方法转发到 Python
    # 这段代码增加了对 `npy_forward_method` 的代码覆盖率
    arr = np.arange(3)
    args = range(1000)
    # 使用 pytest 的断言检查是否会抛出 TypeError 异常
    with pytest.raises(TypeError):
        arr.mean(*args)
```