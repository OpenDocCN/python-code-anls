# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_timedeltas.py`

```
# 导入必要的库和模块
import re  # 导入正则表达式模块

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 测试框架

# 导入 pandas 库中的时间差相关模块和类
from pandas._libs.tslibs.timedeltas import (
    array_to_timedelta64,  # 导入数组转换为 timedelta64 类型的函数
    delta_to_nanoseconds,  # 导入将 Timedelta 对象转换为纳秒的函数
    ints_to_pytimedelta,  # 导入整数转换为 PyTimedelta 类型的函数
)

# 从 pandas 库中导入 Timedelta 和 offsets 类
from pandas import (
    Timedelta,  # 导入 Timedelta 类
    offsets,  # 导入偏移量相关类
)
import pandas._testing as tm  # 导入 pandas 测试模块

# 使用 Pytest 的 parametrize 装饰器定义多个测试用例
@pytest.mark.parametrize(
    "obj,expected",
    [
        (np.timedelta64(14, "D"), 14 * 24 * 3600 * 1e9),  # 测试天数转换为纳秒
        (Timedelta(minutes=-7), -7 * 60 * 1e9),  # 测试负数分钟转换为纳秒
        (Timedelta(minutes=-7).to_pytimedelta(), -7 * 60 * 1e9),  # 测试负数分钟转换为 PyTimedelta 后的纳秒
        (Timedelta(seconds=1234e-9), 1234),  # GH43764, GH40946，测试秒数转换为整数
        (
            Timedelta(seconds=1e-9, milliseconds=1e-5, microseconds=1e-1),
            111,
        ),  # GH43764，测试不同时间单位混合转换为整数
        (
            Timedelta(days=1, seconds=1e-9, milliseconds=1e-5, microseconds=1e-1),
            24 * 3600e9 + 111,
        ),  # GH43764，测试包含天数的时间转换为整数
        (offsets.Nano(125), 125),  # 测试 Nano 偏移量转换为整数
    ],
)
def test_delta_to_nanoseconds(obj, expected):
    result = delta_to_nanoseconds(obj)  # 调用 delta_to_nanoseconds 函数
    assert result == expected  # 断言结果是否与预期相符


# 测试 delta_to_nanoseconds 函数处理错误情况
def test_delta_to_nanoseconds_error():
    obj = np.array([123456789], dtype="m8[ns]")  # 创建一个 numpy 数组

    with pytest.raises(TypeError, match="<class 'numpy.ndarray'>"):
        delta_to_nanoseconds(obj)  # 断言处理 numpy 数组时是否抛出 TypeError

    with pytest.raises(TypeError, match="float"):
        delta_to_nanoseconds(1.5)  # 断言处理浮点数时是否抛出 TypeError
    with pytest.raises(TypeError, match="int"):
        delta_to_nanoseconds(1)  # 断言处理整数时是否抛出 TypeError
    with pytest.raises(TypeError, match="int"):
        delta_to_nanoseconds(np.int64(2))  # 断言处理 int64 类型整数时是否抛出 TypeError
    with pytest.raises(TypeError, match="int"):
        delta_to_nanoseconds(np.int32(3))  # 断言处理 int32 类型整数时是否抛出 TypeError


# 测试 delta_to_nanoseconds 函数对不支持的时间单位（年和月）的处理
def test_delta_to_nanoseconds_td64_MY_raises():
    msg = (
        "delta_to_nanoseconds does not support Y or M units, "
        "as their duration in nanoseconds is ambiguous"
    )

    td = np.timedelta64(1234, "Y")  # 创建年的 timedelta 对象

    with pytest.raises(ValueError, match=msg):
        delta_to_nanoseconds(td)  # 断言处理年单位时是否抛出 ValueError

    td = np.timedelta64(1234, "M")  # 创建月的 timedelta 对象

    with pytest.raises(ValueError, match=msg):
        delta_to_nanoseconds(td)  # 断言处理月单位时是否抛出 ValueError


# 测试不支持的 timedelta64 单位引发异常的情况
@pytest.mark.parametrize("unit", ["Y", "M"])
def test_unsupported_td64_unit_raises(unit):
    # GH 52806
    with pytest.raises(
        ValueError,
        match=f"Unit {unit} is not supported. "
        "Only unambiguous timedelta values durations are supported. "
        "Allowed units are 'W', 'D', 'h', 'm', 's', 'ms', 'us', 'ns'",
    ):
        Timedelta(np.timedelta64(1, unit))  # 断言处理不支持单位时是否抛出 ValueError


# 测试处理超大纳秒数导致溢出的情况
def test_huge_nanoseconds_overflow():
    # GH 32402
    assert delta_to_nanoseconds(Timedelta(1e10)) == 1e10  # 断言处理大数值时是否正确处理
    assert delta_to_nanoseconds(Timedelta(nanoseconds=1e10)) == 1e10  # 断言处理大数值时是否正确处理


# 测试 Timedelta 对象关键字参数错误断言
@pytest.mark.parametrize(
    "kwargs", [{"Seconds": 1}, {"seconds": 1, "Nanoseconds": 1}, {"Foo": 2}]
)
def test_kwarg_assertion(kwargs):
    err_message = (
        "cannot construct a Timedelta from the passed arguments, "
        "allowed keywords are "
        "[weeks, days, hours, minutes, seconds, "
        "milliseconds, microseconds, nanoseconds]"
    )

    with pytest.raises(ValueError, match=re.escape(err_message)):
        Timedelta(**kwargs)  # 断言处理错误关键字参数时是否抛出 ValueError


# 定义 TestArrayToTimedelta64 类
class TestArrayToTimedelta64:
    # 定义测试函数：测试二维数组转换为 timedelta64 字符串时是否正确抛出异常
    def test_array_to_timedelta64_string_with_unit_2d_raises(self):
        # 检查当 'unit' 参数不为 None 且 'errors' 参数不为 "coerce" 时的路径
        # 创建包含字符串和数字的二维 NumPy 数组作为测试数据
        values = np.array([["1", 2], [3, "4"]], dtype=object)
        # 使用 pytest 检查是否正确抛出 ValueError 异常，并验证异常信息中包含特定字符串
        with pytest.raises(ValueError, match="unit must not be specified"):
            array_to_timedelta64(values, unit="s")

    # 定义测试函数：测试非对象类型数组转换为 timedelta64 字符串时是否正确抛出异常
    def test_array_to_timedelta64_non_object_raises(self):
        # 检查是否能正确抛出异常而不是导致段错误
        # 创建一个包含连续整数的 NumPy 数组作为测试数据
        values = np.arange(5)
        # 准备异常信息字符串，用于验证异常类型
        msg = "'values' must have object dtype"
        # 使用 pytest 检查是否正确抛出 TypeError 异常，并验证异常信息中包含特定字符串
        with pytest.raises(TypeError, match=msg):
            array_to_timedelta64(values)
# 使用 pytest.mark.parametrize 装饰器为 test_ints_to_pytimedelta 函数提供参数化测试支持，参数为单位 "s", "ms", "us"
@pytest.mark.parametrize("unit", ["s", "ms", "us"])
# 定义测试函数 test_ints_to_pytimedelta，用于测试非纳秒情况下的 ints_to_pytimedelta 函数
def test_ints_to_pytimedelta(unit):
    # 创建一个包含 0 到 5 的 numpy 数组，数据类型为 np.int64，并转换为指定单位的日期时间格式
    arr = np.arange(6, dtype=np.int64).view(f"m8[{unit}]")

    # 调用 ints_to_pytimedelta 函数，返回结果不包装为对象
    res = ints_to_pytimedelta(arr, box=False)
    # 将 arr 转换为对象类型，期望结果为 pytimedelta 对象而不是整数
    expected = arr.astype(object)
    # 使用 numpy.testing.assert_array_equal 检查 res 和 expected 是否相等
    tm.assert_numpy_array_equal(res, expected)

    # 再次调用 ints_to_pytimedelta 函数，返回结果包装为对象
    res = ints_to_pytimedelta(arr, box=True)
    # 期望结果为一个包含 Timedelta 对象的 numpy 数组
    expected = np.array([Timedelta(x) for x in arr], dtype=object)
    # 使用 numpy.testing.assert_array_equal 检查 res 和 expected 是否相等
    tm.assert_numpy_array_equal(res, expected)


# 使用 pytest.mark.parametrize 装饰器为 test_ints_to_pytimedelta_unsupported 函数提供参数化测试支持，参数为单位 "Y", "M", "ps", "fs", "as"
@pytest.mark.parametrize("unit", ["Y", "M", "ps", "fs", "as"])
# 定义测试函数 test_ints_to_pytimedelta_unsupported，用于测试不支持的单位情况
def test_ints_to_pytimedelta_unsupported(unit):
    # 创建一个包含 0 到 5 的 numpy 数组，数据类型为 np.int64，并尝试转换为指定单位的日期时间格式
    arr = np.arange(6, dtype=np.int64).view(f"m8[{unit}]")

    # 使用 pytest.raises 检查调用 ints_to_pytimedelta 函数时是否抛出 NotImplementedError 异常，并匹配包含数字的错误信息
    with pytest.raises(NotImplementedError, match=r"\d{1,2}"):
        ints_to_pytimedelta(arr, box=False)
    
    # 定义错误消息，表示仅支持 's', 'ms', 'us', 'ns' 这几种时间单位
    msg = "Only resolutions 's', 'ms', 'us', 'ns' are supported"
    # 使用 pytest.raises 检查调用 ints_to_pytimedelta 函数时是否抛出 NotImplementedError 异常，并匹配指定的错误消息
    with pytest.raises(NotImplementedError, match=msg):
        ints_to_pytimedelta(arr, box=True)
```