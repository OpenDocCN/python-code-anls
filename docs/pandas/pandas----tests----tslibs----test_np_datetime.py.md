# `D:\src\scipysrc\pandas\pandas\tests\tslibs\test_np_datetime.py`

```
import numpy as np  # 导入NumPy库，用于处理数值计算
import pytest  # 导入pytest库，用于编写和运行测试用例

from pandas._libs.tslibs.dtypes import NpyDatetimeUnit  # 从pandas库导入NpyDatetimeUnit，处理NumPy日期时间单位
from pandas._libs.tslibs.np_datetime import (  # 从pandas库导入多个日期时间相关的函数和异常处理类
    OutOfBoundsDatetime,  # 日期时间超出范围异常
    OutOfBoundsTimedelta,  # 时间间隔超出范围异常
    astype_overflowsafe,  # 安全类型转换函数
    is_unitless,  # 判断是否是无单位的日期时间类型函数
    py_get_unit_from_dtype,  # 根据dtype获取单位函数
    py_td64_to_tdstruct,  # timedelta64类型转换为td结构函数
)

import pandas._testing as tm  # 导入pandas测试模块，用于测试辅助函数和类


def test_is_unitless():
    dtype = np.dtype("M8[ns]")  # 创建一个dtype对象，表示纳秒级日期时间类型
    assert not is_unitless(dtype)  # 断言纳秒级日期时间类型不是无单位的

    dtype = np.dtype("datetime64")  # 创建一个dtype对象，表示日期时间类型
    assert is_unitless(dtype)  # 断言日期时间类型是无单位的

    dtype = np.dtype("m8[ns]")  # 创建一个dtype对象，表示纳秒级时间间隔类型
    assert not is_unitless(dtype)  # 断言纳秒级时间间隔类型不是无单位的

    dtype = np.dtype("timedelta64")  # 创建一个dtype对象，表示时间间隔类型
    assert is_unitless(dtype)  # 断言时间间隔类型是无单位的

    msg = "dtype must be datetime64 or timedelta64"
    with pytest.raises(ValueError, match=msg):  # 使用pytest断言抛出特定异常和匹配消息
        is_unitless(np.dtype(np.int64))  # 尝试对非日期时间或时间间隔类型进行判断，期望抛出异常

    msg = "Argument 'dtype' has incorrect type"
    with pytest.raises(TypeError, match=msg):  # 使用pytest断言抛出特定异常和匹配消息
        is_unitless("foo")  # 尝试传入非dtype对象进行判断，期望抛出异常


def test_get_unit_from_dtype():
    # datetime64
    assert py_get_unit_from_dtype(np.dtype("M8[Y]")) == NpyDatetimeUnit.NPY_FR_Y.value  # 断言年单位的日期时间类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("M8[M]")) == NpyDatetimeUnit.NPY_FR_M.value  # 断言月单位的日期时间类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("M8[W]")) == NpyDatetimeUnit.NPY_FR_W.value  # 断言周单位的日期时间类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("M8[D]")) == NpyDatetimeUnit.NPY_FR_D.value  # 断言日单位的日期时间类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("M8[h]")) == NpyDatetimeUnit.NPY_FR_h.value  # 断言小时单位的日期时间类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("M8[m]")) == NpyDatetimeUnit.NPY_FR_m.value  # 断言分钟单位的日期时间类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("M8[s]")) == NpyDatetimeUnit.NPY_FR_s.value  # 断言秒单位的日期时间类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("M8[ms]")) == NpyDatetimeUnit.NPY_FR_ms.value  # 断言毫秒单位的日期时间类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("M8[us]")) == NpyDatetimeUnit.NPY_FR_us.value  # 断言微秒单位的日期时间类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("M8[ns]")) == NpyDatetimeUnit.NPY_FR_ns.value  # 断言纳秒单位的日期时间类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("M8[ps]")) == NpyDatetimeUnit.NPY_FR_ps.value  # 断言皮秒单位的日期时间类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("M8[fs]")) == NpyDatetimeUnit.NPY_FR_fs.value  # 断言飞秒单位的日期时间类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("M8[as]")) == NpyDatetimeUnit.NPY_FR_as.value  # 断言阿秒单位的日期时间类型转换为对应的NumPy单位值

    # timedelta64
    assert py_get_unit_from_dtype(np.dtype("m8[Y]")) == NpyDatetimeUnit.NPY_FR_Y.value  # 断言年单位的时间间隔类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("m8[M]")) == NpyDatetimeUnit.NPY_FR_M.value  # 断言月单位的时间间隔类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("m8[W]")) == NpyDatetimeUnit.NPY_FR_W.value  # 断言周单位的时间间隔类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("m8[D]")) == NpyDatetimeUnit.NPY_FR_D.value  # 断言日单位的时间间隔类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("m8[h]")) == NpyDatetimeUnit.NPY_FR_h.value  # 断言小时单位的时间间隔类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("m8[m]")) == NpyDatetimeUnit.NPY_FR_m.value  # 断言分钟单位的时间间隔类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("m8[s]")) == NpyDatetimeUnit.NPY_FR_s.value  # 断言秒单位的时间间隔类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("m8[ms]")) == NpyDatetimeUnit.NPY_FR_ms.value  # 断言毫秒单位的时间间隔类型转换为对应的NumPy单位值
    assert py_get_unit_from_dtype(np.dtype("m8[us]")) == NpyDatetimeUnit.NPY_FR_us.value  # 断言微秒单位的时间间隔类型转换为对应的NumPy单位值
    # 对给定的 NumPy datetime 类型的 dtype，使用 py_get_unit_from_dtype 函数获取其对应的单位值，
    # 并断言该单位值与 NpyDatetimeUnit 中相应单位的值相等
    assert py_get_unit_from_dtype(np.dtype("m8[ns]")) == NpyDatetimeUnit.NPY_FR_ns.value
    # 同上，针对 "m8[ps]" 类型的 dtype 断言单位值与 NpyDatetimeUnit 中相应单位的值相等
    assert py_get_unit_from_dtype(np.dtype("m8[ps]")) == NpyDatetimeUnit.NPY_FR_ps.value
    # 同上，针对 "m8[fs]" 类型的 dtype 断言单位值与 NpyDatetimeUnit 中相应单位的值相等
    assert py_get_unit_from_dtype(np.dtype("m8[fs]")) == NpyDatetimeUnit.NPY_FR_fs.value
    # 同上，针对 "m8[as]" 类型的 dtype 断言单位值与 NpyDatetimeUnit 中相应单位的值相等
    assert py_get_unit_from_dtype(np.dtype("m8[as]")) == NpyDatetimeUnit.NPY_FR_as.value
def test_td64_to_tdstruct():
    val = 12454636234  # 使用任意值进行测试

    # 测试以纳秒为单位转换
    res1 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_ns.value)
    exp1 = {
        "days": 0,
        "hrs": 0,
        "min": 0,
        "sec": 12,
        "ms": 454,
        "us": 636,
        "ns": 234,
        "seconds": 12,
        "microseconds": 454636,
        "nanoseconds": 234,
    }
    assert res1 == exp1

    # 测试以微秒为单位转换
    res2 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_us.value)
    exp2 = {
        "days": 0,
        "hrs": 3,
        "min": 27,
        "sec": 34,
        "ms": 636,
        "us": 234,
        "ns": 0,
        "seconds": 12454,
        "microseconds": 636234,
        "nanoseconds": 0,
    }
    assert res2 == exp2

    # 测试以毫秒为单位转换
    res3 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_ms.value)
    exp3 = {
        "days": 144,
        "hrs": 3,
        "min": 37,
        "sec": 16,
        "ms": 234,
        "us": 0,
        "ns": 0,
        "seconds": 13036,
        "microseconds": 234000,
        "nanoseconds": 0,
    }
    assert res3 == exp3

    # 注意，此处的值对于纳秒级时间增量超出了范围
    res4 = py_td64_to_tdstruct(val, NpyDatetimeUnit.NPY_FR_s.value)
    exp4 = {
        "days": 144150,
        "hrs": 21,
        "min": 10,
        "sec": 34,
        "ms": 0,
        "us": 0,
        "ns": 0,
        "seconds": 76234,
        "microseconds": 0,
        "nanoseconds": 0,
    }
    assert res4 == exp4


class TestAstypeOverflowSafe:
    def test_pass_non_dt64_array(self):
        # 确保我们会抛出异常而不是段错误
        arr = np.arange(5)
        dtype = np.dtype("M8[ns]")

        msg = (
            "astype_overflowsafe values.dtype and dtype must be either "
            "both-datetime64 or both-timedelta64"
        )
        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=True)

        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=False)

    def test_pass_non_dt64_dtype(self):
        # 确保我们会抛出异常而不是段错误
        arr = np.arange(5, dtype="i8").view("M8[D]")
        dtype = np.dtype("m8[ns]")

        msg = (
            "astype_overflowsafe values.dtype and dtype must be either "
            "both-datetime64 or both-timedelta64"
        )
        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=True)

        with pytest.raises(TypeError, match=msg):
            astype_overflowsafe(arr, dtype, copy=False)
    # 定义一个测试函数，用于测试 astype_overflowsafe 函数处理 datetime64 类型数据时的行为
    def test_astype_overflowsafe_dt64(self):
        # 定义一个 datetime64 类型的数据类型
        dtype = np.dtype("M8[ns]")

        # 创建一个 datetime64 对象 dt，表示 2262-04-05
        dt = np.datetime64("2262-04-05", "D")

        # 创建一个包含 10 个元素的 datetime64 数组 arr，每个元素表示自 dt 开始的多少天
        arr = dt + np.arange(10, dtype="m8[D]")

        # 使用 arr.astype 转换到 dtype 类型，会发生静默的溢出
        wrong = arr.astype(dtype)

        # 将类型为 dtype 的数组 wrong 再次转换回原始类型 arr.dtype
        roundtrip = wrong.astype(arr.dtype)

        # 断言 wrong 与 roundtrip 数组不全相等
        assert not (wrong == roundtrip).all()

        # 准备错误消息，用于测试 astype_overflowsafe 函数是否能正确捕获溢出异常
        msg = "Out of bounds nanosecond timestamp"
        
        # 使用 pytest 的断言检查 astype_overflowsafe 函数是否抛出预期的异常
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            astype_overflowsafe(arr, dtype)

        # 将数据转换为微秒级别的 dtype2 类型，并验证结果与 numpy 的预期结果一致
        dtype2 = np.dtype("M8[us]")
        result = astype_overflowsafe(arr, dtype2)
        expected = arr.astype(dtype2)
        tm.assert_numpy_array_equal(result, expected)

    # 定义一个测试函数，用于测试 astype_overflowsafe 函数处理 timedelta64 类型数据时的行为
    def test_astype_overflowsafe_td64(self):
        # 定义一个 timedelta64 类型的数据类型
        dtype = np.dtype("m8[ns]")

        # 创建一个 datetime64 对象 dt，表示 2262-04-05
        dt = np.datetime64("2262-04-05", "D")

        # 创建一个包含 10 个元素的 timedelta64 数组 arr，每个元素表示自 dt 开始的多少天
        arr = dt + np.arange(10, dtype="m8[D]")

        # 将 arr 视图转换为 m8[D] 类型
        arr = arr.view("m8[D]")

        # 使用 arr.astype 转换到 dtype 类型，会发生静默的溢出
        wrong = arr.astype(dtype)

        # 将类型为 dtype 的数组 wrong 再次转换回原始类型 arr.dtype
        roundtrip = wrong.astype(arr.dtype)

        # 断言 wrong 与 roundtrip 数组不全相等
        assert not (wrong == roundtrip).all()

        # 准备错误消息，用于测试 astype_overflowsafe 函数是否能正确捕获溢出异常
        msg = r"Cannot convert 106752 days to timedelta64\[ns\] without overflow"
        
        # 使用 pytest 的断言检查 astype_overflowsafe 函数是否抛出预期的异常
        with pytest.raises(OutOfBoundsTimedelta, match=msg):
            astype_overflowsafe(arr, dtype)

        # 将数据转换为微秒级别的 dtype2 类型，并验证结果与 numpy 的预期结果一致
        dtype2 = np.dtype("m8[us]")
        result = astype_overflowsafe(arr, dtype2)
        expected = arr.astype(dtype2)
        tm.assert_numpy_array_equal(result, expected)

    # 定义一个测试函数，用于测试 astype_overflowsafe 函数在不允许舍入的情况下的行为
    def test_astype_overflowsafe_disallow_rounding(self):
        # 创建一个包含两个元素的 datetime64 数组 arr，类型为 M8[ns]
        arr = np.array([-1500, 1500], dtype="M8[ns]")

        # 定义目标数据类型为 M8[us]
        dtype = np.dtype("M8[us]")

        # 准备错误消息，用于测试 astype_overflowsafe 函数是否能正确捕获溢出异常
        msg = "Cannot losslessly cast '-1500 ns' to us"
        
        # 使用 pytest 的断言检查 astype_overflowsafe 函数是否抛出预期的异常，设置不允许舍入
        with pytest.raises(ValueError, match=msg):
            astype_overflowsafe(arr, dtype, round_ok=False)

        # 调用 astype_overflowsafe 函数，允许舍入，将 arr 转换为 dtype 类型，并验证结果与 numpy 的预期结果一致
        result = astype_overflowsafe(arr, dtype, round_ok=True)
        expected = arr.astype(dtype)
        tm.assert_numpy_array_equal(result, expected)
```