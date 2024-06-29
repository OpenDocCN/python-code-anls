# `D:\src\scipysrc\pandas\pandas\tests\arrays\period\test_constructors.py`

```
# 导入必要的库：numpy 和 pytest
import numpy as np
import pytest

# 导入 pandas 库的特定模块
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.offsets import MonthEnd
from pandas._libs.tslibs.period import IncompatibleFrequency

# 导入 pandas 库及其测试模块
import pandas as pd
import pandas._testing as tm

# 导入 pandas 的数组相关模块
from pandas.core.arrays import (
    PeriodArray,
    period_array,
)

# 使用 pytest 的 parametrize 装饰器，定义多组参数化测试用例
@pytest.mark.parametrize(
    "data, freq, expected",
    [
        # 测试用例1：单个 Period 对象列表，无指定频率
        ([pd.Period("2017", "D")], None, [17167]),
        # 测试用例2：单个 Period 对象列表，指定频率 'D'
        ([pd.Period("2017", "D")], "D", [17167]),
        # 测试用例3：整数列表，指定频率 'D'
        ([2017], "D", [17167]),
        # 测试用例4：字符串列表，指定频率 'D'
        (["2017"], "D", [17167]),
        # 测试用例5：单个 Period 对象列表，频率为 Day()
        ([pd.Period("2017", "D")], pd.tseries.offsets.Day(), [17167]),
        # 测试用例6：包含 None 的 Period 对象列表，无指定频率
        ([pd.Period("2017", "D"), None], None, [17167, iNaT]),
        # 测试用例7：从日期范围创建的 Series，无指定频率
        (pd.Series(pd.date_range("2017", periods=3)), None, [17167, 17168, 17169]),
        # 测试用例8：从日期范围创建的日期列表，无指定频率
        (pd.date_range("2017", periods=3), None, [17167, 17168, 17169]),
        # 测试用例9：从期间范围创建的 PeriodIndex，无指定频率
        (pd.period_range("2017", periods=4, freq="Q"), None, [188, 189, 190, 191]),
    ],
)
def test_period_array_ok(data, freq, expected):
    # 调用 period_array 函数生成 PeriodArray 对象，并获取其整数表示
    result = period_array(data, freq=freq).asi8
    # 将预期结果转换为 numpy 数组，并指定数据类型为 int64
    expected = np.asarray(expected, dtype=np.int64)
    # 使用测试模块的方法验证结果与预期是否相等
    tm.assert_numpy_array_equal(result, expected)


def test_period_array_readonly_object():
    # 测试用例：验证只读对象的处理
    pa = period_array([pd.Period("2019-01-01")])
    # 将 PeriodArray 转换为 object 类型的 numpy 数组，并设置为只读
    arr = np.asarray(pa, dtype="object")
    arr.setflags(write=False)

    # 使用只读数组创建新的 PeriodArray 对象
    result = period_array(arr)
    # 使用测试模块的方法验证两个 PeriodArray 对象是否相等
    tm.assert_period_array_equal(result, pa)

    # 将只读数组转换为 Series 对象，并验证与原始 PeriodArray 对象的 Series 是否相等
    result = pd.Series(arr)
    tm.assert_series_equal(result, pd.Series(pa))

    # 将只读数组转换为 DataFrame 对象，并验证与原始 PeriodArray 对象的 DataFrame 是否相等
    result = pd.DataFrame({"A": arr})
    tm.assert_frame_equal(result, pd.DataFrame({"A": pa}))


def test_from_datetime64_freq_changes():
    # 测试用例：验证从 datetime64 数组创建 PeriodArray 对象，并指定频率变化
    arr = pd.date_range("2017", periods=3, freq="D")
    # 调用 PeriodArray 的静态方法 _from_datetime64，将日期数组转换为 PeriodArray 对象
    result = PeriodArray._from_datetime64(arr, freq="M")
    # 创建预期的 PeriodArray 对象，指定相同的日期字符串和频率 'M'
    expected = period_array(["2017-01-01", "2017-01-01", "2017-01-01"], freq="M")
    # 使用测试模块的方法验证结果与预期是否相等
    tm.assert_period_array_equal(result, expected)


@pytest.mark.parametrize("freq", ["2M", MonthEnd(2)])
def test_from_datetime64_freq_2M(freq):
    # 测试用例：验证从 datetime64 数组创建 PeriodArray 对象，并指定不同的频率
    arr = np.array(
        ["2020-01-01T00:00:00", "2020-01-02T00:00:00"], dtype="datetime64[ns]"
    )
    # 调用 PeriodArray 的静态方法 _from_datetime64，将日期数组转换为 PeriodArray 对象
    result = PeriodArray._from_datetime64(arr, freq)
    # 创建预期的 PeriodArray 对象，指定相同的日期字符串和不同的频率
    expected = period_array(["2020-01", "2020-01"], freq=freq)
    # 使用测试模块的方法验证结果与预期是否相等
    tm.assert_period_array_equal(result, expected)


@pytest.mark.parametrize(
    "data, freq, msg",
    [
        # 测试用例1：包含不同频率的 Period 对象列表，无指定频率
        ([pd.Period("2017", "D"), pd.Period("2017", "Y")], None, "Input has different freq"),
        # 测试用例2：单个 Period 对象列表，指定不同的频率 'Y'
        ([pd.Period("2017", "D")], "Y", "Input has different freq"),
    ],
)
def test_period_array_raises(data, freq, msg):
    # 测试用例：验证 period_array 函数对不同频率的处理，预期引发 IncompatibleFrequency 异常
    with pytest.raises(IncompatibleFrequency, match=msg):
        period_array(data, freq)


def test_period_array_non_period_series_raies():
    # 测试用例：验证处理非 Period 类型的 Series，预期引发 TypeError 异常
    ser = pd.Series([1, 2, 3])
    with pytest.raises(TypeError, match="dtype"):
        # 使用 PeriodArray 尝试从非 Period 类型的 Series 创建对象
        PeriodArray(ser, dtype="period[D]")


def test_period_array_freq_mismatch():
    # 测试用例：验证创建 PeriodArray 对象时指定的频率与数据频率不匹配
    arr = period_array(["2000", "2001"], freq="D")
    # 使用 pytest 来测试期望引发 IncompatibleFrequency 异常，并检查异常消息是否包含字符串 "freq"
    with pytest.raises(IncompatibleFrequency, match="freq"):
        # 创建 PeriodArray 对象，并期望其抛出 IncompatibleFrequency 异常
        PeriodArray(arr, dtype="period[M]")
    
    # 创建一个 PeriodDtype 对象，使用 pd.tseries.offsets.MonthEnd() 作为频率
    dtype = pd.PeriodDtype(pd.tseries.offsets.MonthEnd())
    # 使用 pytest 来测试期望引发 IncompatibleFrequency 异常，并检查异常消息是否包含字符串 "freq"
    with pytest.raises(IncompatibleFrequency, match="freq"):
        # 创建 PeriodArray 对象，指定 dtype 参数为之前创建的 PeriodDtype 对象
        PeriodArray(arr, dtype=dtype)
def test_from_sequence_disallows_i8():
    # 创建一个日期周期数组，包含字符串 "2000" 和 "2001"，频率为每天
    arr = period_array(["2000", "2001"], freq="D")

    # 获取第一个元素的序数值并转换为字符串
    msg = str(arr[0].ordinal)
    
    # 使用 pytest 检查调用 PeriodArray._from_sequence(arr.asi8, dtype=arr.dtype) 是否会抛出 TypeError 异常，并匹配特定的错误消息
    with pytest.raises(TypeError, match=msg):
        PeriodArray._from_sequence(arr.asi8, dtype=arr.dtype)

    # 使用 pytest 检查调用 PeriodArray._from_sequence(list(arr.asi8), dtype=arr.dtype) 是否会抛出 TypeError 异常，并匹配特定的错误消息
    with pytest.raises(TypeError, match=msg):
        PeriodArray._from_sequence(list(arr.asi8), dtype=arr.dtype)


def test_from_td64nat_sequence_raises():
    # GH#44507
    # 创建一个 NaT 值的 np.ndarray，数据类型为 'm8[ns]'
    td = pd.NaT.to_numpy("m8[ns]")

    # 创建一个周期范围，从 "2005-01-01" 开始，包含 3 个周期，频率为每天，并获取其数据类型
    dtype = pd.period_range("2005-01-01", periods=3, freq="D").dtype

    # 创建一个包含单个 None 元素的对象数组 arr
    arr = np.array([None], dtype=object)
    # 将 arr 的第一个元素赋值为 td
    arr[0] = td

    # 定义用于匹配的错误消息
    msg = "Value must be Period, string, integer, or datetime"

    # 使用 pytest 检查调用 PeriodArray._from_sequence(arr, dtype=dtype) 是否会抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        PeriodArray._from_sequence(arr, dtype=dtype)

    # 使用 pytest 检查调用 pd.PeriodIndex(arr, dtype=dtype) 是否会抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        pd.PeriodIndex(arr, dtype=dtype)
    
    # 使用 pytest 检查调用 pd.Index(arr, dtype=dtype) 是否会抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        pd.Index(arr, dtype=dtype)
    
    # 使用 pytest 检查调用 pd.array(arr, dtype=dtype) 是否会抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        pd.array(arr, dtype=dtype)
    
    # 使用 pytest 检查调用 pd.Series(arr, dtype=dtype) 是否会抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        pd.Series(arr, dtype=dtype)
    
    # 使用 pytest 检查调用 pd.DataFrame(arr, dtype=dtype) 是否会抛出 ValueError 异常，并匹配特定的错误消息
    with pytest.raises(ValueError, match=msg):
        pd.DataFrame(arr, dtype=dtype)


def test_period_array_from_datetime64():
    # 创建一个包含两个元素的 np.ndarray，元素为日期时间字符串，数据类型为 'datetime64[ns]'
    arr = np.array(
        ["2020-01-01T00:00:00", "2020-02-02T00:00:00"], dtype="datetime64[ns]"
    )
    # 调用 PeriodArray._from_datetime64(arr, freq=MonthEnd(2)) 返回结果
    result = PeriodArray._from_datetime64(arr, freq=MonthEnd(2))

    # 创建一个预期的周期数组，包含两个周期，频率为每两个月的月底
    expected = period_array(["2020-01-01", "2020-02-01"], freq=MonthEnd(2))
    
    # 使用 tm.assert_period_array_equal 检查 result 和 expected 是否相等
    tm.assert_period_array_equal(result, expected)
```