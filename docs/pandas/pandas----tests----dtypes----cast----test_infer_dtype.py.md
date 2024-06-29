# `D:\src\scipysrc\pandas\pandas\tests\dtypes\cast\test_infer_dtype.py`

```
# 导入需要的模块和函数
from datetime import (
    date,               # 导入日期类型
    datetime,           # 导入日期时间类型
    timedelta,          # 导入时间间隔类型
)

import numpy as np       # 导入NumPy库，用于数值计算
import pytest            # 导入pytest库，用于编写和运行测试

from pandas.core.dtypes.cast import (
    infer_dtype_from,          # 从值推断数据类型
    infer_dtype_from_array,    # 从数组推断数据类型
    infer_dtype_from_scalar,   # 从标量值推断数据类型
)
from pandas.core.dtypes.common import is_dtype_equal   # 判断两种数据类型是否相等

from pandas import (
    Categorical,    # 导入分类数据类型
    Interval,       # 导入区间数据类型
    Period,         # 导入周期数据类型
    Series,         # 导入Series数据类型
    Timedelta,      # 导入时间间隔数据类型
    Timestamp,      # 导入时间戳数据类型
    date_range,     # 生成日期范围
)


def test_infer_dtype_from_int_scalar(any_int_numpy_dtype):
    # 测试infer_dtype_from_scalar函数是否能正确返回整数和浮点数的数据类型
    data = np.dtype(any_int_numpy_dtype).type(12)
    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == type(data)


def test_infer_dtype_from_float_scalar(float_numpy_dtype):
    # 测试infer_dtype_from_scalar函数是否能正确返回浮点数的数据类型
    float_numpy_dtype = np.dtype(float_numpy_dtype).type
    data = float_numpy_dtype(12)

    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == float_numpy_dtype


@pytest.mark.parametrize(
    "data,exp_dtype", [(12, np.int64), (np.float64(12), np.float64)]
)
def test_infer_dtype_from_python_scalar(data, exp_dtype):
    # 使用参数化测试，验证infer_dtype_from_scalar函数对Python标量数据的推断是否正确
    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == exp_dtype


@pytest.mark.parametrize("bool_val", [True, False])
def test_infer_dtype_from_boolean(bool_val):
    # 使用参数化测试，验证infer_dtype_from_scalar函数对布尔值的数据类型推断是否正确
    dtype, val = infer_dtype_from_scalar(bool_val)
    assert dtype == np.bool_


def test_infer_dtype_from_complex(complex_dtype):
    # 测试infer_dtype_from_scalar函数是否能正确返回复数数据类型
    data = np.dtype(complex_dtype).type(1)
    dtype, val = infer_dtype_from_scalar(data)
    assert dtype == np.complex128


def test_infer_dtype_from_datetime():
    # 测试infer_dtype_from_scalar函数对日期时间类型的数据类型推断是否正确
    dt64 = np.datetime64(1, "ns")
    dtype, val = infer_dtype_from_scalar(dt64)
    assert dtype == "M8[ns]"

    ts = Timestamp(1)
    dtype, val = infer_dtype_from_scalar(ts)
    assert dtype == "M8[ns]"

    dt = datetime(2000, 1, 1, 0, 0)
    dtype, val = infer_dtype_from_scalar(dt)
    assert dtype == "M8[us]"


def test_infer_dtype_from_timedelta():
    # 测试infer_dtype_from_scalar函数对时间间隔类型的数据类型推断是否正确
    td64 = np.timedelta64(1, "ns")
    dtype, val = infer_dtype_from_scalar(td64)
    assert dtype == "m8[ns]"

    pytd = timedelta(1)
    dtype, val = infer_dtype_from_scalar(pytd)
    assert dtype == "m8[us]"

    td = Timedelta(1)
    dtype, val = infer_dtype_from_scalar(td)
    assert dtype == "m8[ns]"


@pytest.mark.parametrize("freq", ["M", "D"])
def test_infer_dtype_from_period(freq):
    # 使用参数化测试，验证infer_dtype_from_scalar函数对Period类型的数据类型推断是否正确
    p = Period("2011-01-01", freq=freq)
    dtype, val = infer_dtype_from_scalar(p)

    exp_dtype = f"period[{freq}]"

    assert dtype == exp_dtype
    assert val == p


def test_infer_dtype_misc():
    # 测试infer_dtype_from_scalar函数对其他类型（例如日期、时间戳）的数据类型推断是否正确
    dt = date(2000, 1, 1)
    dtype, val = infer_dtype_from_scalar(dt)
    assert dtype == np.object_

    ts = Timestamp(1, tz="US/Eastern")
    dtype, val = infer_dtype_from_scalar(ts)
    assert dtype == "datetime64[ns, US/Eastern]"


@pytest.mark.parametrize("tz", ["UTC", "US/Eastern", "Asia/Tokyo"])
def test_infer_from_scalar_tz(tz):
    # 使用参数化测试，验证infer_dtype_from_scalar函数对带有时区的时间戳类型的数据类型推断是否正确
    dt = Timestamp(1, tz=tz)
    dtype, val = infer_dtype_from_scalar(dt)

    exp_dtype = f"datetime64[ns, {tz}]"

    assert dtype == exp_dtype
    # 确保变量 val 的值等于变量 dt 的值
    assert val == dt
@pytest.mark.parametrize(
    "left, right, subtype",
    [
        (0, 1, "int64"),  # 定义整数类型的区间
        (0.0, 1.0, "float64"),  # 定义浮点数类型的区间
        (Timestamp(0), Timestamp(1), "datetime64[ns]"),  # 定义日期时间类型的区间
        (Timestamp(0, tz="UTC"), Timestamp(1, tz="UTC"), "datetime64[ns, UTC]"),  # 定义带有时区的日期时间类型的区间
        (Timedelta(0), Timedelta(1), "timedelta64[ns]"),  # 定义时间差类型的区间
    ],
)
def test_infer_from_interval(left, right, subtype, closed):
    # GH 30337
    interval = Interval(left, right, closed)  # 创建区间对象
    result_dtype, result_value = infer_dtype_from_scalar(interval)  # 推断区间对象的数据类型和值
    expected_dtype = f"interval[{subtype}, {closed}]"  # 期望的区间数据类型字符串
    assert result_dtype == expected_dtype  # 断言推断的数据类型与期望的数据类型一致
    assert result_value == interval  # 断言推断的值与原始区间对象一致


def test_infer_dtype_from_scalar_errors():
    msg = "invalid ndarray passed to infer_dtype_from_scalar"  # 错误信息

    with pytest.raises(ValueError, match=msg):  # 断言抛出 ValueError 异常且包含指定错误信息
        infer_dtype_from_scalar(np.array([1]))  # 调用推断函数处理错误的 ndarray


@pytest.mark.parametrize(
    "value, expected",
    [
        ("foo", np.object_),  # 推断字符串值的数据类型为对象类型
        (b"foo", np.object_),  # 推断字节字符串值的数据类型为对象类型
        (1, np.int64),  # 推断整数值的数据类型为 int64
        (1.5, np.float64),  # 推断浮点数值的数据类型为 float64
        (np.datetime64("2016-01-01"), np.dtype("M8[s]")),  # 推断日期时间值的数据类型为秒精度的 datetime64
        (Timestamp("20160101"), np.dtype("M8[s]")),  # 推断 Pandas Timestamp 对象的数据类型为秒精度的 datetime64
        (Timestamp("20160101", tz="UTC"), "datetime64[s, UTC]"),  # 推断带时区的 Pandas Timestamp 对象的数据类型
    ],
)
def test_infer_dtype_from_scalar(value, expected, using_infer_string):
    dtype, _ = infer_dtype_from_scalar(value)  # 推断标量值的数据类型
    if using_infer_string and value == "foo":
        expected = "string"  # 如果使用推断字符串并且值是字符串 "foo"，则期望的类型为字符串
    assert is_dtype_equal(dtype, expected)  # 断言推断的数据类型与期望的数据类型一致

    with pytest.raises(TypeError, match="must be list-like"):  # 断言抛出 TypeError 异常且包含指定错误信息
        infer_dtype_from_array(value)  # 调用推断函数处理非列表型数据


@pytest.mark.parametrize(
    "arr, expected",
    [
        ([1], np.dtype(int)),  # 推断包含整数的列表的数据类型
        (np.array([1], dtype=np.int64), np.int64),  # 推断整数数组的数据类型为 int64
        ([np.nan, 1, ""], np.object_),  # 推断包含 NaN、整数和空字符串的列表的数据类型为对象类型
        (np.array([[1.0, 2.0]]), np.float64),  # 推断浮点数二维数组的数据类型为 float64
        (Categorical(list("aabc")), "category"),  # 推断分类类型的数据类型为 category
        (Categorical([1, 2, 3]), "category"),  # 推断包含整数的分类类型的数据类型为 category
        (date_range("20160101", periods=3), np.dtype("=M8[ns]")),  # 推断日期范围的数据类型为纳秒精度的 datetime64
        (
            date_range("20160101", periods=3, tz="US/Eastern"),
            "datetime64[ns, US/Eastern]",  # 推断带时区的日期范围的数据类型
        ),
        (Series([1.0, 2, 3]), np.float64),  # 推断 Pandas Series 对象的数据类型为 float64
        (Series(list("abc")), np.object_),  # 推断 Pandas Series 对象的数据类型为对象类型
        (
            Series(date_range("20160101", periods=3, tz="US/Eastern")),
            "datetime64[ns, US/Eastern]",  # 推断带时区的 Pandas Series 对象的数据类型
        ),
    ],
)
def test_infer_dtype_from_array(arr, expected, using_infer_string):
    dtype, _ = infer_dtype_from_array(arr)  # 推断数组的数据类型
    if (
        using_infer_string
        and isinstance(arr, Series)
        and arr.tolist() == ["a", "b", "c"]
    ):
        expected = "string"  # 如果使用推断字符串并且数组内容是 ["a", "b", "c"]，则期望的类型为字符串
    assert is_dtype_equal(dtype, expected)  # 断言推断的数据类型与期望的数据类型一致


@pytest.mark.parametrize("cls", [np.datetime64, np.timedelta64])
def test_infer_dtype_from_scalar_zerodim_datetimelike(cls):
    # ndarray.item() can incorrectly return int instead of td64/dt64
    val = cls(1234, "ns")
    arr = np.array(val)

    dtype, res = infer_dtype_from_scalar(arr)  # 推断零维数组的数据类型和结果
    assert dtype.type is cls  # 断言推断的数据类型是指定的日期时间或时间差类型
    assert isinstance(res, cls)  # 断言推断的结果是指定的日期时间或时间差类型

    dtype, res = infer_dtype_from(arr)  # 推断数组的数据类型和结果
    assert dtype.type is cls  # 断言推断的数据类型是指定的日期时间或时间差类型
```