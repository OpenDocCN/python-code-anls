# `D:\src\scipysrc\pandas\pandas\tests\series\test_reductions.py`

```
# 导入必要的库
import numpy as np
import pytest

import pandas as pd
from pandas import Series
import pandas._testing as tm


# 使用pytest的参数化装饰器定义测试函数，测试Series类型数据的reduce操作结果
@pytest.mark.parametrize("operation, expected", [("min", "a"), ("max", "b")])
def test_reductions_series_strings(operation, expected):
    # 创建一个包含字符串类型数据的Series对象
    ser = Series(["a", "b"], dtype="string")
    # 使用getattr函数根据operation参数调用对应的reduce操作（例如min或max）
    res_operation_serie = getattr(ser, operation)()
    # 断言操作后的结果与期望值相等
    assert res_operation_serie == expected


# 使用pytest的参数化装饰器定义测试函数，测试mode方法的扩展数据类型
@pytest.mark.parametrize("as_period", [True, False])
def test_mode_extension_dtype(as_period):
    # 创建一个包含日期时间类型数据的Series对象
    ser = Series([pd.Timestamp(1979, 4, n) for n in range(1, 5)])

    if as_period:
        # 如果as_period为True，则将日期时间转换为周期类型
        ser = ser.dt.to_period("D")
    else:
        # 否则将日期时间本地化为指定时区
        ser = ser.dt.tz_localize("US/Central")

    # 对Series对象执行mode操作
    res = ser.mode()
    # 断言结果的数据类型与原始Series对象的数据类型相同
    assert res.dtype == ser.dtype
    # 使用pandas测试模块中的函数验证结果与原始Series对象相等
    tm.assert_series_equal(res, ser)


# 测试mode方法处理空值及不同数据类型情况
def test_mode_nullable_dtype(any_numeric_ea_dtype):
    # 创建一个包含数字和pd.NA空值的Series对象
    ser = Series([1, 3, 2, pd.NA, 3, 2, pd.NA], dtype=any_numeric_ea_dtype)
    # 对包含空值的Series对象执行mode操作，保留空值
    result = ser.mode(dropna=False)
    expected = Series([2, 3, pd.NA], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)

    # 对包含空值的Series对象执行mode操作，忽略空值
    result = ser.mode(dropna=True)
    expected = Series([2, 3], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)

    # 修改Series对象中的最后一个元素为pd.NA
    ser[-1] = pd.NA

    # 再次执行mode操作，忽略空值
    result = ser.mode(dropna=True)
    expected = Series([2, 3], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)

    # 再次执行mode操作，保留空值
    result = ser.mode(dropna=False)
    expected = Series([pd.NA], dtype=any_numeric_ea_dtype)
    tm.assert_series_equal(result, expected)


# 测试处理推断字符串数据类型的mode方法
def test_mode_infer_string():
    # 使用pytest的importorskip函数，如果pyarrow库不可用则跳过测试
    pytest.importorskip("pyarrow")
    # 创建一个包含对象类型数据的Series对象
    ser = Series(["a", "b"], dtype=object)
    # 设置pandas的选项，启用推断字符串的功能
    with pd.option_context("future.infer_string", True):
        # 执行mode操作
        result = ser.mode()
    expected = Series(["a", "b"], dtype=object)
    # 使用pandas测试模块中的函数验证结果与期望值相等
    tm.assert_series_equal(result, expected)


# 测试处理包含NaT的时间数据类型的reduce操作
def test_reductions_td64_with_nat():
    # 创建一个包含时间数据类型的Series对象，其中包含一个NaT值
    ser = Series([0, pd.NaT], dtype="m8[ns]")
    # 获取Series对象中第一个非NaT值作为期望值
    exp = ser[0]
    # 断言median、min和max方法的结果与期望值相等
    assert ser.median() == exp
    assert ser.min() == exp
    assert ser.max() == exp


# 测试空Series对象的timedelta64[ns]类型的sum方法
def test_td64_sum_empty(skipna):
    # 创建一个空的timedelta64[ns]类型的Series对象
    ser = Series([], dtype="timedelta64[ns]")

    # 执行sum方法，跳过空值的计算
    result = ser.sum(skipna=skipna)
    # 断言结果的数据类型为pd.Timedelta
    assert isinstance(result, pd.Timedelta)
    # 断言结果为0
    assert result == pd.Timedelta(0)


# 测试处理时间数据类型的sum方法，验证是否会溢出
def test_td64_summation_overflow():
    # 创建一个包含大量时间数据的Series对象
    ser = Series(pd.date_range("20130101", periods=100000, freq="h"))
    # 将第一个元素的时间增加1秒1毫秒
    ser[0] += pd.Timedelta("1s 1ms")

    # 计算（ser - ser.min()）的平均值
    result = (ser - ser.min()).mean()
    # 使用TimedeltaIndex计算期望值的平均值
    expected = pd.Timedelta((pd.TimedeltaIndex(ser - ser.min()).asi8 / len(ser)).sum())

    # 由于计算转换为浮点数，可能会有精度损失，使用np.allclose进行断言
    assert np.allclose(result._value / 1000, expected._value / 1000)

    # 执行sum方法，验证是否会溢出，预期会引发ValueError异常
    msg = "overflow in timedelta operation"
    with pytest.raises(ValueError, match=msg):
        (ser - ser.min()).sum()

    # 创建较小范围的Series对象，并执行sum方法，同样预期引发ValueError异常
    s1 = ser[0:10000]
    with pytest.raises(ValueError, match=msg):
        (s1 - s1.min()).sum()
    s2 = ser[0:1000]
    (s2 - s2.min()).sum()
# 测试解决 NumPy 1.6 中的 bug，确保 ser.prod() 返回的不是 Series 类型
def test_prod_numpy16_bug():
    ser = Series([1.0, 1.0, 1.0], index=range(3))
    result = ser.prod()

    assert not isinstance(result, Series)


# 使用 pytest 参数化装饰器，对 np.any 和 np.all 函数进行参数化测试
# 对于每个函数，测试两个不同的参数组合
@pytest.mark.parametrize("func", [np.any, np.all])
@pytest.mark.parametrize("kwargs", [{"keepdims": True}, {"out": object()}])
def test_validate_any_all_out_keepdims_raises(kwargs, func):
    ser = Series([1, 2])
    param = next(iter(kwargs))
    name = func.__name__

    # 构造错误消息，指出在 pandas 的实现中不支持特定的参数
    msg = (
        f"the '{param}' parameter is not "
        "supported in the pandas "
        rf"implementation of {name}\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        func(ser, **kwargs)


# 测试 np.sum 函数在使用 initial 参数时是否引发 ValueError
def test_validate_sum_initial():
    ser = Series([1, 2])
    msg = (
        r"the 'initial' parameter is not "
        r"supported in the pandas "
        r"implementation of sum\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        np.sum(ser, initial=10)


# 测试 Series.median 方法在使用 overwrite_input 参数时是否引发 ValueError
def test_validate_median_initial():
    ser = Series([1, 2])
    msg = (
        r"the 'overwrite_input' parameter is not "
        r"supported in the pandas "
        r"implementation of median\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        # 由于 np.median 不会派发，因此我们使用方法而不是 ufunc。
        ser.median(overwrite_input=True)


# 测试 np.sum 函数在使用 keepdims 参数时是否引发 ValueError
def test_validate_stat_keepdims():
    ser = Series([1, 2])
    msg = (
        r"the 'keepdims' parameter is not "
        r"supported in the pandas "
        r"implementation of sum\(\)"
    )
    with pytest.raises(ValueError, match=msg):
        np.sum(ser, keepdims=True)


# 使用 infer_string 参数化装饰器测试处理可转换为字符串的 Series
# 验证在不同条件下对 sum() 和 mean() 的行为
def test_mean_with_convertible_string_raises(using_infer_string):
    # GH#44008
    ser = Series(["1", "2"])
    if using_infer_string:
        msg = "does not support"
        with pytest.raises(TypeError, match=msg):
            ser.sum()
    else:
        assert ser.sum() == "12"
    msg = "Could not convert string '12' to numeric|does not support"
    with pytest.raises(TypeError, match=msg):
        ser.mean()

    df = ser.to_frame()
    msg = r"Could not convert \['12'\] to numeric|does not support"
    with pytest.raises(TypeError, match=msg):
        df.mean()


# 测试处理含有可转换为复杂类型的字符串的情况，验证对 median() 的行为
def test_mean_dont_convert_j_to_complex():
    # GH#36703
    df = pd.DataFrame([{"db": "J", "numeric": 123}])
    msg = r"Could not convert \['J'\] to numeric|does not support"
    with pytest.raises(TypeError, match=msg):
        df.mean()

    with pytest.raises(TypeError, match=msg):
        df.agg("mean")

    msg = "Could not convert string 'J' to numeric|does not support"
    with pytest.raises(TypeError, match=msg):
        df["db"].mean()
    msg = "Could not convert string 'J' to numeric|ufunc 'divide'"
    with pytest.raises(TypeError, match=msg):
        np.mean(df["db"].astype("string").array)


# 测试处理含有可转换为字符串的情况下，验证对 median() 的行为
def test_median_with_convertible_string_raises():
    # GH#34671 this _could_ return a string "2", but definitely not float 2.0
    msg = r"Cannot convert \['1' '2' '3'\] to numeric|does not support"
    # 创建一个包含字符串数据的 pandas Series 对象
    ser = Series(["1", "2", "3"])
    
    # 使用 pytest 框架的断言语法，验证在调用 ser.median() 方法时是否抛出 TypeError 异常，
    # 并且异常消息需要与变量 msg 匹配
    with pytest.raises(TypeError, match=msg):
        ser.median()
    
    # 定义一个正则表达式字符串，用于匹配特定的错误消息，指示无法将数据转换为数值或不支持的操作
    msg = r"Cannot convert \[\['1' '2' '3'\]\] to numeric|does not support"
    
    # 将 Series 转换为 DataFrame
    df = ser.to_frame()
    
    # 使用 pytest 框架的断言语法，验证在调用 df.median() 方法时是否抛出 TypeError 异常，
    # 并且异常消息需要与变量 msg 匹配
    with pytest.raises(TypeError, match=msg):
        df.median()
```