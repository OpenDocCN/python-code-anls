# `D:\src\scipysrc\pandas\pandas\tests\dtypes\test_missing.py`

```
from contextlib import nullcontext  # 导入 nullcontext 上下文管理器，用于创建一个空的上下文，无需实际操作
from datetime import datetime  # 导入 datetime 模块中的 datetime 类
from decimal import Decimal  # 导入 Decimal 类，用于高精度的十进制运算

import numpy as np  # 导入 NumPy 库，用于科学计算
import pytest  # 导入 pytest 框架，用于编写和运行测试

from pandas._libs import missing as libmissing  # 导入 pandas 库中的 missing 模块
from pandas._libs.tslibs import iNaT  # 导入 pandas 库中的时间戳无效值 iNaT
from pandas.compat.numpy import np_version_gte1p25  # 导入 pandas 兼容模块中的 NumPy 版本检查函数

from pandas.core.dtypes.common import (  # 导入 pandas 中的数据类型常用函数
    is_float,
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import (  # 导入 pandas 中的数据类型类
    CategoricalDtype,
    DatetimeTZDtype,
    IntervalDtype,
    PeriodDtype,
)
from pandas.core.dtypes.missing import (  # 导入 pandas 中的数据缺失处理函数
    array_equivalent,
    is_valid_na_for_dtype,
    isna,
    isnull,
    na_value_for_dtype,
    notna,
    notnull,
)

import pandas as pd  # 导入 pandas 库作为 pd 别名，用于数据处理和分析
from pandas import (  # 从 pandas 库中导入多个类和函数
    DatetimeIndex,
    Index,
    NaT,
    Series,
    TimedeltaIndex,
    date_range,
    period_range,
)
import pandas._testing as tm  # 导入 pandas 测试模块作为 tm 别名

fix_now = pd.Timestamp("2021-01-01")  # 创建一个固定的时间戳对象 fix_now，表示 2021-01-01
fix_utcnow = pd.Timestamp("2021-01-01", tz="UTC")  # 创建一个带有时区的固定时间戳对象 fix_utcnow，表示 2021-01-01 UTC 时间


@pytest.mark.parametrize("notna_f", [notna, notnull])
def test_notna_notnull(notna_f):
    assert notna_f(1.0)  # 断言 notna_f(1.0) 应该为 True
    assert not notna_f(None)  # 断言 notna_f(None) 应该为 False
    assert not notna_f(np.nan)  # 断言 notna_f(np.nan) 应该为 False


@pytest.mark.parametrize("null_func", [notna, notnull, isna, isnull])
@pytest.mark.parametrize(
    "ser",
    [
        Series(  # 创建一个 Series 对象，使用字符串作为值和索引
            [str(i) for i in range(5)],
            index=Index([str(i) for i in range(5)], dtype=object),
            dtype=object,
        ),
        Series(range(5), date_range("2020-01-01", periods=5)),  # 创建一个时间序列的 Series 对象
        Series(range(5), period_range("2020-01-01", periods=5)),  # 创建一个时间段序列的 Series 对象
    ],
)
def test_null_check_is_series(null_func, ser):
    assert isinstance(null_func(ser), Series)  # 断言 null_func(ser) 返回的对象是一个 Series 对象


class TestIsNA:
    def test_0d_array(self):
        assert isna(np.array(np.nan))  # 断言 np.array(np.nan) 是否为 NaN
        assert not isna(np.array(0.0))  # 断言 np.array(0.0) 是否不是 NaN
        assert not isna(np.array(0))  # 断言 np.array(0) 是否不是 NaN
        # test object dtype
        assert isna(np.array(np.nan, dtype=object))  # 断言 np.array(np.nan, dtype=object) 是否为 NaN
        assert not isna(np.array(0.0, dtype=object))  # 断言 np.array(0.0, dtype=object) 是否不是 NaN
        assert not isna(np.array(0, dtype=object))  # 断言 np.array(0, dtype=object) 是否不是 NaN

    @pytest.mark.parametrize("shape", [(4, 0), (4,)])
    def test_empty_object(self, shape):
        arr = np.empty(shape=shape, dtype=object)  # 创建一个空的对象数组
        result = isna(arr)  # 调用 isna 函数检查数组中的 NaN 值
        expected = np.ones(shape=shape, dtype=bool)  # 创建一个期望的布尔数组，表示预期的 NaN 值
        tm.assert_numpy_array_equal(result, expected)  # 使用测试模块中的函数进行断言，检查结果与期望是否一致

    @pytest.mark.parametrize("isna_f", [isna, isnull])
    def test_isna_isnull(self, isna_f):
        assert not isna_f(1.0)  # 断言 isna_f(1.0) 应该为 False
        assert isna_f(None)  # 断言 isna_f(None) 应该为 True
        assert isna_f(np.nan)  # 断言 isna_f(np.nan) 应该为 True
        assert float("nan")  # 断言 float("nan") 是否为 NaN
        assert not isna_f(np.inf)  # 断言 isna_f(np.inf) 是否为 False
        assert not isna_f(-np.inf)  # 断言 isna_f(-np.inf) 是否为 False

        # type
        assert not isna_f(type(Series(dtype=object)))  # 断言 isna_f(Series(dtype=object)) 是否为 False
        assert not isna_f(type(Series(dtype=np.float64)))  # 断言 isna_f(Series(dtype=np.float64)) 是否为 False
        assert not isna_f(type(pd.DataFrame()))  # 断言 isna_f(type(pd.DataFrame())) 是否为 False

    @pytest.mark.parametrize("isna_f", [isna, isnull])
    @pytest.mark.parametrize(
        "data",
        [
            np.arange(4, dtype=float),  # 创建一个 NumPy 浮点型数组
            [0.0, 1.0, 0.0, 1.0],  # 创建一个 Python 列表
            Series(list("abcd"), dtype=object),  # 创建一个包含对象类型数据的 Series 对象
            date_range("2020-01-01", periods=4),  # 创建一个日期范围的 Series 对象
        ],
    )
    @pytest.mark.parametrize(
        "index",
        [
            date_range("2020-01-01", periods=4),  # 生成一个日期范围，从 '2020-01-01' 开始，长度为 4
            range(4),  # 创建一个包含 0 到 3 的整数范围
            period_range("2020-01-01", periods=4),  # 生成一个日期周期范围，从 '2020-01-01' 开始，长度为 4
        ],
    )
    def test_isna_isnull_frame(self, isna_f, data, index):
        # 创建一个包含给定数据和索引的 Pandas DataFrame
        df = pd.DataFrame(data, index=index)
        # 调用测试中传入的函数 isna_f 处理 DataFrame，并记录结果
        result = isna_f(df)
        # 使用 DataFrame 的 apply 方法调用 isna_f 处理每一列，并记录期望结果
        expected = df.apply(isna_f)
        # 断言处理结果与期望结果是否相等
        tm.assert_frame_equal(result, expected)

    def test_isna_lists(self):
        # 测试处理包含嵌套列表的情况
        result = isna([[False]])
        exp = np.array([[False]])
        tm.assert_numpy_array_equal(result, exp)

        result = isna([[1], [2]])
        exp = np.array([[False], [False]])
        tm.assert_numpy_array_equal(result, exp)

        # 测试处理包含字符串或 Unicode 字符串的列表
        result = isna(["foo", "bar"])
        exp = np.array([False, False])
        tm.assert_numpy_array_equal(result, exp)

        result = isna(["foo", "bar"])
        exp = np.array([False, False])
        tm.assert_numpy_array_equal(result, exp)

        # GH20675
        # 测试处理包含 NaN 和字符串的列表
        result = isna([np.nan, "world"])
        exp = np.array([True, False])
        tm.assert_numpy_array_equal(result, exp)

    def test_isna_nat(self):
        # 测试处理包含 NaT（Not a Time）的情况
        result = isna([NaT])
        exp = np.array([True])
        tm.assert_numpy_array_equal(result, exp)

        result = isna(np.array([NaT], dtype=object))
        exp = np.array([True])
        tm.assert_numpy_array_equal(result, exp)

    def test_isna_numpy_nat(self):
        # 测试处理包含 numpy 数组中的 NaT（Not a Time）的情况
        arr = np.array(
            [
                NaT,
                np.datetime64("NaT"),
                np.timedelta64("NaT"),
                np.datetime64("NaT", "s"),
            ]
        )
        result = isna(arr)
        expected = np.array([True] * 4)
        tm.assert_numpy_array_equal(result, expected)

    def test_isna_datetime(self):
        # 测试处理日期时间类型的情况
        assert not isna(datetime.now())
        assert notna(datetime.now())

        idx = date_range("1/1/1990", periods=20)
        exp = np.ones(len(idx), dtype=bool)
        tm.assert_numpy_array_equal(notna(idx), exp)

        idx = np.asarray(idx)
        idx[0] = iNaT  # 将第一个索引设置为 iNaT（Invalid NaT）
        idx = DatetimeIndex(idx)
        mask = isna(idx)
        assert mask[0]
        exp = np.array([True] + [False] * (len(idx) - 1), dtype=bool)
        tm.assert_numpy_array_equal(mask, exp)

        # GH 9129
        pidx = idx.to_period(freq="M")
        mask = isna(pidx)
        assert mask[0]
        exp = np.array([True] + [False] * (len(idx) - 1), dtype=bool)
        tm.assert_numpy_array_equal(mask, exp)

        mask = isna(pidx[1:])
        exp = np.zeros(len(mask), dtype=bool)
        tm.assert_numpy_array_equal(mask, exp)
    # 测试 isna_old 方法对于 dt64tz、td64 和 period 的工作是否正常，而不仅限于 tznaive 类型
    def test_isna_old_datetimelike(self):
        # 创建一个日期范围，从 "2016-01-01" 开始，共 3 个时间点
        dti = date_range("2016-01-01", periods=3)
        # 获取日期范围内部的数据
        dta = dti._data
        # 将最后一个时间点置为 NaT（Not a Time，即缺失时间）
        dta[-1] = NaT
        # 预期结果是一个布尔类型的数组，表示每个时间点是否为 NaT
        expected = np.array([False, False, True], dtype=bool)

        # 创建一个包含不同对象的列表，用于测试 isna 方法
        objs = [dta, dta.tz_localize("US/Eastern"), dta - dta, dta.to_period("D")]

        # 对于每个对象进行测试
        for obj in objs:
            # 调用 isna 方法，获取结果
            result = isna(obj)
            # 使用 assert_numpy_array_equal 断言方法比较结果是否与预期相符
            tm.assert_numpy_array_equal(result, expected)

    # 使用 pytest 的参数化装饰器，对多个输入值进行测试
    @pytest.mark.parametrize(
        "value, expected",
        [
            (np.complex128(np.nan), True),
            (np.float64(1), False),
            (np.array([1, 1 + 0j, np.nan, 3]), np.array([False, False, True, False])),
            (
                np.array([1, 1 + 0j, np.nan, 3], dtype=object),
                np.array([False, False, True, False]),
            ),
            (
                np.array([1, 1 + 0j, np.nan, 3]).astype(object),
                np.array([False, False, True, False]),
            ),
        ],
    )
    # 测试复杂数据类型的 isna 方法
    def test_complex(self, value, expected):
        # 调用 isna 方法，获取结果
        result = isna(value)
        # 如果返回值是标量，则使用 assert 断言确保结果与预期相符
        if is_scalar(result):
            assert result is expected
        else:
            # 否则，使用 assert_numpy_array_equal 断言方法比较结果是否与预期相符
            tm.assert_numpy_array_equal(result, expected)

    # 测试 DatetimeIndex 中其他时间单位的 isna 和 notna 方法
    def test_datetime_other_units(self):
        # 创建一个 DatetimeIndex 对象
        idx = DatetimeIndex(["2011-01-01", "NaT", "2011-01-02"])
        # 预期结果是一个布尔类型的数组，表示每个时间点是否为 NaT
        exp = np.array([False, True, False])
        # 使用 assert_numpy_array_equal 断言方法比较 isna 方法的结果是否与预期相符
        tm.assert_numpy_array_equal(isna(idx), exp)
        # 使用 assert_numpy_array_equal 断言方法比较 notna 方法的结果是否与预期相符
        tm.assert_numpy_array_equal(notna(idx), ~exp)
        # 使用 assert_numpy_array_equal 断言方法比较 isna 方法对值数组的结果是否与预期相符
        tm.assert_numpy_array_equal(isna(idx.values), exp)
        # 使用 assert_numpy_array_equal 断言方法比较 notna 方法对值数组的结果是否与预期相符
        tm.assert_numpy_array_equal(notna(idx.values), ~exp)

    # 使用 pytest 的参数化装饰器，测试不同的 datetime 单位转换为指定 dtype 后的 isna 和 notna 方法
    @pytest.mark.parametrize(
        "dtype",
        [
            "datetime64[D]",
            "datetime64[h]",
            "datetime64[m]",
            "datetime64[s]",
            "datetime64[ms]",
            "datetime64[us]",
            "datetime64[ns]",
        ],
    )
    # 测试 DatetimeIndex 中其他时间单位转换后的 isna 和 notna 方法
    def test_datetime_other_units_astype(self, dtype):
        # 创建一个 DatetimeIndex 对象
        idx = DatetimeIndex(["2011-01-01", "NaT", "2011-01-02"])
        # 将该对象的值转换为指定的 dtype
        values = idx.values.astype(dtype)

        # 预期结果是一个布尔类型的数组，表示每个时间点是否为 NaT
        exp = np.array([False, True, False])
        # 使用 assert_numpy_array_equal 断言方法比较 isna 方法的结果是否与预期相符
        tm.assert_numpy_array_equal(isna(values), exp)
        # 使用 assert_numpy_array_equal 断言方法比较 notna 方法的结果是否与预期相符
        tm.assert_numpy_array_equal(notna(values), ~exp)

        # 创建一个 Series 对象，使用该对象的 isna 和 notna 方法进行测试
        exp = Series([False, True, False])
        s = Series(values)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

        # 将 Series 对象的 dtype 设置为 object 类型，再次使用 isna 和 notna 方法进行测试
        s = Series(values, dtype=object)
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

    # 测试 TimedeltaIndex 中其他时间单位的 isna 和 notna 方法
    def test_timedelta_other_units(self):
        # 创建一个 TimedeltaIndex 对象
        idx = TimedeltaIndex(["1 days", "NaT", "2 days"])
        # 预期结果是一个布尔类型的数组，表示每个时间点是否为 NaT
        exp = np.array([False, True, False])
        # 使用 assert_numpy_array_equal 断言方法比较 isna 方法的结果是否与预期相符
        tm.assert_numpy_array_equal(isna(idx), exp)
        # 使用 assert_numpy_array_equal 断言方法比较 notna 方法的结果是否与预期相符
        tm.assert_numpy_array_equal(notna(idx), ~exp)
        # 使用 assert_numpy_array_equal 断言方法比较 isna 方法对值数组的结果是否与预期相符
        tm.assert_numpy_array_equal(isna(idx.values), exp)
        # 使用 assert_numpy_array_equal 断言方法比较 notna 方法对值数组的结果是否与预期相符
        tm.assert_numpy_array_equal(notna(idx.values), ~exp)
    # 使用 pytest 的参数化装饰器，为测试方法提供多个数据类型作为参数
    @pytest.mark.parametrize(
        "dtype",
        [
            "timedelta64[D]",  # 以天为单位的时间增量数据类型
            "timedelta64[h]",  # 以小时为单位的时间增量数据类型
            "timedelta64[m]",  # 以分钟为单位的时间增量数据类型
            "timedelta64[s]",  # 以秒为单位的时间增量数据类型
            "timedelta64[ms]",  # 以毫秒为单位的时间增量数据类型
            "timedelta64[us]",  # 以微秒为单位的时间增量数据类型
            "timedelta64[ns]",  # 以纳秒为单位的时间增量数据类型
        ],
    )
    # 测试不同单位时间增量数据类型的方法
    def test_timedelta_other_units_dtype(self, dtype):
        # 创建时间增量索引对象，包含三个时间字符串和一个 NaT（Not a Time）值
        idx = TimedeltaIndex(["1 days", "NaT", "2 days"])
        # 将时间增量索引的值转换为指定数据类型的 numpy 数组
        values = idx.values.astype(dtype)

        # 期望的布尔数组结果
        exp = np.array([False, True, False])
        # 检查 numpy 数组是否与期望结果相等
        tm.assert_numpy_array_equal(isna(values), exp)
        tm.assert_numpy_array_equal(notna(values), ~exp)

        # 期望的布尔 Series 结果
        exp = Series([False, True, False])
        # 将 numpy 数组转换为 Series 对象
        s = Series(values)
        # 检查 Series 是否与期望结果相等
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)
        # 将 Series 对象的数据类型设置为对象类型
        s = Series(values, dtype=object)
        # 再次检查 Series 是否与期望结果相等
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

    # 测试 Period 类型数据的方法
    def test_period(self):
        # 创建 PeriodIndex 对象，包含两个时间字符串和一个 NaT（Not a Time）值，频率为每月
        idx = pd.PeriodIndex(["2011-01", "NaT", "2012-01"], freq="M")
        # 期望的布尔数组结果
        exp = np.array([False, True, False])
        # 检查 numpy 数组是否与期望结果相等
        tm.assert_numpy_array_equal(isna(idx), exp)
        tm.assert_numpy_array_equal(notna(idx), ~exp)

        # 期望的布尔 Series 结果
        exp = Series([False, True, False])
        # 将 PeriodIndex 对象转换为 Series 对象
        s = Series(idx)
        # 检查 Series 是否与期望结果相等
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)
        # 将 Series 对象的数据类型设置为对象类型
        s = Series(idx, dtype=object)
        # 再次检查 Series 是否与期望结果相等
        tm.assert_series_equal(isna(s), exp)
        tm.assert_series_equal(notna(s), ~exp)

    # 测试 Decimal 类型数据的方法
    def test_decimal(self):
        # 测试标量值 GH#23530
        a = Decimal(1.0)
        # 断言标量值是否不是 NaN
        assert isna(a) is False
        assert notna(a) is True

        b = Decimal("NaN")
        # 断言标量值是否是 NaN
        assert isna(b) is True
        assert notna(b) is False

        # 数组
        arr = np.array([a, b])
        expected = np.array([False, True])
        # 检查 numpy 数组是否与期望结果相等
        result = isna(arr)
        tm.assert_numpy_array_equal(result, expected)

        result = notna(arr)
        tm.assert_numpy_array_equal(result, ~expected)

        # Series
        ser = Series(arr)
        expected = Series(expected)
        # 检查 Series 是否与期望结果相等
        result = isna(ser)
        tm.assert_series_equal(result, expected)

        result = notna(ser)
        tm.assert_series_equal(result, ~expected)

        # 索引
        idx = Index(arr)
        expected = np.array([False, True])
        # 检查 numpy 数组是否与期望结果相等
        result = isna(idx)
        tm.assert_numpy_array_equal(result, expected)

        result = notna(idx)
        tm.assert_numpy_array_equal(result, ~expected)
# 使用 pytest 提供的 @pytest.mark.parametrize 装饰器，为 test_array_equivalent 函数添加参数化测试
@pytest.mark.parametrize("dtype_equal", [True, False])
def test_array_equivalent(dtype_equal):
    # 断言两个包含 NaN 的 numpy 数组在使用 array_equivalent 函数比较时应该相等
    assert array_equivalent(
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]), dtype_equal=dtype_equal
    )
    # 断言两个包含 NaN 和整数的 numpy 数组在使用 array_equivalent 函数比较时应该相等
    assert array_equivalent(
        np.array([np.nan, 1, np.nan]),
        np.array([np.nan, 1, np.nan]),
        dtype_equal=dtype_equal,
    )
    # 断言两个包含 NaN 和 None 的对象类型 numpy 数组在使用 array_equivalent 函数比较时应该相等
    assert array_equivalent(
        np.array([np.nan, None], dtype="object"),
        np.array([np.nan, None], dtype="object"),
        dtype_equal=dtype_equal,
    )
    # 检查 array_equivalent_object 函数对嵌套数组的处理
    assert array_equivalent(
        np.array([np.array([np.nan, None], dtype="object"), None], dtype="object"),
        np.array([np.array([np.nan, None], dtype="object"), None], dtype="object"),
        dtype_equal=dtype_equal,
    )
    # 断言两个包含 NaN 和复数的 numpy 数组在使用 array_equivalent 函数比较时应该相等
    assert array_equivalent(
        np.array([np.nan, 1 + 1j], dtype="complex"),
        np.array([np.nan, 1 + 1j], dtype="complex"),
        dtype_equal=dtype_equal,
    )
    # 断言两个复数数组在存在差异时，使用 array_equivalent 函数比较应该不相等
    assert not array_equivalent(
        np.array([np.nan, 1 + 1j], dtype="complex"),
        np.array([np.nan, 1 + 2j], dtype="complex"),
        dtype_equal=dtype_equal,
    )
    # 断言两个包含 NaN 和整数的 numpy 数组在存在差异时，使用 array_equivalent 函数比较应该不相等
    assert not array_equivalent(
        np.array([np.nan, 1, np.nan]),
        np.array([np.nan, 2, np.nan]),
        dtype_equal=dtype_equal,
    )
    # 断言两个不相同长度的字符串数组在使用 array_equivalent 函数比较时应该不相等
    assert not array_equivalent(
        np.array(["a", "b", "c", "d"]), np.array(["e", "e"]), dtype_equal=dtype_equal
    )
    # 断言两个包含整数和 NaN 的 Index 对象在使用 array_equivalent 函数比较时应该相等
    assert array_equivalent(
        Index([0, np.nan]), Index([0, np.nan]), dtype_equal=dtype_equal
    )
    # 断言两个不相同的 Index 对象在使用 array_equivalent 函数比较时应该不相等
    assert not array_equivalent(
        Index([0, np.nan]), Index([1, np.nan]), dtype_equal=dtype_equal
    )


# 使用 pytest 提供的 @pytest.mark.parametrize 装饰器，为 test_array_equivalent_tdi 函数添加参数化测试
@pytest.mark.parametrize("dtype_equal", [True, False])
def test_array_equivalent_tdi(dtype_equal):
    # 断言两个 TimedeltaIndex 对象在使用 array_equivalent 函数比较时应该相等
    assert array_equivalent(
        TimedeltaIndex([0, np.nan]),
        TimedeltaIndex([0, np.nan]),
        dtype_equal=dtype_equal,
    )
    # 断言两个 TimedeltaIndex 对象在存在差异时，使用 array_equivalent 函数比较应该不相等
    assert not array_equivalent(
        TimedeltaIndex([0, np.nan]),
        TimedeltaIndex([1, np.nan]),
        dtype_equal=dtype_equal,
    )


# 使用 pytest 提供的 @pytest.mark.parametrize 装饰器，为 test_array_equivalent_dti 函数添加参数化测试
@pytest.mark.parametrize("dtype_equal", [True, False])
def test_array_equivalent_dti(dtype_equal):
    # 断言两个 DatetimeIndex 对象在使用 array_equivalent 函数比较时应该相等
    assert array_equivalent(
        DatetimeIndex([0, np.nan]), DatetimeIndex([0, np.nan]), dtype_equal=dtype_equal
    )
    # 断言两个 DatetimeIndex 对象在存在差异时，使用 array_equivalent 函数比较应该不相等
    assert not array_equivalent(
        DatetimeIndex([0, np.nan]), DatetimeIndex([1, np.nan]), dtype_equal=dtype_equal
    )

    # 创建带有时区信息的 DatetimeIndex 对象
    dti1 = DatetimeIndex([0, np.nan], tz="US/Eastern")
    dti2 = DatetimeIndex([0, np.nan], tz="CET")
    dti3 = DatetimeIndex([1, np.nan], tz="US/Eastern")

    # 断言两个具有相同时区信息的 DatetimeIndex 对象在使用 array_equivalent 函数比较时应该相等
    assert array_equivalent(
        dti1,
        dti1,
        dtype_equal=dtype_equal,
    )
    # 断言具有不同值的 DatetimeIndex 对象在使用 array_equivalent 函数比较时应该不相等
    assert not array_equivalent(
        dti1,
        dti3,
        dtype_equal=dtype_equal,
    )
    # 断言未指定 dtype_equal 参数的情况下，两个 DatetimeIndex 对象在使用 array_equivalent 函数比较时应该不相等
    assert not array_equivalent(DatetimeIndex([0, np.nan]), dti1)
    # 断言具有不同时区信息的 DatetimeIndex 对象在使用 array_equivalent 函数比较时应该不相等
    assert array_equivalent(
        dti2,
        dti1,
    )


这里是对每个测试函数和其内部的断言语句进行了详细的注释，描述了它们的作用和预期的比较结果。
    # 断言：确保 DatetimeIndex([0, np.nan]) 和 TimedeltaIndex([0, np.nan]) 不是等价的
    assert not array_equivalent(DatetimeIndex([0, np.nan]), TimedeltaIndex([0, np.nan]))
# 使用 pytest.mark.parametrize 装饰器，为 test_array_equivalent_series 函数定义多组参数化测试数据
@pytest.mark.parametrize(
    "val", [1, 1.1, 1 + 1j, True, "abc", [1, 2], (1, 2), {1, 2}, {"a": 1}, None]
)
# 定义测试函数 test_array_equivalent_series，验证 array_equivalent 函数的行为
def test_array_equivalent_series(val):
    # 创建包含整数数组 [1, 2] 的 NumPy 数组
    arr = np.array([1, 2])
    # 定义错误消息字符串
    msg = "elementwise comparison failed"
    # 创建上下文管理器 cm，根据 val 是否为字符串和 np_version_gte1p25 的值决定是否生成 FutureWarning 警告
    cm = (
        tm.assert_produces_warning(FutureWarning, match=msg, check_stacklevel=False)
        if isinstance(val, str) and not np_version_gte1p25
        else nullcontext()
    )
    # 进入上下文管理器 cm
    with cm:
        # 断言 Series([arr, arr]) 与 Series([arr, val]) 不等效
        assert not array_equivalent(Series([arr, arr]), Series([arr, val]))


# 定义测试函数 test_array_equivalent_array_mismatched_shape，验证 array_equivalent 函数处理数组形状不匹配的情况
def test_array_equivalent_array_mismatched_shape():
    # 创建不同长度的 NumPy 数组，以触发错误的 bug
    first = np.array([1, 2, 3])
    second = np.array([1, 2])

    # 创建元素类型为对象的 Series 对象 left 和 right
    left = Series([first, "a"], dtype=object)
    right = Series([second, "a"], dtype=object)
    # 断言 left 和 right 不等效
    assert not array_equivalent(left, right)


# 定义测试函数 test_array_equivalent_array_mismatched_dtype，验证 array_equivalent 函数处理数据类型不匹配但值相等的情况
def test_array_equivalent_array_mismatched_dtype():
    # 创建具有不同数据类型的 NumPy 数组，但值相同
    first = np.array([1, 2], dtype=np.float64)
    second = np.array([1, 2])

    # 创建元素类型为对象的 Series 对象 left 和 right
    left = Series([first, "a"], dtype=object)
    right = Series([second, "a"], dtype=object)
    # 断言 left 和 right 等效
    assert array_equivalent(left, right)


# 定义测试函数 test_array_equivalent_different_dtype_but_equal，验证 array_equivalent 函数处理数据类型不同但值相等的情况
def test_array_equivalent_different_dtype_but_equal():
    # 断言 NumPy 数组 np.array([1, 2]) 和 np.array([1.0, 2.0]) 等效
    assert array_equivalent(np.array([1, 2]), np.array([1.0, 2.0]))


# 使用 pytest.mark.parametrize 装饰器，为 test_array_equivalent_tzawareness 函数定义多组参数化测试数据
@pytest.mark.parametrize(
    "lvalue, rvalue",
    [
        # 包含多个 lvalue 和 rvalue 变体，用于测试时区感知的情况
        (fix_now, fix_utcnow),
        (fix_now.to_datetime64(), fix_utcnow),
        (fix_now.to_pydatetime(), fix_utcnow),
        (fix_now.to_datetime64(), fix_utcnow.to_pydatetime()),
        (fix_now.to_pydatetime(), fix_utcnow.to_pydatetime()),
    ],
)
# 定义测试函数 test_array_equivalent_tzawareness，验证 array_equivalent 函数处理时区感知日期时间的情况
def test_array_equivalent_tzawareness(lvalue, rvalue):
    # 创建元素类型为对象的 NumPy 数组 left 和 right，分别包含 lvalue 和 rvalue
    left = np.array([lvalue], dtype=object)
    right = np.array([rvalue], dtype=object)

    # 断言 left 和 right 在严格 NaN 模式下不等效
    assert not array_equivalent(left, right, strict_nan=True)
    # 断言 left 和 right 在非严格 NaN 模式下不等效
    assert not array_equivalent(left, right, strict_nan=False)


# 定义测试函数 test_array_equivalent_compat，验证 array_equivalent 函数的兼容性
def test_array_equivalent_compat():
    # 创建结构化 NumPy 数组 m 和 n，用于测试严格和非严格 NaN 模式下的等效性
    m = np.array([(1, 2), (3, 4)], dtype=[("a", int), ("b", float)])
    n = np.array([(1, 2), (3, 4)], dtype=[("a", int), ("b", float)])
    # 断言 m 和 n 在严格 NaN 模式下等效
    assert array_equivalent(m, n, strict_nan=True)
    # 断言 m 和 n 在非严格 NaN 模式下等效
    assert array_equivalent(m, n, strict_nan=False)

    # 创建结构化 NumPy 数组 m 和 n，测试它们在严格和非严格 NaN 模式下的不等效性
    m = np.array([(1, 2), (3, 4)], dtype=[("a", int), ("b", float)])
    n = np.array([(1, 2), (4, 3)], dtype=[("a", int), ("b", float)])
    # 断言 m 和 n 在严格 NaN 模式下不等效
    assert not array_equivalent(m, n, strict_nan=True)
    # 断言 m 和 n 在非严格 NaN 模式下不等效
    assert not array_equivalent(m, n, strict_nan=False)

    # 创建结构化 NumPy 数组 m 和 n，测试它们在严格和非严格 NaN 模式下的不等效性
    m = np.array([(1, 2), (3, 4)], dtype=[("a", int), ("b", float)])
    n = np.array([(1, 2), (3, 4)], dtype=[("b", int), ("a", float)])
    # 使用 array_equivalent 函数比较 m 和 n 两个数组是否在严格 NaN 模式下等价
    assert not array_equivalent(m, n, strict_nan=True)
    
    # 使用 array_equivalent 函数比较 m 和 n 两个数组是否在非严格 NaN 模式下等价
    assert not array_equivalent(m, n, strict_nan=False)
@pytest.mark.parametrize("dtype", ["O", "S", "U"])
# 定义测试函数 test_array_equivalent_str，使用参数化测试数据类型为 "O", "S", "U"
def test_array_equivalent_str(dtype):
    # 断言两个相同类型的数组是否等价
    assert array_equivalent(
        np.array(["A", "B"], dtype=dtype), np.array(["A", "B"], dtype=dtype)
    )
    # 断言两个不同的数组是否不等价
    assert not array_equivalent(
        np.array(["A", "B"], dtype=dtype), np.array(["A", "X"], dtype=dtype)
    )


@pytest.mark.parametrize("strict_nan", [True, False])
# 定义测试函数 test_array_equivalent_nested，使用参数化测试数据 strict_nan 为 True 和 False
def test_array_equivalent_nested(strict_nan):
    # 对于嵌套数组的测试，确保在组合聚合中使用 np.any 进行比较时返回真值
    left = np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object)
    right = np.array([np.array([50, 70, 90]), np.array([20, 30])], dtype=object)

    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    left = np.empty(2, dtype=object)
    left[:] = [np.array([50, 70, 90]), np.array([20, 30, 40])]
    right = np.empty(2, dtype=object)
    right[:] = [np.array([50, 70, 90]), np.array([20, 30, 40])]
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    left = np.array([np.array([50, 50, 50]), np.array([40, 40])], dtype=object)
    right = np.array([50, 40])
    assert not array_equivalent(left, right, strict_nan=strict_nan)


@pytest.mark.filterwarnings("ignore:elementwise comparison failed:DeprecationWarning")
@pytest.mark.parametrize("strict_nan", [True, False])
# 定义测试函数 test_array_equivalent_nested2，使用参数化测试数据 strict_nan 为 True 和 False
def test_array_equivalent_nested2(strict_nan):
    # 测试多层嵌套的数组是否等价
    left = np.array(
        [
            np.array([np.array([50, 70]), np.array([90])], dtype=object),
            np.array([np.array([20, 30])], dtype=object),
        ],
        dtype=object,
    )
    right = np.array(
        [
            np.array([np.array([50, 70]), np.array([90])], dtype=object),
            np.array([np.array([20, 30])], dtype=object),
        ],
        dtype=object,
    )
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    left = np.array([np.array([np.array([50, 50, 50])], dtype=object)], dtype=object)
    right = np.array([50])
    assert not array_equivalent(left, right, strict_nan=strict_nan)


@pytest.mark.parametrize("strict_nan", [True, False])
# 定义测试函数 test_array_equivalent_nested_list，使用参数化测试数据 strict_nan 为 True 和 False
def test_array_equivalent_nested_list(strict_nan):
    # 测试嵌套列表的数组是否等价
    left = np.array([[50, 70, 90], [20, 30]], dtype=object)
    right = np.array([[50, 70, 90], [20, 30]], dtype=object)

    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    left = np.array([[50, 50, 50], [40, 40]], dtype=object)
    right = np.array([50, 40])
    assert not array_equivalent(left, right, strict_nan=strict_nan)


@pytest.mark.filterwarnings("ignore:elementwise comparison failed:DeprecationWarning")
@pytest.mark.xfail(reason="failing")
# 标记该测试为预期失败，原因是 "failing"
# 使用 pytest 的参数化装饰器，指定 strict_nan 参数为 True 和 False 两种情况
@pytest.mark.parametrize("strict_nan", [True, False])
def test_array_equivalent_nested_mixed_list(strict_nan):
    # 创建一个包含不同类型的数组和列表的 numpy 数组 left 和 right
    # 这里涉及的 GitHub 问题链接为 https://github.com/pandas-dev/pandas/issues/50360
    left = np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object)
    right = np.array([[1, 2, 3], [4, 5]], dtype=object)

    # 断言 left 和 right 在 strict_nan 模式下是等价的
    assert array_equivalent(left, right, strict_nan=strict_nan)
    # 断言 left 和 right 逆序时不等价
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    # 多层嵌套的例子
    left = np.array(
        [
            np.array([np.array([1, 2, 3]), np.array([4, 5])], dtype=object),
            np.array([np.array([6]), np.array([7, 8]), np.array([9])], dtype=object),
        ],
        dtype=object,
    )
    right = np.array([[[1, 2, 3], [4, 5]], [[6], [7, 8], [9]]], dtype=object)
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    # 长度相同的列表
    subarr = np.empty(2, dtype=object)
    subarr[:] = [
        np.array([None, "b"], dtype=object),
        np.array(["c", "d"], dtype=object),
    ]
    left = np.array([subarr, None], dtype=object)
    right = np.array([[[None, "b"], ["c", "d"]], None], dtype=object)
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)


# 标记为预期失败的测试用例
@pytest.mark.xfail(reason="failing")
@pytest.mark.parametrize("strict_nan", [True, False])
def test_array_equivalent_nested_dicts(strict_nan):
    # 创建包含嵌套字典的 numpy 数组 left 和 right
    left = np.array([{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object)
    right = np.array(
        [{"f1": 1, "f2": np.array(["a", "b"], dtype=object)}], dtype=object
    )
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)

    # 修改 right2 以测试不相等的情况
    right2 = np.array([{"f1": 1, "f2": ["a", "b"]}], dtype=object)
    assert array_equivalent(left, right2, strict_nan=strict_nan)
    assert not array_equivalent(left, right2[::-1], strict_nan=strict_nan)


def test_array_equivalent_index_with_tuples():
    # GH#48446 测试 Index 类与包含元组的 numpy 数组的相等性
    idx1 = Index(np.array([(pd.NA, 4), (1, 1)], dtype="object"))
    idx2 = Index(np.array([(1, 1), (pd.NA, 4)], dtype="object"))
    assert not array_equivalent(idx1, idx2)
    assert not idx1.equals(idx2)
    assert not array_equivalent(idx2, idx1)
    assert not idx2.equals(idx1)

    idx1 = Index(np.array([(4, pd.NA), (1, 1)], dtype="object"))
    idx2 = Index(np.array([(1, 1), (4, pd.NA)], dtype="object"))
    assert not array_equivalent(idx1, idx2)
    assert not idx1.equals(idx2)
    assert not array_equivalent(idx2, idx1)
    assert not idx2.equals(idx1)


@pytest.mark.parametrize(
    "dtype, na_value",
    [
        # Datetime-like 数据类型为日期时间
        (np.dtype("M8[ns]"), np.datetime64("NaT", "ns")),  # 初始化为 NaT 的日期时间数据类型
        (np.dtype("m8[ns]"), np.timedelta64("NaT", "ns")),  # 初始化为 NaT 的时间间隔数据类型
        (DatetimeTZDtype.construct_from_string("datetime64[ns, US/Eastern]"), NaT),  # 初始化为 NaT 的美东时区日期时间数据类型
        (PeriodDtype("M"), NaT),  # 初始化为 NaT 的周期数据类型
        # Integer 整数数据类型
        ("u1", 0),  # 无符号 1 字节整数，初始值为 0
        ("u2", 0),  # 无符号 2 字节整数，初始值为 0
        ("u4", 0),  # 无符号 4 字节整数，初始值为 0
        ("u8", 0),  # 无符号 8 字节整数，初始值为 0
        ("i1", 0),  # 有符号 1 字节整数，初始值为 0
        ("i2", 0),  # 有符号 2 字节整数，初始值为 0
        ("i4", 0),  # 有符号 4 字节整数，初始值为 0
        ("i8", 0),  # 有符号 8 字节整数，初始值为 0
        # Bool 布尔值数据类型
        ("bool", False),  # 布尔类型，初始值为 False
        # Float 浮点数数据类型
        ("f2", np.nan),  # 2 字节浮点数，初始值为 NaN
        ("f4", np.nan),  # 4 字节浮点数，初始值为 NaN
        ("f8", np.nan),  # 8 字节浮点数，初始值为 NaN
        # Complex 复数数据类型
        ("c8", np.nan),  # 8 字节复数，初始值为 NaN
        ("c16", np.nan),  # 16 字节复数，初始值为 NaN
        # Object 对象数据类型
        ("O", np.nan),  # 对象类型，初始值为 NaN
        # Interval 区间数据类型
        (IntervalDtype(), np.nan),  # 区间类型，初始值为 NaN
    ],
)
def test_na_value_for_dtype(dtype, na_value):
    # 调用函数 na_value_for_dtype 处理给定的 dtype，返回处理结果
    result = na_value_for_dtype(pandas_dtype(dtype))
    # 检查处理结果是否与 na_value 相同，或者是空值且类型相同
    # 对于 datetime64/timedelta64("NaT") 无法使用身份检查，因为它们不是单例对象
    assert result is na_value or (
        isna(result) and isna(na_value) and type(result) is type(na_value)
    )


class TestNAObj:
    def _check_behavior(self, arr, expected):
        # 调用 libmissing.isnaobj 函数检查数组 arr 的缺失值情况，并与期望值 expected 比较
        result = libmissing.isnaobj(arr)
        tm.assert_numpy_array_equal(result, expected)

        # 将 arr 和 expected 至少转换为二维数组
        arr = np.atleast_2d(arr)
        expected = np.atleast_2d(expected)

        # 再次检查转换后的数组的缺失值情况
        result = libmissing.isnaobj(arr)
        tm.assert_numpy_array_equal(result, expected)

        # 测试 Fortran 顺序下的数组缺失值情况
        arr = arr.copy(order="F")
        result = libmissing.isnaobj(arr)
        tm.assert_numpy_array_equal(result, expected)

    def test_basic(self):
        # 创建包含各种数据类型和缺失值的 numpy 数组 arr
        arr = np.array([1, None, "foo", -5.1, NaT, np.nan])
        # 创建预期的布尔类型的 numpy 数组，表示 arr 中每个元素是否为缺失值
        expected = np.array([False, True, False, False, True, True])

        # 调用 _check_behavior 方法检查数组 arr 的缺失值情况是否符合预期
        self._check_behavior(arr, expected)

    def test_non_obj_dtype(self):
        # 创建包含非对象类型的 numpy 数组 arr
        arr = np.array([1, 3, np.nan, 5], dtype=float)
        # 创建预期的布尔类型的 numpy 数组，表示 arr 中每个元素是否为缺失值
        expected = np.array([False, False, True, False])

        # 调用 _check_behavior 方法检查数组 arr 的缺失值情况是否符合预期
        self._check_behavior(arr, expected)

    def test_empty_arr(self):
        # 创建空的 numpy 数组 arr
        arr = np.array([])
        # 创建空的布尔类型的 numpy 数组 expected
        expected = np.array([], dtype=bool)

        # 调用 _check_behavior 方法检查数组 arr 的缺失值情况是否符合预期
        self._check_behavior(arr, expected)

    def test_empty_str_inp(self):
        # 创建包含空字符串的 numpy 数组 arr
        arr = np.array([""])  # empty but not na
        # 创建预期的布尔类型的 numpy 数组，表示 arr 中每个元素是否为缺失值
        expected = np.array([False])

        # 调用 _check_behavior 方法检查数组 arr 的缺失值情况是否符合预期
        self._check_behavior(arr, expected)

    def test_empty_like(self):
        # 创建与给定数组形状相同但元素未初始化的 numpy 数组 arr
        # 参考 gh-13717: 不会发生段错误！
        arr = np.empty_like([None])
        # 创建预期的布尔类型的 numpy 数组，表示 arr 中每个元素是否为缺失值
        expected = np.array([True])

        # 调用 _check_behavior 方法检查数组 arr 的缺失值情况是否符合预期
        self._check_behavior(arr, expected)


m8_units = ["as", "ps", "ns", "us", "ms", "s", "m", "h", "D", "W", "M", "Y"]

na_vals = (
    [
        None,
        NaT,
        float("NaN"),
        complex("NaN"),
        np.nan,
        np.float64("NaN"),
        np.float32("NaN"),
        np.complex64(np.nan),
        np.complex128(np.nan),
        np.datetime64("NaT"),
        np.timedelta64("NaT"),
    ]
    + [np.datetime64("NaT", unit) for unit in m8_units]
    + [np.timedelta64("NaT", unit) for unit in m8_units]
)

inf_vals = [
    float("inf"),
    float("-inf"),
    complex("inf"),
    complex("-inf"),
    np.inf,
    -np.inf,
]

int_na_vals = [
    # 匹配 iNaT 的值，在特定情况下我们将其视为空值
    np.int64(NaT._value),
    int(NaT._value),
]

sometimes_na_vals = [Decimal("NaN")]

never_na_vals = [
    # 当视为 int64 时与 iNaT 匹配的 float/complex 值
    -0.0,
    np.float64("-0.0"),
    -0j,
    np.complex64(-0j),
]


class TestLibMissing:
    @pytest.mark.parametrize("func", [libmissing.checknull, isna])
    @pytest.mark.parametrize(
        "value",
        na_vals + sometimes_na_vals,  # type: ignore[operator]
    )
    def test_checknull_na_vals(self, func, value):
        # 断言 func(value) 返回 True，即 value 被识别为缺失值
        assert func(value)

    @pytest.mark.parametrize("func", [libmissing.checknull, isna])
    # 使用 pytest 的参数化功能，对 inf_vals 列表中的每个值进行测试
    @pytest.mark.parametrize("value", inf_vals)
    def test_checknull_inf_vals(self, func, value):
        # 断言 func(value) 的返回值为 False
        assert not func(value)
    
    # 使用 pytest 的参数化功能，对 int_na_vals 列表中的每个值进行测试
    @pytest.mark.parametrize("func", [libmissing.checknull, isna])
    @pytest.mark.parametrize("value", int_na_vals)
    def test_checknull_intna_vals(self, func, value):
        # 断言 func(value) 的返回值为 False
        assert not func(value)
    
    # 使用 pytest 的参数化功能，对 never_na_vals 列表中的每个值进行测试
    @pytest.mark.parametrize("func", [libmissing.checknull, isna])
    @pytest.mark.parametrize("value", never_na_vals)
    def test_checknull_never_na_vals(self, func, value):
        # 断言 func(value) 的返回值为 False
        assert not func(value)
    
    # 使用 pytest 的参数化功能，对 na_vals + sometimes_na_vals 列表中的每个值进行测试
    @pytest.mark.parametrize(
        "value",
        na_vals + sometimes_na_vals,  # type: ignore[operator]
    )
    def test_checknull_old_na_vals(self, value):
        # 断言 libmissing.checknull(value) 的返回值为 True
        assert libmissing.checknull(value)
    
    # 使用 pytest 的参数化功能，对 int_na_vals 列表中的每个值进行测试
    @pytest.mark.parametrize("value", int_na_vals)
    def test_checknull_old_intna_vals(self, value):
        # 断言 libmissing.checknull(value) 的返回值为 False
        assert not libmissing.checknull(value)
    
    # 测试 libmissing.is_matching_na 函数，使用 nulls_fixture 和 nulls_fixture2 作为参数
    def test_is_matching_na(self, nulls_fixture, nulls_fixture2):
        left = nulls_fixture
        right = nulls_fixture2
    
        # 断言 libmissing.is_matching_na(left, left) 的返回值为 True
        assert libmissing.is_matching_na(left, left)
    
        # 根据不同情况进行断言 libmissing.is_matching_na(left, right) 的返回值
        if left is right:
            assert libmissing.is_matching_na(left, right)
        elif is_float(left) and is_float(right):
            # 对于 np.nan 和 float("NaN") 视为匹配
            assert libmissing.is_matching_na(left, right)
        elif type(left) is type(right):
            # 对于类型相同的情况（如 Decimal("NaN")）
            assert libmissing.is_matching_na(left, right)
        else:
            # 其他情况视为不匹配
            assert not libmissing.is_matching_na(left, right)
    
    # 测试 libmissing.is_matching_na 函数，特别测试 np.nan 和 None 的匹配情况
    def test_is_matching_na_nan_matches_none(self):
        # 断言 libmissing.is_matching_na(None, np.nan) 的返回值为 False
        assert not libmissing.is_matching_na(None, np.nan)
        # 断言 libmissing.is_matching_na(np.nan, None) 的返回值为 False
        assert not libmissing.is_matching_na(np.nan, None)
    
        # 断言 libmissing.is_matching_na(None, np.nan, nan_matches_none=True) 的返回值为 True
        assert libmissing.is_matching_na(None, np.nan, nan_matches_none=True)
        # 断言 libmissing.is_matching_na(np.nan, None, nan_matches_none=True) 的返回值为 True
        assert libmissing.is_matching_na(np.nan, None, nan_matches_none=True)
class TestIsValidNAForDtype:
    # 测试是否对于特定数据类型有效的 NA 值（缺失值）

    def test_is_valid_na_for_dtype_interval(self):
        # 测试区间数据类型的有效 NA 值判断
        dtype = IntervalDtype("int64", "left")
        # 断言 NaT（Not a Time）对于指定的区间数据类型不是有效的 NA 值
        assert not is_valid_na_for_dtype(NaT, dtype)

        dtype = IntervalDtype("datetime64[ns]", "both")
        # 断言 NaT 对于指定的日期时间区间数据类型不是有效的 NA 值
        assert not is_valid_na_for_dtype(NaT, dtype)

    def test_is_valid_na_for_dtype_categorical(self):
        # 测试分类数据类型的有效 NA 值判断
        dtype = CategoricalDtype(categories=[0, 1, 2])
        # 断言 np.nan 对于指定的分类数据类型是有效的 NA 值
        assert is_valid_na_for_dtype(np.nan, dtype)

        # 对于分类数据类型，断言以下情况不是有效的 NA 值
        assert not is_valid_na_for_dtype(NaT, dtype)
        assert not is_valid_na_for_dtype(np.datetime64("NaT", "ns"), dtype)
        assert not is_valid_na_for_dtype(np.timedelta64("NaT", "ns"), dtype)
```