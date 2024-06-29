# `D:\src\scipysrc\pandas\pandas\tests\arithmetic\test_datetime64.py`

```
# 引入需要的库和模块

from datetime import (
    datetime,    # 导入 datetime 类
    time,        # 导入 time 类
    timedelta,   # 导入 timedelta 类
    timezone,    # 导入 timezone 类
)
from itertools import (
    product,     # 导入 product 函数
    starmap,     # 导入 starmap 函数
)
import operator   # 导入 operator 模块

import numpy as np   # 导入 numpy 库
import pytest        # 导入 pytest 测试框架

# 导入 pandas 库及其子模块和函数
import pandas as pd
from pandas import (
    DateOffset,
    DatetimeIndex,
    NaT,
    Period,
    Series,
    Timedelta,
    TimedeltaIndex,
    Timestamp,
    date_range,
)

import pandas._testing as tm   # 导入 pandas 测试工具模块
from pandas.core import roperator   # 导入 pandas 核心 roperator 模块
from pandas.tests.arithmetic.common import (   # 导入通用算术测试函数
    assert_cannot_add,
    assert_invalid_addsub_type,
    assert_invalid_comparison,
    get_upcast_box,
)

# ------------------------------------------------------------------
# 比较测试

class TestDatetime64ArrayLikeComparisons:
    # datetime64 向量的比较测试，完全参数化，涵盖 DataFrame/Series/DatetimeIndex/DatetimeArray
    
    def test_compare_zerodim(self, tz_naive_fixture, box_with_array):
        # 测试与零维数组的比较，确保解除装箱
        tz = tz_naive_fixture   # 获取时区
        box = box_with_array    # 获取装箱后的数组
        dti = date_range("20130101", periods=3, tz=tz)   # 创建具有时区的日期范围

        other = np.array(dti.to_numpy()[0])   # 使用 dti 的第一个元素创建 NumPy 数组

        dtarr = tm.box_expected(dti, box)   # 获得预期的装箱结果
        xbox = get_upcast_box(dtarr, other, True)   # 获取升级后的装箱盒子
        result = dtarr <= other   # 执行比较操作
        expected = np.array([True, False, False])   # 预期结果数组
        expected = tm.box_expected(expected, xbox)   # 使用升级后的盒子装箱预期结果
        tm.assert_equal(result, expected)   # 使用测试工具模块断言结果相等

    @pytest.mark.parametrize(
        "other",
        [
            "foo",   # 字符串无效比较
            -1,      # 整数无效比较
            99,      # 整数无效比较
            4.0,     # 浮点数无效比较
            object(),   # 对象无效比较
            timedelta(days=2),   # timedelta 对象比较
            datetime(2001, 1, 1).date(),   # datetime.date 对象比较
            None,    # None 比较
            np.nan,  # NaN 比较
        ],
    )
    def test_dt64arr_cmp_scalar_invalid(self, other, tz_naive_fixture, box_with_array):
        # 测试 datetime64 数组与标量的无效比较
        tz = tz_naive_fixture   # 获取时区

        rng = date_range("1/1/2000", periods=10, tz=tz)   # 创建具有时区的日期范围
        dtarr = tm.box_expected(rng, box_with_array)   # 获得预期的装箱结果
        assert_invalid_comparison(dtarr, other, box_with_array)   # 使用测试工具模块断言比较无效
    @pytest.mark.parametrize(
        "other",
        [
            # GH#4968 invalid date/int comparisons
            # 创建一个参数化测试，用于测试各种数据类型与日期时间数组的比较
            list(range(10)),  # 创建一个包含整数的列表
            np.arange(10),  # 创建一个 NumPy 数组，包含从 0 到 9 的整数
            np.arange(10).astype(np.float32),  # 创建一个浮点数类型的 NumPy 数组
            np.arange(10).astype(object),  # 创建一个对象类型的 NumPy 数组
            pd.timedelta_range("1ns", periods=10).array,  # 创建一个包含 Timedelta 的数组
            np.array(pd.timedelta_range("1ns", periods=10)),  # 创建一个 NumPy 数组，包含 Timedelta 对象
            list(pd.timedelta_range("1ns", periods=10)),  # 创建一个包含 Timedelta 对象的列表
            pd.timedelta_range("1 Day", periods=10).astype(object),  # 创建一个对象类型的 Timedelta 数组
            pd.period_range("1971-01-01", freq="D", periods=10).array,  # 创建一个包含 Period 的数组
            pd.period_range("1971-01-01", freq="D", periods=10).astype(object),  # 创建一个对象类型的 Period 数组
        ],
    )
    def test_dt64arr_cmp_arraylike_invalid(
        self, other, tz_naive_fixture, box_with_array
    ):
        # 为日期时间数组与各种数据类型的比较进行无效性测试
        tz = tz_naive_fixture

        # 创建一个日期范围对象的数据数组
        dta = date_range("1970-01-01", freq="ns", periods=10, tz=tz)._data
        # 将数据数组封装成预期的盒子对象
        obj = tm.box_expected(dta, box_with_array)
        # 断言无效的比较操作
        assert_invalid_comparison(obj, other, box_with_array)

    def test_dt64arr_cmp_mixed_invalid(self, tz_naive_fixture):
        # 为混合类型数据与日期时间数组的比较进行无效性测试
        tz = tz_naive_fixture

        # 创建一个日期范围对象的数据数组
        dta = date_range("1970-01-01", freq="h", periods=5, tz=tz)._data

        # 创建一个混合数据类型的 NumPy 数组
        other = np.array([0, 1, 2, dta[3], Timedelta(days=1)])
        # 进行等于比较操作，并断言结果
        result = dta == other
        expected = np.array([False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 进行不等于比较操作，并断言结果
        result = dta != other
        tm.assert_numpy_array_equal(result, ~expected)

        # 断言不支持的比较操作引发 TypeError 异常
        msg = "Invalid comparison between|Cannot compare type|not supported between"
        with pytest.raises(TypeError, match=msg):
            dta < other
        with pytest.raises(TypeError, match=msg):
            dta > other
        with pytest.raises(TypeError, match=msg):
            dta <= other
        with pytest.raises(TypeError, match=msg):
            dta >= other

    def test_dt64arr_nat_comparison(self, tz_naive_fixture, box_with_array):
        # GH#22242, GH#22163 DataFrame considered NaT == ts incorrectly
        # 为 NaT 与时间戳的比较进行测试，解决 DataFrame 中 NaT 与时间戳相等的问题
        tz = tz_naive_fixture
        box = box_with_array

        # 创建一个带时区信息的时间戳对象
        ts = Timestamp("2021-01-01", tz=tz)
        # 创建一个包含时间戳和 NaT 的序列
        ser = Series([ts, NaT])

        # 将序列封装成预期的盒子对象
        obj = tm.box_expected(ser, box)
        # 获取与对象相关的升级盒子
        xbox = get_upcast_box(obj, ts, True)

        # 创建预期的布尔类型序列
        expected = Series([True, False], dtype=np.bool_)
        expected = tm.box_expected(expected, xbox)

        # 进行等于比较操作，并断言结果
        result = obj == ts
        tm.assert_equal(result, expected)
# 定义一个测试类，用于测试Datetime64SeriesComparison相关功能
class TestDatetime64SeriesComparison:
    # TODO: moved from tests.series.test_operators; needs cleanup

    # 使用pytest.mark.parametrize装饰器，定义参数化测试
    @pytest.mark.parametrize(
        "pair",  # 参数为pair，包含多组测试数据对
        [
            (
                [Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")],  # 第一组测试数据
                [NaT, NaT, Timestamp("2011-01-03")],  # 第一组期望结果
            ),
            (
                [Timedelta("1 days"), NaT, Timedelta("3 days")],  # 第二组测试数据
                [NaT, NaT, Timedelta("3 days")],  # 第二组期望结果
            ),
            (
                [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")],  # 第三组测试数据
                [NaT, NaT, Period("2011-03", freq="M")],  # 第三组期望结果
            ),
        ],
    )
    # 使用pytest.mark.parametrize装饰器，定义参数化测试
    @pytest.mark.parametrize("reverse", [True, False])
    # 使用pytest.mark.parametrize装饰器，定义参数化测试
    @pytest.mark.parametrize("dtype", [None, object])
    # 使用pytest.mark.parametrize装饰器，定义参数化测试
    @pytest.mark.parametrize(
        "op, expected",  # 参数为op和expected，包含多组操作符和期望结果
        [
            (operator.eq, [False, False, True]),  # 等于操作符的期望结果
            (operator.ne, [True, True, False]),  # 不等于操作符的期望结果
            (operator.lt, [False, False, False]),  # 小于操作符的期望结果
            (operator.gt, [False, False, False]),  # 大于操作符的期望结果
            (operator.ge, [False, False, True]),  # 大于等于操作符的期望结果
            (operator.le, [False, False, True]),  # 小于等于操作符的期望结果
        ],
    )
    # 定义测试方法，用于测试NaN值的比较操作
    def test_nat_comparisons(
        self,
        dtype,
        index_or_series,
        reverse,
        pair,
        op,
        expected,
    ):
        box = index_or_series  # 将index_or_series赋值给box变量
        lhs, rhs = pair  # 解包pair，分别赋值给lhs和rhs
        if reverse:
            # 如果reverse为True，则交换lhs和rhs的值
            lhs, rhs = rhs, lhs

        left = Series(lhs, dtype=dtype)  # 使用lhs创建Series对象left
        right = box(rhs, dtype=dtype)  # 使用rhs创建Series对象right

        result = op(left, right)  # 使用op进行left和right的比较操作，得到结果

        tm.assert_series_equal(result, Series(expected))  # 断言result与期望的Series对象相等

    # 使用pytest.mark.parametrize装饰器，定义参数化测试
    @pytest.mark.parametrize(
        "data",  # 参数为data，包含多组测试数据
        [
            [Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")],  # 第一组测试数据
            [Timedelta("1 days"), NaT, Timedelta("3 days")],  # 第二组测试数据
            [Period("2011-01", freq="M"), NaT, Period("2011-03", freq="M")],  # 第三组测试数据
        ],
    )
    # 使用pytest.mark.parametrize装饰器，定义参数化测试
    @pytest.mark.parametrize("dtype", [None, object])
    # 测试比较标量值与 NaT（Not a Time）的行为
    def test_nat_comparisons_scalar(self, dtype, data, box_with_array):
        # 从参数中获取带有数组的盒子
        box = box_with_array

        # 创建一个 Series 对象，使用指定的数据和数据类型
        left = Series(data, dtype=dtype)
        # 根据盒子预期处理 left 对象
        left = tm.box_expected(left, box)
        # 获取左操作数的升级盒子
        xbox = get_upcast_box(left, NaT, True)

        # 预期的比较结果列表
        expected = [False, False, False]
        # 对预期结果进行盒子预期处理
        expected = tm.box_expected(expected, xbox)
        # 如果盒子是 pd.array 并且数据类型是 object，将预期结果转换为布尔型 pd.array
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype="bool")

        # 断言 left == NaT 的结果与预期一致
        tm.assert_equal(left == NaT, expected)
        # 断言 NaT == left 的结果与预期一致
        tm.assert_equal(NaT == left, expected)

        # 设置新的预期结果列表
        expected = [True, True, True]
        # 对预期结果进行盒子预期处理
        expected = tm.box_expected(expected, xbox)
        # 如果盒子是 pd.array 并且数据类型是 object，将预期结果转换为布尔型 pd.array
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype="bool")
        # 断言 left != NaT 的结果与预期一致
        tm.assert_equal(left != NaT, expected)
        # 断言 NaT != left 的结果与预期一致
        tm.assert_equal(NaT != left, expected)

        # 设置新的预期结果列表
        expected = [False, False, False]
        # 对预期结果进行盒子预期处理
        expected = tm.box_expected(expected, xbox)
        # 如果盒子是 pd.array 并且数据类型是 object，将预期结果转换为布尔型 pd.array
        if box is pd.array and dtype is object:
            expected = pd.array(expected, dtype="bool")
        # 断言 left < NaT 的结果与预期一致
        tm.assert_equal(left < NaT, expected)
        # 断言 NaT > left 的结果与预期一致
        tm.assert_equal(NaT > left, expected)
        # 断言 left <= NaT 的结果与预期一致
        tm.assert_equal(left <= NaT, expected)
        # 断言 NaT >= left 的结果与预期一致
        tm.assert_equal(NaT >= left, expected)

        # 断言 left > NaT 的结果与预期一致
        tm.assert_equal(left > NaT, expected)
        # 断言 NaT < left 的结果与预期一致
        tm.assert_equal(NaT < left, expected)
        # 断言 left >= NaT 的结果与预期一致
        tm.assert_equal(left >= NaT, expected)
        # 断言 NaT <= left 的结果与预期一致
        tm.assert_equal(NaT <= left, expected)
    # 定义一个测试方法，用于测试日期时间数组与时间戳的比较
    def test_dt64arr_timestamp_equality(self, box_with_array):
        # GH#11034
        # 从参数中获取带有数组的盒子对象
        box = box_with_array

        # 创建一个包含时间戳对象的序列
        ser = Series([Timestamp("2000-01-29 01:59:00"), Timestamp("2000-01-30"), NaT])
        # 将序列数据进行盒化处理，以适应特定盒子对象的类型要求
        ser = tm.box_expected(ser, box)
        # 获取升级的盒子对象
        xbox = get_upcast_box(ser, ser, True)

        # 检查序列中的元素是否不相等，生成布尔结果序列
        result = ser != ser
        # 生成预期的结果序列，使用测试辅助方法来进行盒化
        expected = tm.box_expected([False, False, True], xbox)
        # 断言结果序列与预期序列相等
        tm.assert_equal(result, expected)

        # 如果盒子类型是 DataFrame
        if box is pd.DataFrame:
            # 对于数据框与序列的比较，不再对齐将会引发警告
            # 详见 GH#46795，在版本 2.0 中将会强制执行
            # 期望引发 ValueError 异常，异常信息中包含 "not aligned"
            with pytest.raises(ValueError, match="not aligned"):
                # 检查序列与序列的第一个元素是否不相等
                ser != ser[0]
        else:
            # 否则，检查序列与序列的第一个元素是否不相等
            result = ser != ser[0]
            # 生成预期的结果序列，使用测试辅助方法来进行盒化
            expected = tm.box_expected([False, True, True], xbox)
            # 断言结果序列与预期序列相等
            tm.assert_equal(result, expected)

        # 如果盒子类型是 DataFrame
        if box is pd.DataFrame:
            # 对于数据框与序列的比较，不再对齐将会引发警告
            # 详见 GH#46795，在版本 2.0 中将会强制执行
            # 期望引发 ValueError 异常，异常信息中包含 "not aligned"
            with pytest.raises(ValueError, match="not aligned"):
                # 检查序列与序列的第三个元素是否不相等
                ser != ser[2]
        else:
            # 否则，检查序列与序列的第三个元素是否不相等
            result = ser != ser[2]
            # 生成预期的结果序列，使用测试辅助方法来进行盒化
            expected = tm.box_expected([True, True, True], xbox)
            # 断言结果序列与预期序列相等
            tm.assert_equal(result, expected)

        # 检查序列中的元素是否相等，生成布尔结果序列
        result = ser == ser
        # 生成预期的结果序列，使用测试辅助方法来进行盒化
        expected = tm.box_expected([True, True, False], xbox)
        # 断言结果序列与预期序列相等
        tm.assert_equal(result, expected)

        # 如果盒子类型是 DataFrame
        if box is pd.DataFrame:
            # 对于数据框与序列的比较，不再对齐将会引发警告
            # 详见 GH#46795，在版本 2.0 中将会强制执行
            # 期望引发 ValueError 异常，异常信息中包含 "not aligned"
            with pytest.raises(ValueError, match="not aligned"):
                # 检查序列与序列的第一个元素是否相等
                ser == ser[0]
        else:
            # 否则，检查序列与序列的第一个元素是否相等
            result = ser == ser[0]
            # 生成预期的结果序列，使用测试辅助方法来进行盒化
            expected = tm.box_expected([True, False, False], xbox)
            # 断言结果序列与预期序列相等
            tm.assert_equal(result, expected)

        # 如果盒子类型是 DataFrame
        if box is pd.DataFrame:
            # 对于数据框与序列的比较，不再对齐将会引发警告
            # 详见 GH#46795，在版本 2.0 中将会强制执行
            # 期望引发 ValueError 异常，异常信息中包含 "not aligned"
            with pytest.raises(ValueError, match="not aligned"):
                # 检查序列与序列的第三个元素是否相等
                ser == ser[2]
        else:
            # 否则，检查序列与序列的第三个元素是否相等
            result = ser == ser[2]
            # 生成预期的结果序列，使用测试辅助方法来进行盒化
            expected = tm.box_expected([False, False, False], xbox)
            # 断言结果序列与预期序列相等
            tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "datetimelike",
        [
            # 指定不同类型的日期时间对象作为测试参数
            Timestamp("20130101"),
            datetime(2013, 1, 1),
            np.datetime64("2013-01-01T00:00", "ns"),
        ],
    )
    @pytest.mark.parametrize(
        "op,expected",
        [
            # 指定不同的操作符和预期结果列表作为测试参数
            (operator.lt, [True, False, False, False]),
            (operator.le, [True, True, False, False]),
            (operator.eq, [False, True, False, False]),
            (operator.gt, [False, False, False, True]),
        ],
    )
    # 定义一个测试方法，用于比较 datetime64[ns] 列和 datetimelike 对象的能力
    def test_dt64_compare_datetime_scalar(self, datetimelike, op, expected):
        # GH#17965, test for ability to compare datetime64[ns] columns
        #  to datetimelike
        # 创建一个包含 Timestamp 对象的 Series，表示日期时间序列
        ser = Series(
            [
                Timestamp("20120101"),
                Timestamp("20130101"),
                np.nan,
                Timestamp("20130103"),
            ],
            name="A",
        )
        # 使用给定的操作符 op 对序列 ser 和 datetimelike 进行比较
        result = op(ser, datetimelike)
        # 创建一个预期的 Series，用于与结果进行比较
        expected = Series(expected, name="A")
        # 使用测试框架中的方法检查两个 Series 是否相等
        tm.assert_series_equal(result, expected)
# 定义一个测试类 TestDatetimeIndexComparisons，用于测试日期时间索引的比较操作
class TestDatetimeIndexComparisons:
    # TODO: moved from tests.indexes.test_base; parametrize and de-duplicate
    # 定义一个测试方法 test_comparators，测试比较操作
    def test_comparators(self, comparison_op):
        # 创建一个日期范围索引，从"2020-01-01"开始，包含10个日期
        index = date_range("2020-01-01", periods=10)
        # 获取索引中间的元素
        element = index[len(index) // 2]
        # 将元素转换为 Timestamp 对象，再转换为 datetime64 格式
        element = Timestamp(element).to_datetime64()

        # 将索引转换为 numpy 数组
        arr = np.array(index)
        # 使用给定的比较操作符比较数组和元素
        arr_result = comparison_op(arr, element)
        # 使用给定的比较操作符比较索引和元素
        index_result = comparison_op(index, element)

        # 断言索引结果的类型为 numpy 数组
        assert isinstance(index_result, np.ndarray)
        # 断言数组结果和索引结果相等
        tm.assert_numpy_array_equal(arr_result, index_result)

    # 使用参数化标记，定义一个测试方法 test_dti_cmp_datetimelike，测试日期时间索引与日期时间对象的比较操作
    @pytest.mark.parametrize(
        "other",
        [datetime(2016, 1, 1), Timestamp("2016-01-01"), np.datetime64("2016-01-01")],
    )
    def test_dti_cmp_datetimelike(self, other, tz_naive_fixture):
        # 获取时区信息
        tz = tz_naive_fixture
        # 创建一个时区感知的日期范围索引，从"2016-01-01"开始，包含2个日期，使用给定的时区
        dti = date_range("2016-01-01", periods=2, tz=tz)
        
        # 如果时区不为空
        if tz is not None:
            # 如果 other 是 numpy 的 datetime64 类型，跳过测试并显示跳过信息
            if isinstance(other, np.datetime64):
                pytest.skip(f"{type(other).__name__} is not tz aware")
            # 将 Python 的 datetime 对象转换为时区感知的对象
            other = localize_pydatetime(other, dti.tzinfo)

        # 执行日期时间索引与 other 的等于比较，将结果与预期结果进行比较
        result = dti == other
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 执行日期时间索引与 other 的大于比较，将结果与预期结果进行比较
        result = dti > other
        expected = np.array([False, True])
        tm.assert_numpy_array_equal(result, expected)

        # 执行日期时间索引与 other 的大于等于比较，将结果与预期结果进行比较
        result = dti >= other
        expected = np.array([True, True])
        tm.assert_numpy_array_equal(result, expected)

        # 执行日期时间索引与 other 的小于比较，将结果与预期结果进行比较
        result = dti < other
        expected = np.array([False, False])
        tm.assert_numpy_array_equal(result, expected)

        # 执行日期时间索引与 other 的小于等于比较，将结果与预期结果进行比较
        result = dti <= other
        expected = np.array([True, False])
        tm.assert_numpy_array_equal(result, expected)

    # 使用参数化标记，定义一个测试方法 test_dti_cmp_datetimelike，测试日期时间索引与不同数据类型的比较操作
    @pytest.mark.parametrize("dtype", [None, object])
    # 定义一个测试方法，用于比较日期时间索引和 NaT（Not a Time）的行为
    def test_dti_cmp_nat(self, dtype, box_with_array):
        # 创建左侧日期时间索引，包含三个时间戳，其中一个为 NaT
        left = DatetimeIndex([Timestamp("2011-01-01"), NaT, Timestamp("2011-01-03")])
        # 创建右侧日期时间索引，所有时间戳均为 NaT
        right = DatetimeIndex([NaT, NaT, Timestamp("2011-01-03")])

        # 使用给定的盒子对象对左侧日期时间索引进行封装处理
        left = tm.box_expected(left, box_with_array)
        # 使用给定的盒子对象对右侧日期时间索引进行封装处理
        right = tm.box_expected(right, box_with_array)
        # 获取升级类型的盒子，用于比较左侧和右侧索引
        xbox = get_upcast_box(left, right, True)

        # 如果数据类型为对象类型，则将左侧和右侧索引转换为对象类型
        if dtype is object:
            lhs, rhs = left.astype(object), right.astype(object)
        else:
            lhs, rhs = left, right

        # 比较右侧和左侧索引，生成布尔型结果
        result = rhs == lhs
        # 预期结果是一个布尔数组，用盒子对象处理后返回
        expected = np.array([False, False, True])
        expected = tm.box_expected(expected, xbox)
        # 断言实际结果与预期结果相等
        tm.assert_equal(result, expected)

        # 比较左侧和右侧索引，生成不等比较的布尔结果
        result = lhs != rhs
        # 预期结果是一个布尔数组，用盒子对象处理后返回
        expected = np.array([True, True, False])
        expected = tm.box_expected(expected, xbox)
        # 断言实际结果与预期结果相等
        tm.assert_equal(result, expected)

        # 预期结果是一个布尔数组，用盒子对象处理后返回
        expected = np.array([False, False, False])
        expected = tm.box_expected(expected, xbox)
        # 断言左侧索引与 NaT 比较的结果与预期结果相等
        tm.assert_equal(lhs == NaT, expected)
        # 断言 NaT 与右侧索引比较的结果与预期结果相等
        tm.assert_equal(NaT == rhs, expected)

        # 预期结果是一个布尔数组，用盒子对象处理后返回
        expected = np.array([True, True, True])
        expected = tm.box_expected(expected, xbox)
        # 断言左侧索引与 NaT 比较的结果与预期结果相等
        tm.assert_equal(lhs != NaT, expected)
        # 断言 NaT 与左侧索引比较的结果与预期结果相等
        tm.assert_equal(NaT != lhs, expected)

        # 预期结果是一个布尔数组，用盒子对象处理后返回
        expected = np.array([False, False, False])
        expected = tm.box_expected(expected, xbox)
        # 断言左侧索引与 NaT 比较的结果与预期结果相等
        tm.assert_equal(lhs < NaT, expected)
        # 断言 NaT 与左侧索引比较的结果与预期结果相等
        tm.assert_equal(NaT > lhs, expected)
    # 测试时区感知比较的兼容性
    def test_comparison_tzawareness_compat(self, comparison_op, box_with_array):
        # GH#18162
        # 设置操作符和数据框/数组的实例
        op = comparison_op
        box = box_with_array

        # 创建一个日期范围对象，不带时区信息
        dr = date_range("2016-01-01", periods=6)
        # 将日期范围对象本地化到"US/Pacific"时区
        dz = dr.tz_localize("US/Pacific")

        # 根据测试环境调整数据框/数组的期望形式
        dr = tm.box_expected(dr, box)
        dz = tm.box_expected(dz, box)

        # 如果 box 是 DataFrame 类型，则定义转换函数 tolist 为将对象转换为对象数组并返回其第一个元素的列表
        if box is pd.DataFrame:
            tolist = lambda x: x.astype(object).values.tolist()[0]
        else:
            tolist = list

        # 如果操作符不是等于或不等于，预期会引发类型错误异常，消息指定了相关的错误信息
        if op not in [operator.eq, operator.ne]:
            msg = (
                r"Invalid comparison between dtype=datetime64\[ns.*\] "
                "and (Timestamp|DatetimeArray|list|ndarray)"
            )
            # 使用 pytest 来验证操作是否会引发预期的类型错误异常
            with pytest.raises(TypeError, match=msg):
                op(dr, dz)

            with pytest.raises(TypeError, match=msg):
                op(dr, tolist(dz))
            with pytest.raises(TypeError, match=msg):
                op(dr, np.array(tolist(dz), dtype=object))
            with pytest.raises(TypeError, match=msg):
                op(dz, dr)

            with pytest.raises(TypeError, match=msg):
                op(dz, tolist(dr))
            with pytest.raises(TypeError, match=msg):
                op(dz, np.array(tolist(dr), dtype=object))

        # 断言：在 aware==aware 和 naive==naive 比较中不会引发异常
        assert np.all(dr == dr)
        assert np.all(dr == tolist(dr))
        assert np.all(tolist(dr) == dr)
        assert np.all(np.array(tolist(dr), dtype=object) == dr)
        assert np.all(dr == np.array(tolist(dr), dtype=object))

        assert np.all(dz == dz)
        assert np.all(dz == tolist(dz))
        assert np.all(tolist(dz) == dz)
        assert np.all(np.array(tolist(dz), dtype=object) == dz)
        assert np.all(dz == np.array(tolist(dz), dtype=object))

    # 测试时区感知比较的兼容性（标量版本）
    def test_comparison_tzawareness_compat_scalars(self, comparison_op, box_with_array):
        # GH#18162
        # 设置操作符
        op = comparison_op

        # 创建一个日期范围对象，不带时区信息
        dr = date_range("2016-01-01", periods=6)
        # 将日期范围对象本地化到 box_with_array 指定的时区
        dz = dr.tz_localize("US/Pacific")

        # 根据测试环境调整数据框/数组的期望形式
        dr = tm.box_expected(dr, box_with_array)
        dz = tm.box_expected(dz, box_with_array)

        # 检查与标量 Timestamp 对象的比较
        ts = Timestamp("2000-03-14 01:59")
        ts_tz = Timestamp("2000-03-14 01:59", tz="Europe/Amsterdam")

        # 断言：所有 dr 中的日期都应大于 ts
        assert np.all(dr > ts)
        # 如果操作符不是等于或不等于，预期会引发类型错误异常，消息指定了相关的错误信息
        msg = r"Invalid comparison between dtype=datetime64\[ns.*\] and Timestamp"
        if op not in [operator.eq, operator.ne]:
            with pytest.raises(TypeError, match=msg):
                op(dr, ts_tz)

        # 断言：所有 dz 中的日期都应大于 ts_tz
        assert np.all(dz > ts_tz)
        if op not in [operator.eq, operator.ne]:
            with pytest.raises(TypeError, match=msg):
                op(dz, ts)

        if op not in [operator.eq, operator.ne]:
            # GH#12601: 检查与 Timestamp 和 DatetimeIndex 的比较
            with pytest.raises(TypeError, match=msg):
                op(ts, dz)
    # 使用 pytest 的参数化装饰器，为测试函数提供多组参数输入
    @pytest.mark.parametrize(
        "other",
        [datetime(2016, 1, 1), Timestamp("2016-01-01"), np.datetime64("2016-01-01")],
    )
    # 在比较标量时可能出现的 NumPy 缺陷，详见 https://github.com/numpy/numpy/issues/13841
    # 在 __eq__ 方法中引发异常将回退到 NumPy，这会导致警告和失败，因此我们需要忽略这些警告
    @pytest.mark.filterwarnings("ignore:elementwise comp:DeprecationWarning")
    # 测试标量比较时考虑时区感知性的情况
    def test_scalar_comparison_tzawareness(
        self, comparison_op, other, tz_aware_fixture, box_with_array
    ):
        # 获取比较操作符
        op = comparison_op
        # 获取时区信息
        tz = tz_aware_fixture
        # 创建一个时区感知的日期范围对象
        dti = date_range("2016-01-01", periods=2, tz=tz)

        # 将日期范围对象封装成指定类型的盒子
        dtarr = tm.box_expected(dti, box_with_array)
        # 获取升级后的盒子对象，用于与其他对象进行比较
        xbox = get_upcast_box(dtarr, other, True)
        
        # 如果操作符是相等或不相等
        if op in [operator.eq, operator.ne]:
            # 确定预期的布尔值结果
            exbool = op is operator.ne
            expected = np.array([exbool, exbool], dtype=bool)
            expected = tm.box_expected(expected, xbox)

            # 进行数组与标量的比较，并断言结果与预期一致
            result = op(dtarr, other)
            tm.assert_equal(result, expected)

            # 进行标量与数组的比较，并断言结果与预期一致
            result = op(other, dtarr)
            tm.assert_equal(result, expected)
        else:
            # 如果操作符不是相等或不相等，则抛出类型错误异常，带有特定的错误信息
            msg = (
                r"Invalid comparison between dtype=datetime64\[ns, .*\] "
                f"and {type(other).__name__}"
            )
            with pytest.raises(TypeError, match=msg):
                op(dtarr, other)
            with pytest.raises(TypeError, match=msg):
                op(other, dtarr)

    # 测试 NaT（Not a Time）与时区感知日期索引对象的比较
    def test_nat_comparison_tzawareness(self, comparison_op):
        # 获取比较操作符
        op = comparison_op

        # 创建一个包含 NaT 的日期索引对象
        dti = DatetimeIndex(
            ["2014-01-01", NaT, "2014-03-01", NaT, "2014-05-01", "2014-07-01"]
        )
        # 创建预期的布尔数组，用于比较操作的预期结果
        expected = np.array([op == operator.ne] * len(dti))
        
        # 对日期索引对象与 NaT 进行比较，并断言结果与预期一致
        result = op(dti, NaT)
        tm.assert_numpy_array_equal(result, expected)

        # 对本地化到指定时区的日期索引对象与 NaT 进行比较，并断言结果与预期一致
        result = op(dti.tz_localize("US/Pacific"), NaT)
        tm.assert_numpy_array_equal(result, expected)
    # 定义一个测试方法，用于测试日期时间索引与字符串的比较
    def test_dti_cmp_str(self, tz_naive_fixture):
        # GH#22074
        # 无论时区如何，我们期望这些比较都是有效的
        tz = tz_naive_fixture
        # 创建一个带有时区的日期时间范围
        rng = date_range("1/1/2000", periods=10, tz=tz)
        # 设置另一个字符串日期
        other = "1/1/2000"

        # 进行相等比较
        result = rng == other
        # 预期结果是一个布尔数组，第一个为True，其余为False
        expected = np.array([True] + [False] * 9)
        tm.assert_numpy_array_equal(result, expected)

        # 进行不等比较
        result = rng != other
        # 预期结果是一个布尔数组，第一个为False，其余为True
        expected = np.array([False] + [True] * 9)
        tm.assert_numpy_array_equal(result, expected)

        # 进行小于比较
        result = rng < other
        # 预期结果是一个全部为False的布尔数组
        expected = np.array([False] * 10)
        tm.assert_numpy_array_equal(result, expected)

        # 进行小于等于比较
        result = rng <= other
        # 预期结果是一个布尔数组，第一个为True，其余为False
        expected = np.array([True] + [False] * 9)
        tm.assert_numpy_array_equal(result, expected)

        # 进行大于比较
        result = rng > other
        # 预期结果是一个布尔数组，第一个为False，其余为True
        expected = np.array([False] + [True] * 9)
        tm.assert_numpy_array_equal(result, expected)

        # 进行大于等于比较
        result = rng >= other
        # 预期结果是一个全部为True的布尔数组
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)

    # 定义一个测试方法，用于测试日期时间索引与列表的比较
    def test_dti_cmp_list(self):
        # 创建一个日期时间范围
        rng = date_range("1/1/2000", periods=10)

        # 进行相等比较
        result = rng == list(rng)
        # 预期结果是一个布尔数组，与自身的相等比较结果相同
        expected = rng == rng
        tm.assert_numpy_array_equal(result, expected)

    # 使用参数化装饰器定义一个测试方法，用于测试日期时间索引与时间增量索引的比较和错误处理
    @pytest.mark.parametrize(
        "other",
        [
            pd.timedelta_range("1D", periods=10),
            pd.timedelta_range("1D", periods=10).to_series(),
            pd.timedelta_range("1D", periods=10).asi8.view("m8[ns]"),
        ],
        # 为参数化设置 ID 以显示对象类型的名称
        ids=lambda x: type(x).__name__,
    )
    def test_dti_cmp_tdi_tzawareness(self, other):
        # GH#22074
        # 反向测试，确保我们在与TimedeltaIndex比较时不调用_assert_tzawareness_compat
        # 创建一个带有时区的日期时间索引
        dti = date_range("2000-01-01", periods=10, tz="Asia/Tokyo")

        # 进行相等比较
        result = dti == other
        # 预期结果是一个全部为False的布尔数组
        expected = np.array([False] * 10)
        tm.assert_numpy_array_equal(result, expected)

        # 进行不等比较
        result = dti != other
        # 预期结果是一个全部为True的布尔数组
        expected = np.array([True] * 10)
        tm.assert_numpy_array_equal(result, expected)

        # 使用 pytest 来确保在比较时会抛出 TypeError 异常，并包含特定的错误消息
        msg = "Invalid comparison between"
        with pytest.raises(TypeError, match=msg):
            dti < other
        with pytest.raises(TypeError, match=msg):
            dti <= other
        with pytest.raises(TypeError, match=msg):
            dti > other
        with pytest.raises(TypeError, match=msg):
            dti >= other
    # 定义一个测试方法，用于比较日期时间索引对象的数据类型
    def test_dti_cmp_object_dtype(self):
        # GH#22074: GitHub issue reference
        # 创建一个包含 10 个日期范围的时间索引对象，设定时区为 "Asia/Tokyo"
        dti = date_range("2000-01-01", periods=10, tz="Asia/Tokyo")

        # 将时间索引对象转换为对象类型（"O" 表示对象类型）
        other = dti.astype("O")

        # 比较原始时间索引对象和转换后的对象是否相等，返回布尔数组
        result = dti == other
        # 预期结果是一个包含 10 个 True 的 NumPy 数组
        expected = np.array([True] * 10)
        # 使用断言检查结果是否与预期相等
        tm.assert_numpy_array_equal(result, expected)

        # 将时间索引对象的时区信息去除
        other = dti.tz_localize(None)
        # 比较原始时间索引对象和去除时区后的对象是否不相等，返回布尔数组
        result = dti != other
        # 使用断言检查结果是否与预期相等
        tm.assert_numpy_array_equal(result, expected)

        # 创建一个包含前 5 个日期时间和后 5 个增加了一天的 Timedelta 对象的 NumPy 数组
        other = np.array(list(dti[:5]) + [Timedelta(days=1)] * 5)
        # 比较时间索引对象和新创建的数组是否相等，返回布尔数组
        result = dti == other
        # 预期结果是一个包含前 5 个 True 和后 5 个 False 的 NumPy 数组
        expected = np.array([True] * 5 + [False] * 5)
        # 使用断言检查结果是否与预期相等
        tm.assert_numpy_array_equal(result, expected)

        # 设置错误消息字符串，用于检查是否会抛出 TypeError 异常
        msg = ">=' not supported between instances of 'Timestamp' and 'Timedelta'"
        # 使用 pytest 检查在执行 dti >= other 时是否会抛出 TypeError 异常，并且异常消息与预期匹配
        with pytest.raises(TypeError, match=msg):
            dti >= other
# ------------------------------------------------------------------
# Arithmetic

# 定义一个测试类 TestDatetime64Arithmetic，用于测试日期时间相关的算术运算
class TestDatetime64Arithmetic:
    # This class is intended for "finished" tests that are fully parametrized
    # over DataFrame/Series/Index/DatetimeArray
    # 此类旨在进行已完全参数化的测试，涵盖了DataFrame/Series/Index/DatetimeArray的所有情况

    # -------------------------------------------------------------
    # Addition/Subtraction of timedelta-like

    @pytest.mark.arm_slow
    # 标记为 arm_slow 的测试函数，较慢的测试用例
    def test_dt64arr_add_timedeltalike_scalar(
        self, tz_naive_fixture, two_hours, box_with_array
    ):
        # GH#22005, GH#22163 check DataFrame doesn't raise TypeError
        # 检查 DataFrame 是否不会引发 TypeError
        tz = tz_naive_fixture

        # 创建一个时区感知的日期范围 rng，从 "2000-01-01" 到 "2000-02-01"
        rng = date_range("2000-01-01", "2000-02-01", tz=tz)
        # 创建预期结果，增加两个小时后的日期范围
        expected = date_range("2000-01-01 02:00", "2000-02-01 02:00", tz=tz)

        # 将 rng 和 expected 使用 box_with_array 进行包装
        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        # 测试 rng + two_hours 的结果是否与 expected 相等
        result = rng + two_hours
        tm.assert_equal(result, expected)

        # 测试 two_hours + rng 的结果是否与 expected 相等
        result = two_hours + rng
        tm.assert_equal(result, expected)

        # 将 rng 自身增加 two_hours，测试结果是否与 expected 相等
        rng += two_hours
        tm.assert_equal(rng, expected)

    # 测试从日期范围中减去 timedelta 类似的标量
    def test_dt64arr_sub_timedeltalike_scalar(
        self, tz_naive_fixture, two_hours, box_with_array
    ):
        tz = tz_naive_fixture

        # 创建一个时区感知的日期范围 rng，从 "2000-01-01" 到 "2000-02-01"
        rng = date_range("2000-01-01", "2000-02-01", tz=tz)
        # 创建预期结果，减去两个小时后的日期范围
        expected = date_range("1999-12-31 22:00", "2000-01-31 22:00", tz=tz)

        # 将 rng 和 expected 使用 box_with_array 进行包装
        rng = tm.box_expected(rng, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        # 测试 rng - two_hours 的结果是否与 expected 相等
        result = rng - two_hours
        tm.assert_equal(result, expected)

        # 将 rng 自身减去 two_hours，测试结果是否与 expected 相等
        rng -= two_hours
        tm.assert_equal(rng, expected)

    # 测试日期时间数组之间的减法，涉及不同时区的情况
    def test_dt64_array_sub_dt_with_different_timezone(self, box_with_array):
        # 创建一个本地化为 "US/Eastern" 时区的日期范围 t1
        t1 = date_range("20130101", periods=3).tz_localize("US/Eastern")
        t1 = tm.box_expected(t1, box_with_array)
        # 创建一个本地化为 "CET" 时区的时间戳 t2
        t2 = Timestamp("20130101").tz_localize("CET")
        # 创建一个没有时区信息的时间戳 tnaive
        tnaive = Timestamp(20130101)

        # 测试 t1 - t2 的结果是否与预期的时间差索引相等
        result = t1 - t2
        expected = TimedeltaIndex(
            ["0 days 06:00:00", "1 days 06:00:00", "2 days 06:00:00"]
        )
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        # 测试 t2 - t1 的结果是否与预期的时间差索引相等
        result = t2 - t1
        expected = TimedeltaIndex(
            ["-1 days +18:00:00", "-2 days +18:00:00", "-3 days +18:00:00"]
        )
        expected = tm.box_expected(expected, box_with_array)
        tm.assert_equal(result, expected)

        # 测试不能对时区感知与无时区信息的 datetime-like 对象进行减法操作时是否引发 TypeError
        msg = "Cannot subtract tz-naive and tz-aware datetime-like objects"
        with pytest.raises(TypeError, match=msg):
            t1 - tnaive

        with pytest.raises(TypeError, match=msg):
            tnaive - t1
    # 定义一个测试方法，用于测试日期时间数组与不同时区之间的日期时间操作
    def test_dt64_array_sub_dt64_array_with_different_timezone(self, box_with_array):
        # 创建一个包含三个日期的时间范围，使用美国东部时区进行本地化
        t1 = date_range("20130101", periods=3).tz_localize("US/Eastern")
        # 将 t1 作为预期结果与给定的数组盒子进行打包处理
        t1 = tm.box_expected(t1, box_with_array)
        # 创建另一个包含三个日期的时间范围，使用中欧时间进行本地化
        t2 = date_range("20130101", periods=3).tz_localize("CET")
        # 将 t2 作为预期结果与给定的数组盒子进行打包处理
        t2 = tm.box_expected(t2, box_with_array)
        # 创建一个包含三个日期的非时区感知时间范围
        tnaive = date_range("20130101", periods=3)

        # 计算 t1 与 t2 的日期时间差值
        result = t1 - t2
        # 创建一个预期的时间增量索引，包含相同的时间差值
        expected = TimedeltaIndex(
            ["0 days 06:00:00", "0 days 06:00:00", "0 days 06:00:00"]
        )
        # 将预期结果与给定的数组盒子进行打包处理
        expected = tm.box_expected(expected, box_with_array)
        # 断言计算结果与预期结果相等
        tm.assert_equal(result, expected)

        # 计算 t2 与 t1 的日期时间差值
        result = t2 - t1
        # 创建一个预期的时间增量索引，包含相同的时间差值
        expected = TimedeltaIndex(
            ["-1 days +18:00:00", "-1 days +18:00:00", "-1 days +18:00:00"]
        )
        # 将预期结果与给定的数组盒子进行打包处理
        expected = tm.box_expected(expected, box_with_array)
        # 断言计算结果与预期结果相等
        tm.assert_equal(result, expected)

        # 创建一个错误消息，用于测试时区非感知与时区感知日期时间对象相减的情况
        msg = "Cannot subtract tz-naive and tz-aware datetime-like objects"
        # 使用 pytest 断言应该抛出 TypeError 异常，并包含特定错误消息
        with pytest.raises(TypeError, match=msg):
            t1 - tnaive

        # 使用 pytest 断言应该抛出 TypeError 异常，并包含特定错误消息
        with pytest.raises(TypeError, match=msg):
            tnaive - t1

    # -----------------------------------------------------------------
    # Subtraction of datetime-like scalars

    def test_dt64arr_add_sub_td64_nat(self, box_with_array, tz_naive_fixture):
        # GH#23320 special handling for timedelta64("NaT")
        # 获取时区信息
        tz = tz_naive_fixture

        # 创建一个时区感知的日期时间索引，包含从 "1994-04-01" 开始的九个季度的日期
        dti = date_range("1994-04-01", periods=9, tz=tz, freq="QS")
        # 创建一个表示 "NaT"（Not a Time）的 timedelta64 对象
        other = np.timedelta64("NaT")
        # 创建一个预期的日期时间索引，包含九个 "NaT" 值，并应用指定的时区
        expected = DatetimeIndex(["NaT"] * 9, tz=tz).as_unit("ns")

        # 将 dti 视为预期结果与给定的数组盒子进行打包处理
        obj = tm.box_expected(dti, box_with_array)
        # 将 expected 视为预期结果与给定的数组盒子进行打包处理
        expected = tm.box_expected(expected, box_with_array)

        # 计算 obj 与 other 的日期时间加法
        result = obj + other
        # 断言计算结果与预期结果相等
        tm.assert_equal(result, expected)
        # 计算 other 与 obj 的日期时间加法
        result = other + obj
        # 断言计算结果与预期结果相等
        tm.assert_equal(result, expected)
        # 计算 obj 与 other 的日期时间减法
        result = obj - other
        # 断言计算结果与预期结果相等
        tm.assert_equal(result, expected)
        # 创建一个错误消息，用于测试不支持 other - obj 操作的情况
        msg = "cannot subtract"
        # 使用 pytest 断言应该抛出 TypeError 异常，并包含特定错误消息
        with pytest.raises(TypeError, match=msg):
            other - obj

    def test_dt64arr_add_sub_td64ndarray(self, tz_naive_fixture, box_with_array):
        # 获取时区信息
        tz = tz_naive_fixture
        # 创建一个时区感知的日期时间索引，包含从 "2016-01-01" 开始的三个日期
        dti = date_range("2016-01-01", periods=3, tz=tz)
        # 创建一个表示 "-1 Day" 的 timedelta 索引
        tdi = TimedeltaIndex(["-1 Day", "-1 Day", "-1 Day"])
        # 获取 tdi 的值作为数组
        tdarr = tdi.values

        # 创建一个预期的日期时间索引，包含从 "2015-12-31" 到 "2016-01-02" 的三个日期，应用指定的时区
        expected = date_range("2015-12-31", "2016-01-02", periods=3, tz=tz)

        # 将 dti 视为预期结果与给定的数组盒子进行打包处理
        dtarr = tm.box_expected(dti, box_with_array)
        # 将 expected 视为预期结果与给定的数组盒子进行打包处理
        expected = tm.box_expected(expected, box_with_array)

        # 计算 dtarr 与 tdarr 的日期时间加法
        result = dtarr + tdarr
        # 断言计算结果与预期结果相等
        tm.assert_equal(result, expected)
        # 计算 tdarr 与 dtarr 的日期时间加法
        result = tdarr + dtarr
        # 断言计算结果与预期结果相等
        tm.assert_equal(result, expected)

        # 创建一个预期的日期时间索引，包含从 "2016-01-02" 到 "2016-01-04" 的三个日期，应用指定的时区
        expected = date_range("2016-01-02", "2016-01-04", periods=3, tz=tz)
        # 将 expected 视为预期结果与给定的数组盒子进行打包处理
        expected = tm.box_expected(expected, box_with_array)

        # 计算 dtarr 与 tdarr 的日期时间减法
        result = dtarr - tdarr
        # 断言计算结果与预期结果相等
        tm.assert_equal(result, expected)
        # 创建一个错误消息，用于测试不支持 tdarr - dtarr 操作的情况
        msg = "cannot subtract|(bad|unsupported) operand type for unary"
        # 使用 pytest 断言应该抛出 TypeError 异常，并包含特定错误消息
        with pytest.raises(TypeError, match=msg):
            tdarr - dtarr
    @pytest.mark.parametrize(
        "ts",
        [
            Timestamp("2013-01-01"),  # 创建一个 pandas Timestamp 对象，表示特定日期
            Timestamp("2013-01-01").to_pydatetime(),  # 将 Timestamp 转换为 Python 的 datetime 对象
            Timestamp("2013-01-01").to_datetime64(),  # 将 Timestamp 转换为 datetime64 格式
            # GH#7996, GH#22163 确保非纳秒精度的 datetime64 被转换为纳秒精度以进行 DataFrame 操作
            np.datetime64("2013-01-01", "D"),  # 创建一个 datetime64 对象，表示特定日期
        ],
    )
    def test_dt64arr_sub_dtscalar(self, box_with_array, ts):
        # GH#8554, GH#22163 DataFrame 操作不应返回 datetime64 类型
        idx = date_range("2013-01-01", periods=3)._with_freq(None)  # 创建一个日期范围索引对象
        idx = tm.box_expected(idx, box_with_array)  # 将索引对象与预期的盒子（数据结构）进行比较

        expected = TimedeltaIndex(["0 Days", "1 Day", "2 Days"])  # 创建一个时间增量索引对象
        expected = tm.box_expected(expected, box_with_array)  # 将时间增量索引与预期的盒子进行比较

        result = idx - ts  # 执行索引与时间戳的减法操作
        tm.assert_equal(result, expected)  # 使用测试工具断言结果与预期相等

        result = ts - idx  # 执行时间戳与索引的减法操作
        tm.assert_equal(result, -expected)  # 使用测试工具断言结果与预期相等

        tm.assert_equal(result, -expected)  # 再次使用测试工具断言结果与预期相等

    def test_dt64arr_sub_timestamp_tzaware(self, box_with_array):
        ser = date_range("2014-03-17", periods=2, freq="D", tz="US/Eastern")  # 创建一个带时区信息的日期序列
        ser = ser._with_freq(None)  # 移除日期序列的频率信息
        ts = ser[0]  # 获取日期序列的第一个时间戳

        ser = tm.box_expected(ser, box_with_array)  # 将日期序列与预期的盒子进行比较

        delta_series = Series([np.timedelta64(0, "D"), np.timedelta64(1, "D")])  # 创建一个时间增量序列
        expected = tm.box_expected(delta_series, box_with_array)  # 将时间增量序列与预期的盒子进行比较

        tm.assert_equal(ser - ts, expected)  # 使用测试工具断言结果与预期相等
        tm.assert_equal(ts - ser, -expected)  # 使用测试工具断言结果与预期相等

    def test_dt64arr_sub_NaT(self, box_with_array, unit):
        # GH#18808
        dti = DatetimeIndex([NaT, Timestamp("19900315")]).as_unit(unit)  # 创建一个时间索引，并转换为指定单位的时间单位
        ser = tm.box_expected(dti, box_with_array)  # 将时间索引与预期的盒子进行比较

        result = ser - NaT  # 执行时间索引与 NaT 的减法操作
        expected = Series([NaT, NaT], dtype=f"timedelta64[{unit}]")  # 创建一个预期的时间增量序列
        expected = tm.box_expected(expected, box_with_array)  # 将预期的时间增量序列与预期的盒子进行比较
        tm.assert_equal(result, expected)  # 使用测试工具断言结果与预期相等

        dti_tz = dti.tz_localize("Asia/Tokyo")  # 将时间索引本地化为指定时区
        ser_tz = tm.box_expected(dti_tz, box_with_array)  # 将本地化的时间索引与预期的盒子进行比较

        result = ser_tz - NaT  # 执行本地化时间索引与 NaT 的减法操作
        expected = Series([NaT, NaT], dtype=f"timedelta64[{unit}]")  # 创建一个预期的时间增量序列
        expected = tm.box_expected(expected, box_with_array)  # 将预期的时间增量序列与预期的盒子进行比较
        tm.assert_equal(result, expected)  # 使用测试工具断言结果与预期相等

    # -------------------------------------------------------------
    # Subtraction of datetime-like array-like

    def test_dt64arr_sub_dt64object_array(
        self, performance_warning, box_with_array, tz_naive_fixture
    ):
        dti = date_range("2016-01-01", periods=3, tz=tz_naive_fixture)  # 创建一个带有时区信息的日期范围索引对象
        expected = dti - dti  # 执行日期范围索引对象与自身的减法操作

        obj = tm.box_expected(dti, box_with_array)  # 将日期范围索引对象与预期的盒子进行比较
        expected = tm.box_expected(expected, box_with_array).astype(object)  # 将预期的结果与预期的盒子进行比较，并转换为对象类型

        with tm.assert_produces_warning(performance_warning):  # 使用测试工具断言产生了警告
            result = obj - obj.astype(object)  # 执行对象与对象转换为对象类型的减法操作
        tm.assert_equal(result, expected)  # 使用测试工具断言结果与预期相等
    # 定义一个测试方法，用于测试 datetime64 数组与 datetime64 数组的减法操作
    def test_dt64arr_naive_sub_dt64ndarray(self, box_with_array):
        # 创建一个包含三个日期的日期范围，时区为 None
        dti = date_range("2016-01-01", periods=3, tz=None)
        # 提取日期范围的 datetime64 值数组
        dt64vals = dti.values

        # 将日期范围转换为数组形式，并进行包装
        dtarr = tm.box_expected(dti, box_with_array)

        # 计算期望的结果，即数组与自身的减法操作
        expected = dtarr - dtarr
        # 执行 datetime64 值数组与数组的减法操作
        result = dtarr - dt64vals
        # 断言减法操作的结果与期望结果相等
        tm.assert_equal(result, expected)
        # 执行 datetime64 值数组与数组的减法操作（顺序颠倒）
        result = dt64vals - dtarr
        # 断言减法操作的结果与期望结果相等
        tm.assert_equal(result, expected)

    # 定义一个测试方法，用于测试 datetime64 数组与包含时区信息的 datetime64 数组的减法操作引发异常
    def test_dt64arr_aware_sub_dt64ndarray_raises(
        self, tz_aware_fixture, box_with_array
    ):
        # 获取测试用的时区信息
        tz = tz_aware_fixture
        # 创建一个包含三个日期的日期范围，指定时区
        dti = date_range("2016-01-01", periods=3, tz=tz)
        # 提取日期范围的 datetime64 值数组
        dt64vals = dti.values

        # 将日期范围转换为数组形式，并进行包装
        dtarr = tm.box_expected(dti, box_with_array)
        # 定义预期的错误消息
        msg = "Cannot subtract tz-naive and tz-aware datetime"
        # 使用 pytest 的上下文管理器，断言减法操作会引发 TypeError 异常，并匹配预期的错误消息
        with pytest.raises(TypeError, match=msg):
            dtarr - dt64vals
        # 使用 pytest 的上下文管理器，断言减法操作会引发 TypeError 异常，并匹配预期的错误消息（顺序颠倒）
        with pytest.raises(TypeError, match=msg):
            dt64vals - dtarr

    # -------------------------------------------------------------
    # 添加日期时间类（datetime-like）对象的操作（无效）

    def test_dt64arr_add_dtlike_raises(self, tz_naive_fixture, box_with_array):
        # 用于确保 DataFrame 不会将 Timestamp 强制转换为 i8 的 GH#22163
        # GH#9631
        # 获取测试用的时区信息
        tz = tz_naive_fixture

        # 创建一个包含三个日期的日期范围，指定时区
        dti = date_range("2016-01-01", periods=3, tz=tz)
        # 根据时区是否为 None，调整日期范围的时区信息
        if tz is None:
            dti2 = dti.tz_localize("US/Eastern")
        else:
            dti2 = dti.tz_localize(None)
        # 将日期范围转换为数组形式，并进行包装
        dtarr = tm.box_expected(dti, box_with_array)

        # 断言无法将数组与日期时间类对象相加的方法
        assert_cannot_add(dtarr, dti.values)
        assert_cannot_add(dtarr, dti)
        assert_cannot_add(dtarr, dtarr)
        assert_cannot_add(dtarr, dti[0])
        assert_cannot_add(dtarr, dti[0].to_pydatetime())
        assert_cannot_add(dtarr, dti[0].to_datetime64())
        assert_cannot_add(dtarr, dti2[0])
        assert_cannot_add(dtarr, dti2[0].to_pydatetime())
        assert_cannot_add(dtarr, np.datetime64("2011-01-01", "D"))

    # -------------------------------------------------------------
    # 其它无效的加法/减法操作

    # 注意：这里的频率包括 Tick 和非 Tick 偏移量；这是有关的，因为历史上如果我们有一个频率，
    #      允许整数加法。
    @pytest.mark.parametrize("freq", ["h", "D", "W", "2ME", "MS", "QE", "B", None])
    @pytest.mark.parametrize("dtype", [None, "uint8"])
    def test_dt64arr_addsub_intlike(
        self, dtype, index_or_series_or_array, freq, tz_naive_fixture
    ):
    ):
        # 处理 GH#19959, GH#19123, GH#19012 问题
        # 使用 index_or_series_or_array 替代 box_with_array，因为 DataFrame 对齐使其不适用
        tz = tz_naive_fixture

        # 如果频率为空，则创建一个带时区信息的 DatetimeIndex 对象
        if freq is None:
            dti = DatetimeIndex(["NaT", "2017-04-05 06:07:08"], tz=tz)
        else:
            # 否则，根据指定的频率创建一个 DatetimeIndex 对象
            dti = date_range("2016-01-01", periods=2, freq=freq, tz=tz)

        # 根据 DatetimeIndex 对象获取对应的 index_or_series_or_array 对象
        obj = index_or_series_or_array(dti)

        # 创建一个包含 [4, -1] 的 NumPy 数组作为 other
        other = np.array([4, -1])

        # 如果指定了 dtype，则将 other 转换为指定的数据类型
        if dtype is not None:
            other = other.astype(dtype)

        # 设置错误消息，用于断言异常情况
        msg = "|".join(
            [
                "Addition/subtraction of integers",
                "cannot subtract DatetimeArray from",
                # IntegerArray
                "can only perform ops with numeric values",
                "unsupported operand type.*Categorical",
                r"unsupported operand type\(s\) for -: 'int' and 'Timestamp'",
            ]
        )

        # 断言不同类型的操作会抛出异常
        assert_invalid_addsub_type(obj, 1, msg)
        assert_invalid_addsub_type(obj, np.int64(2), msg)
        assert_invalid_addsub_type(obj, np.array(3, dtype=np.int64), msg)
        assert_invalid_addsub_type(obj, other, msg)
        assert_invalid_addsub_type(obj, np.array(other), msg)
        assert_invalid_addsub_type(obj, pd.array(other), msg)
        assert_invalid_addsub_type(obj, pd.Categorical(other), msg)
        assert_invalid_addsub_type(obj, pd.Index(other), msg)
        assert_invalid_addsub_type(obj, Series(other), msg)

    @pytest.mark.parametrize(
        "other",
        [
            3.14,
            np.array([2.0, 3.0]),
            # GH#13078 datetime +/- Period is invalid
            Period("2011-01-01", freq="D"),
            # https://github.com/pandas-dev/pandas/issues/10329
            time(1, 2, 3),
        ],
    )
    @pytest.mark.parametrize("dti_freq", [None, "D"])
    def test_dt64arr_add_sub_invalid(self, dti_freq, other, box_with_array):
        # 创建一个带有指定频率的 DatetimeIndex 对象
        dti = DatetimeIndex(["2011-01-01", "2011-01-02"], freq=dti_freq)
        # 使用 box_expected 函数对 dti 进行包装
        dtarr = tm.box_expected(dti, box_with_array)

        # 设置错误消息，用于断言异常情况
        msg = "|".join(
            [
                "unsupported operand type",
                "cannot (add|subtract)",
                "cannot use operands with types",
                "ufunc '?(add|subtract)'? cannot use operands with types",
                "Concatenation operation is not implemented for NumPy arrays",
            ]
        )

        # 断言不同类型的操作会抛出异常
        assert_invalid_addsub_type(dtarr, other, msg)

    @pytest.mark.parametrize("pi_freq", ["D", "W", "Q", "h"])
    @pytest.mark.parametrize("dti_freq", [None, "D"])
    def test_dt64arr_add_sub_parr(
        self, dti_freq, pi_freq, box_with_array, box_with_array2
    ):
        # GH#20049 subtracting PeriodIndex should raise TypeError
        # 创建一个包含两个日期的 DatetimeIndex 对象
        dti = DatetimeIndex(["2011-01-01", "2011-01-02"], freq=dti_freq)
        # 将 DatetimeIndex 转换为 PeriodIndex 对象
        pi = dti.to_period(pi_freq)

        # 使用 box_expected 函数将 dti 转换为数组形式的日期时间对象
        dtarr = tm.box_expected(dti, box_with_array)
        # 使用 box_expected 函数将 pi 转换为数组形式的日期时间周期对象
        parr = tm.box_expected(pi, box_with_array2)

        # 准备断言失败时的错误消息
        msg = "|".join(
            [
                "cannot (add|subtract)",  # 不能执行加法或减法
                "unsupported operand",    # 不支持的操作数
                "descriptor.*requires",   # 描述符需要
                "ufunc.*cannot use operands",  # ufunc 不能使用操作数
            ]
        )
        # 断言执行无效的加减法操作时会抛出特定类型的错误
        assert_invalid_addsub_type(dtarr, parr, msg)

    @pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
    def test_dt64arr_addsub_time_objects_raises(self, box_with_array, tz_naive_fixture):
        # https://github.com/pandas-dev/pandas/issues/10329

        # 获取时区信息
        tz = tz_naive_fixture

        # 创建一个日期范围对象，带有时区信息
        obj1 = date_range("2012-01-01", periods=3, tz=tz)
        # 创建一个包含多个 time 对象的列表
        obj2 = [time(i, i, i) for i in range(3)]

        # 使用 box_expected 函数将 obj1 转换为数组形式的日期时间对象
        obj1 = tm.box_expected(obj1, box_with_array)
        # 使用 box_expected 函数将 obj2 转换为数组形式的日期时间对象
        obj2 = tm.box_expected(obj2, box_with_array)

        # 准备断言失败时的错误消息
        msg = "|".join(
            [
                "unsupported operand",  # 不支持的操作数
                "cannot subtract DatetimeArray from ndarray",  # 不能从 ndarray 中减去 DatetimeArray
            ]
        )
        # 断言执行无效的加减法操作时会抛出特定类型的错误，忽略 PerformanceWarning
        assert_invalid_addsub_type(obj1, obj2, msg=msg)

    # -------------------------------------------------------------
    # Other invalid operations

    @pytest.mark.parametrize(
        "dt64_series",
        [
            Series([Timestamp("19900315"), Timestamp("19900315")]),
            Series([NaT, Timestamp("19900315")]),
            Series([NaT, NaT], dtype="datetime64[ns]"),
        ],
    )
    @pytest.mark.parametrize("one", [1, 1.0, np.array(1)])
    def test_dt64_mul_div_numeric_invalid(self, one, dt64_series, box_with_array):
        # 使用 box_expected 函数将 dt64_series 转换为数组形式的日期时间对象
        obj = tm.box_expected(dt64_series, box_with_array)

        # 准备断言失败时的错误消息
        msg = "cannot perform .* with this index type"

        # 断言执行无效的乘除法操作时会抛出特定类型的错误
        # 乘法
        with pytest.raises(TypeError, match=msg):
            obj * one
        with pytest.raises(TypeError, match=msg):
            one * obj

        # 除法
        with pytest.raises(TypeError, match=msg):
            obj / one
        with pytest.raises(TypeError, match=msg):
            one / obj
class TestDatetime64DateOffsetArithmetic:
    # -------------------------------------------------------------
    # Tick DateOffsets

    # TODO: parametrize over timezone?
    # 测试将日期偏移量添加到时间序列中
    def test_dt64arr_series_add_tick_DateOffset(self, box_with_array, unit):
        # GH#4532
        # 使用 pd.offsets 进行操作
        # 创建包含时间戳的序列，并转换为指定单位（unit）
        ser = Series(
            [Timestamp("20130101 9:01"), Timestamp("20130101 9:02")]
        ).dt.as_unit(unit)
        # 创建期望的序列，加上了 5 秒的日期偏移量
        expected = Series(
            [Timestamp("20130101 9:01:05"), Timestamp("20130101 9:02:05")]
        ).dt.as_unit(unit)

        # 将序列进行盒化处理，以适应不同的数据盒
        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        # 将 5 秒的日期偏移量加到序列上，并进行断言比较
        result = ser + pd.offsets.Second(5)
        tm.assert_equal(result, expected)

        # 另一种方式，将 5 秒的日期偏移量加到序列上，并进行断言比较
        result2 = pd.offsets.Second(5) + ser
        tm.assert_equal(result2, expected)

    # 测试从时间序列中减去日期偏移量
    def test_dt64arr_series_sub_tick_DateOffset(self, box_with_array):
        # GH#4532
        # 使用 pd.offsets 进行操作
        ser = Series([Timestamp("20130101 9:01"), Timestamp("20130101 9:02")])
        # 创建期望的序列，减去了 5 秒的日期偏移量
        expected = Series(
            [Timestamp("20130101 9:00:55"), Timestamp("20130101 9:01:55")]
        )

        # 将序列进行盒化处理，以适应不同的数据盒
        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        # 从序列中减去 5 秒的日期偏移量，并进行断言比较
        result = ser - pd.offsets.Second(5)
        tm.assert_equal(result, expected)

        # 另一种方式，从序列中减去 5 秒的日期偏移量，并进行断言比较
        result2 = -pd.offsets.Second(5) + ser
        tm.assert_equal(result2, expected)
        
        # 测试不支持一元操作符的情况
        msg = "(bad|unsupported) operand type for unary"
        with pytest.raises(TypeError, match=msg):
            pd.offsets.Second(5) - ser

    @pytest.mark.parametrize(
        "cls_name", ["Day", "Hour", "Minute", "Second", "Milli", "Micro", "Nano"]
    )
    # 测试各种日期偏移量的基本功能
    def test_dt64arr_add_sub_tick_DateOffset_smoke(self, cls_name, box_with_array):
        # GH#4532
        # 烟雾测试以验证有效的日期偏移量
        ser = Series([Timestamp("20130101 9:01"), Timestamp("20130101 9:02")])
        # 将序列进行盒化处理，以适应不同的数据盒
        ser = tm.box_expected(ser, box_with_array)

        # 根据类名动态获取对应的日期偏移量类
        offset_cls = getattr(pd.offsets, cls_name)
        # 将 5 单位的日期偏移量添加到序列中
        ser + offset_cls(5)
        # 另一种方式，将 5 单位的日期偏移量添加到序列中
        offset_cls(5) + ser
        # 从序列中减去 5 单位的日期偏移量
        ser - offset_cls(5)
    # 定义一个测试方法，用于测试时区感知功能和增加时间偏移的操作
    def test_dti_add_tick_tzaware(self, tz_aware_fixture, box_with_array):
        # GH#21610, GH#22163 确保 DataFrame 不返回对象类型的列
        tz = tz_aware_fixture
        # 如果时区是 "US/Pacific"
        if tz == "US/Pacific":
            # 创建一个日期范围，从 '2012-11-01' 开始，3个时间段，使用指定时区
            dates = date_range("2012-11-01", periods=3, tz=tz)
            # 偏移时间，将日期时间对象的每个日期偏移5小时
            offset = dates + pd.offsets.Hour(5)
            # 断言偏移后的第一个日期时间与预期结果相等
            assert dates[0] + pd.offsets.Hour(5) == offset[0]

        # 创建一个日期范围，从 '2010-11-01 00:00' 开始，3个时间段，使用指定时区和频率为小时
        dates = date_range("2010-11-01 00:00", periods=3, tz=tz, freq="h")
        # 创建预期的日期时间索引对象，包含指定日期时间字符串，时区为 tz，频率为小时
        expected = DatetimeIndex(
            ["2010-11-01 05:00", "2010-11-01 06:00", "2010-11-01 07:00"],
            freq="h",
            tz=tz,
        ).as_unit("ns")

        # 使用 box_with_array 对象对 dates 进行包装处理
        dates = tm.box_expected(dates, box_with_array)
        # 使用 box_with_array 对象对 expected 进行包装处理
        expected = tm.box_expected(expected, box_with_array)

        # 针对 [pd.offsets.Hour(5), np.timedelta64(5, "h"), timedelta(hours=5)] 中的每个标量进行循环
        for scalar in [pd.offsets.Hour(5), np.timedelta64(5, "h"), timedelta(hours=5)]:
            # 对日期时间对象 dates 添加标量偏移
            offset = dates + scalar
            # 断言偏移后的结果与预期的结果相等
            tm.assert_equal(offset, expected)
            # 对标量 scalar 加上日期时间对象 dates 的偏移
            offset = scalar + dates
            # 断言偏移后的结果与预期的结果相等
            tm.assert_equal(offset, expected)

            # 将 offset 减去 scalar 后的结果，应与原始 dates 相等
            roundtrip = offset - scalar
            # 断言往返操作后的结果与原始 dates 相等
            tm.assert_equal(roundtrip, dates)

            # 准备错误消息，用于捕获预期的异常情况
            msg = "|".join(
                ["bad operand type for unary -", "cannot subtract DatetimeArray"]
            )
            # 使用 pytest 捕获 TypeError 异常，并匹配预期的错误消息
            with pytest.raises(TypeError, match=msg):
                # 尝试执行 scalar - dates 操作，预期会抛出 TypeError 异常
                scalar - dates

    # -------------------------------------------------------------
    # RelativeDelta DateOffsets
    # 定义测试函数，用于测试DatetimeIndex数组在加减DateOffset相对偏移时的行为
    def test_dt64arr_add_sub_relativedelta_offsets(self, box_with_array, unit):
        # GH#10699: GitHub issue reference

        # 创建DatetimeIndex数组，包含一组时间戳
        vec = DatetimeIndex(
            [
                Timestamp("2000-01-05 00:15:00"),
                Timestamp("2000-01-31 00:23:00"),
                Timestamp("2000-01-01"),
                Timestamp("2000-03-31"),
                Timestamp("2000-02-29"),
                Timestamp("2000-12-31"),
                Timestamp("2000-05-15"),
                Timestamp("2001-06-15"),
            ]
        ).as_unit(unit)
        
        # 将vec根据box_with_array的类型进行处理
        vec = tm.box_expected(vec, box_with_array)
        vec_items = vec.iloc[0] if box_with_array is pd.DataFrame else vec

        # 定义DateOffset的偏移单位和值
        relative_kwargs = [
            ("years", 2),
            ("months", 5),
            ("days", 3),
            ("hours", 5),
            ("minutes", 10),
            ("seconds", 2),
            ("microseconds", 5),
        ]

        # 遍历relative_kwargs中的偏移单位和值
        for i, (offset_unit, value) in enumerate(relative_kwargs):
            # 创建DateOffset对象
            off = DateOffset(**{offset_unit: value})

            # 根据偏移单位调整exp_unit的值
            exp_unit = unit
            if offset_unit == "microseconds" and unit != "ns":
                exp_unit = "us"

            # 构建预期的DatetimeIndex，分别进行加法和减法操作
            expected = DatetimeIndex([x + off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + off)

            expected = DatetimeIndex([x - off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - off)

            # 根据当前偏移单位创建DateOffset对象
            off = DateOffset(**dict(relative_kwargs[: i + 1]))

            # 再次进行加法和减法操作，验证累积偏移效果
            expected = DatetimeIndex([x + off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec + off)

            expected = DatetimeIndex([x - off for x in vec_items]).as_unit(exp_unit)
            expected = tm.box_expected(expected, box_with_array)
            tm.assert_equal(expected, vec - off)

            # 验证off与vec进行减法操作时抛出TypeError异常
            msg = "(bad|unsupported) operand type for unary"
            with pytest.raises(TypeError, match=msg):
                off - vec

    # -------------------------------------------------------------
    # Non-Tick, Non-RelativeDelta DateOffsets

    # TODO: redundant with test_dt64arr_add_sub_DateOffset?  that includes
    #  tz-aware cases which this does not
    @pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
    # 使用 pytest.mark.parametrize 装饰器定义测试参数化，测试不同的日期偏移类和参数组合
    @pytest.mark.parametrize(
        "cls_and_kwargs",
        [
            "YearBegin",  # 年初日期偏移类
            ("YearBegin", {"month": 5}),  # 年初日期偏移类，指定月份为5
            "YearEnd",  # 年末日期偏移类
            ("YearEnd", {"month": 5}),  # 年末日期偏移类，指定月份为5
            "MonthBegin",  # 月初日期偏移类
            "MonthEnd",  # 月末日期偏移类
            "SemiMonthEnd",  # 半月末日期偏移类
            "SemiMonthBegin",  # 半月初日期偏移类
            "Week",  # 周日期偏移类
            ("Week", {"weekday": 3}),  # 周日期偏移类，指定工作日为星期三
            ("Week", {"weekday": 6}),  # 周日期偏移类，指定工作日为星期六
            "BusinessDay",  # 工作日日期偏移类
            "BDay",  # 工作日日期偏移类的简写形式
            "QuarterEnd",  # 季度末日期偏移类
            "QuarterBegin",  # 季度初日期偏移类
            "CustomBusinessDay",  # 自定义工作日日期偏移类
            "CDay",  # 自定义工作日日期偏移类的简写形式
            "CBMonthEnd",  # 自定义工作日月末日期偏移类
            "CBMonthBegin",  # 自定义工作日月初日期偏移类
            "BMonthBegin",  # 工作日月初日期偏移类
            "BMonthEnd",  # 工作日月末日期偏移类
            "BusinessHour",  # 工作小时日期偏移类
            "BYearBegin",  # 工作日年初日期偏移类
            "BYearEnd",  # 工作日年末日期偏移类
            "BQuarterBegin",  # 工作日季度初日期偏移类
            ("LastWeekOfMonth", {"weekday": 2}),  # 月末的最后一个指定工作日的日期偏移类
            (
                "FY5253Quarter",  # FY5253季度日期偏移类
                {
                    "qtr_with_extra_week": 1,
                    "startingMonth": 1,
                    "weekday": 2,
                    "variation": "nearest",
                },
            ),  # FY5253季度日期偏移类，指定有额外周数，从1开始，起始月份为1，工作日为星期二，变化方式为最近
            ("FY5253", {"weekday": 0, "startingMonth": 2, "variation": "nearest"}),  # FY5253日期偏移类，工作日为星期日，起始月份为2，变化方式为最近
            ("WeekOfMonth", {"weekday": 2, "week": 2}),  # 每月第二周指定工作日的日期偏移类
            "Easter",  # 复活节日期偏移类
            ("DateOffset", {"day": 4}),  # 指定天数的日期偏移类
            ("DateOffset", {"month": 5}),  # 指定月数的日期偏移类
        ],
    )
    # 参数化测试用例中的 normalize 参数，测试是否进行规范化
    @pytest.mark.parametrize("normalize", [True, False])
    # 参数化测试用例中的 n 参数，测试偏移量的数量
    @pytest.mark.parametrize("n", [0, 5])
    # 参数化测试用例中的 tz 参数，测试不同的时区设置
    @pytest.mark.parametrize("tz", [None, "US/Central"])
    # 定义测试方法，测试对日期时间数组进行添加和减去日期偏移的操作
    def test_dt64arr_add_sub_DateOffsets(
        self, box_with_array, n, normalize, cls_and_kwargs, unit, tz
    ):
        # GH#10699
        # assert vectorized operation matches pointwise operations

        if isinstance(cls_and_kwargs, tuple):
            # 如果 cls_and_kwargs 是元组，则第二个条目是偏移构造函数的关键字参数
            cls_name, kwargs = cls_and_kwargs
        else:
            # 否则，cls_name 是 cls_and_kwargs 的唯一参数，kwargs 为空字典
            cls_name = cls_and_kwargs
            kwargs = {}

        if n == 0 and cls_name in [
            "WeekOfMonth",
            "LastWeekOfMonth",
            "FY5253Quarter",
            "FY5253",
        ]:
            # 如果 n 等于 0 并且 cls_name 在指定的无效偏移类列表中，则直接返回
            return

        vec = (
            DatetimeIndex(
                [
                    Timestamp("2000-01-05 00:15:00"),
                    Timestamp("2000-01-31 00:23:00"),
                    Timestamp("2000-01-01"),
                    Timestamp("2000-03-31"),
                    Timestamp("2000-02-29"),
                    Timestamp("2000-12-31"),
                    Timestamp("2000-05-15"),
                    Timestamp("2001-06-15"),
                ]
            )
            .as_unit(unit)  # 将时间索引转换为指定单位
            .tz_localize(tz)  # 设置时区
        )
        vec = tm.box_expected(vec, box_with_array)  # 按照指定的方式包装 vec
        vec_items = vec.iloc[0] if box_with_array is pd.DataFrame else vec

        offset_cls = getattr(pd.offsets, cls_name)  # 获取偏移类对象
        offset = offset_cls(n, normalize=normalize, **kwargs)  # 使用给定参数创建偏移对象

        # TODO(GH#55564): as_unit will be unnecessary
        # 构建预期结果，应用偏移量到 vec_items 中的每个时间戳，并转换为指定单位
        expected = DatetimeIndex([x + offset for x in vec_items]).as_unit(unit)
        expected = tm.box_expected(expected, box_with_array)  # 按照指定的方式包装 expected
        tm.assert_equal(expected, vec + offset)  # 断言期望结果与 vec + offset 相等
        tm.assert_equal(expected, offset + vec)  # 断言期望结果与 offset + vec 相等

        # 构建预期结果，从 vec_items 中的每个时间戳中减去偏移量，并转换为指定单位
        expected = DatetimeIndex([x - offset for x in vec_items]).as_unit(unit)
        expected = tm.box_expected(expected, box_with_array)  # 按照指定的方式包装 expected
        tm.assert_equal(expected, vec - offset)  # 断言期望结果与 vec - offset 相等

        # 构建预期结果，将偏移量加到 vec_items 中的每个时间戳中，并转换为指定单位
        expected = DatetimeIndex([offset + x for x in vec_items]).as_unit(unit)
        expected = tm.box_expected(expected, box_with_array)  # 按照指定的方式包装 expected
        tm.assert_equal(expected, offset + vec)  # 断言期望结果与 offset + vec 相等

        # 预期的错误信息，用于测试在偏移量与 vec 不兼容时是否引发 TypeError
        msg = "(bad|unsupported) operand type for unary"
        with pytest.raises(TypeError, match=msg):
            offset - vec

    @pytest.mark.parametrize(
        "other",
        [
            [pd.offsets.MonthEnd(), pd.offsets.Day(n=2)],
            [pd.offsets.DateOffset(years=1), pd.offsets.MonthEnd()],
            # matching offsets
            [pd.offsets.DateOffset(years=1), pd.offsets.DateOffset(years=1)],
        ],
    )
    @pytest.mark.parametrize("op", [operator.add, roperator.radd, operator.sub])
    def test_dt64arr_add_sub_offset_array(
        self, performance_warning, tz_naive_fixture, box_with_array, op, other
    ):
        # GH#18849
        # GH#10699 array of offsets

        # 使用固定时区的 Fixture
        tz = tz_naive_fixture
        # 创建一个日期范围，从 "2017-01-01" 开始，两个时间段，使用给定的时区
        dti = date_range("2017-01-01", periods=2, tz=tz)
        # 使用 tm.box_expected 将 dti 打包成数组，使用 box_with_array
        dtarr = tm.box_expected(dti, box_with_array)
        # 转换 other 到 numpy 数组
        other = np.array(other)
        # 生成一个期望的 DatetimeIndex，对 dti 和 other 的每个元素应用 op 操作
        expected = DatetimeIndex([op(dti[n], other[n]) for n in range(len(dti))])
        # 使用 tm.box_expected 将期望的结果打包成数组，使用 box_with_array，并转换成对象类型
        expected = tm.box_expected(expected, box_with_array).astype(object)

        # 断言产生性能警告
        with tm.assert_produces_warning(performance_warning):
            res = op(dtarr, other)
        # 断言 res 和 expected 相等
        tm.assert_equal(res, expected)

        # 类似的操作，但是对 other 进行打包
        other = tm.box_expected(other, box_with_array)
        # 如果 box_with_array 是 pd.array 并且 op 是 roperator.radd
        if box_with_array is pd.array and op is roperator.radd:
            # 我们期望一个 NumpyExtensionArray，而不是 ndarray[object]
            expected = pd.array(expected, dtype=object)
        # 再次断言产生性能警告
        with tm.assert_produces_warning(performance_warning):
            res = op(dtarr, other)
        # 再次断言 res 和 expected 相等
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize(
        "op, offset, exp, exp_freq",
        [
            (
                "__add__",
                DateOffset(months=3, days=10),
                [
                    Timestamp("2014-04-11"),
                    Timestamp("2015-04-11"),
                    Timestamp("2016-04-11"),
                    Timestamp("2017-04-11"),
                ],
                None,
            ),
            (
                "__add__",
                DateOffset(months=3),
                [
                    Timestamp("2014-04-01"),
                    Timestamp("2015-04-01"),
                    Timestamp("2016-04-01"),
                    Timestamp("2017-04-01"),
                ],
                "YS-APR",
            ),
            (
                "__sub__",
                DateOffset(months=3, days=10),
                [
                    Timestamp("2013-09-21"),
                    Timestamp("2014-09-21"),
                    Timestamp("2015-09-21"),
                    Timestamp("2016-09-21"),
                ],
                None,
            ),
            (
                "__sub__",
                DateOffset(months=3),
                [
                    Timestamp("2013-10-01"),
                    Timestamp("2014-10-01"),
                    Timestamp("2015-10-01"),
                    Timestamp("2016-10-01"),
                ],
                "YS-OCT",
            ),
        ],
    )
    def test_dti_add_sub_nonzero_mth_offset(
        self, op, offset, exp, exp_freq, tz_aware_fixture, box_with_array
    ):
        # GH 26258
        # 使用 tz_aware_fixture 设置时区
        tz = tz_aware_fixture
        # 创建一个年度频率的日期范围，从 "01 Jan 2014" 到 "01 Jan 2017"，使用给定的时区
        date = date_range(start="01 Jan 2014", end="01 Jan 2017", freq="YS", tz=tz)
        # 使用 tm.box_expected 将 date 打包成数组，使用 box_with_array，False 表示不拆箱
        date = tm.box_expected(date, box_with_array, False)
        # 获取 date 对象的 op 方法
        mth = getattr(date, op)
        # 对 date 应用 offset 进行操作
        result = mth(offset)

        # 生成期望的 DatetimeIndex，设置时区为 tz，并转换为纳秒单位
        expected = DatetimeIndex(exp, tz=tz).as_unit("ns")
        # 使用 tm.box_expected 将期望的结果打包成数组，使用 box_with_array，False 表示不拆箱
        expected = tm.box_expected(expected, box_with_array, False)
        # 断言 result 和 expected 相等
        tm.assert_equal(result, expected)
    def test_dt64arr_series_add_DateOffset_with_milli(self):
        # 定义测试方法：验证DatetimeIndex与DateOffset的加法运算（毫秒级）
        dti = DatetimeIndex(
            [
                "2000-01-01 00:00:00.012345678",
                "2000-01-31 00:00:00.012345678",
                "2000-02-29 00:00:00.012345678",
            ],
            dtype="datetime64[ns]",
        )
        # 执行DatetimeIndex加上毫秒级DateOffset的运算，生成结果
        result = dti + DateOffset(milliseconds=4)
        # 预期结果DatetimeIndex，毫秒级加4
        expected = DatetimeIndex(
            [
                "2000-01-01 00:00:00.016345678",
                "2000-01-31 00:00:00.016345678",
                "2000-02-29 00:00:00.016345678",
            ],
            dtype="datetime64[ns]",
        )
        # 断言验证result与expected是否相等
        tm.assert_index_equal(result, expected)

        # 执行DatetimeIndex加上天数与毫秒级DateOffset的运算，生成结果
        result = dti + DateOffset(days=1, milliseconds=4)
        # 预期结果DatetimeIndex，天数加1，毫秒级加4
        expected = DatetimeIndex(
            [
                "2000-01-02 00:00:00.016345678",
                "2000-02-01 00:00:00.016345678",
                "2000-03-01 00:00:00.016345678",
            ],
            dtype="datetime64[ns]",
        )
        # 断言验证result与expected是否相等
        tm.assert_index_equal(result, expected)
class TestDatetime64OverflowHandling:
    # TODO: box + de-duplicate

    def test_dt64_overflow_masking(self, box_with_array):
        # GH#25317
        # 创建包含单个时间戳的 Series 对象，数据类型为 'M8[ns]'
        left = Series([Timestamp("1969-12-31")], dtype="M8[ns]")
        # 创建包含 NaT（Not a Time）的 Series 对象
        right = Series([NaT])

        # 对 left 和 right 进行盒装处理，使用 box_with_array 函数
        left = tm.box_expected(left, box_with_array)
        right = tm.box_expected(right, box_with_array)

        # 创建预期的 TimedeltaIndex 对象，包含一个 NaT，数据类型为 'm8[ns]'
        expected = TimedeltaIndex([NaT], dtype="m8[ns]")
        expected = tm.box_expected(expected, box_with_array)

        # 计算 left 与 right 的差值，期望结果为 expected
        result = left - right
        tm.assert_equal(result, expected)

    def test_dt64_series_arith_overflow(self):
        # GH#12534, fixed by GH#19024
        # 创建一个 Timestamp 对象，表示日期为 '1700-01-31'
        dt = Timestamp("1700-01-31")
        # 创建一个 Timedelta 对象，表示 20000 天
        td = Timedelta("20000 Days")
        # 创建一个日期范围，从 '1949-09-30' 开始，100 年一次，共 4 个周期
        dti = date_range("1949-09-30", freq="100YE", periods=4)
        # 创建一个包含 dti 的 Series 对象
        ser = Series(dti)
        # 错误消息字符串
        msg = "Overflow in int64 addition"
        
        # 检查在执行各种运算时是否会引发 OverflowError 异常，异常信息匹配 msg
        with pytest.raises(OverflowError, match=msg):
            ser - dt
        with pytest.raises(OverflowError, match=msg):
            dt - ser
        with pytest.raises(OverflowError, match=msg):
            ser + td
        with pytest.raises(OverflowError, match=msg):
            td + ser

        # 将 Series 对象的最后一个元素设为 NaT
        ser.iloc[-1] = NaT
        # 创建预期结果的 Series 对象，数据类型为 'datetime64[ns]'
        expected = Series(
            ["2004-10-03", "2104-10-04", "2204-10-04", "NaT"], dtype="datetime64[ns]"
        )
        # 执行 ser + td 运算，期望结果与 expected 相等
        res = ser + td
        tm.assert_series_equal(res, expected)
        # 执行 td + ser 运算，期望结果与 expected 相等
        res = td + ser
        tm.assert_series_equal(res, expected)

        # 将 Series 对象的第二个及其后的元素设为 NaT
        ser.iloc[1:] = NaT
        # 创建预期结果的 Series 对象，数据类型为 'timedelta64[ns]'
        expected = Series(["91279 Days", "NaT", "NaT", "NaT"], dtype="timedelta64[ns]")
        # 执行 ser - dt 运算，期望结果与 expected 相等
        res = ser - dt
        tm.assert_series_equal(res, expected)
        # 执行 dt - ser 运算，期望结果与 -expected 相等
        res = dt - ser
        tm.assert_series_equal(res, -expected)
    # 定义测试函数，用于检查在DateTimeIndex与Timestamp之间的减法操作中是否会发生溢出错误

    # 创建包含 Timestamp.max 和指定日期时间的 DateTimeIndex 对象，并转换为纳秒单位
    dtimax = pd.to_datetime(["2021-12-28 17:19", Timestamp.max]).as_unit("ns")
    # 创建包含 Timestamp.min 和指定日期时间的 DateTimeIndex 对象，并转换为纳秒单位
    dtimin = pd.to_datetime(["2021-12-28 17:19", Timestamp.min]).as_unit("ns")

    # 创建一个 Timestamp 对象，表示1950年1月1日，并将其转换为纳秒单位
    tsneg = Timestamp("1950-01-01").as_unit("ns")
    # 构建多种表示1950年1月1日的 Timestamp 对象，以测试不同类型的输入
    ts_neg_variants = [
        tsneg,
        tsneg.to_pydatetime(),
        tsneg.to_datetime64().astype("datetime64[ns]"),
        tsneg.to_datetime64().astype("datetime64[D]"),
    ]

    # 创建一个 Timestamp 对象，表示1980年1月1日，并将其转换为纳秒单位
    tspos = Timestamp("1980-01-01").as_unit("ns")
    # 构建多种表示1980年1月1日的 Timestamp 对象，以测试不同类型的输入
    ts_pos_variants = [
        tspos,
        tspos.to_pydatetime(),
        tspos.to_datetime64().astype("datetime64[ns]"),
        tspos.to_datetime64().astype("datetime64[D]"),
    ]

    # 错误信息字符串，用于断言异常抛出时进行匹配
    msg = "Overflow in int64 addition"

    # 对于每种负数时间戳的变体，预期在与 dtimax 相减时会抛出 OverflowError
    for variant in ts_neg_variants:
        with pytest.raises(OverflowError, match=msg):
            dtimax - variant

    # 计算预期结果并断言：在与 tspos 相减时，结果的第二个元素的数值应等于预期值
    expected = Timestamp.max._value - tspos._value
    for variant in ts_pos_variants:
        res = dtimax - variant
        assert res[1]._value == expected

    # 计算预期结果并断言：在与 tsneg 相减时，结果的第二个元素的数值应等于预期值
    expected = Timestamp.min._value - tsneg._value
    for variant in ts_neg_variants:
        res = dtimin - variant
        assert res[1]._value == expected

    # 对于每种正数时间戳的变体，预期在与 dtimin 相减时会抛出 OverflowError
    for variant in ts_pos_variants:
        with pytest.raises(OverflowError, match=msg):
            dtimin - variant

    # 定义测试函数，用于检查在DateTimeIndex与DateTimeIndex之间的减法操作中是否会发生溢出错误

    # 创建包含 Timestamp.max 和指定日期时间的 DateTimeIndex 对象，并转换为纳秒单位
    dtimax = pd.to_datetime(["2021-12-28 17:19", Timestamp.max]).as_unit("ns")
    # 创建包含 Timestamp.min 和指定日期时间的 DateTimeIndex 对象，并转换为纳秒单位
    dtimin = pd.to_datetime(["2021-12-28 17:19", Timestamp.min]).as_unit("ns")

    # 创建包含两个负数日期时间的 DateTimeIndex 对象，并转换为纳秒单位
    ts_neg = pd.to_datetime(["1950-01-01", "1950-01-01"]).as_unit("ns")
    # 创建包含两个正数日期时间的 DateTimeIndex 对象，并转换为纳秒单位
    ts_pos = pd.to_datetime(["1980-01-01", "1980-01-01"]).as_unit("ns")

    # 一般测试情况下，计算预期结果并断言：在与 ts_pos 的第二个元素相减时，结果的数值应等于预期值
    expected = Timestamp.max._value - ts_pos[1]._value
    result = dtimax - ts_pos
    assert result[1]._value == expected

    # 一般测试情况下，计算预期结果并断言：在与 ts_neg 的第二个元素相减时，结果的数值应等于预期值
    expected = Timestamp.min._value - ts_neg[1]._value
    result = dtimin - ts_neg
    assert result[1]._value == expected

    # 断言：在与 ts_neg 相减时，预期会抛出 OverflowError 异常
    with pytest.raises(OverflowError, match=msg):
        dtimax - ts_neg

    # 断言：在与 ts_pos 相减时，预期会抛出 OverflowError 异常
    with pytest.raises(OverflowError, match=msg):
        dtimin - ts_pos

    # 边缘情况测试

    # 创建包含 Timestamp.min 的 DateTimeIndex 对象
    tmin = pd.to_datetime([Timestamp.min])
    # 创建一个新的日期时间对象，其值为 tmin + Timedelta.max + Timedelta("1us")
    t1 = tmin + Timedelta.max + Timedelta("1us")
    # 断言：在 t1 与 tmin 相减时，预期会抛出 OverflowError 异常
    with pytest.raises(OverflowError, match=msg):
        t1 - tmin

    # 创建包含 Timestamp.max 的 DateTimeIndex 对象
    tmax = pd.to_datetime([Timestamp.max])
    # 创建一个新的日期时间对象，其值为 tmax + Timedelta.min - Timedelta("1us")
    t2 = tmax + Timedelta.min - Timedelta("1us")
    # 断言：在 tmax 与 t2 相减时，预期会抛出 OverflowError 异常
    with pytest.raises(OverflowError, match=msg):
        tmax - t2
class TestTimestampSeriesArithmetic:
    # 测试空时间序列的加法和减法操作
    def test_empty_series_add_sub(self, box_with_array):
        # GH#13844，参考GitHub issue编号
        # 创建一个空的时间序列 a，数据类型为 "M8[ns]"
        a = Series(dtype="M8[ns]")
        # 创建一个空的时间序列 b，数据类型为 "m8[ns]"
        b = Series(dtype="m8[ns]")
        # 使用 box_with_array 函数处理时间序列 a
        a = box_with_array(a)
        # 使用 box_with_array 函数处理时间序列 b
        b = box_with_array(b)
        # 断言操作：a 等于 a 加 b
        tm.assert_equal(a, a + b)
        # 断言操作：a 等于 a 减 b
        tm.assert_equal(a, a - b)
        # 断言操作：a 等于 b 加 a
        tm.assert_equal(a, b + a)
        # 设置错误消息
        msg = "cannot subtract"
        # 使用 pytest 来断言抛出 TypeError 异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            # b 减 a 应该引发 TypeError 异常
            b - a

    # 测试时间序列的日期时间操作
    def test_operators_datetimelike(self):
        # ## timedelta64 ###
        # 创建一个时间增量序列 td1，元素为 5分钟3秒的时间增量，有3个元素
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        # 将 td1 中第三个元素设置为 NaN
        td1.iloc[2] = np.nan

        # ## datetime64 ###
        # 创建一个日期时间序列 dt1，包含三个日期时间戳
        dt1 = Series(
            [
                Timestamp("20111230"),
                Timestamp("20120101"),
                Timestamp("20120103"),
            ]
        )
        # 将 dt1 中第三个元素设置为 NaN
        dt1.iloc[2] = np.nan
        # 创建另一个日期时间序列 dt2，包含三个日期时间戳
        dt2 = Series(
            [
                Timestamp("20111231"),
                Timestamp("20120102"),
                Timestamp("20120104"),
            ]
        )
        # dt1 减 dt2，计算日期时间差
        dt1 - dt2
        # dt2 减 dt1，计算日期时间差

        # datetime64 与 timedelta64 的加法操作
        # dt1 加 td1，计算日期时间加时间增量
        dt1 + td1
        # td1 加 dt1，计算时间增量加日期时间
        td1 + dt1
        # dt1 减 td1，计算日期时间减时间增量

        # timedelta64 与 datetime64 的加法操作
        # td1 加 dt1，计算时间增量加日期时间
        td1 + dt1
        # dt1 加 td1，计算日期时间加时间增量

    # 测试时间序列与日期时间数据类型的减法操作
    def test_dt64ser_sub_datetime_dtype(self, unit):
        # 创建一个时间戳 ts，表示1993年1月7日13点30分
        ts = Timestamp(datetime(1993, 1, 7, 13, 30, 00))
        # 创建一个日期时间 dt，表示1993年6月22日13点30分
        dt = datetime(1993, 6, 22, 13, 30)
        # 创建一个时间序列 ser，包含 ts 时间戳，数据类型为指定的 unit
        ser = Series([ts], dtype=f"M8[{unit}]")
        # 计算 ser 减去 dt 的结果
        result = ser - dt

        # 预期的时间单位是 unit 和 dt 的时间单位中的最大值，这里为 "us"
        exp_unit = tm.get_finest_unit(unit, "us")
        # 断言结果的数据类型应为 timedelta64[exp_unit]
        assert result.dtype == f"timedelta64[{exp_unit}]"

    # -------------------------------------------------------------
    # TODO: 下面的测试块来自 tests.series.test_operators，需要去重和参数化 box 类

    @pytest.mark.parametrize(
        "left, right, op_fail",
        [
            [
                [Timestamp("20111230"), Timestamp("20120101"), NaT],
                [Timestamp("20111231"), Timestamp("20120102"), Timestamp("20120104")],
                ["__sub__", "__rsub__"],
            ],
            [
                [Timestamp("20111230"), Timestamp("20120101"), NaT],
                [timedelta(minutes=5, seconds=3), timedelta(minutes=5, seconds=3), NaT],
                ["__add__", "__radd__", "__sub__"],
            ],
            [
                [
                    Timestamp("20111230", tz="US/Eastern"),
                    Timestamp("20111230", tz="US/Eastern"),
                    NaT,
                ],
                [timedelta(minutes=5, seconds=3), NaT, timedelta(minutes=5, seconds=3)],
                ["__add__", "__radd__", "__sub__"],
            ],
        ],
    )
    def test_operators_datetimelike_invalid(
        self, left, right, op_fail, all_arithmetic_operators
    ):
    ):
        # 这些都是 TypeError 操作
        op_str = all_arithmetic_operators
        arg1 = Series(left)
        arg2 = Series(right)
        # 检查我们是否得到了 TypeError
        # 对于那些未定义的操作符，使用 'operate'（来自 core/ops.py）
        op = getattr(arg1, op_str, None)
        # 以前，core/indexes/base.py 中的 _validate_for_numeric_binop 为我们执行了这些操作
        if op_str not in op_fail:
            # 使用 pytest 来断言抛出 TypeError 异常，匹配特定的错误信息
            with pytest.raises(
                TypeError, match="operate|[cC]annot|unsupported operand"
            ):
                op(arg2)
        else:
            # 简单测试
            op(arg2)

    def test_sub_single_tz(self, unit):
        # GH#12290
        # 创建两个包含时区信息的 Series 对象
        s1 = Series([Timestamp("2016-02-10", tz="America/Sao_Paulo")]).dt.as_unit(unit)
        s2 = Series([Timestamp("2016-02-08", tz="America/Sao_Paulo")]).dt.as_unit(unit)
        # 计算两个 Series 的差值
        result = s1 - s2
        # 创建预期的 Series 对象，包含 Timedelta 类型的数据，单位为指定的 unit
        expected = Series([Timedelta("2days")]).dt.as_unit(unit)
        # 断言两个 Series 对象是否相等
        tm.assert_series_equal(result, expected)
        # 再次进行相减操作，验证负数结果
        result = s2 - s1
        expected = Series([Timedelta("-2days")]).dt.as_unit(unit)
        tm.assert_series_equal(result, expected)

    def test_dt64tz_series_sub_dtitz(self):
        # GH#19071 从带时区信息的 DatetimeIndex 中减去带时区信息的 Series 对象
        # (两者时区相同) 应该抛出异常，已经由 #19024 修复
        dti = date_range("1999-09-30", periods=10, tz="US/Pacific")
        ser = Series(dti)
        expected = Series(TimedeltaIndex(["0days"] * 10))

        # 执行减法操作
        res = dti - ser
        # 断言结果与预期相等
        tm.assert_series_equal(res, expected)
        # 再次进行减法操作，验证对称性
        res = ser - dti
        tm.assert_series_equal(res, expected)

    def test_sub_datetime_compat(self, unit):
        # 见 GH#14088
        # 创建包含时区信息的 Series 对象
        ser = Series([datetime(2016, 8, 23, 12, tzinfo=timezone.utc), NaT]).dt.as_unit(
            unit
        )
        dt = datetime(2016, 8, 22, 12, tzinfo=timezone.utc)
        # datetime 对象有 "us" 精度，所以我们升级为更高精度的单位
        exp_unit = tm.get_finest_unit(unit, "us")
        # 创建预期的 Series 对象，包含 Timedelta 类型的数据，单位为 exp_unit
        exp = Series([Timedelta("1 days"), NaT]).dt.as_unit(exp_unit)
        # 执行减法操作
        result = ser - dt
        # 断言结果与预期相等
        tm.assert_series_equal(result, exp)
        # 再次执行减法操作，传入 Timestamp 对象
        result2 = ser - Timestamp(dt)
        tm.assert_series_equal(result2, exp)

    def test_dt64_series_add_mixed_tick_DateOffset(self):
        # GH#4532
        # 使用 pd.offsets 进行操作
        # 创建一个包含 Timestamp 对象的 Series 对象
        s = Series([Timestamp("20130101 9:01"), Timestamp("20130101 9:02")])

        # 执行加法操作，添加 Milli(5) 偏移量
        result = s + pd.offsets.Milli(5)
        result2 = pd.offsets.Milli(5) + s
        # 创建预期的 Series 对象，包含 Timestamp 类型的数据，带有毫秒精度的时间
        expected = Series(
            [Timestamp("20130101 9:01:00.005"), Timestamp("20130101 9:02:00.005")]
        )
        # 断言结果与预期相等
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)

        # 进行连续的加法操作，添加 Minute(5) 和 Milli(5) 偏移量
        result = s + pd.offsets.Minute(5) + pd.offsets.Milli(5)
        # 创建预期的 Series 对象，包含 Timestamp 类型的数据，带有分钟和毫秒精度的时间
        expected = Series(
            [Timestamp("20130101 9:06:00.005"), Timestamp("20130101 9:07:00.005")]
        )
        # 断言结果与预期相等
        tm.assert_series_equal(result, expected)
    # 定义测试函数，用于测试 datetime64 操作中的 NaT（Not a Time）特性
    def test_datetime64_ops_nat(self, unit):
        # 创建包含 NaT 和具体日期时间戳的 Series，并将其转换为指定单位的 datetime64 类型
        datetime_series = Series([NaT, Timestamp("19900315")]).dt.as_unit(unit)
        
        # 创建包含 NaT 的 Series，并指定其数据类型为特定单位的 datetime64
        nat_series_dtype_timestamp = Series([NaT, NaT], dtype=f"datetime64[{unit}]")
        
        # 创建只包含 NaT 的 Series，并指定其数据类型为特定单位的 datetime64
        single_nat_dtype_datetime = Series([NaT], dtype=f"datetime64[{unit}]")

        # 执行减法操作
        tm.assert_series_equal(-NaT + datetime_series, nat_series_dtype_timestamp)
        msg = "bad operand type for unary -: 'DatetimeArray'"
        # 使用 pytest 断言异常抛出，确保对单个 NaT 执行减法时会引发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            -single_nat_dtype_datetime + datetime_series

        # 再次执行减法操作，验证 NaT 与指定类型的 datetime64 Series 进行运算的结果
        tm.assert_series_equal(
            -NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp
        )
        with pytest.raises(TypeError, match=msg):
            -single_nat_dtype_datetime + nat_series_dtype_timestamp

        # 执行加法操作，验证指定类型的 datetime64 Series 与 NaT 进行运算的结果
        tm.assert_series_equal(
            nat_series_dtype_timestamp + NaT, nat_series_dtype_timestamp
        )
        tm.assert_series_equal(
            NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp
        )

        # 再次验证加法操作，确保 NaT 与指定类型的 datetime64 Series 进行加法运算的结果
        tm.assert_series_equal(
            nat_series_dtype_timestamp + NaT, nat_series_dtype_timestamp
        )
        tm.assert_series_equal(
            NaT + nat_series_dtype_timestamp, nat_series_dtype_timestamp
        )

    # -------------------------------------------------------------
    # Timezone-Centric Tests
    # 定义测试函数，用于测试带有时区的日期时间操作
    def test_operators_datetimelike_with_timezones(self):
        # 设置时区字符串
        tz = "US/Eastern"
        # 创建包含时区信息的日期时间序列 dt1
        dt1 = Series(date_range("2000-01-01 09:00:00", periods=5, tz=tz), name="foo")
        # 复制 dt1 到 dt2
        dt2 = dt1.copy()
        # 将 dt2 中的第三个元素置为 NaN
        dt2.iloc[2] = np.nan

        # 创建包含时间增量的时间增量序列 td1
        td1 = Series(pd.timedelta_range("1 days 1 min", periods=5, freq="h"))
        # 复制 td1 到 td2
        td2 = td1.copy()
        # 将 td2 中的第二个元素置为 NaN
        td2.iloc[1] = np.nan
        # 断言 td2 的频率属性为 None
        assert td2._values.freq is None

        # 对 dt1 应用时间增量 td1 的第一个元素，得到结果 result
        result = dt1 + td1[0]
        # 计算期望的结果 exp，将 dt1 的时区信息去除后加上 td1 的第一个元素，并重新设置时区为 tz
        exp = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
        # 使用测试框架检查 result 是否与 exp 相等
        tm.assert_series_equal(result, exp)

        # 对 dt2 应用时间增量 td2 的第一个元素，得到结果 result
        result = dt2 + td2[0]
        # 计算期望的结果 exp，将 dt2 的时区信息去除后加上 td2 的第一个元素，并重新设置时区为 tz
        exp = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
        # 使用测试框架检查 result 是否与 exp 相等
        tm.assert_series_equal(result, exp)

        # 对于 numpy 中与标量时间增量相关的奇怪行为
        result = td1[0] + dt1
        # 计算期望的结果 exp，将 dt1 的时区信息去除后加上 td1 的第一个元素，并重新设置时区为 tz
        exp = (dt1.dt.tz_localize(None) + td1[0]).dt.tz_localize(tz)
        # 使用测试框架检查 result 是否与 exp 相等
        tm.assert_series_equal(result, exp)

        result = td2[0] + dt2
        # 计算期望的结果 exp，将 dt2 的时区信息去除后加上 td2 的第一个元素，并重新设置时区为 tz
        exp = (dt2.dt.tz_localize(None) + td2[0]).dt.tz_localize(tz)
        # 使用测试框架检查 result 是否与 exp 相等
        tm.assert_series_equal(result, exp)

        # 对 dt1 减去时间增量 td1 的第一个元素，得到结果 result
        result = dt1 - td1[0]
        # 计算期望的结果 exp，将 dt1 的时区信息去除后减去 td1 的第一个元素，并重新设置时区为 tz
        exp = (dt1.dt.tz_localize(None) - td1[0]).dt.tz_localize(tz)
        # 使用测试框架检查 result 是否与 exp 相等
        tm.assert_series_equal(result, exp)
        # 定义异常消息
        msg = "(bad|unsupported) operand type for unary"
        # 使用 pytest 框架检查是否抛出 TypeError 异常，并匹配异常消息
        with pytest.raises(TypeError, match=msg):
            td1[0] - dt1

        # 对 dt2 减去时间增量 td2 的第一个元素，得到结果 result
        result = dt2 - td2[0]
        # 计算期望的结果 exp，将 dt2 的时区信息去除后减去 td2 的第一个元素，并重新设置时区为 tz
        exp = (dt2.dt.tz_localize(None) - td2[0]).dt.tz_localize(tz)
        # 使用测试框架检查 result 是否与 exp 相等
        tm.assert_series_equal(result, exp)
        # 使用 pytest 框架检查是否抛出 TypeError 异常，并匹配异常消息
        with pytest.raises(TypeError, match=msg):
            td2[0] - dt2

        # 对 dt1 应用时间增量 td1，得到结果 result
        result = dt1 + td1
        # 计算期望的结果 exp，将 dt1 的时区信息去除后加上 td1，并重新设置时区为 tz
        exp = (dt1.dt.tz_localize(None) + td1).dt.tz_localize(tz)
        # 使用测试框架检查 result 是否与 exp 相等
        tm.assert_series_equal(result, exp)

        # 对 dt2 应用时间增量 td2，得到结果 result
        result = dt2 + td2
        # 计算期望的结果 exp，将 dt2 的时区信息去除后加上 td2，并重新设置时区为 tz
        exp = (dt2.dt.tz_localize(None) + td2).dt.tz_localize(tz)
        # 使用测试框架检查 result 是否与 exp 相等
        tm.assert_series_equal(result, exp)

        # 对 dt1 减去时间增量 td1，得到结果 result
        result = dt1 - td1
        # 计算期望的结果 exp，将 dt1 的时区信息去除后减去 td1，并重新设置时区为 tz
        exp = (dt1.dt.tz_localize(None) - td1).dt.tz_localize(tz)
        # 使用测试框架检查 result 是否与 exp 相等
        tm.assert_series_equal(result, exp)

        # 对 dt2 减去时间增量 td2，得到结果 result
        result = dt2 - td2
        # 计算期望的结果 exp，将 dt2 的时区信息去除后减去 td2，并重新设置时区为 tz
        exp = (dt2.dt.tz_localize(None) - td2).dt.tz_localize(tz)
        # 使用测试框架检查 result 是否与 exp 相等
        tm.assert_series_equal(result, exp)
        # 定义异常消息
        msg = "cannot (add|subtract)"
        # 使用 pytest 框架检查是否抛出 TypeError 异常，并匹配异常消息
        with pytest.raises(TypeError, match=msg):
            td1 - dt1
        with pytest.raises(TypeError, match=msg):
            td2 - dt2
class TestDatetimeIndexArithmetic:
    # -------------------------------------------------------------
    # Binary operations DatetimeIndex and TimedeltaIndex/array

    def test_dti_add_tdi(self, tz_naive_fixture):
        # GH#17558
        # 设置时区
        tz = tz_naive_fixture
        # 创建包含相同日期的DatetimeIndex对象
        dti = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        # 创建一个TimedeltaIndex，范围为10天
        tdi = pd.timedelta_range("0 days", periods=10)
        # 创建预期的日期范围，设定时区
        expected = date_range("2017-01-01", periods=10, tz=tz)
        expected = expected._with_freq(None)

        # 使用TimedeltaIndex执行加法操作
        result = dti + tdi
        tm.assert_index_equal(result, expected)

        # 反向顺序执行加法操作
        result = tdi + dti
        tm.assert_index_equal(result, expected)

        # 使用tdi的值数组执行加法操作
        result = dti + tdi.values
        tm.assert_index_equal(result, expected)

        # 反向顺序，使用tdi的值数组执行加法操作
        result = tdi.values + dti
        tm.assert_index_equal(result, expected)

    def test_dti_iadd_tdi(self, tz_naive_fixture):
        # GH#17558
        # 设置时区
        tz = tz_naive_fixture
        # 创建包含相同日期的DatetimeIndex对象
        dti = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        # 创建一个TimedeltaIndex，范围为10天
        tdi = pd.timedelta_range("0 days", periods=10)
        # 创建预期的日期范围，设定时区
        expected = date_range("2017-01-01", periods=10, tz=tz)
        expected = expected._with_freq(None)

        # 使用iadd操作与TimedeltaIndex相加
        result = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        result += tdi
        tm.assert_index_equal(result, expected)

        # 使用iadd操作与tdi相加
        result = pd.timedelta_range("0 days", periods=10)
        result += dti
        tm.assert_index_equal(result, expected)

        # 使用tdi的值数组执行iadd操作
        result = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        result += tdi.values
        tm.assert_index_equal(result, expected)

        # 使用tdi与日期范围执行iadd操作
        result = pd.timedelta_range("0 days", periods=10)
        result += dti
        tm.assert_index_equal(result, expected)

    def test_dti_sub_tdi(self, tz_naive_fixture):
        # GH#17558
        # 设置时区
        tz = tz_naive_fixture
        # 创建包含相同日期的DatetimeIndex对象
        dti = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10)
        # 创建一个TimedeltaIndex，范围为10天
        tdi = pd.timedelta_range("0 days", periods=10)
        # 创建预期的日期范围，设定时区，频率为每天
        expected = date_range("2017-01-01", periods=10, tz=tz, freq="-1D")
        expected = expected._with_freq(None)

        # 使用TimedeltaIndex执行减法操作
        result = dti - tdi
        tm.assert_index_equal(result, expected)

        # 检查类型错误异常
        msg = "cannot subtract .*TimedeltaArray"
        with pytest.raises(TypeError, match=msg):
            tdi - dti

        # 使用tdi的值数组执行减法操作
        result = dti - tdi.values
        tm.assert_index_equal(result, expected)

        # 检查类型错误异常
        msg = "cannot subtract a datelike from a TimedeltaArray"
        with pytest.raises(TypeError, match=msg):
            tdi.values - dti
    # 定义一个测试方法，用于测试 DatetimeIndex 和 TimedeltaIndex 的减法操作
    def test_dti_isub_tdi(self, tz_naive_fixture, unit):
        # GH#17558
        # 设置时区为 tz_naive_fixture
        tz = tz_naive_fixture
        # 创建一个 DatetimeIndex 对象，包含10个相同的日期时间戳，转换为指定时间单位
        dti = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10).as_unit(unit)
        # 创建一个 TimedeltaIndex 对象，包含10个时间间隔，指定时间单位
        tdi = pd.timedelta_range("0 days", periods=10, unit=unit)
        # 创建一个预期的 DatetimeIndex 对象，从指定日期开始，包含10个日期，带有时区信息和频率
        expected = date_range("2017-01-01", periods=10, tz=tz, freq="-1D", unit=unit)
        # 移除预期对象的频率信息
        expected = expected._with_freq(None)

        # 使用 TimedeltaIndex 对象进行 DatetimeIndex 对象的减法操作
        result = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10).as_unit(unit)
        result -= tdi
        # 断言结果与预期相等
        tm.assert_index_equal(result, expected)

        # DTA.__isub__ GH#43904
        # 复制 dti 对象的数据
        dta = dti._data.copy()
        # 对数据进行减法操作
        dta -= tdi
        # 断言处理后的日期时间数组与预期的数据数组相等
        tm.assert_datetime_array_equal(dta, expected._data)

        # 复制 dti 对象的数据
        out = dti._data.copy()
        # 使用 numpy 的减法函数，将 tdi 减去到 out 数组
        np.subtract(out, tdi, out=out)
        # 断言处理后的日期时间数组与预期的数据数组相等
        tm.assert_datetime_array_equal(out, expected._data)

        # 准备错误消息
        msg = "cannot subtract a datelike from a TimedeltaArray"
        # 预期引发 TypeError 异常，异常消息匹配 msg
        with pytest.raises(TypeError, match=msg):
            tdi -= dti

        # 使用 timedelta64 数组进行 DatetimeIndex 对象的减法操作
        result = DatetimeIndex([Timestamp("2017-01-01", tz=tz)] * 10).as_unit(unit)
        result -= tdi.values
        # 断言结果与预期相等
        tm.assert_index_equal(result, expected)

        # 预期引发 TypeError 异常，异常消息匹配 msg
        with pytest.raises(TypeError, match=msg):
            tdi.values -= dti

        # 预期引发 TypeError 异常，异常消息匹配 msg
        with pytest.raises(TypeError, match=msg):
            tdi._values -= dti

    # -------------------------------------------------------------
    # 日期时间索引和日期时间操作的二进制运算
    # TODO: 这一部分还应包含几个其他的测试，将它们移动到一个没有巨大差异的 PR 中。
    # -------------------------------------------------------------

    # 定义一个测试方法，测试 DatetimeArray 对象与索引类的交互
    def test_dta_add_sub_index(self, tz_naive_fixture):
        # 检查 DatetimeArray 对象如何延迟到索引类
        # 创建一个日期时间索引对象
        dti = date_range("20130101", periods=3, tz=tz_naive_fixture)
        # 获取该索引对象的 DatetimeArray
        dta = dti.array
        # 执行减法操作，结果与预期相等
        result = dta - dti
        expected = dti - dti
        tm.assert_index_equal(result, expected)

        # 将结果存入 tdi 变量
        tdi = result
        # 执行加法操作，结果与预期相等
        result = dta + tdi
        expected = tdi + tdi
        tm.assert_index_equal(result, expected)

        # 执行减法操作，结果与预期相等
        result = dta - tdi
        expected = tdi - tdi
        tm.assert_index_equal(result, expected)
    # 定义一个测试方法，用于测试日期时间索引相减的功能
    def test_sub_dti_dti(self, unit):
        # 之前使用过的setop（在0.16.0版本中已弃用），现在改为返回减法操作 -> 时间增量索引（GH ...）

        # 创建一个不同时区的日期时间索引对象
        dti = date_range("20130101", periods=3, unit=unit)
        # 创建一个带有时区信息的日期时间索引对象
        dti_tz = date_range("20130101", periods=3, unit=unit).tz_localize("US/Eastern")
        # 创建预期的时间增量索引对象，其值都为0
        expected = TimedeltaIndex([0, 0, 0]).as_unit(unit)

        # 对不同的日期时间索引对象进行减法操作，并断言结果与预期相等
        result = dti - dti
        tm.assert_index_equal(result, expected)

        # 对带有时区信息的日期时间索引对象进行减法操作，并断言引发TypeError异常
        result = dti_tz - dti_tz
        tm.assert_index_equal(result, expected)
        msg = "Cannot subtract tz-naive and tz-aware datetime-like objects"
        with pytest.raises(TypeError, match=msg):
            dti_tz - dti

        with pytest.raises(TypeError, match=msg):
            dti - dti_tz

        # 原地减法操作（isub）
        dti -= dti
        tm.assert_index_equal(dti, expected)

        # 如果索引对象长度不同，应引发ValueError异常
        dti1 = date_range("20130101", periods=3, unit=unit)
        dti2 = date_range("20130101", periods=4, unit=unit)
        msg = "cannot add indices of unequal length"
        with pytest.raises(ValueError, match=msg):
            dti1 - dti2

        # NaN传播测试
        dti1 = DatetimeIndex(["2012-01-01", np.nan, "2012-01-03"]).as_unit(unit)
        dti2 = DatetimeIndex(["2012-01-02", "2012-01-03", np.nan]).as_unit(unit)
        expected = TimedeltaIndex(["1 days", np.nan, np.nan]).as_unit(unit)
        result = dti2 - dti1
        tm.assert_index_equal(result, expected)

    # -------------------------------------------------------------------
    # TODO: 大部分代码块从序列或框架测试中移植过来，需要进行清理、盒子参数化和去重复

    # 参数化测试，测试timedelta64与支持的timedelta操作相等性
    @pytest.mark.parametrize("op", [operator.add, operator.sub])
    def test_timedelta64_equal_timedelta_supported_ops(self, op, box_with_array):
        # 创建一个包含时间戳的序列对象
        ser = Series(
            [
                Timestamp("20130301"),
                Timestamp("20130228 23:00:00"),
                Timestamp("20130228 22:00:00"),
                Timestamp("20130228 21:00:00"),
            ]
        )
        # 使用box_with_array函数封装序列对象
        obj = box_with_array(ser)

        # 时间间隔单位列表
        intervals = ["D", "h", "m", "s", "us"]

        # 定义timedelta64函数，根据参数和时间间隔单位列表返回np.timedelta64对象的总和
        def timedelta64(*args):
            # 见NumPy gh-12927中的类型转换说明
            return np.sum(list(starmap(np.timedelta64, zip(args, intervals))))

        # 使用product函数迭代不同的时间间隔组合
        for d, h, m, s, us in product(*([range(2)] * 5)):
            # 计算np.timedelta64对象和datetime.timedelta对象
            nptd = timedelta64(d, h, m, s, us)
            pytd = timedelta(days=d, hours=h, minutes=m, seconds=s, microseconds=us)
            # 对对象执行操作op（add或sub），并比较其结果是否相等
            lhs = op(obj, nptd)
            rhs = op(obj, pytd)

            tm.assert_equal(lhs, rhs)
    # 定义测试函数，测试混合的自然时间与日期时间类型操作
    def test_ops_nat_mixed_datetime64_timedelta64(self):
        # 创建包含 NaT 和 Timedelta("1s") 的 Series 对象
        timedelta_series = Series([NaT, Timedelta("1s")])
        # 创建包含 NaT 和 Timestamp("19900315") 的 Series 对象
        datetime_series = Series([NaT, Timestamp("19900315")])
        # 创建数据类型为 'timedelta64[ns]' 的 Series 对象，包含两个 NaT 值
        nat_series_dtype_timedelta = Series([NaT, NaT], dtype="timedelta64[ns]")
        # 创建数据类型为 'datetime64[ns]' 的 Series 对象，包含两个 NaT 值
        nat_series_dtype_timestamp = Series([NaT, NaT], dtype="datetime64[ns]")
        # 创建数据类型为 'datetime64[ns]' 的 Series 对象，包含一个 NaT 值
        single_nat_dtype_datetime = Series([NaT], dtype="datetime64[ns]")
        # 创建数据类型为 'timedelta64[ns]' 的 Series 对象，包含一个 NaT 值
        single_nat_dtype_timedelta = Series([NaT], dtype="timedelta64[ns]")

        # 时间减法操作
        tm.assert_series_equal(
            datetime_series - single_nat_dtype_datetime, nat_series_dtype_timedelta
        )

        tm.assert_series_equal(
            datetime_series - single_nat_dtype_timedelta, nat_series_dtype_timestamp
        )
        tm.assert_series_equal(
            -single_nat_dtype_timedelta + datetime_series, nat_series_dtype_timestamp
        )

        # 没有 Series 包装的 NaT 值时，其被解释为 timedelta64 或 datetime64 是不明确的，
        # 默认解释为 timedelta64
        tm.assert_series_equal(
            nat_series_dtype_timestamp - single_nat_dtype_datetime,
            nat_series_dtype_timedelta,
        )

        tm.assert_series_equal(
            nat_series_dtype_timestamp - single_nat_dtype_timedelta,
            nat_series_dtype_timestamp,
        )
        tm.assert_series_equal(
            -single_nat_dtype_timedelta + nat_series_dtype_timestamp,
            nat_series_dtype_timestamp,
        )

        # 出现错误情况，尝试用 NaT 减去日期时间类型会导致 TypeError
        msg = "cannot subtract a datelike"
        with pytest.raises(TypeError, match=msg):
            timedelta_series - single_nat_dtype_datetime

        # 时间加法操作
        tm.assert_series_equal(
            nat_series_dtype_timestamp + single_nat_dtype_timedelta,
            nat_series_dtype_timestamp,
        )
        tm.assert_series_equal(
            single_nat_dtype_timedelta + nat_series_dtype_timestamp,
            nat_series_dtype_timestamp,
        )

        tm.assert_series_equal(
            nat_series_dtype_timedelta + single_nat_dtype_datetime,
            nat_series_dtype_timestamp,
        )
        tm.assert_series_equal(
            single_nat_dtype_datetime + nat_series_dtype_timedelta,
            nat_series_dtype_timestamp,
        )
    # 定义一个测试函数，用于测试时间索引的数值运算和强制转换
    def test_ufunc_coercions(self, unit):
        # 创建一个时间索引对象 idx，从指定日期开始，周期为3天，频率为每2天一次，名称为"x"，单位为 unit
        idx = date_range("2011-01-01", periods=3, freq="2D", name="x", unit=unit)

        # 创建一个表示一天时间间隔的 timedelta 对象 delta
        delta = np.timedelta64(1, "D")
        # 创建一个期望的时间索引对象 exp，从 "2011-01-02" 开始，周期为3天，频率为每2天一次，名称为"x"，单位为 unit
        exp = date_range("2011-01-02", periods=3, freq="2D", name="x", unit=unit)
        # 对于 idx 加上 delta 后的结果，以及 np.add(idx, delta) 的结果
        for result in [idx + delta, np.add(idx, delta)]:
            # 断言结果是 DatetimeIndex 类型
            assert isinstance(result, DatetimeIndex)
            # 使用测试模块中的函数检查 result 是否等于 exp
            tm.assert_index_equal(result, exp)
            # 断言 result 的频率为 "2D"
            assert result.freq == "2D"

        # 创建一个期望的时间索引对象 exp，从 "2010-12-31" 开始，周期为3天，频率为每2天一次，名称为"x"，单位为 unit
        exp = date_range("2010-12-31", periods=3, freq="2D", name="x", unit=unit)
        # 对于 idx 减去 delta 后的结果，以及 np.subtract(idx, delta) 的结果
        for result in [idx - delta, np.subtract(idx, delta)]:
            # 断言结果是 DatetimeIndex 类型
            assert isinstance(result, DatetimeIndex)
            # 使用测试模块中的函数检查 result 是否等于 exp
            tm.assert_index_equal(result, exp)
            # 断言 result 的频率为 "2D"

        # 当对一个没有 .freq 属性的 ndarray 进行加减操作时，结果不会推断频率
        # 将 idx 的频率设置为 None
        idx = idx._with_freq(None)
        # 创建一个包含时间间隔数组的 numpy.ndarray delta
        delta = np.array(
            [np.timedelta64(1, "D"), np.timedelta64(2, "D"), np.timedelta64(3, "D")]
        )
        # 创建一个期望的时间索引对象 exp，从 ["2011-01-02", "2011-01-05", "2011-01-08"] 开始，名称为"x"，单位为 unit
        exp = DatetimeIndex(
            ["2011-01-02", "2011-01-05", "2011-01-08"], name="x"
        ).as_unit(unit)
        # 对于 idx 加上 delta 后的结果，以及 np.add(idx, delta) 的结果
        for result in [idx + delta, np.add(idx, delta)]:
            # 使用测试模块中的函数检查 result 是否等于 exp
            tm.assert_index_equal(result, exp)
            # 断言 result 的频率等于 exp 的频率

        # 创建一个期望的时间索引对象 exp，从 ["2010-12-31", "2011-01-01", "2011-01-02"] 开始，名称为"x"，单位为 unit
        exp = DatetimeIndex(
            ["2010-12-31", "2011-01-01", "2011-01-02"], name="x"
        ).as_unit(unit)
        # 对于 idx 减去 delta 后的结果，以及 np.subtract(idx, delta) 的结果
        for result in [idx - delta, np.subtract(idx, delta)]:
            # 断言结果是 DatetimeIndex 类型
            assert isinstance(result, DatetimeIndex)
            # 使用测试模块中的函数检查 result 是否等于 exp
            tm.assert_index_equal(result, exp)
            # 断言 result 的频率等于 exp 的频率
        # GH#18849, GH#19744
        other_box = index_or_series
        # 设置时区为 tz_naive_fixture
        tz = tz_naive_fixture
        # 创建一个包含两个日期的日期范围对象，带有指定的时区和名称
        dti = date_range("2017-01-01", periods=2, tz=tz, name=names[0])
        # 使用 other_box 创建另一个对象，包含两个时间偏移量
        other = other_box([pd.offsets.MonthEnd(), pd.offsets.Day(n=2)], name=names[1])
        # 获取两个对象的升级类型后的对象
        xbox = get_upcast_box(dti, other)

        # 使用 assert_produces_warning 上下文管理器，检查操作执行时是否会产生性能警告
        with tm.assert_produces_warning(performance_warning):
            # 执行操作 op，并获取结果
            res = op(dti, other)

        # 创建预期的 DatetimeIndex 对象，包含操作结果中每个日期时间对象的操作结果
        expected = DatetimeIndex(
            [op(dti[n], other[n]) for n in range(len(dti))], name=names[2], freq="infer"
        )
        # 将预期结果转换为与 xbox 类型相匹配的对象
        expected = tm.box_expected(expected, xbox).astype(object)
        # 使用 assert_equal 断言方法，检查实际结果与预期结果是否一致
        tm.assert_equal(res, expected)

    @pytest.mark.parametrize("other_box", [pd.Index, np.array])
    def test_dti_addsub_object_arraylike(
        self, performance_warning, tz_naive_fixture, box_with_array, other_box
    ):
        # 设置时区为 tz_naive_fixture
        tz = tz_naive_fixture

        # 创建一个包含两个日期的日期范围对象，带有指定的时区
        dti = date_range("2017-01-01", periods=2, tz=tz)
        # 将 dti 与 box_with_array 结合，获取新的对象
        dtarr = tm.box_expected(dti, box_with_array)
        # 使用 other_box 创建另一个对象，包含两个时间偏移量或日期时间差
        other = other_box([pd.offsets.MonthEnd(), Timedelta(days=4)])
        # 获取两个对象的升级类型后的对象
        xbox = get_upcast_box(dtarr, other)

        # 创建预期的 DatetimeIndex 对象，包含执行加法操作后的预期结果
        expected = DatetimeIndex(["2017-01-31", "2017-01-06"], tz=tz_naive_fixture)
        expected = tm.box_expected(expected, xbox).astype(object)

        # 使用 assert_produces_warning 上下文管理器，检查执行加法操作时是否会产生性能警告
        with tm.assert_produces_warning(performance_warning):
            # 执行 dtarr 与 other 的加法操作，并获取结果
            result = dtarr + other
        # 使用 assert_equal 断言方法，检查加法操作的实际结果与预期结果是否一致
        tm.assert_equal(result, expected)

        # 创建预期的 DatetimeIndex 对象，包含执行减法操作后的预期结果
        expected = DatetimeIndex(["2016-12-31", "2016-12-29"], tz=tz_naive_fixture)
        expected = tm.box_expected(expected, xbox).astype(object)

        # 使用 assert_produces_warning 上下文管理器，检查执行减法操作时是否会产生性能警告
        with tm.assert_produces_warning(performance_warning):
            # 执行 dtarr 与 other 的减法操作，并获取结果
            result = dtarr - other
        # 使用 assert_equal 断言方法，检查减法操作的实际结果与预期结果是否一致
        tm.assert_equal(result, expected)
@pytest.mark.parametrize("years", [-1, 0, 1])
@pytest.mark.parametrize("months", [-2, 0, 2])
def test_shift_months(years, months, unit):
    # 创建一个包含特定日期时间戳的DatetimeIndex对象
    dti = DatetimeIndex(
        [
            Timestamp("2000-01-05 00:15:00"),
            Timestamp("2000-01-31 00:23:00"),
            Timestamp("2000-01-01"),
            Timestamp("2000-02-29"),
            Timestamp("2000-12-31"),
        ]
    ).as_unit(unit)
    # 对DatetimeIndex对象中的日期时间戳进行月份偏移操作
    shifted = shift_months(dti.asi8, years * 12 + months, reso=dti._data._creso)
    # 将偏移后的日期时间戳视图转换为指定单位的numpy datetime64数组
    shifted_dt64 = shifted.view(f"M8[{dti.unit}]")
    # 创建新的DatetimeIndex对象，包含偏移后的日期时间戳
    actual = DatetimeIndex(shifted_dt64)

    # 计算预期的偏移结果
    raw = [x + pd.offsets.DateOffset(years=years, months=months) for x in dti]
    expected = DatetimeIndex(raw).as_unit(dti.unit)
    # 断言实际结果与预期结果相等
    tm.assert_index_equal(actual, expected)


def test_dt64arr_addsub_object_dtype_2d(performance_warning):
    # 对DataFrame块操作需要在2D的DatetimeArray/TimedeltaArray上进行，因此特别检查这种情况
    dti = date_range("1994-02-13", freq="2W", periods=4)
    # 将DatetimeIndex的内部数据重塑为2维数组
    dta = dti._data.reshape((4, 1))

    # 创建一个包含偏移量的numpy数组
    other = np.array([[pd.offsets.Day(n)] for n in range(4)])
    assert other.shape == dta.shape

    with tm.assert_produces_warning(performance_warning):
        # 执行块操作并检查警告
        result = dta + other
    with tm.assert_produces_warning(performance_warning):
        expected = (dta[:, 0] + other[:, 0]).reshape(-1, 1)

    # 断言numpy数组相等
    tm.assert_numpy_array_equal(result, expected)

    with tm.assert_produces_warning(performance_warning):
        # 预期返回TimedeltaArray的情况
        result2 = dta - dta.astype(object)

    assert result2.shape == (4, 1)
    assert all(td._value == 0 for td in result2.ravel())


def test_non_nano_dt64_addsub_np_nat_scalars():
    # GH 52295
    # 创建包含时间戳的Series对象
    ser = Series([1233242342344, 232432434324, 332434242344], dtype="datetime64[ms]")
    # 使用np.datetime64("nat", "ms")计算时间戳与自然时间的差异
    result = ser - np.datetime64("nat", "ms")
    # 创建预期的timedelta64[ms]类型的Series对象
    expected = Series([NaT] * 3, dtype="timedelta64[ms]")
    # 断言Series对象相等
    tm.assert_series_equal(result, expected)

    result = ser + np.timedelta64("nat", "ms")
    expected = Series([NaT] * 3, dtype="datetime64[ms]")
    tm.assert_series_equal(result, expected)


def test_non_nano_dt64_addsub_np_nat_scalars_unitless():
    # GH 52295
    # TODO: Can we default to the ser unit?
    # 创建包含时间戳的Series对象
    ser = Series([1233242342344, 232432434324, 332434242344], dtype="datetime64[ms]")
    # 使用np.datetime64("nat")计算时间戳与自然时间的差异
    result = ser - np.datetime64("nat")
    # 创建预期的timedelta64[ns]类型的Series对象
    expected = Series([NaT] * 3, dtype="timedelta64[ns]")
    # 断言Series对象相等
    tm.assert_series_equal(result, expected)

    result = ser + np.timedelta64("nat")
    expected = Series([NaT] * 3, dtype="datetime64[ns]")
    tm.assert_series_equal(result, expected)


def test_non_nano_dt64_addsub_np_nat_scalars_unsupported_unit():
    # GH 52295
    # 创建包含时间戳的Series对象
    ser = Series([12332, 23243, 33243], dtype="datetime64[s]")
    # 使用np.datetime64("nat", "D")计算时间戳与自然时间的差异
    result = ser - np.datetime64("nat", "D")
    # 创建预期的timedelta64[s]类型的Series对象
    expected = Series([NaT] * 3, dtype="timedelta64[s]")
    # 断言Series对象相等
    tm.assert_series_equal(result, expected)

    result = ser + np.timedelta64("nat", "D")
    expected = Series([NaT] * 3, dtype="datetime64[s]")
    tm.assert_series_equal(result, expected)
    # 使用测试工具库中的函数比较两个 Pandas Series 对象是否相等
    tm.assert_series_equal(result, expected)
```