# `D:\src\scipysrc\pandas\pandas\tests\arithmetic\test_timedelta64.py`

```
# 导入所需模块和库
from datetime import (
    datetime,    # 导入datetime类
    timedelta,   # 导入timedelta类
)

import numpy as np   # 导入NumPy库，并使用np作为别名
import pytest         # 导入pytest模块

from pandas.compat import WASM   # 导入WASM兼容模块
from pandas.errors import OutOfBoundsDatetime   # 导入OutOfBoundsDatetime异常类

import pandas as pd   # 导入Pandas库，并使用pd作为别名
from pandas import (   # 导入Pandas模块中的多个类和函数
    DataFrame,            # 导入DataFrame类
    DatetimeIndex,        # 导入DatetimeIndex类
    Index,                # 导入Index类
    NaT,                  # 导入NaT对象
    Series,               # 导入Series类
    Timedelta,            # 导入Timedelta类
    TimedeltaIndex,       # 导入TimedeltaIndex类
    Timestamp,            # 导入Timestamp类
    offsets,              # 导入offsets模块
    timedelta_range,      # 导入timedelta_range函数
)
import pandas._testing as tm   # 导入Pandas测试模块中的_tm别名
from pandas.core.arrays import NumpyExtensionArray   # 导入NumpyExtensionArray类
from pandas.tests.arithmetic.common import (   # 导入arithmetic.common模块中的多个函数和类
    assert_invalid_addsub_type,    # 导入assert_invalid_addsub_type函数
    assert_invalid_comparison,     # 导入assert_invalid_comparison函数
    get_upcast_box,                # 导入get_upcast_box函数
)


def assert_dtype(obj, expected_dtype):
    """
    Helper to check the dtype for a Series, Index, or single-column DataFrame.
    检查Series、Index或单列DataFrame的dtype是否符合预期。
    """
    dtype = tm.get_dtype(obj)   # 调用_tm.get_dtype函数获取对象的dtype

    assert dtype == expected_dtype   # 断言获取的dtype与预期dtype相等


def get_expected_name(box, names):
    """
    Determine expected name based on the type of 'box' and given 'names'.
    根据'box'的类型和给定的'names'确定预期的名称。
    """
    if box is DataFrame:
        # Since we are operating with a DataFrame and a non-DataFrame,
        # the non-DataFrame is cast to Series and its name ignored.
        # 由于我们操作的是DataFrame和非DataFrame，
        # 非DataFrame被转换为Series并忽略其名称。
        exname = names[0]
    elif box in [tm.to_array, pd.array]:
        exname = names[1]
    else:
        exname = names[2]
    return exname


# ------------------------------------------------------------------
# Timedelta64[ns] dtype Comparisons
# Timedelta64[ns] dtype比较

class TestTimedelta64ArrayLikeComparisons:
    """
    Comparison tests for timedelta64[ns] vectors fully parametrized over
    DataFrame/Series/TimedeltaIndex/TimedeltaArray.  Ideally all comparison
    tests will eventually end up here.
    """
    # 对timedelta64[ns]向量的比较测试，完全参数化为DataFrame/Series/TimedeltaIndex/TimedeltaArray。
    # 理想情况下，所有比较测试最终将在这里完成。

    def test_compare_timedelta64_zerodim(self, box_with_array):
        """
        Test for comparison of timedelta64[ns] with zero-dimensional array.
        测试timedelta64[ns]与零维数组的比较。
        """
        box = box_with_array   # 将box_with_array参数赋值给box
        xbox = box_with_array if box_with_array not in [Index, pd.array] else np.ndarray   # 根据条件选择赋值给xbox

        tdi = timedelta_range("2h", periods=4)   # 创建一个时间增量范围
        other = np.array(tdi.to_numpy()[0])   # 将tdi转换为NumPy数组，并选择第一个元素

        tdi = tm.box_expected(tdi, box)   # 使用_tm.box_expected函数将tdi装箱
        res = tdi <= other   # 比较tdi与other的大小关系
        expected = np.array([True, False, False, False])   # 预期的比较结果数组
        expected = tm.box_expected(expected, xbox)   # 使用_tm.box_expected函数将expected装箱
        tm.assert_equal(res, expected)   # 使用_tm.assert_equal函数断言res与expected相等

    @pytest.mark.parametrize(
        "td_scalar",
        [
            timedelta(days=1),                   # 创建一个时间增量为1天的timedelta对象
            Timedelta(days=1),                   # 创建一个时间增量为1天的Timedelta对象
            Timedelta(days=1).to_timedelta64(),  # 将1天转换为timedelta64格式
            offsets.Hour(24),                    # 创建一个24小时的偏移量对象
        ],
    )
    def test_compare_timedeltalike_scalar(self, box_with_array, td_scalar):
        """
        Test for comparison of timedeltalike object with scalar.
        测试timedeltalike对象与标量的比较。
        """
        box = box_with_array   # 将box_with_array参数赋值给box
        xbox = box if box not in [Index, pd.array] else np.ndarray   # 根据条件选择赋值给xbox

        ser = Series([timedelta(days=1), timedelta(days=2)])   # 创建一个包含两个时间增量的Series对象
        ser = tm.box_expected(ser, box)   # 使用_tm.box_expected函数将ser装箱
        actual = ser > td_scalar   # 比较ser与td_scalar的大小关系
        expected = Series([False, True])   # 预期的比较结果Series
        expected = tm.box_expected(expected, xbox)   # 使用_tm.box_expected函数将expected装箱
        tm.assert_equal(actual, expected)   # 使用_tm.assert_equal函数断言actual与expected相等
    @pytest.mark.parametrize(
        "invalid",
        [
            345600000000000,  # 一个整数，不符合预期的数据类型
            "a",  # 一个字符串，不符合预期的数据类型
            Timestamp("2021-01-01"),  # 一个 pandas Timestamp 对象
            Timestamp("2021-01-01").now("UTC"),  # 当前 UTC 时间的 Timestamp 对象
            Timestamp("2021-01-01").now().to_datetime64(),  # 转换为 datetime64 的 Timestamp 对象
            Timestamp("2021-01-01").now().to_pydatetime(),  # 转换为 Python datetime 的 Timestamp 对象
            Timestamp("2021-01-01").date(),  # 转换为日期的 Timestamp 对象
            np.array(4),  # 一个零维数组，类型与预期不匹配
        ],
    )
    def test_td64_comparisons_invalid(self, box_with_array, invalid):
        # GH#13624 for str
        box = box_with_array  # 将 box_with_array 赋值给 box

        rng = timedelta_range("1 days", periods=10)  # 创建一个时间范围对象，包含 10 个周期
        obj = tm.box_expected(rng, box)  # 使用 tm 模块中的 box_expected 函数进行处理

        assert_invalid_comparison(obj, invalid, box)  # 断言 obj 和 invalid 的比较结果与 box 有关

    @pytest.mark.parametrize(
        "other",
        [
            list(range(10)),  # 包含 0 到 9 的列表
            np.arange(10),  # 生成的 numpy 数组，包含 0 到 9
            np.arange(10).astype(np.float32),  # 类型转换为 np.float32 的 numpy 数组
            np.arange(10).astype(object),  # 类型转换为对象类型的 numpy 数组
            pd.date_range("1970-01-01", periods=10, tz="UTC").array,  # pandas 生成的日期时间范围的数组
            np.array(pd.date_range("1970-01-01", periods=10)),  # 转换为 numpy 数组的日期时间范围
            list(pd.date_range("1970-01-01", periods=10)),  # 转换为列表的日期时间范围
            pd.date_range("1970-01-01", periods=10).astype(object),  # 类型转换为对象类型的日期时间范围
            pd.period_range("1971-01-01", freq="D", periods=10).array,  # pandas 生成的周期范围的数组
            pd.period_range("1971-01-01", freq="D", periods=10).astype(object),  # 类型转换为对象类型的周期范围
        ],
    )
    def test_td64arr_cmp_arraylike_invalid(self, other, box_with_array):
        # 不使用 box_with_array 进行参数化，因为 listlike other 与 assert_invalid_comparison 的反向检查不兼容

        rng = timedelta_range("1 days", periods=10)._data  # 获取时间范围对象的数据
        rng = tm.box_expected(rng, box_with_array)  # 使用 tm 模块中的 box_expected 函数处理 rng 和 box_with_array

        assert_invalid_comparison(rng, other, box_with_array)  # 断言 rng 和 other 的比较结果与 box_with_array 有关

    def test_td64arr_cmp_mixed_invalid(self):
        rng = timedelta_range("1 days", periods=5)._data  # 获取时间范围对象的数据，包含 5 个周期
        other = np.array([0, 1, 2, rng[3], Timestamp("2021-01-01")])  # 包含多种类型数据的 numpy 数组

        result = rng == other  # 执行元素级的等于比较
        expected = np.array([False, False, False, True, False])  # 期望的比较结果数组
        tm.assert_numpy_array_equal(result, expected)  # 使用 tm 模块中的 assert_numpy_array_equal 断言结果与期望相同

        result = rng != other  # 执行元素级的不等于比较
        tm.assert_numpy_array_equal(result, ~expected)  # 使用 tm 模块中的 assert_numpy_array_equal 断言结果与期望相反

        msg = "Invalid comparison between|Cannot compare type|not supported between"
        with pytest.raises(TypeError, match=msg):  # 断言抛出 TypeError 异常并包含指定的错误消息
            rng < other
        with pytest.raises(TypeError, match=msg):  # 断言抛出 TypeError 异常并包含指定的错误消息
            rng > other
        with pytest.raises(TypeError, match=msg):  # 断言抛出 TypeError 异常并包含指定的错误消息
            rng <= other
        with pytest.raises(TypeError, match=msg):  # 断言抛出 TypeError 异常并包含指定的错误消息
            rng >= other
# 定义一个测试类，用于比较 Timedelta64 数组的操作
class TestTimedelta64ArrayComparisons:
    # TODO: All of these need to be parametrized over box

    # 使用 pytest 的参数化装饰器，测试比较自然时间差的情况
    @pytest.mark.parametrize("dtype", [None, object])
    def test_comp_nat(self, dtype):
        # 创建 TimedeltaIndex 对象 left 和 right，包含了不同的时间差和 NaT（Not a Time）值
        left = TimedeltaIndex([Timedelta("1 days"), NaT, Timedelta("3 days")])
        right = TimedeltaIndex([NaT, NaT, Timedelta("3 days")])

        # 根据 dtype 参数选择是否转换为 object 类型
        lhs, rhs = left, right
        if dtype is object:
            lhs, rhs = left.astype(object), right.astype(object)

        # 执行等于运算符的比较
        result = rhs == lhs
        expected = np.array([False, False, True])
        tm.assert_numpy_array_equal(result, expected)

        # 执行不等于运算符的比较
        result = rhs != lhs
        expected = np.array([True, True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 对 NaT 和 TimedeltaIndex 进行比较
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs == NaT, expected)
        tm.assert_numpy_array_equal(NaT == rhs, expected)

        # 对 NaT 和 TimedeltaIndex 进行不等于比较
        expected = np.array([True, True, True])
        tm.assert_numpy_array_equal(lhs != NaT, expected)
        tm.assert_numpy_array_equal(NaT != lhs, expected)

        # 对 NaT 和 TimedeltaIndex 进行大小比较
        expected = np.array([False, False, False])
        tm.assert_numpy_array_equal(lhs < NaT, expected)
        tm.assert_numpy_array_equal(NaT > lhs, expected)

    # 使用 pytest 的参数化装饰器，测试不同类型的 TimedeltaIndex
    @pytest.mark.parametrize(
        "idx2",
        [
            # 创建不同内容的 TimedeltaIndex 对象或者 numpy 数组
            TimedeltaIndex(
                ["2 day", "2 day", NaT, NaT, "1 day 00:00:02", "5 days 00:00:03"]
            ),
            np.array(
                [
                    np.timedelta64(2, "D"),
                    np.timedelta64(2, "D"),
                    np.timedelta64("nat"),
                    np.timedelta64("nat"),
                    np.timedelta64(1, "D") + np.timedelta64(2, "s"),
                    np.timedelta64(5, "D") + np.timedelta64(3, "s"),
                ]
            ),
        ],
    )
    # 定义一个测试方法，用于比较时间增量索引和给定的时间增量索引 idx2
    def test_comparisons_nat(self, idx2):
        # 创建一个时间增量索引 idx1，包含多个时间增量字符串和 NaT（Not a Time）值
        idx1 = TimedeltaIndex(
            [
                "1 day",
                NaT,
                "1 day 00:00:01",
                NaT,
                "1 day 00:00:01",
                "5 day 00:00:03",
            ]
        )
        # 检查 pd.NaT 是否和 np.nan 处理方式一致
        result = idx1 < idx2
        expected = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 比较 idx2 是否大于 idx1
        result = idx2 > idx1
        expected = np.array([True, False, False, False, True, False])
        tm.assert_numpy_array_equal(result, expected)

        # 比较 idx1 是否小于等于 idx2
        result = idx1 <= idx2
        expected = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)

        # 比较 idx2 是否大于等于 idx1
        result = idx2 >= idx1
        expected = np.array([True, False, False, False, True, True])
        tm.assert_numpy_array_equal(result, expected)

        # 比较 idx1 是否等于 idx2
        result = idx1 == idx2
        expected = np.array([False, False, False, False, False, True])
        tm.assert_numpy_array_equal(result, expected)

        # 比较 idx1 是否不等于 idx2
        result = idx1 != idx2
        expected = np.array([True, True, True, True, True, False])
        tm.assert_numpy_array_equal(result, expected)

    # TODO: better name
    # 定义一个测试方法，用于测试时间增量范围与指定索引列表的比较
    def test_comparisons_coverage(self):
        # 创建一个时间增量范围 rng，从 "1 days" 开始，包含 10 个时间增量周期
        rng = timedelta_range("1 days", periods=10)

        # 检查 rng 是否小于 rng[3]
        result = rng < rng[3]
        expected = np.array([True, True, True] + [False] * 7)
        tm.assert_numpy_array_equal(result, expected)

        # 检查 rng 是否等于其转换为列表后的结果
        result = rng == list(rng)
        exp = rng == rng
        tm.assert_numpy_array_equal(result, exp)
# ------------------------------------------------------------------
# Timedelta64[ns] dtype Arithmetic Operations

# 定义一个测试类，用于测试 Timedelta64 类型的算术操作，主要包括乘法、除法、取反和绝对值等操作
class TestTimedelta64ArithmeticUnsorted:
    # Tests moved from type-specific test files but not
    #  yet sorted/parametrized/de-duplicated

    # 测试函数：测试在 TimedeltaIndex 上的 ufunc 类型的协同操作
    def test_ufunc_coercions(self):
        # 创建一个 TimedeltaIndex 对象 idx，包含时间增量字符串和频率参数
        idx = TimedeltaIndex(["2h", "4h", "6h", "8h", "10h"], freq="2h", name="x")

        # 测试乘法操作：分别使用 * 运算符和 np.multiply 函数
        for result in [idx * 2, np.multiply(idx, 2)]:
            # 断言结果是 TimedeltaIndex 类型
            assert isinstance(result, TimedeltaIndex)
            # 期望的结果，是一个 TimedeltaIndex 对象，包含更新后的时间增量字符串和频率参数
            exp = TimedeltaIndex(["4h", "8h", "12h", "16h", "20h"], freq="4h", name="x")
            tm.assert_index_equal(result, exp)
            # 断言结果的频率正确更新为 "4h"
            assert result.freq == "4h"

        # 测试除法操作：分别使用 / 运算符和 np.divide 函数
        for result in [idx / 2, np.divide(idx, 2)]:
            # 断言结果是 TimedeltaIndex 类型
            assert isinstance(result, TimedeltaIndex)
            # 期望的结果，是一个 TimedeltaIndex 对象，包含更新后的时间增量字符串和频率参数
            exp = TimedeltaIndex(["1h", "2h", "3h", "4h", "5h"], freq="h", name="x")
            tm.assert_index_equal(result, exp)
            # 断言结果的频率正确更新为 "h"
            assert result.freq == "h"

        # 测试取反操作：分别使用 - 运算符和 np.negative 函数
        for result in [-idx, np.negative(idx)]:
            # 断言结果是 TimedeltaIndex 类型
            assert isinstance(result, TimedeltaIndex)
            # 期望的结果，是一个 TimedeltaIndex 对象，包含取反后的时间增量字符串和频率参数
            exp = TimedeltaIndex(
                ["-2h", "-4h", "-6h", "-8h", "-10h"], freq="-2h", name="x"
            )
            tm.assert_index_equal(result, exp)
            # 断言结果的频率正确更新为 "-2h"
            assert result.freq == "-2h"

        # 重新设置 idx 为一个包含负数时间增量的 TimedeltaIndex 对象
        idx = TimedeltaIndex(["-2h", "-1h", "0h", "1h", "2h"], freq="h", name="x")
        # 测试绝对值操作：分别使用 abs 函数和 np.absolute 函数
        for result in [abs(idx), np.absolute(idx)]:
            # 断言结果是 TimedeltaIndex 类型
            assert isinstance(result, TimedeltaIndex)
            # 期望的结果，是一个 TimedeltaIndex 对象，包含取绝对值后的时间增量字符串和频率参数
            exp = TimedeltaIndex(["2h", "1h", "0h", "1h", "2h"], freq=None, name="x")
            tm.assert_index_equal(result, exp)
            # 断言结果的频率正确更新为 None
            assert result.freq is None
    def test_subtraction_ops(self):
        # 创建一个 TimedeltaIndex 对象，包含三个元素，分别为 "1 days", NaT, "2 days"，并指定名称为 "foo"
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        # 创建一个 DatetimeIndex 对象，从 "20130101" 开始，包含三个日期时间元素，名称为 "bar"
        dti = pd.date_range("20130101", periods=3, name="bar")
        # 创建一个 Timedelta 对象，表示时间差为 1 天
        td = Timedelta("1 days")
        # 创建一个 Timestamp 对象，表示日期时间为 "20130101"
        dt = Timestamp("20130101")

        # 设置错误消息，指示从 TimedeltaArray 中减去 datelike 对象时会引发 TypeError
        msg = "cannot subtract a datelike from a TimedeltaArray"
        # 使用 pytest 检查 tdi 减去 dt 是否引发 TypeError，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            tdi - dt
        # 使用 pytest 检查 tdi 减去 dti 是否引发 TypeError，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            tdi - dti

        # 设置错误消息，指示 Timedelta 对象减去 Timestamp 对象时会引发 TypeError
        msg = r"unsupported operand type\(s\) for -"
        # 使用 pytest 检查 td 减去 dt 是否引发 TypeError，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            td - dt

        # 设置错误消息，指示 Timedelta 对象减去 TimedeltaIndex 对象时会引发 TypeError
        msg = "(bad|unsupported) operand type for unary"
        # 使用 pytest 检查 td 减去 dti 是否引发 TypeError，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            td - dti

        # 计算 dt 减去 dti 的结果
        result = dt - dti
        # 创建一个预期的 TimedeltaIndex 对象，包含 ["0 days", "-1 days", "-2 days"]，并指定名称为 "bar"
        expected = TimedeltaIndex(["0 days", "-1 days", "-2 days"], name="bar")
        # 使用 tm.assert_index_equal 检查计算结果是否与预期相等
        tm.assert_index_equal(result, expected)

        # 计算 dti 减去 dt 的结果
        result = dti - dt
        # 创建一个预期的 TimedeltaIndex 对象，包含 ["0 days", "1 days", "2 days"]，并指定名称为 "bar"
        expected = TimedeltaIndex(["0 days", "1 days", "2 days"], name="bar")
        # 使用 tm.assert_index_equal 检查计算结果是否与预期相等
        tm.assert_index_equal(result, expected)

        # 计算 tdi 减去 td 的结果
        result = tdi - td
        # 创建一个预期的 TimedeltaIndex 对象，包含 ["0 days", NaT, "1 days"]，并指定名称为 "foo"
        expected = TimedeltaIndex(["0 days", NaT, "1 days"], name="foo")
        # 使用 tm.assert_index_equal 检查计算结果是否与预期相等
        tm.assert_index_equal(result, expected)

        # 计算 td 减去 tdi 的结果
        result = td - tdi
        # 创建一个预期的 TimedeltaIndex 对象，包含 ["0 days", NaT, "-1 days"]，并指定名称为 "foo"
        expected = TimedeltaIndex(["0 days", NaT, "-1 days"], name="foo")
        # 使用 tm.assert_index_equal 检查计算结果是否与预期相等
        tm.assert_index_equal(result, expected)

        # 计算 dti 减去 td 的结果
        result = dti - td
        # 创建一个预期的 DatetimeIndex 对象，包含 ["20121231", "20130101", "20130102"]，数据类型为 "M8[ns]"，频率为 "D"，名称为 "bar"
        expected = DatetimeIndex(
            ["20121231", "20130101", "20130102"], dtype="M8[ns]", freq="D", name="bar"
        )
        # 使用 tm.assert_index_equal 检查计算结果是否与预期相等
        tm.assert_index_equal(result, expected)

        # 计算 dt 减去 tdi 的结果
        result = dt - tdi
        # 创建一个预期的 DatetimeIndex 对象，包含 ["20121231", NaT, "20121230"]，数据类型为 "M8[ns]"，名称为 "foo"
        expected = DatetimeIndex(
            ["20121231", NaT, "20121230"], dtype="M8[ns]", name="foo"
        )
        # 使用 tm.assert_index_equal 检查计算结果是否与预期相等
        tm.assert_index_equal(result, expected)

    def test_dti_tdi_numeric_ops(self):
        # 创建一个 TimedeltaIndex 对象，包含三个元素，分别为 "1 days", NaT, "2 days"，并指定名称为 "foo"
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        # 创建一个 DatetimeIndex 对象，从 "20130101" 开始，包含三个日期时间元素，名称为 "bar"
        dti = pd.date_range("20130101", periods=3, name="bar")

        # 计算 tdi 减去 tdi 的结果
        result = tdi - tdi
        # 创建一个预期的 TimedeltaIndex 对象，包含 ["0 days", NaT, "0 days"]，并指定名称为 "foo"
        expected = TimedeltaIndex(["0 days", NaT, "0 days"], name="foo")
        # 使用 tm.assert_index_equal 检查计算结果是否与预期相等
        tm.assert_index_equal(result, expected)

        # 计算 tdi 加上 tdi 的结果
        result = tdi + tdi
        # 创建一个预期的 TimedeltaIndex 对象，包含 ["2 days", NaT, "4 days"]，并指定名称为 "foo"
        expected = TimedeltaIndex(["2 days", NaT, "4 days"], name="foo")
        # 使用 tm.assert_index_equal 检查计算结果是否与预期相等
        tm.assert_index_equal(result, expected)

        # 计算 dti 减去 tdi 的结果，由于结果的名称会被重置
        result = dti - tdi
        # 创建一个预期的 DatetimeIndex 对象，包含 ["20121231", NaT, "20130101"]，数据类型为 "M8[ns]"
        expected = DatetimeIndex(["20121231", NaT, "20130101"], dtype="M8[ns]")
        # 使用 tm.assert_index_equal 检查计算结果是否与预期相等
        tm.assert_index_equal(result, expected)
    # 定义测试方法，用于测试日期时间和时间增量索引的加法操作
    def test_addition_ops(self):
        # 创建时间增量索引对象，包含字符串表示的时间增量和空值，设置索引名称为"foo"
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        # 创建日期时间索引对象，从"20130101"开始，周期为3天，设置索引名称为"bar"
        dti = pd.date_range("20130101", periods=3, name="bar")
        # 创建时间增量对象，表示1天的时间增量
        td = Timedelta("1 days")
        # 创建时间戳对象，表示日期为"20130101"
        dt = Timestamp("20130101")

        # 执行时间增量索引和时间戳的加法操作，得到结果索引对象
        result = tdi + dt
        # 创建预期的日期时间索引对象，包含对应的日期和空值，数据类型为"datetime64[ns]"，设置索引名称为"foo"
        expected = DatetimeIndex(
            ["20130102", NaT, "20130103"], dtype="M8[ns]", name="foo"
        )
        # 使用测试工具检查结果与预期是否相等
        tm.assert_index_equal(result, expected)

        # 执行时间戳和时间增量索引的加法操作，得到结果索引对象
        result = dt + tdi
        # 创建预期的日期时间索引对象，包含对应的日期和空值，数据类型为"datetime64[ns]"，设置索引名称为"foo"
        expected = DatetimeIndex(
            ["20130102", NaT, "20130103"], dtype="M8[ns]", name="foo"
        )
        # 使用测试工具检查结果与预期是否相等
        tm.assert_index_equal(result, expected)

        # 执行时间增量索引和时间增量的加法操作，得到结果索引对象
        result = td + tdi
        # 创建预期的时间增量索引对象，包含对应的时间增量和空值，设置索引名称为"foo"
        expected = TimedeltaIndex(["2 days", NaT, "3 days"], name="foo")
        # 使用测试工具检查结果与预期是否相等
        tm.assert_index_equal(result, expected)

        # 执行时间增量和时间增量索引的加法操作，得到结果索引对象
        result = tdi + td
        # 创建预期的时间增量索引对象，包含对应的时间增量和空值，设置索引名称为"foo"
        expected = TimedeltaIndex(["2 days", NaT, "3 days"], name="foo")
        # 使用测试工具检查结果与预期是否相等
        tm.assert_index_equal(result, expected)

        # 测试不同长度的索引相加会引发值错误异常
        msg = "cannot add indices of unequal length"
        with pytest.raises(ValueError, match=msg):
            tdi + dti[0:1]
        with pytest.raises(ValueError, match=msg):
            tdi[0:1] + dti

        # 测试整数和整数数组与索引相加会引发类型错误异常
        msg = "Addition/subtraction of integers and integer-arrays"
        with pytest.raises(TypeError, match=msg):
            tdi + Index([1, 2, 3], dtype=np.int64)

        # 此处是对索引的联合操作，暂时注释掉的测试用例
        # FIXME: don't leave commented-out
        # pytest.raises(TypeError, lambda : Index([1,2,3]) + tdi)

        # 执行日期时间索引和日期时间索引的加法操作，得到结果索引对象，索引名称将会重置
        result = tdi + dti  # name will be reset
        # 创建预期的日期时间索引对象，包含对应的日期和空值，数据类型为"datetime64[ns]"
        expected = DatetimeIndex(["20130102", NaT, "20130105"], dtype="M8[ns]")
        # 使用测试工具检查结果与预期是否相等
        tm.assert_index_equal(result, expected)

        # 执行日期时间索引和日期时间索引的加法操作，得到结果索引对象，索引名称将会重置
        result = dti + tdi  # name will be reset
        # 创建预期的日期时间索引对象，包含对应的日期和空值，数据类型为"datetime64[ns]"
        expected = DatetimeIndex(["20130102", NaT, "20130105"], dtype="M8[ns]")
        # 使用测试工具检查结果与预期是否相等
        tm.assert_index_equal(result, expected)

        # 执行时间戳和时间增量的加法操作，得到结果时间戳对象
        result = dt + td
        # 创建预期的时间戳对象，表示日期为"20130102"
        expected = Timestamp("20130102")
        # 检查结果与预期是否相等
        assert result == expected

        # 执行时间增量和时间戳的加法操作，得到结果时间戳对象
        result = td + dt
        # 创建预期的时间戳对象，表示日期为"20130102"
        expected = Timestamp("20130102")
        # 检查结果与预期是否相等
        assert result == expected

    # TODO: 需要更具信息性的名称，可能需要拆分成更具针对性的测试
    @pytest.mark.parametrize("freq", ["D", "B"])
    # 测试 timedelta 的功能
    def test_timedelta(self, freq):
        # 创建一个日期范围索引，从"2000年1月1日"开始，包含50个时间点，频率由参数freq指定
        index = pd.date_range("1/1/2000", periods=50, freq=freq)

        # 将索引向前偏移一天
        shifted = index + timedelta(1)
        # 将偏移后的索引再向后偏移一天
        back = shifted + timedelta(-1)
        # 使用推断的频率重新设置back索引
        back = back._with_freq("infer")
        # 检查原始索引和调整后的back索引是否相等
        tm.assert_index_equal(index, back)

        # 如果频率为'D'（天）
        if freq == "D":
            # 期望的频率为1天
            expected = pd.tseries.offsets.Day(1)
            # 检查索引和偏移后的索引的频率是否符合期望
            assert index.freq == expected
            assert shifted.freq == expected
            assert back.freq == expected
        else:  # 如果频率为'B'（工作日）
            # 期望的频率为1个工作日
            assert index.freq == pd.tseries.offsets.BusinessDay(1)
            # 偏移后的索引不应该有频率信息
            assert shifted.freq is None
            assert back.freq == pd.tseries.offsets.BusinessDay(1)

        # 对索引进行减去一天的操作
        result = index - timedelta(1)
        # 期望的结果是索引加上向前一天的结果
        expected = index + timedelta(-1)
        # 检查减法操作的结果是否符合预期
        tm.assert_index_equal(result, expected)

    # 测试 timedelta 在 tick 算术中的使用
    def test_timedelta_tick_arithmetic(self):
        # GH#4134, 与 timedelta 有 bug 的问题
        rng = pd.date_range("2013", "2014")
        s = Series(rng)
        # 使用 Hour 偏移1小时
        result1 = rng - offsets.Hour(1)
        # 从 Series 中减去 np.timedelta64(100000000)
        result2 = DatetimeIndex(s - np.timedelta64(100000000))
        # 从日期范围中减去 np.timedelta64(100000000)
        result3 = rng - np.timedelta64(100000000)
        # 从 Series 中减去 Hour(1)
        result4 = DatetimeIndex(s - offsets.Hour(1))

        # 检查 result1 和原始日期范围的频率是否相同
        assert result1.freq == rng.freq
        # 移除 result1 的频率信息
        result1 = result1._with_freq(None)
        # 检查移除频率后的 result1 是否等于 result4
        tm.assert_index_equal(result1, result4)

        # 检查 result3 和原始日期范围的频率是否相同
        assert result3.freq == rng.freq
        # 移除 result3 的频率信息
        result3 = result3._with_freq(None)
        # 检查移除频率后的 result2 是否等于 result3
        tm.assert_index_equal(result2, result3)

    # 测试 TimedeltaArray 对象与索引的加减操作
    def test_tda_add_sub_index(self):
        # 检查 TimedeltaArray 对象在算术操作时是否推迟到索引上
        tdi = TimedeltaIndex(["1 days", NaT, "2 days"])
        tda = tdi.array

        # 创建一个日期范围索引，从"1999-12-31"开始，包含3个时间点，频率为天
        dti = pd.date_range("1999-12-31", periods=3, freq="D")

        # 对 tda 和 dti 进行加法操作
        result = tda + dti
        expected = tdi + dti
        # 检查加法操作后的结果是否与期望相等
        tm.assert_index_equal(result, expected)

        # 对 tda 和 tdi 进行加法操作
        result = tda + tdi
        expected = tdi + tdi
        # 检查加法操作后的结果是否与期望相等
        tm.assert_index_equal(result, expected)

        # 对 tda 和 tdi 进行减法操作
        result = tda - tdi
        expected = tdi - tdi
        # 检查减法操作后的结果是否与期望相等
        tm.assert_index_equal(result, expected)

    # 测试 TimedeltaArray 与 datetime64 对象数组的加法操作
    def test_tda_add_dt64_object_array(
        self, performance_warning, box_with_array, tz_naive_fixture
    ):
        # 结果应该转换回 DatetimeArray 类型
        box = box_with_array

        # 创建一个带有时区的日期范围索引，从"2016-01-01"开始，包含3个时间点
        dti = pd.date_range("2016-01-01", periods=3, tz=tz_naive_fixture)
        # 移除 dti 的频率信息
        dti = dti._with_freq(None)
        # tdi 是 dti 减去自身的结果
        tdi = dti - dti

        # 使用 tm.box_expected 函数对 tdi 和 dti 进行包装
        obj = tm.box_expected(tdi, box)
        other = tm.box_expected(dti, box)

        # 使用 assert_produces_warning 上下文检查性能警告
        with tm.assert_produces_warning(performance_warning):
            # 对 obj 和 other 执行加法操作，other 被强制转换为 object 类型
            result = obj + other.astype(object)
        # 检查结果是否等于 other 的 object 类型结果
        tm.assert_equal(result, other.astype(object))
    # 定义测试方法，用于测试 timedelta_range 对象的加法操作
    def test_tdi_iadd_timedeltalike(self, two_hours, box_with_array):
        # 只测试加法操作，因为现在加号可以进行数值运算了
        rng = timedelta_range("1 days", "10 days")  # 创建一个时间间隔范围对象 rng，从 1 天到 10 天
        expected = timedelta_range("1 days 02:00:00", "10 days 02:00:00", freq="D")  # 创建预期的时间间隔范围对象 expected

        rng = tm.box_expected(rng, box_with_array)  # 调用 tm 模块的方法，对 rng 进行包装
        expected = tm.box_expected(expected, box_with_array)  # 调用 tm 模块的方法，对 expected 进行包装

        orig_rng = rng  # 记录原始的 rng 对象
        rng += two_hours  # 执行加法操作
        tm.assert_equal(rng, expected)  # 断言 rng 等于 expected
        if box_with_array is not Index:
            # 检查加法操作是否真的是原地操作
            tm.assert_equal(orig_rng, expected)  # 断言原始的 rng 等于 expected

    # 定义测试方法，用于测试 timedelta_range 对象的减法操作
    def test_tdi_isub_timedeltalike(self, two_hours, box_with_array):
        # 只测试减法操作，因为现在减号可以进行数值运算了
        rng = timedelta_range("1 days", "10 days")  # 创建一个时间间隔范围对象 rng，从 1 天到 10 天
        expected = timedelta_range("0 days 22:00:00", "9 days 22:00:00")  # 创建预期的时间间隔范围对象 expected

        rng = tm.box_expected(rng, box_with_array)  # 调用 tm 模块的方法，对 rng 进行包装
        expected = tm.box_expected(expected, box_with_array)  # 调用 tm 模块的方法，对 expected 进行包装

        orig_rng = rng  # 记录原始的 rng 对象
        rng -= two_hours  # 执行减法操作
        tm.assert_equal(rng, expected)  # 断言 rng 等于 expected
        if box_with_array is not Index:
            # 检查减法操作是否真的是原地操作
            tm.assert_equal(orig_rng, expected)  # 断言原始的 rng 等于 expected

    # -------------------------------------------------------------

    # 定义测试方法，用于测试 timedelta_range 对象的各种运算和属性
    def test_tdi_ops_attributes(self):
        rng = timedelta_range("2 days", periods=5, freq="2D", name="x")  # 创建一个时间间隔范围对象 rng，从 2 天开始，周期为 5，频率为 2 天，名称为 x

        result = rng + 1 * rng.freq  # 执行加法操作
        exp = timedelta_range("4 days", periods=5, freq="2D", name="x")  # 创建预期的时间间隔范围对象 exp
        tm.assert_index_equal(result, exp)  # 断言 result 等于 exp
        assert result.freq == "2D"  # 检查 result 的频率属性是否为 "2D"

        result = rng - 2 * rng.freq  # 执行减法操作
        exp = timedelta_range("-2 days", periods=5, freq="2D", name="x")  # 创建预期的时间间隔范围对象 exp
        tm.assert_index_equal(result, exp)  # 断言 result 等于 exp
        assert result.freq == "2D"  # 检查 result 的频率属性是否为 "2D"

        result = rng * 2  # 执行乘法操作
        exp = timedelta_range("4 days", periods=5, freq="4D", name="x")  # 创建预期的时间间隔范围对象 exp
        tm.assert_index_equal(result, exp)  # 断言 result 等于 exp
        assert result.freq == "4D"  # 检查 result 的频率属性是否为 "4D"

        result = rng / 2  # 执行除法操作
        exp = timedelta_range("1 days", periods=5, freq="D", name="x")  # 创建预期的时间间隔范围对象 exp
        tm.assert_index_equal(result, exp)  # 断言 result 等于 exp
        assert result.freq == "D"  # 检查 result 的频率属性是否为 "D"

        result = -rng  # 执行取负操作
        exp = timedelta_range("-2 days", periods=5, freq="-2D", name="x")  # 创建预期的时间间隔范围对象 exp
        tm.assert_index_equal(result, exp)  # 断言 result 等于 exp
        assert result.freq == "-2D"  # 检查 result 的频率属性是否为 "-2D"

        rng = timedelta_range("-2 days", periods=5, freq="D", name="x")  # 创建一个时间间隔范围对象 rng，从 -2 天开始，周期为 5，频率为 天，名称为 x

        result = abs(rng)  # 执行取绝对值操作
        exp = TimedeltaIndex(
            ["2 days", "1 days", "0 days", "1 days", "2 days"], name="x"
        )  # 创建预期的 TimedeltaIndex 对象 exp
        tm.assert_index_equal(result, exp)  # 断言 result 等于 exp
        assert result.freq is None  # 检查 result 的频率属性是否为 None
class TestAddSubNaTMasking:
    # TODO: parametrize over boxes

    @pytest.mark.parametrize("str_ts", ["1950-01-01", "1980-01-01"])
    def test_tdarr_add_timestamp_nat_masking(self, box_with_array, str_ts):
        # GH#17991 checking for overflow-masking with NaT
        # 创建一个包含两个元素的时间增量数组，其中一个元素为 NaT
        tdinat = pd.to_timedelta(["24658 days 11:15:00", "NaT"])
        # 使用给定的数组和数据结构对象创建预期的时间增量对象
        tdobj = tm.box_expected(tdinat, box_with_array)

        # 将字符串时间戳转换为 Timestamp 对象
        ts = Timestamp(str_ts)
        # 创建包含不同变体的时间戳列表
        ts_variants = [
            ts,
            ts.to_pydatetime(),
            ts.to_datetime64().astype("datetime64[ns]"),
            ts.to_datetime64().astype("datetime64[D]"),
        ]

        # 对于每个时间戳变体执行以下操作
        for variant in ts_variants:
            # 执行时间增量对象与时间戳变体的加法操作
            res = tdobj + variant
            # 如果数据结构对象是 DataFrame
            if box_with_array is DataFrame:
                # 断言结果的特定位置是否为 NaT
                assert res.iloc[1, 1] is NaT
            else:
                # 断言结果的特定位置是否为 NaT
                assert res[1] is NaT

    def test_tdi_add_overflow(self):
        # See GH#14068
        # preliminary test scalar analogue of vectorized tests below
        # TODO: Make raised error message more informative and test
        # 使用 pytest 检查在日期时间溢出情况下是否引发了指定异常
        with pytest.raises(OutOfBoundsDatetime, match="10155196800000000000"):
            pd.to_timedelta(106580, "D") + Timestamp("2000")
        with pytest.raises(OutOfBoundsDatetime, match="10155196800000000000"):
            Timestamp("2000") + pd.to_timedelta(106580, "D")

        # 定义一个超出范围的 NaT
        _NaT = NaT._value + 1
        msg = "Overflow in int64 addition"
        # 使用 pytest 检查在整数溢出情况下是否引发了指定异常
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([106580], "D") + Timestamp("2000")
        with pytest.raises(OverflowError, match=msg):
            Timestamp("2000") + pd.to_timedelta([106580], "D")
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta([_NaT]) - Timedelta("1 days")
        with pytest.raises(OverflowError, match=msg):
            pd.to_timedelta(["5 days", _NaT]) - Timedelta("1 days")
        with pytest.raises(OverflowError, match=msg):
            (
                pd.to_timedelta([_NaT, "5 days", "1 hours"])
                - pd.to_timedelta(["7 seconds", _NaT, "4 hours"])
            )

        # These should not overflow!
        # 验证以下操作不会引发溢出
        exp = TimedeltaIndex([NaT])
        result = pd.to_timedelta([NaT]) - Timedelta("1 days")
        tm.assert_index_equal(result, exp)

        exp = TimedeltaIndex(["4 days", NaT])
        result = pd.to_timedelta(["5 days", NaT]) - Timedelta("1 days")
        tm.assert_index_equal(result, exp)

        exp = TimedeltaIndex([NaT, NaT, "5 hours"])
        result = pd.to_timedelta([NaT, "5 days", "1 hours"]) + pd.to_timedelta(
            ["7 seconds", NaT, "4 hours"]
        )
        tm.assert_index_equal(result, exp)


class TestTimedeltaArraylikeAddSubOps:
    # Tests for timedelta64[ns] __add__, __sub__, __radd__, __rsub__

    def test_sub_nat_retain_unit(self):
        # 将 Series 对象转换为 timedelta 类型并指定单位为秒
        ser = pd.to_timedelta(Series(["00:00:01"])).astype("m8[s]")

        # 对 Series 中的每个元素执行与 NaT 的减法操作
        result = ser - NaT
        # 创建一个预期结果的 Series 对象，其值为 NaT，数据类型为 timedelta 类型，单位为秒
        expected = Series([NaT], dtype="m8[s]")
        # 断言操作结果与预期结果相等
        tm.assert_series_equal(result, expected)
    # TODO: moved from tests.indexes.timedeltas.test_arithmetic; needs
    #  parametrization+de-duplication
    # TODO: moved from tests.series.test_operators, needs splitting, cleanup,
    # de-duplication, box-parametrization...
    # 定义一个测试函数，测试timedelta64类型的操作符重载功能
    def test_operators_timedelta64(self):
        # series ops
        # 创建一个日期范围，频率为每天，生成一个Series对象v1
        v1 = pd.date_range("2012-1-1", periods=3, freq="D")
        # 创建另一个日期范围，频率为每天，生成一个Series对象v2
        v2 = pd.date_range("2012-1-2", periods=3, freq="D")
        # 计算两个Series对象的差，结果为一个timedelta64类型的Series对象rs
        rs = Series(v2) - Series(v1)
        # 创建一个与rs相同索引的Series对象，每个元素都是1e9 * 3600 * 24的整数型timedelta64[ns]
        xp = Series(1e9 * 3600 * 24, rs.index).astype("int64").astype("timedelta64[ns]")
        # 检查rs和xp是否相等
        tm.assert_series_equal(rs, xp)
        # 检查rs的数据类型是否为timedelta64[ns]
        assert rs.dtype == "timedelta64[ns]"

        # 创建一个DataFrame对象，包含一个列"A"，其值为v1的元素
        df = DataFrame({"A": v1})
        # 创建一个包含timedelta对象的Series对象td，每个元素为一天的时间间隔
        td = Series([timedelta(days=i) for i in range(3)])
        # 检查td的数据类型是否为timedelta64[ns]
        assert td.dtype == "timedelta64[ns]"

        # series on the rhs
        # 计算DataFrame列"A"与其向上移动一位后的差，结果为一个timedelta64类型的Series对象
        result = df["A"] - df["A"].shift()
        # 检查result的数据类型是否为timedelta64[ns]
        assert result.dtype == "timedelta64[ns]"

        # 计算DataFrame列"A"与Series对象td的和，结果为一个M8[ns]类型的Series对象
        result = df["A"] + td
        # 检查result的数据类型是否为M8[ns]
        assert result.dtype == "M8[ns]"

        # scalar Timestamp on rhs
        # 计算DataFrame列"A"与其最大值的差，结果为一个timedelta64类型的Series对象
        maxa = df["A"].max()
        # 检查maxa是否为Timestamp类型
        assert isinstance(maxa, Timestamp)

        # 计算DataFrame列"A"与其最大值的差，结果为一个timedelta64类型的Series对象
        resultb = df["A"] - df["A"].max()
        # 检查resultb的数据类型是否为timedelta64[ns]
        assert resultb.dtype == "timedelta64[ns]"

        # timestamp on lhs
        # 计算resultb与DataFrame列"A"的和，结果为一个包含Timestamp对象的Series对象
        result = resultb + df["A"]
        # 创建一个Timestamp对象列表
        values = [Timestamp("20111230"), Timestamp("20120101"), Timestamp("20120103")]
        # 创建一个期望的Series对象，包含上述Timestamp对象，数据类型为M8[ns]
        expected = Series(values, dtype="M8[ns]", name="A")
        # 检查result与expected是否相等
        tm.assert_series_equal(result, expected)

        # datetimes on rhs
        # 计算DataFrame列"A"与指定日期的差，结果为一个包含timedelta对象的Series对象
        result = df["A"] - datetime(2001, 1, 1)
        # 创建一个期望的Series对象，包含与指定日期的差对应的timedelta对象，数据类型为m8[ns]
        expected = Series([timedelta(days=4017 + i) for i in range(3)], name="A")
        # 检查result与expected是否相等
        tm.assert_series_equal(result, expected)
        # 检查result的数据类型是否为m8[ns]
        assert result.dtype == "m8[ns]"

        # 创建一个指定日期和时间的datetime对象
        d = datetime(2001, 1, 1, 3, 4)
        # 计算DataFrame列"A"与d的差，结果为一个timedelta64类型的Series对象
        resulta = df["A"] - d
        # 检查resulta的数据类型是否为m8[ns]
        assert resulta.dtype == "m8[ns]"

        # roundtrip
        # 计算resulta与d的和，结果应该等于原始DataFrame列"A"的值
        resultb = resulta + d
        # 检查计算结果与原始DataFrame列"A"是否相等
        tm.assert_series_equal(df["A"], resultb)

        # timedeltas on rhs
        # 创建一个时间间隔对象，表示一天的时间间隔
        td = timedelta(days=1)
        # 计算DataFrame列"A"与td的和，结果为一个M8[ns]类型的Series对象
        resulta = df["A"] + td
        # 计算resulta与td的差，结果应该等于原始DataFrame列"A"的值
        resultb = resulta - td
        # 检查计算结果与原始DataFrame列"A"是否相等
        tm.assert_series_equal(resultb, df["A"])
        # 检查resultb的数据类型是否为M8[ns]
        assert resultb.dtype == "M8[ns]"

        # roundtrip
        # 创建一个时间间隔对象，表示5分钟3秒的时间间隔
        td = timedelta(minutes=5, seconds=3)
        # 计算DataFrame列"A"与td的和，结果为一个M8[ns]类型的Series对象
        resulta = df["A"] + td
        # 计算resulta与td的差，结果应该等于原始DataFrame列"A"的值
        resultb = resulta - td
        # 检查计算结果与原始DataFrame列"A"是否相等
        tm.assert_series_equal(df["A"], resultb)
        # 检查resultb的数据类型是否为M8[ns]
        assert resultb.dtype == "M8[ns]"

        # inplace
        # 计算rs[2]与一个时间间隔对象的和，结果作为新的rs[2]值
        value = rs[2] + np.timedelta64(timedelta(minutes=5, seconds=1))
        rs[2] += np.timedelta64(timedelta(minutes=5, seconds=1))
        # 检查更新后的rs[2]是否等于预期值value
        assert rs[2] == value
    ):
        # GH#11925, GH#29558, GH#23215
        # 选择一个时区，用于测试
        tz = tz_naive_fixture

        # 创建一个具有时区信息的时间戳对象
        dt_scalar = Timestamp("2012-01-01", tz=tz)
        
        # 根据类别创建不同的时间对象
        if cls is datetime:
            # 如果是 datetime 类型，转换为 Python 的 datetime 对象
            ts = dt_scalar.to_pydatetime()
        elif cls is np.datetime64:
            # 如果是 np.datetime64 类型，根据时区信息进行相应的处理
            if tz_naive_fixture is not None:
                pytest.skip(f"{cls} doesn support {tz_naive_fixture}")
            ts = dt_scalar.to_datetime64()
        else:
            # 否则直接使用时间戳对象
            ts = dt_scalar

        # 创建一个时间间隔范围对象
        tdi = timedelta_range("1 day", periods=3)
        # 创建一个预期的日期范围对象，带有时区信息
        expected = pd.date_range("2012-01-02", periods=3, tz=tz)

        # 对时间间隔范围对象应用预期的函数，返回一个处理后的数组
        tdarr = tm.box_expected(tdi, box_with_array)
        # 对预期日期范围对象应用预期的函数，返回一个处理后的数组
        expected = tm.box_expected(expected, box_with_array)

        # 断言时间戳加上时间间隔数组后的结果是否与预期相等
        tm.assert_equal(ts + tdarr, expected)
        # 断言时间间隔数组加上时间戳后的结果是否与预期相等
        tm.assert_equal(tdarr + ts, expected)

        # 创建另一个预期的日期范围对象，带有时区信息，但日期向前一天
        expected2 = pd.date_range("2011-12-31", periods=3, freq="-1D", tz=tz)
        # 对预期日期范围对象应用预期的函数，返回一个处理后的数组
        expected2 = tm.box_expected(expected2, box_with_array)

        # 断言时间戳减去时间间隔数组后的结果是否与预期相等
        tm.assert_equal(ts - tdarr, expected2)
        # 断言时间戳加上负的时间间隔数组后的结果是否与预期相等
        tm.assert_equal(ts + (-tdarr), expected2)

        # 设置错误消息，测试时间间隔数组减去时间戳的操作是否会引发 TypeError 异常
        msg = "cannot subtract a datelike"
        with pytest.raises(TypeError, match=msg):
            tdarr - ts

    def test_td64arr_add_datetime64_nat(self, box_with_array):
        # GH#23215
        # 创建一个 np.datetime64 类型的 NaT（Not a Time）对象
        other = np.datetime64("NaT")

        # 创建一个时间间隔范围对象
        tdi = timedelta_range("1 day", periods=3)
        # 创建一个预期的日期时间索引对象，其中包含三个 NaT
        expected = DatetimeIndex(["NaT", "NaT", "NaT"], dtype="M8[ns]")

        # 对时间间隔范围对象应用预期的函数，返回一个处理后的数组
        tdser = tm.box_expected(tdi, box_with_array)
        # 对预期的日期时间索引对象应用预期的函数，返回一个处理后的数组
        expected = tm.box_expected(expected, box_with_array)

        # 断言时间间隔数组加上 NaT 后的结果是否与预期相等
        tm.assert_equal(tdser + other, expected)
        # 断言 NaT 加上时间间隔数组后的结果是否与预期相等
        tm.assert_equal(other + tdser, expected)

    def test_td64arr_sub_dt64_array(self, box_with_array):
        # 创建一个日期时间索引对象
        dti = pd.date_range("2016-01-01", periods=3)
        # 创建一个时间间隔索引对象
        tdi = TimedeltaIndex(["-1 Day"] * 3)
        # 从日期时间索引对象获取其数值，形成一个 np.datetime64 类型的数组
        dtarr = dti.values
        # 创建一个预期的日期时间索引对象，通过数组减去时间间隔索引对象得到
        expected = DatetimeIndex(dtarr) - tdi

        # 对时间间隔索引对象应用预期的函数，返回一个处理后的数组
        tdi = tm.box_expected(tdi, box_with_array)
        # 对预期的日期时间索引对象应用预期的函数，返回一个处理后的数组
        expected = tm.box_expected(expected, box_with_array)

        # 设置错误消息，测试时间间隔索引对象减去日期时间索引数组是否会引发 TypeError 异常
        msg = "cannot subtract a datelike from"
        with pytest.raises(TypeError, match=msg):
            tdi - dtarr

        # 执行日期时间索引数组减去时间间隔索引对象的操作
        result = dtarr - tdi
        # 断言结果是否与预期相等
        tm.assert_equal(result, expected)

    def test_td64arr_add_dt64_array(self, box_with_array):
        # 创建一个日期时间索引对象
        dti = pd.date_range("2016-01-01", periods=3)
        # 创建一个时间间隔索引对象
        tdi = TimedeltaIndex(["-1 Day"] * 3)
        # 从日期时间索引对象获取其数值，形成一个 np.datetime64 类型的数组
        dtarr = dti.values
        # 创建一个预期的日期时间索引对象，通过数组加上时间间隔索引对象得到
        expected = DatetimeIndex(dtarr) + tdi

        # 对时间间隔索引对象应用预期的函数，返回一个处理后的数组
        tdi = tm.box_expected(tdi, box_with_array)
        # 对预期的日期时间索引对象应用预期的函数，返回一个处理后的数组
        expected = tm.box_expected(expected, box_with_array)

        # 执行时间间隔索引对象加上日期时间索引数组的操作
        result = tdi + dtarr
        # 断言结果是否与预期相等
        tm.assert_equal(result, expected)
        # 执行日期时间索引数组加上时间间隔索引对象的操作
        result = dtarr + tdi
        # 断言结果是否与预期相等
        tm.assert_equal(result, expected)

    # ------------------------------------------------------------------
    # Invalid __add__/__sub__ operations

    @pytest.mark.parametrize("pi_freq", ["D", "W", "Q", "h"])
    @pytest.mark.parametrize("tdi_freq", [None, "h"])
    def test_td64arr_sub_periodlike(
        self, box_with_array, box_with_array2, tdi_freq, pi_freq
    ):
        # GH#20049 subtracting PeriodIndex should raise TypeError
        # 创建一个时间增量索引对象，包含两个小时的时间增量，使用给定的频率参数
        tdi = TimedeltaIndex(["1 hours", "2 hours"], freq=tdi_freq)
        # 创建一个时间戳对象，并加上时间增量索引对象
        dti = Timestamp("2018-03-07 17:16:40") + tdi
        # 将时间戳对象转换为周期对象，使用给定的周期频率参数
        pi = dti.to_period(pi_freq)
        # 获取周期对象的第一个周期
        per = pi[0]

        # 对时间增量索引对象和周期对象进行类型检查，预期会引发 TypeError 异常，异常信息包含特定字符串
        tdi = tm.box_expected(tdi, box_with_array)
        pi = tm.box_expected(pi, box_with_array2)
        msg = "cannot subtract|unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            tdi - pi

        # GH#13078 subtraction of Period scalar not supported
        # 对周期对象和周期标量进行减法操作的类型检查，预期会引发 TypeError 异常，异常信息包含特定字符串
        with pytest.raises(TypeError, match=msg):
            tdi - per

    @pytest.mark.parametrize(
        "other",
        [
            # GH#12624 for str case
            "a",
            # GH#19123
            1,
            1.5,
            np.array(2),
        ],
    )
    def test_td64arr_addsub_numeric_scalar_invalid(self, box_with_array, other):
        # 创建一个时间序列对象，包含"59 Days"、"59 Days"和"NaT"，数据类型为'm8[ns]'
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        # 将时间序列对象包装成期望的类型，使用给定的数组盒子
        tdarr = tm.box_expected(tdser, box_with_array)

        # 调用函数进行无效的加法和减法类型检查，预期会引发异常
        assert_invalid_addsub_type(tdarr, other)

    @pytest.mark.parametrize(
        "vec",
        [
            np.array([1, 2, 3]),
            Index([1, 2, 3]),
            Series([1, 2, 3]),
            DataFrame([[1, 2, 3]]),
        ],
        ids=lambda x: type(x).__name__,
    )
    def test_td64arr_addsub_numeric_arr_invalid(
        self, box_with_array, vec, any_real_numpy_dtype
    ):
        # 创建一个时间序列对象，包含"59 Days"、"59 Days"和"NaT"，数据类型为'm8[ns]'
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        # 将时间序列对象包装成期望的类型，使用给定的数组盒子
        tdarr = tm.box_expected(tdser, box_with_array)

        # 将向量对象转换为指定的NumPy数据类型
        vector = vec.astype(any_real_numpy_dtype)
        # 调用函数进行无效的加法和减法类型检查，预期会引发异常
        assert_invalid_addsub_type(tdarr, vector)

    def test_td64arr_add_sub_int(self, box_with_array, one):
        # 创建一个时间增量范围对象，从指定时间点开始，每小时频率，共10个时间点
        rng = timedelta_range("1 days 09:00:00", freq="h", periods=10)
        # 将时间增量范围对象包装成期望的类型，使用给定的数组盒子
        tdarr = tm.box_expected(rng, box_with_array)

        msg = "Addition/subtraction of integers"
        # 调用函数进行无效的加法和减法类型检查，预期会引发 TypeError 异常，异常信息包含特定字符串
        assert_invalid_addsub_type(tdarr, one, msg)

        # TODO: get inplace ops into assert_invalid_addsub_type
        with pytest.raises(TypeError, match=msg):
            tdarr += one
        with pytest.raises(TypeError, match=msg):
            tdarr -= one

    def test_td64arr_add_sub_integer_array(self, box_with_array):
        # GH#19959, deprecated GH#22535
        # GH#22696 for DataFrame case, check that we don't dispatch to numpy
        #  implementation, which treats int64 as m8[ns]
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        # 创建一个时间增量范围对象，从指定时间点开始，每小时频率，共3个时间点
        rng = timedelta_range("1 days 09:00:00", freq="h", periods=3)
        # 将时间增量范围对象包装成期望的类型，使用给定的数组盒子
        tdarr = tm.box_expected(rng, box)
        # 创建一个包含整数的数组对象，使用给定的数组盒子
        other = tm.box_expected([4, 3, 2], xbox)

        msg = "Addition/subtraction of integers and integer-arrays"
        # 调用函数进行无效的加法和减法类型检查，预期会引发异常
        assert_invalid_addsub_type(tdarr, other, msg)
    # 定义一个测试方法，用于测试不带频率的整数数组的加减运算
    def test_td64arr_addsub_integer_array_no_freq(self, box_with_array):
        # GH#19959
        # 获取传入的 box_with_array 参数并赋值给 box 变量
        box = box_with_array
        # 如果 box 是 pd.array，则将 np.ndarray 赋值给 xbox，否则直接使用 box
        xbox = np.ndarray if box is pd.array else box

        # 创建一个时间增量索引对象 tdi，包含 "1 Day", "NaT", "3 Hours" 三个时间增量
        tdi = TimedeltaIndex(["1 Day", "NaT", "3 Hours"])
        # 使用测试数据和 box 创建时间增量数组 tdarr
        tdarr = tm.box_expected(tdi, box)
        # 创建另一个数组 other，包含 [14, -1, 16]，并根据 box 类型进行转换
        other = tm.box_expected([14, -1, 16], xbox)

        # 设置错误消息提示
        msg = "Addition/subtraction of integers"
        # 断言 tdarr 和 other 不能进行加减运算，验证错误消息
        assert_invalid_addsub_type(tdarr, other, msg)

    # ------------------------------------------------------------------
    # Operations with timedelta-like others

    # 定义一个测试方法，用于测试时间增量数组与 td64 数组的加减运算
    def test_td64arr_add_sub_td64_array(self, box_with_array):
        # 获取传入的 box_with_array 参数并赋值给 box 变量
        box = box_with_array
        # 创建一个日期范围对象 dti，从 "2016-01-01" 开始的三天日期
        dti = pd.date_range("2016-01-01", periods=3)
        # 计算日期范围对象 dti 的时间增量，并赋值给 tdi
        tdi = dti - dti.shift(1)
        # 获取 tdi 的值作为 tdarr，即时间增量数组
        tdarr = tdi.values

        # 计算预期的结果 expected，是 tdi 的两倍
        expected = 2 * tdi
        # 根据 box 类型将 tdi 和 expected 转换为相应类型
        tdi = tm.box_expected(tdi, box)
        expected = tm.box_expected(expected, box)

        # 执行 tdi + tdarr 运算，验证结果与预期相等
        result = tdi + tdarr
        tm.assert_equal(result, expected)
        # 执行 tdarr + tdi 运算，验证结果与预期相等
        result = tdarr + tdi
        tm.assert_equal(result, expected)

        # 计算预期的减法结果 expected_sub，为 tdi 减去 tdarr 的结果
        expected_sub = 0 * tdi
        # 执行 tdi - tdarr 运算，验证结果与预期相等
        result = tdi - tdarr
        tm.assert_equal(result, expected_sub)
        # 执行 tdarr - tdi 运算，验证结果与预期相等
        result = tdarr - tdi
        tm.assert_equal(result, expected_sub)

    # 定义一个测试方法，用于测试时间增量数组与时间增量索引对象 tdi 的加减运算
    def test_td64arr_add_sub_tdi(self, box_with_array, names):
        # GH#17250 确保结果的数据类型正确
        # GH#19043 确保名称正确传播
        # 获取传入的 box_with_array 参数并赋值给 box 变量
        box = box_with_array
        # 根据 box 和 names 获取预期的名称 exname
        exname = get_expected_name(box, names)

        # 创建一个时间增量索引对象 tdi，包含 ["0 days", "1 day"]，并设置名称
        tdi = TimedeltaIndex(["0 days", "1 day"], name=names[1])
        # 根据 box 的类型转换 tdi 的数据类型为 np.array 或保持原样
        tdi = np.array(tdi) if box in [tm.to_array, pd.array] else tdi
        # 创建一个时间增量序列对象 ser，包含 [Timedelta(hours=3), Timedelta(hours=4)]，并设置名称
        ser = Series([Timedelta(hours=3), Timedelta(hours=4)], name=names[0])
        # 创建预期的时间增量序列对象 expected，包含 [Timedelta(hours=3), Timedelta(days=1, hours=4)]，并设置名称
        expected = Series([Timedelta(hours=3), Timedelta(days=1, hours=4)], name=exname)

        # 根据 box 的类型将 ser 和 expected 转换为相应类型
        ser = tm.box_expected(ser, box)
        expected = tm.box_expected(expected, box)

        # 执行 tdi + ser 运算，验证结果与预期相等
        result = tdi + ser
        tm.assert_equal(result, expected)
        # 断言结果的数据类型为 timedelta64[ns]
        assert_dtype(result, "timedelta64[ns]")

        # 执行 ser + tdi 运算，验证结果与预期相等
        result = ser + tdi
        tm.assert_equal(result, expected)
        # 断言结果的数据类型为 timedelta64[ns]

        # 创建预期的减法结果 expected，包含 [Timedelta(hours=-3), Timedelta(days=1, hours=-4)]，并设置名称
        expected = Series(
            [Timedelta(hours=-3), Timedelta(days=1, hours=-4)], name=exname
        )
        # 根据 box 的类型将 expected 转换为相应类型
        expected = tm.box_expected(expected, box)

        # 执行 tdi - ser 运算，验证结果与预期相等
        result = tdi - ser
        tm.assert_equal(result, expected)
        # 断言结果的数据类型为 timedelta64[ns]

        # 执行 ser - tdi 运算，验证结果为预期的相反数
        tm.assert_equal(result, -expected)
        # 断言结果的数据类型为 timedelta64[ns]

    @pytest.mark.parametrize("tdnat", [np.timedelta64("NaT"), NaT])
    def test_td64arr_add_sub_td64_nat(self, box_with_array, tdnat):
        # GH#18808, GH#23320 special handling for timedelta64("NaT")
        # 设置测试用例的基础对象为输入的数组盒子
        box = box_with_array
        # 创建一个时间增量索引，包含 NaT 和 1 秒的时间增量
        tdi = TimedeltaIndex([NaT, Timedelta("1s")])
        # 期望的结果是一个时间增量索引，包含两个 "NaT" 值
        expected = TimedeltaIndex(["NaT"] * 2)

        # 对基础对象进行封装，返回封装后的对象
        obj = tm.box_expected(tdi, box)
        # 对期望结果进行封装，返回封装后的对象
        expected = tm.box_expected(expected, box)

        # 执行对象加上 NaT 的操作，并断言结果是否与期望相同
        result = obj + tdnat
        tm.assert_equal(result, expected)
        # 执行 NaT 加上对象的操作，并断言结果是否与期望相同
        result = tdnat + obj
        tm.assert_equal(result, expected)
        # 执行对象减去 NaT 的操作，并断言结果是否与期望相同
        result = obj - tdnat
        tm.assert_equal(result, expected)
        # 执行 NaT 减去对象的操作，并断言结果是否与期望相同
        result = tdnat - obj
        tm.assert_equal(result, expected)

    def test_td64arr_add_timedeltalike(self, two_hours, box_with_array):
        # only test adding/sub offsets as + is now numeric
        # GH#10699 for Tick cases
        # 设置测试用例的基础对象为输入的数组盒子
        box = box_with_array
        # 创建一个时间增量范围，从 "1 days" 到 "10 days"
        rng = timedelta_range("1 days", "10 days")
        # 期望的结果是一个时间增量范围，从 "1 days 02:00:00" 到 "10 days 02:00:00"，频率为每天
        expected = timedelta_range("1 days 02:00:00", "10 days 02:00:00", freq="D")
        # 对时间增量范围进行封装，返回封装后的对象
        rng = tm.box_expected(rng, box)
        # 对期望结果进行封装，返回封装后的对象
        expected = tm.box_expected(expected, box)

        # 执行时间增量范围加上两小时的操作，并断言结果是否与期望相同
        result = rng + two_hours
        tm.assert_equal(result, expected)

        # 执行两小时加上时间增量范围的操作，并断言结果是否与期望相同
        result = two_hours + rng
        tm.assert_equal(result, expected)

    def test_td64arr_sub_timedeltalike(self, two_hours, box_with_array):
        # only test adding/sub offsets as - is now numeric
        # GH#10699 for Tick cases
        # 设置测试用例的基础对象为输入的数组盒子
        box = box_with_array
        # 创建一个时间增量范围，从 "1 days" 到 "10 days"
        rng = timedelta_range("1 days", "10 days")
        # 期望的结果是一个时间增量范围，从 "0 days 22:00:00" 到 "9 days 22:00:00"
        expected = timedelta_range("0 days 22:00:00", "9 days 22:00:00")

        # 对时间增量范围进行封装，返回封装后的对象
        rng = tm.box_expected(rng, box)
        # 对期望结果进行封装，返回封装后的对象
        expected = tm.box_expected(expected, box)

        # 执行时间增量范围减去两小时的操作，并断言结果是否与期望相同
        result = rng - two_hours
        tm.assert_equal(result, expected)

        # 执行两小时减去时间增量范围的操作，并断言结果是否与期望相反
        result = two_hours - rng
        tm.assert_equal(result, -expected)

    # ------------------------------------------------------------------
    # __add__/__sub__ with DateOffsets and arrays of DateOffsets

    def test_td64arr_add_sub_offset_index(
        self, performance_warning, names, box_with_array
    ):
        # GH#18849, GH#19744
        # 将 box_with_array 参数赋值给 box 变量
        box = box_with_array
        # 根据 box 和 names 获取预期的名称
        exname = get_expected_name(box, names)

        # 创建一个时间增量索引对象 tdi，包含两个时间间隔字符串，名称为 names[0]
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"], name=names[0])
        # 创建一个索引对象 other，包含两个偏移量对象，名称为 names[1]
        other = Index([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])
        # 如果 box 在 [tm.to_array, pd.array] 中，则将 other 转换为 numpy 数组，否则保持不变
        other = np.array(other) if box in [tm.to_array, pd.array] else other

        # 创建预期的时间增量索引对象 expected，其元素为 tdi[n] + other[n] 的结果，频率为 "infer"，名称为 exname
        expected = TimedeltaIndex(
            [tdi[n] + other[n] for n in range(len(tdi))], freq="infer", name=exname
        )
        # 创建预期的时间增量索引对象 expected_sub，其元素为 tdi[n] - other[n] 的结果，频率为 "infer"，名称为 exname
        expected_sub = TimedeltaIndex(
            [tdi[n] - other[n] for n in range(len(tdi))], freq="infer", name=exname
        )

        # 使用 tm.box_expected 函数将 tdi 转换为预期的数据类型
        tdi = tm.box_expected(tdi, box)
        # 使用 tm.box_expected 函数将 expected 转换为预期的数据类型，并转换为对象类型
        expected = tm.box_expected(expected, box).astype(object)
        # 使用 tm.box_expected 函数将 expected_sub 转换为预期的数据类型，并转换为对象类型
        expected_sub = tm.box_expected(expected_sub, box).astype(object)

        # 使用 assert_produces_warning 上下文管理器确保执行加法操作时产生性能警告
        with tm.assert_produces_warning(performance_warning):
            res = tdi + other
        # 断言 res 等于 expected
        tm.assert_equal(res, expected)

        # 使用 assert_produces_warning 上下文管理器确保执行加法操作时产生性能警告
        with tm.assert_produces_warning(performance_warning):
            res2 = other + tdi
        # 断言 res2 等于 expected
        tm.assert_equal(res2, expected)

        # 使用 assert_produces_warning 上下文管理器确保执行减法操作时产生性能警告
        with tm.assert_produces_warning(performance_warning):
            res_sub = tdi - other
        # 断言 res_sub 等于 expected_sub
        tm.assert_equal(res_sub, expected_sub)

    # 测试处理时间增量数组与偏移量数组的加减法操作，包含性能警告参数和 box_with_array 参数
    def test_td64arr_add_sub_offset_array(self, performance_warning, box_with_array):
        # GH#18849, GH#18824
        # 将 box_with_array 参数赋值给 box 变量
        box = box_with_array
        # 创建时间增量索引对象 tdi，包含两个时间间隔字符串
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"])
        # 创建一个包含两个偏移量对象的 numpy 数组 other
        other = np.array([offsets.Hour(n=1), offsets.Minute(n=-2)])

        # 创建预期的时间增量索引对象 expected，其元素为 tdi[n] + other[n] 的结果，频率为 "infer"
        expected = TimedeltaIndex(
            [tdi[n] + other[n] for n in range(len(tdi))], freq="infer"
        )
        # 创建预期的时间增量索引对象 expected_sub，其元素为 tdi[n] - other[n] 的结果，频率为 "infer"
        expected_sub = TimedeltaIndex(
            [tdi[n] - other[n] for n in range(len(tdi))], freq="infer"
        )

        # 使用 tm.box_expected 函数将 tdi 转换为预期的数据类型
        tdi = tm.box_expected(tdi, box)

        # 使用 assert_produces_warning 上下文管理器确保执行加法操作时产生性能警告
        with tm.assert_produces_warning(performance_warning):
            res = tdi + other
        # 断言 res 等于 expected
        tm.assert_equal(res, expected)

        # 使用 assert_produces_warning 上下文管理器确保执行加法操作时产生性能警告
        with tm.assert_produces_warning(performance_warning):
            res2 = other + tdi
        # 断言 res2 等于 expected
        tm.assert_equal(res2, expected)

        # 使用 box_with_array 参数作为 box 执行 tm.box_expected 函数转换 expected_sub
        expected_sub = tm.box_expected(expected_sub, box_with_array).astype(object)
        # 使用 assert_produces_warning 上下文管理器确保执行减法操作时产生性能警告
        with tm.assert_produces_warning(performance_warning):
            res_sub = tdi - other
        # 断言 res_sub 等于 expected_sub
        tm.assert_equal(res_sub, expected_sub)

    # 测试时间增量数组与偏移量序列的加减法操作，包含性能警告参数、名称和 box_with_array 参数
    def test_td64arr_with_offset_series(
        self, performance_warning, names, box_with_array
    ):
        # GH#18849
        # 将 box_with_array 赋值给 box
        box = box_with_array
        # 如果 box 在 [Index, tm.to_array, pd.array] 中，则将 box2 设为 Series，否则保持 box 不变
        box2 = Series if box in [Index, tm.to_array, pd.array] else box
        # 获取预期的名称
        exname = get_expected_name(box, names)

        # 创建一个 TimedeltaIndex，包含两个时间差对象
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"], name=names[0])
        # 创建一个包含两个时间偏移对象的 Series
        other = Series([offsets.Hour(n=1), offsets.Minute(n=-2)], name=names[1])

        # 创建一个预期结果的 Series，包含 tdi 和 other 对应位置元素相加的结果
        expected_add = Series(
            [tdi[n] + other[n] for n in range(len(tdi))], name=exname, dtype=object
        )
        # 使用 tm.box_expected 封装 tdi 对象，并根据 box 类型返回对应结果的 obj
        obj = tm.box_expected(tdi, box)
        # 使用 tm.box_expected 封装 expected_add 对象，并根据 box2 类型返回对应结果
        expected_add = tm.box_expected(expected_add, box2).astype(object)

        # 使用 assert_produces_warning 检查性能警告
        with tm.assert_produces_warning(performance_warning):
            # 执行加法操作
            res = obj + other
        # 使用 assert_equal 检查 res 和 expected_add 是否相等
        tm.assert_equal(res, expected_add)

        # 使用 assert_produces_warning 检查性能警告
        with tm.assert_produces_warning(performance_warning):
            # 执行加法操作（顺序相反）
            res2 = other + obj
        # 使用 assert_equal 检查 res2 和 expected_add 是否相等
        tm.assert_equal(res2, expected_add)

        # 创建一个预期结果的 Series，包含 tdi 和 other 对应位置元素相减的结果
        expected_sub = Series(
            [tdi[n] - other[n] for n in range(len(tdi))], name=exname, dtype=object
        )
        # 使用 tm.box_expected 封装 expected_sub 对象，并根据 box2 类型返回对应结果
        expected_sub = tm.box_expected(expected_sub, box2).astype(object)

        # 使用 assert_produces_warning 检查性能警告
        with tm.assert_produces_warning(performance_warning):
            # 执行减法操作
            res3 = obj - other
        # 使用 assert_equal 检查 res3 和 expected_sub 是否相等
        tm.assert_equal(res3, expected_sub)

    @pytest.mark.parametrize("obox", [np.array, Index, Series])
    def test_td64arr_addsub_anchored_offset_arraylike(
        self, performance_warning, obox, box_with_array
    ):
        # GH#18824
        # 创建一个 TimedeltaIndex，包含两个时间差对象
        tdi = TimedeltaIndex(["1 days 00:00:00", "3 days 04:00:00"])
        # 使用 tm.box_expected 封装 tdi 对象，并根据 box_with_array 类型返回对应结果
        tdi = tm.box_expected(tdi, box_with_array)

        # 创建一个 obox 对象，包含两个时间偏移对象 MonthEnd 和 Day(n=2)
        anchored = obox([offsets.MonthEnd(), offsets.Day(n=2)])

        # 使用 pytest.raises 检查 TypeError 异常，匹配指定的错误消息，确保加法和减法操作会发出性能警告并引发 TypeError
        msg = "has incorrect type|cannot add the type MonthEnd"
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                # 执行 tdi + anchored 加法操作
                tdi + anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                # 执行 anchored + tdi 加法操作（顺序相反）
                anchored + tdi
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                # 执行 tdi - anchored 减法操作
                tdi - anchored
        with pytest.raises(TypeError, match=msg):
            with tm.assert_produces_warning(performance_warning):
                # 执行 anchored - tdi 减法操作（顺序相反）
                anchored - tdi
    # 定义测试函数，用于测试 timedelta 数组与对象数组的加法和减法操作
    def test_td64arr_add_sub_object_array(self, performance_warning, box_with_array):
        # 将 box_with_array 参数赋值给变量 box
        box = box_with_array
        # 如果 box 是 pd.array，则将 np.ndarray 赋给 xbox，否则将 box 赋给 xbox
        xbox = np.ndarray if box is pd.array else box

        # 创建一个时间增量范围，从 "1 day" 开始，总共 3 个时间点，频率为每天一次
        tdi = timedelta_range("1 day", periods=3, freq="D")
        # 根据 box 对象包装 tdi，返回一个时间增量数组 tdarr
        tdarr = tm.box_expected(tdi, box)

        # 创建另一个包含时间增量对象的 numpy 数组 other
        other = np.array([Timedelta(days=1), offsets.Day(2), Timestamp("2000-01-04")])

        # 使用 pytest 的 assert_produces_warning 上下文，检查是否产生性能警告
        with tm.assert_produces_warning(performance_warning):
            # 执行 tdarr 与 other 的加法操作，将结果赋给 result
            result = tdarr + other

        # 创建预期结果 Index 对象，包含预期的时间增量和时间戳对象
        expected = Index(
            [Timedelta(days=2), Timedelta(days=4), Timestamp("2000-01-07")]
        )
        # 将预期结果经过 box_expected 包装，并转换为 object 类型
        expected = tm.box_expected(expected, xbox).astype(object)
        # 使用 tm.assert_equal 断言 result 和 expected 相等
        tm.assert_equal(result, expected)

        # 准备用于错误检查的消息字符串
        msg = "unsupported operand type|cannot subtract a datelike"
        # 使用 pytest.raises 检查是否引发 TypeError，并匹配预期的错误消息
        with pytest.raises(TypeError, match=msg):
            # 使用 assert_produces_warning 上下文，检查减法操作是否产生性能警告
            with tm.assert_produces_warning(performance_warning):
                # 执行 tdarr 与 other 的减法操作
                tdarr - other

        # 使用 assert_produces_warning 上下文，检查减法操作是否产生性能警告
        with tm.assert_produces_warning(performance_warning):
            # 执行 other 与 tdarr 的减法操作
            result = other - tdarr

        # 创建预期结果 Index 对象，包含预期的时间增量和时间戳对象
        expected = Index([Timedelta(0), Timedelta(0), Timestamp("2000-01-01")])
        # 将预期结果经过 box_expected 包装，并转换为 object 类型
        expected = tm.box_expected(expected, xbox).astype(object)
        # 使用 tm.assert_equal 断言 result 和 expected 相等
        tm.assert_equal(result, expected)
class TestTimedeltaArraylikeMulDivOps:
    # 测试 timedelta64[ns] 类型的乘法和除法操作
    # 包括 __mul__, __rmul__, __div__, __rdiv__, __floordiv__, __rfloordiv__

    # ------------------------------------------------------------------
    # Multiplication
    # organized with scalar others first, then array-like

    # 测试时间增量索引乘以整数的情况
    def test_td64arr_mul_int(self, box_with_array):
        # 创建一个从0到4的 timedelta64 索引对象
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        # 将索引对象放入测试框架，根据测试框架的要求进行封装
        idx = tm.box_expected(idx, box_with_array)

        # 测试 timedelta64 索引乘以整数 1 的结果是否等于原索引
        result = idx * 1
        tm.assert_equal(result, idx)

        # 测试整数 1 乘以 timedelta64 索引的结果是否等于原索引
        result = 1 * idx
        tm.assert_equal(result, idx)

    # 测试时间增量索引乘以 timedelta 类型的标量会抛出 TypeError 异常
    def test_td64arr_mul_tdlike_scalar_raises(self, two_hours, box_with_array):
        # 创建一个时间增量范围对象 rng，包含一系列的时间增量
        rng = timedelta_range("1 days", "10 days", name="foo")
        # 将时间增量范围对象放入测试框架，根据测试框架的要求进行封装
        rng = tm.box_expected(rng, box_with_array)
        # 准备错误信息
        msg = "argument must be an integer|cannot use operands with types dtype"
        # 测试乘法操作是否会抛出 TypeError 异常，匹配预期的错误信息
        with pytest.raises(TypeError, match=msg):
            rng * two_hours

    # 测试时间增量索引乘以一维整数数组的情况
    def test_tdi_mul_int_array_zerodim(self, box_with_array):
        # 创建一个包含从0到4的一维整数数组 rng5
        rng5 = np.arange(5, dtype="int64")
        # 创建时间增量索引对象 idx，使用一维整数数组 rng5 初始化
        idx = TimedeltaIndex(rng5)
        # 创建预期的时间增量索引对象，将 rng5 中的每个元素乘以 5
        expected = TimedeltaIndex(rng5 * 5)

        # 将 idx 和 expected 对象放入测试框架，根据测试框架的要求进行封装
        idx = tm.box_expected(idx, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        # 测试时间增量索引 idx 乘以一个整数数组 [5] 的结果是否等于 expected
        result = idx * np.array(5, dtype="int64")
        tm.assert_equal(result, expected)

    # 测试时间增量索引乘以整数数组的情况
    def test_tdi_mul_int_array(self, box_with_array):
        # 创建一个包含从0到4的一维整数数组 rng5
        rng5 = np.arange(5, dtype="int64")
        # 创建时间增量索引对象 idx，使用一维整数数组 rng5 初始化
        idx = TimedeltaIndex(rng5)
        # 创建预期的时间增量索引对象，将 rng5 中的每个元素平方
        expected = TimedeltaIndex(rng5**2)

        # 将 idx 和 expected 对象放入测试框架，根据测试框架的要求进行封装
        idx = tm.box_expected(idx, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        # 测试时间增量索引 idx 乘以整数数组 rng5 的结果是否等于 expected
        result = idx * rng5
        tm.assert_equal(result, expected)

    # 测试时间增量索引乘以整数 Series 的情况
    def test_tdi_mul_int_series(self, box_with_array):
        box = box_with_array
        xbox = Series if box in [Index, tm.to_array, pd.array] else box

        # 创建一个包含从0到4的一维整数数组的时间增量索引对象 idx
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        # 创建预期的时间增量索引对象，将从0到4的一维整数数组每个元素平方
        expected = TimedeltaIndex(np.arange(5, dtype="int64") ** 2)

        # 将 idx 和 expected 对象放入测试框架，根据测试框架的要求进行封装
        idx = tm.box_expected(idx, box)
        expected = tm.box_expected(expected, xbox)

        # 测试时间增量索引 idx 乘以整数 Series 对象的结果是否等于 expected
        result = idx * Series(np.arange(5, dtype="int64"))
        tm.assert_equal(result, expected)

    # 测试时间增量索引乘以浮点数 Series 的情况
    def test_tdi_mul_float_series(self, box_with_array):
        box = box_with_array
        xbox = Series if box in [Index, tm.to_array, pd.array] else box

        # 创建一个包含从0到4的一维整数数组的时间增量索引对象 idx
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        # 将时间增量索引对象 idx 放入测试框架，根据测试框架的要求进行封装
        idx = tm.box_expected(idx, box)

        # 创建一个包含从0到4的一维浮点数数组 rng5f
        rng5f = np.arange(5, dtype="float64")
        # 创建预期的时间增量索引对象，将 rng5f 中的每个元素乘以 (rng5f + 1.0)
        expected = TimedeltaIndex(rng5f * (rng5f + 1.0))
        expected = tm.box_expected(expected, xbox)

        # 测试时间增量索引 idx 乘以浮点数 Series 对象 rng5f + 1.0 的结果是否等于 expected
        result = idx * Series(rng5f + 1.0)
        tm.assert_equal(result, expected)

    # TODO: Put Series/DataFrame in others?
    @pytest.mark.parametrize(
        "other",
        [
            np.arange(1, 11),  # 参数化测试的参数：一个 numpy 数组，从 1 到 10
            Index(np.arange(1, 11), np.int64),  # 参数化测试的参数：一个索引对象，使用 np.int64 类型
            Index(range(1, 11), np.uint64),  # 参数化测试的参数：一个索引对象，使用 np.uint64 类型
            Index(range(1, 11), np.float64),  # 参数化测试的参数：一个索引对象，使用 np.float64 类型
            pd.RangeIndex(1, 11),  # 参数化测试的参数：一个 Pandas 的 RangeIndex 对象，从 1 到 10
        ],
        ids=lambda x: type(x).__name__,  # 每个参数的标识符，使用对象类型的名称
    )
    def test_tdi_rmul_arraylike(self, other, box_with_array):
        box = box_with_array

        tdi = TimedeltaIndex(["1 Day"] * 10)  # 创建一个 TimedeltaIndex 对象，包含 10 个 "1 Day" 的时间增量
        expected = timedelta_range("1 days", "10 days")._with_freq(None)  # 期望的时间增量范围，从 "1 days" 到 "10 days"

        tdi = tm.box_expected(tdi, box)  # 将 tdi 对象装箱为期望的类型
        xbox = get_upcast_box(tdi, other)  # 获取 tdi 和 other 的广播后的类型

        expected = tm.box_expected(expected, xbox)  # 将期望的结果装箱为 xbox 类型

        result = other * tdi  # 执行乘法操作
        tm.assert_equal(result, expected)  # 断言结果与期望相等
        commute = tdi * other  # 交换顺序的乘法操作
        tm.assert_equal(commute, expected)  # 断言交换顺序操作的结果与期望相等

    # ------------------------------------------------------------------
    # __div__, __rdiv__

    def test_td64arr_div_nat_invalid(self, box_with_array):
        # 不允许通过 NaT 进行除法操作（可能未来会允许）
        rng = timedelta_range("1 days", "10 days", name="foo")  # 创建一个时间增量范围对象 rng，并命名为 "foo"
        rng = tm.box_expected(rng, box_with_array)  # 将 rng 对象装箱为期望的类型

        with pytest.raises(TypeError, match="unsupported operand type"):  # 检查是否引发 TypeError 异常，匹配指定的错误信息
            rng / NaT
        with pytest.raises(TypeError, match="Cannot divide NaTType by"):  # 检查是否引发 TypeError 异常，匹配指定的错误信息
            NaT / rng

        dt64nat = np.datetime64("NaT", "ns")  # 创建一个表示 NaT 的 np.datetime64 对象
        msg = "|".join(
            [
                # 'divide' on npdev as of 2021-12-18
                "ufunc '(true_divide|divide)' cannot use operands",  # 拼接匹配的错误信息
                "cannot perform __r?truediv__",  # 拼接匹配的错误信息
                "Cannot divide datetime64 by TimedeltaArray",  # 拼接匹配的错误信息
            ]
        )
        with pytest.raises(TypeError, match=msg):  # 检查是否引发 TypeError 异常，匹配复杂的错误信息
            rng / dt64nat
        with pytest.raises(TypeError, match=msg):  # 检查是否引发 TypeError 异常，匹配复杂的错误信息
            dt64nat / rng

    def test_td64arr_div_td64nat(self, box_with_array):
        # GH#23829
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        rng = timedelta_range("1 days", "10 days")  # 创建一个时间增量范围对象 rng，从 "1 days" 到 "10 days"
        rng = tm.box_expected(rng, box)  # 将 rng 对象装箱为期望的类型

        other = np.timedelta64("NaT")  # 创建一个表示 NaT 的 np.timedelta64 对象

        expected = np.array([np.nan] * 10)  # 创建一个包含 10 个 NaN 值的 numpy 数组
        expected = tm.box_expected(expected, xbox)  # 将期望的结果装箱为 xbox 类型

        result = rng / other  # 执行除法操作
        tm.assert_equal(result, expected)  # 断言结果与期望相等

        result = other / rng  # 执行反向除法操作
        tm.assert_equal(result, expected)  # 断言反向操作的结果与期望相等

    def test_td64arr_div_int(self, box_with_array):
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))  # 创建一个时间增量索引对象，包含从 0 到 4 的 int64 类型数据
        idx = tm.box_expected(idx, box_with_array)  # 将 idx 对象装箱为期望的类型

        result = idx / 1  # 执行除法操作
        tm.assert_equal(result, idx)  # 断言结果与 idx 相等

        with pytest.raises(TypeError, match="Cannot divide"):  # 检查是否引发 TypeError 异常，匹配指定的错误信息
            # GH#23829
            1 / idx
    # 测试函数，用于验证时间增量对象与标量的除法操作
    def test_td64arr_div_tdlike_scalar(self, two_hours, box_with_array):
        # GH#20088, GH#22163 确保 DataFrame 返回正确的数据类型
        box = box_with_array
        # 如果 box 是 pd.array，则设置 xbox 为 np.ndarray，否则为 box 自身
        xbox = np.ndarray if box is pd.array else box

        # 创建一个时间增量范围对象 rng，表示从 "1 days" 到 "10 days" 的时间间隔，名称为 "foo"
        rng = timedelta_range("1 days", "10 days", name="foo")
        # 创建期望的索引对象 expected，包含 10 个元素，每个元素为 (1 到 10) * 12，数据类型为 np.float64，名称为 "foo"
        expected = Index((np.arange(10) + 1) * 12, dtype=np.float64, name="foo")

        # 将 rng 和 expected 根据 box 进行适当的封装
        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)

        # 计算 rng 除以 two_hours 的结果
        result = rng / two_hours
        # 验证计算结果与期望值是否相等
        tm.assert_equal(result, expected)

        # 计算 two_hours 除以 rng 的结果
        result = two_hours / rng
        # 计算期望值为 1 除以 expected
        expected = 1 / expected
        # 验证计算结果与期望值是否相等
        tm.assert_equal(result, expected)

    # 使用 pytest 的参数化装饰器，测试时间增量对象与时间增量标量的除法操作
    @pytest.mark.parametrize("m", [1, 3, 10])
    @pytest.mark.parametrize("unit", ["D", "h", "m", "s", "ms", "us", "ns"])
    def test_td64arr_div_td64_scalar(self, m, unit, box_with_array):
        box = box_with_array
        # 如果 box 是 pd.array，则设置 xbox 为 np.ndarray，否则为 box 自身
        xbox = np.ndarray if box is pd.array else box

        # 创建一个包含三个 Timedelta(days=59) 的 Series 对象 ser
        ser = Series([Timedelta(days=59)] * 3)
        # 将 ser 中第二个元素设置为 NaN
        ser[2] = np.nan
        # 将 flat 指向 ser 的引用
        flat = ser
        # 根据 box 对 ser 进行适当的封装
        ser = tm.box_expected(ser, box)

        # 计算每个 flat 中的元素除以 np.timedelta64(m, unit)，生成期望的 Series 对象 expected
        expected = Series([x / np.timedelta64(m, unit) for x in flat])
        # 根据 box 对 expected 进行适当的封装
        expected = tm.box_expected(expected, xbox)
        # 计算 ser 除以 np.timedelta64(m, unit)，得到结果 result
        result = ser / np.timedelta64(m, unit)
        # 验证计算结果与期望值是否相等
        tm.assert_equal(result, expected)

        # 计算 np.timedelta64(m, unit) 除以 ser 中的每个元素，生成期望的 Series 对象 expected
        expected = Series([Timedelta(np.timedelta64(m, unit)) / x for x in flat])
        # 根据 box 对 expected 进行适当的封装
        expected = tm.box_expected(expected, xbox)
        # 计算 np.timedelta64(m, unit) 除以 ser，得到结果 result
        result = np.timedelta64(m, unit) / ser
        # 验证计算结果与期望值是否相等
        tm.assert_equal(result, expected)

    # 测试函数，用于验证时间增量对象与标量的除法操作（包含 NaN）
    def test_td64arr_div_tdlike_scalar_with_nat(self, two_hours, box_with_array):
        box = box_with_array
        # 如果 box 是 pd.array，则设置 xbox 为 np.ndarray，否则为 box 自身
        xbox = np.ndarray if box is pd.array else box

        # 创建一个 TimedeltaIndex 对象 rng，包含 ["1 days", NaT, "2 days"]，名称为 "foo"
        rng = TimedeltaIndex(["1 days", NaT, "2 days"], name="foo")
        # 创建期望的索引对象 expected，包含 [12, NaN, 24]，数据类型为 np.float64，名称为 "foo"
        expected = Index([12, np.nan, 24], dtype=np.float64, name="foo")

        # 将 rng 和 expected 根据 box 进行适当的封装
        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)

        # 计算 rng 除以 two_hours 的结果
        result = rng / two_hours
        # 验证计算结果与期望值是否相等
        tm.assert_equal(result, expected)

        # 计算 two_hours 除以 rng 的结果
        result = two_hours / rng
        # 计算期望值为 1 除以 expected
        expected = 1 / expected
        # 验证计算结果与期望值是否相等
        tm.assert_equal(result, expected)
    # 测试方法，用于测试 TimedeltaIndex 对象与其他数据类型的除法操作
    def test_td64arr_div_td64_ndarray(self, box_with_array):
        # GH#22631
        # 从参数中获取 box_with_array 对象
        box = box_with_array
        # 根据 box 的类型选择 numpy.ndarray 或 box 对象
        xbox = np.ndarray if box is pd.array else box

        # 创建 TimedeltaIndex 对象，包含字符串表示的时间增量和 NaT（Not a Time）
        rng = TimedeltaIndex(["1 days", NaT, "2 days"])
        # 创建预期的 Index 对象，包含浮点数和 NaN 值
        expected = Index([12, np.nan, 24], dtype=np.float64)

        # 将 rng 和 expected 分别传入 box_expected 函数，处理后返回
        rng = tm.box_expected(rng, box)
        expected = tm.box_expected(expected, xbox)

        # 创建 numpy 数组 other，包含时间增量类型的数据
        other = np.array([2, 4, 2], dtype="m8[h]")
        # 执行 rng 与 other 数组的除法操作
        result = rng / other
        # 使用 tm.assert_equal 断言结果与期望值相等
        tm.assert_equal(result, expected)

        # 执行 rng 与 other 数组包装后的 box 操作的除法
        result = rng / tm.box_expected(other, box)
        tm.assert_equal(result, expected)

        # 执行 rng 与 other 数组转换为 object 类型后的除法
        result = rng / other.astype(object)
        tm.assert_equal(result, expected.astype(object))

        # 执行 rng 与 other 列表的除法
        result = rng / list(other)
        tm.assert_equal(result, expected)

        # 执行反向操作，expected 对象中的值取倒数
        expected = 1 / expected
        # 执行 other 数组与 rng 的除法
        result = other / rng
        # 使用 tm.assert_equal 断言结果与期望值相等
        tm.assert_equal(result, expected)

        # 执行 other 数组包装后的 box 操作与 rng 的除法
        result = tm.box_expected(other, box) / rng
        tm.assert_equal(result, expected)

        # 执行 other 数组转换为 object 类型后与 rng 的除法
        result = other.astype(object) / rng
        tm.assert_equal(result, expected)

        # 执行 other 列表与 rng 的除法
        result = list(other) / rng
        tm.assert_equal(result, expected)

    # 测试方法，用于测试 TimedeltaIndex 对象与长度不匹配的其他数据类型的除法操作
    def test_tdarr_div_length_mismatch(self, box_with_array):
        # 创建 TimedeltaIndex 对象，包含字符串表示的时间增量和 NaT（Not a Time）
        rng = TimedeltaIndex(["1 days", NaT, "2 days"])
        # 创建长度不匹配的列表 mismatched
        mismatched = [1, 2, 3, 4]

        # 将 rng 传入 box_expected 函数，处理后返回
        rng = tm.box_expected(rng, box_with_array)
        # 设置异常消息
        msg = "Cannot divide vectors|Unable to coerce to Series"
        # 遍历 mismatched 列表和其子集
        for obj in [mismatched, mismatched[:2]]:
            # 对于每个 obj，创建不同类型的 other 对象进行测试
            # 一个比 rng 长，一个比 rng 短
            for other in [obj, np.array(obj), Index(obj)]:
                # 使用 pytest.raises 断言除法操作会抛出 ValueError 异常，且异常消息匹配 msg
                with pytest.raises(ValueError, match=msg):
                    rng / other
                with pytest.raises(ValueError, match=msg):
                    other / rng
    # 测试函数，用于测试混合对象和 timedelta64 类型数据的除法运算
    def test_td64_div_object_mixed_result(self, box_with_array):
        # 创建一个带有 NaT（Not a Time）值的 timedelta_range 对象，期望在结果中出现 NaT 而不是 timedelta64("NaT")，这可能会引起误导
        orig = timedelta_range("1 Day", periods=3).insert(1, NaT)
        # 对原始数据进行箱式化处理，以便与预期结果进行比较，transpose=False 表示不进行转置操作
        tdi = tm.box_expected(orig, box_with_array, transpose=False)

        # 创建一个包含对象类型的 numpy 数组，包括原始数据的部分值和一些浮点数
        other = np.array([orig[0], 1.5, 2.0, orig[2]], dtype=object)
        other = tm.box_expected(other, box_with_array, transpose=False)

        # 进行除法运算
        res = tdi / other

        # 创建预期的索引对象，包含特定的浮点数和 timedelta64("NaT") 对象
        expected = Index([1.0, np.timedelta64("NaT", "ns"), orig[0], 1.5], dtype=object)
        expected = tm.box_expected(expected, box_with_array, transpose=False)
        # 如果预期结果是 NumpyExtensionArray 类型，则转换为 numpy 数组
        if isinstance(expected, NumpyExtensionArray):
            expected = expected.to_numpy()
        # 检查 res 和 expected 是否相等
        tm.assert_equal(res, expected)
        # 如果 box_with_array 是 DataFrame 类型
        if box_with_array is DataFrame:
            # 确保结果的特定位置是 np.timedelta64 类型而不是 pd.NaT
            assert isinstance(res.iloc[1, 0], np.timedelta64)

        # 进行整除运算
        res = tdi // other

        # 创建预期的索引对象，包含特定的整数和 timedelta64("NaT") 对象
        expected = Index([1, np.timedelta64("NaT", "ns"), orig[0], 1], dtype=object)
        expected = tm.box_expected(expected, box_with_array, transpose=False)
        # 如果预期结果是 NumpyExtensionArray 类型，则转换为 numpy 数组
        if isinstance(expected, NumpyExtensionArray):
            expected = expected.to_numpy()
        # 检查 res 和 expected 是否相等
        tm.assert_equal(res, expected)
        # 如果 box_with_array 是 DataFrame 类型
        if box_with_array is DataFrame:
            # 确保结果的特定位置是 np.timedelta64 类型而不是 pd.NaT
            assert isinstance(res.iloc[1, 0], np.timedelta64)

    # ------------------------------------------------------------------
    # __floordiv__, __rfloordiv__

    # 标记为跳过测试，如果在 wasm 环境下，原因是 wasm 不支持浮点异常
    @pytest.mark.skipif(WASM, reason="no fp exception support in wasm")
    def test_td64arr_floordiv_td64arr_with_nat(self, box_with_array):
        # GH#35529
        # 根据 box_with_array 类型选择不同的箱式化方法
        box = box_with_array
        xbox = np.ndarray if box is pd.array else box

        # 创建两个包含 timedelta64[ns] 类型数据的 Series 对象，其中右侧数据包含 None 值
        left = Series([1000, 222330, 30], dtype="timedelta64[ns]")
        right = Series([1000, 222330, None], dtype="timedelta64[ns]")

        # 对 left 和 right 进行箱式化处理
        left = tm.box_expected(left, box)
        right = tm.box_expected(right, box)

        # 创建预期的 numpy 数组，包含浮点数和 NaN 值
        expected = np.array([1.0, 1.0, np.nan], dtype=np.float64)
        expected = tm.box_expected(expected, xbox)

        # 可能会产生 RuntimeWarning 的上下文，检查结果并与预期结果比较
        with tm.maybe_produces_warning(
            RuntimeWarning, box is pd.array, check_stacklevel=False
        ):
            result = left // right

        # 检查 result 和 expected 是否相等
        tm.assert_equal(result, expected)

        # 使用 __rfloordiv__ 处理数组的情况
        with tm.maybe_produces_warning(
            RuntimeWarning, box is pd.array, check_stacklevel=False
        ):
            result = np.asarray(left) // right
        # 检查 result 和 expected 是否相等
        tm.assert_equal(result, expected)

    # 忽略特定的 RuntimeWarning 警告
    @pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
    # 测试方法：用于测试 timedelta64 数组与 timedelta 标量的整数除法操作
    def test_td64arr_floordiv_tdscalar(self, box_with_array, scalar_td):
        # GH#18831, GH#19125
        # 使用 box_with_array 作为 box 变量
        box = box_with_array
        # 如果 box 是 pd.array 则使用 np.ndarray，否则使用 box 本身
        xbox = np.ndarray if box is pd.array else box
        # 创建一个 Timedelta 对象，表示 5 分钟 3 秒
        td = Timedelta("5m3s")  # i.e. (scalar_td - 1sec) / 2

        # 创建一个包含 Timedelta 对象的 Series，数据类型为 "m8[ns]"
        td1 = Series([td, td, NaT], dtype="m8[ns]")
        # 根据 box 参数将 td1 进行适配
        td1 = tm.box_expected(td1, box, transpose=False)

        # 创建一个预期的 Series 对象，包含 [0, 0, NaN]
        expected = Series([0, 0, np.nan])
        # 根据 xbox 参数将 expected 进行适配
        expected = tm.box_expected(expected, xbox, transpose=False)

        # 计算 td1 与 scalar_td 的整数除法结果
        result = td1 // scalar_td
        # 断言 result 与 expected 相等
        tm.assert_equal(result, expected)

        # 反向操作
        # 创建一个预期的 Series 对象，包含 [2, 2, NaN]
        expected = Series([2, 2, np.nan])
        # 根据 xbox 参数将 expected 进行适配
        expected = tm.box_expected(expected, xbox, transpose=False)

        # 计算 scalar_td 与 td1 的整数除法结果
        result = scalar_td // td1
        # 断言 result 与 expected 相等
        tm.assert_equal(result, expected)

        # 与上述操作相同，但显式调用 __rfloordiv__ 方法
        result = td1.__rfloordiv__(scalar_td)
        # 断言 result 与 expected 相等
        tm.assert_equal(result, expected)

    # 测试方法：测试 timedelta64 数组与整数的整数除法操作
    def test_td64arr_floordiv_int(self, box_with_array):
        # 创建一个包含 [0, 1, 2, 3, 4] 的 TimedeltaIndex
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        # 根据 box_with_array 参数将 idx 进行适配
        idx = tm.box_expected(idx, box_with_array)
        # 计算 idx 与整数 1 的整数除法结果
        result = idx // 1
        # 断言 result 与 idx 相等
        tm.assert_equal(result, idx)

        # 设置预期的错误信息模式
        pattern = "floor_divide cannot use operands|Cannot divide int by Timedelta*"
        # 使用 pytest 断言，检查是否会抛出预期的 TypeError 异常
        with pytest.raises(TypeError, match=pattern):
            # 尝试计算整数 1 与 idx 的整数除法结果，预期抛出异常
            1 // idx

    # ------------------------------------------------------------------
    # mod, divmod
    # TODO: operations with timedelta-like arrays, numeric arrays,
    #  reversed ops

    # 测试方法：测试 timedelta64 数组与 timedelta 标量的取模操作
    def test_td64arr_mod_tdscalar(
        self, performance_warning, box_with_array, three_days
    ):
        # 创建一个包含从 "1 Day" 到 "9 days" 的 TimedeltaIndex
        tdi = timedelta_range("1 Day", "9 days")
        # 根据 box_with_array 参数将 tdi 进行适配
        tdarr = tm.box_expected(tdi, box_with_array)

        # 创建一个预期的 TimedeltaIndex，包含 ["1 Day", "2 Days", "0 Days"] 重复三次的数据
        expected = TimedeltaIndex(["1 Day", "2 Days", "0 Days"] * 3)
        # 根据 box_with_array 参数将 expected 进行适配
        expected = tm.box_expected(expected, box_with_array)

        # 计算 tdarr 与 three_days 的取模结果
        result = tdarr % three_days
        # 断言 result 与 expected 相等
        tm.assert_equal(result, expected)

        # 如果 box_with_array 是 DataFrame 类型，并且 three_days 是 pd.DateOffset 类型
        if box_with_array is DataFrame and isinstance(three_days, pd.DateOffset):
            # TODO: making expected be object here a result of DataFrame.__divmod__
            #  being defined in a naive way that does not dispatch to the underlying
            #  array's __divmod__
            # 将 expected 转换为 object 类型
            expected = expected.astype(object)
        else:
            # 关闭性能警告
            performance_warning = False

        # 使用 tm.assert_produces_warning 断言，检查是否会产生性能警告
        with tm.assert_produces_warning(performance_warning):
            # 计算 tdarr 与 three_days 的 divmod 结果
            result = divmod(tdarr, three_days)

        # 断言 result[1] 与 expected 相等
        tm.assert_equal(result[1], expected)
        # 断言 result[0] 与 tdarr // three_days 相等
        tm.assert_equal(result[0], tdarr // three_days)
    # 定义一个测试函数，用于测试 TimedeltaIndex 对象与整数之间的取模操作
    def test_td64arr_mod_int(self, box_with_array):
        # 创建一个时间增量范围对象，表示从 "1 ns" 到 "10 ns"，共 10 个时间点
        tdi = timedelta_range("1 ns", "10 ns", periods=10)
        # 将时间增量范围对象转换为指定盒子包装后的对象
        tdarr = tm.box_expected(tdi, box_with_array)

        # 创建一个预期的 TimedeltaIndex 对象，包含交替的 "1 ns" 和 "0 ns"，共 10 个时间点
        expected = TimedeltaIndex(["1 ns", "0 ns"] * 5)
        # 将预期的 TimedeltaIndex 对象转换为指定盒子包装后的对象
        expected = tm.box_expected(expected, box_with_array)

        # 对 tdarr 中的每个元素取模 2
        result = tdarr % 2
        # 断言结果与预期相等
        tm.assert_equal(result, expected)

        # 创建一个错误消息字符串
        msg = "Cannot divide int by"
        # 断言执行 2 % tdarr 时会引发 TypeError 异常，并且异常消息包含 msg
        with pytest.raises(TypeError, match=msg):
            2 % tdarr

        # 使用 divmod 函数计算 tdarr 与 2 的商和余数
        result = divmod(tdarr, 2)
        # 断言结果的余数部分与预期相等
        tm.assert_equal(result[1], expected)
        # 断言结果的商部分与 tdarr 除以 2 的预期相等
        tm.assert_equal(result[0], tdarr // 2)

    # 定义一个测试函数，用于测试 TimedeltaIndex 对象与单个时间增量对象之间的取模操作
    def test_td64arr_rmod_tdscalar(self, box_with_array, three_days):
        # 创建一个时间增量范围对象，表示从 "1 Day" 到 "9 days"，默认 1 天的间隔
        tdi = timedelta_range("1 Day", "9 days")
        # 将时间增量范围对象转换为指定盒子包装后的对象
        tdarr = tm.box_expected(tdi, box_with_array)

        # 创建一个预期的 TimedeltaIndex 对象，包含特定的时间增量组合
        expected = ["0 Days", "1 Day", "0 Days"] + ["3 Days"] * 6
        expected = TimedeltaIndex(expected)
        # 将预期的 TimedeltaIndex 对象转换为指定盒子包装后的对象
        expected = tm.box_expected(expected, box_with_array)

        # 计算 three_days 与 tdarr 中每个元素的取模结果
        result = three_days % tdarr
        # 断言结果与预期相等
        tm.assert_equal(result, expected)

        # 使用 divmod 函数计算 three_days 与 tdarr 的商和余数
        result = divmod(three_days, tdarr)
        # 断言结果的余数部分与预期相等
        tm.assert_equal(result[1], expected)
        # 断言结果的商部分与 three_days 除以 tdarr 的预期相等
        tm.assert_equal(result[0], three_days // tdarr)

    # ------------------------------------------------------------------
    # Operations with invalid others

    # 定义一个测试函数，用于测试 TimedeltaIndex 对象与单个时间增量对象相乘时的异常情况
    def test_td64arr_mul_tdscalar_invalid(self, box_with_array, scalar_td):
        # 创建一个包含 3 个 timedelta(minutes=5, seconds=3) 对象的 Series
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        # 将 Series 对象转换为指定盒子包装后的对象
        td1 = tm.box_expected(td1, box_with_array)

        # 检查是否会因未定义的操作而引发 TypeError 异常
        pattern = "operate|unsupported|cannot|not supported"
        with pytest.raises(TypeError, match=pattern):
            # 尝试将 td1 与 scalar_td 相乘，预期引发 TypeError 异常
            td1 * scalar_td
        with pytest.raises(TypeError, match=pattern):
            # 尝试将 scalar_td 与 td1 相乘，预期引发 TypeError 异常
            scalar_td * td1

    # 定义一个测试函数，用于测试 TimedeltaIndex 对象与自身的部分相乘时的异常情况
    def test_td64arr_mul_too_short_raises(self, box_with_array):
        # 创建一个包含 5 个元素的 TimedeltaIndex 对象
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        # 将 TimedeltaIndex 对象转换为指定盒子包装后的对象
        idx = tm.box_expected(idx, box_with_array)
        # 创建一个用于匹配的错误消息模式
        msg = "|".join(
            [
                "cannot use operands with types dtype",
                "Cannot multiply with unequal lengths",
                "Unable to coerce to Series",
            ]
        )
        with pytest.raises(TypeError, match=msg):
            # 尝试将 idx 与 idx[:3] 的部分进行相乘，预期引发 TypeError 或 ValueError 异常
            idx * idx[:3]
        with pytest.raises(ValueError, match=msg):
            # 尝试将 idx 与长度不匹配的数组进行相乘，预期引发 TypeError 或 ValueError 异常
            idx * np.array([1, 2])

    # 定义一个测试函数，用于测试 TimedeltaIndex 对象与自身进行完整相乘时的异常情况
    def test_td64arr_mul_td64arr_raises(self, box_with_array):
        # 创建一个包含 5 个元素的 TimedeltaIndex 对象
        idx = TimedeltaIndex(np.arange(5, dtype="int64"))
        # 将 TimedeltaIndex 对象转换为指定盒子包装后的对象
        idx = tm.box_expected(idx, box_with_array)
        # 创建一个用于匹配的错误消息模式
        msg = "cannot use operands with types dtype"
        with pytest.raises(TypeError, match=msg):
            # 尝试将 idx 与自身进行相乘，预期引发 TypeError 异常
            idx * idx

    # ------------------------------------------------------------------
    # Operations with numeric others
    def test_td64arr_mul_numeric_scalar(self, box_with_array, one):
        # GH#4521
        # divide/multiply by integers
        
        # 创建一个包含时间差数据的Series对象，指定数据类型为纳秒级别的时间差
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        # 创建一个预期结果的Series对象，数据类型为纳秒级别的时间差
        expected = Series(["-59 Days", "-59 Days", "NaT"], dtype="timedelta64[ns]")

        # 将时间差Series对象进行包装处理，可能是与box_with_array相关的处理
        tdser = tm.box_expected(tdser, box_with_array)
        # 将预期结果的Series对象进行包装处理，可能是与box_with_array相关的处理
        expected = tm.box_expected(expected, box_with_array)

        # 计算时间差Series对象与负数标量的乘积
        result = tdser * (-one)
        # 断言结果与预期相等
        tm.assert_equal(result, expected)
        # 计算负数标量与时间差Series对象的乘积
        result = (-one) * tdser
        # 断言结果与预期相等
        tm.assert_equal(result, expected)

        # 更新预期结果的Series对象，改变时间差的数值
        expected = Series(["118 Days", "118 Days", "NaT"], dtype="timedelta64[ns]")
        # 将更新后的预期结果进行包装处理，可能是与box_with_array相关的处理
        expected = tm.box_expected(expected, box_with_array)

        # 计算时间差Series对象与2倍标量的乘积
        result = tdser * (2 * one)
        # 断言结果与预期相等
        tm.assert_equal(result, expected)
        # 计算2倍标量与时间差Series对象的乘积
        result = (2 * one) * tdser
        # 断言结果与预期相等
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("two", [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_div_numeric_scalar(self, box_with_array, two):
        # GH#4521
        # divide/multiply by integers
        
        # 创建一个包含时间差数据的Series对象，指定数据类型为纳秒级别的时间差
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        # 创建一个预期结果的Series对象，数据类型为纳秒级别的时间差，但数值为浮点数
        expected = Series(["29.5D", "29.5D", "NaT"], dtype="timedelta64[ns]")

        # 将时间差Series对象进行包装处理，可能是与box_with_array相关的处理
        tdser = tm.box_expected(tdser, box_with_array)
        # 将预期结果的Series对象进行包装处理，可能是与box_with_array相关的处理
        expected = tm.box_expected(expected, box_with_array)

        # 计算时间差Series对象与标量two的除法
        result = tdser / two
        # 断言结果与预期相等
        tm.assert_equal(result, expected)

        # 使用pytest的断言检查，验证除法运算中出现的TypeError异常
        with pytest.raises(TypeError, match="Cannot divide"):
            two / tdser

    @pytest.mark.parametrize("two", [2, 2.0, np.array(2), np.array(2.0)])
    def test_td64arr_floordiv_numeric_scalar(self, box_with_array, two):
        # 创建一个包含时间差数据的Series对象，指定数据类型为纳秒级别的时间差
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        # 创建一个预期结果的Series对象，数据类型为纳秒级别的时间差，但数值为浮点数
        expected = Series(["29.5D", "29.5D", "NaT"], dtype="timedelta64[ns]")

        # 将时间差Series对象进行包装处理，可能是与box_with_array相关的处理
        tdser = tm.box_expected(tdser, box_with_array)
        # 将预期结果的Series对象进行包装处理，可能是与box_with_array相关的处理
        expected = tm.box_expected(expected, box_with_array)

        # 计算时间差Series对象与标量two的整数除法
        result = tdser // two
        # 断言结果与预期相等
        tm.assert_equal(result, expected)

        # 使用pytest的断言检查，验证整数除法运算中出现的TypeError异常
        with pytest.raises(TypeError, match="Cannot divide"):
            two // tdser

    @pytest.mark.parametrize(
        "klass",
        [np.array, Index, Series],
        ids=lambda x: x.__name__,
    )
    def test_td64arr_rmul_numeric_array(
        self,
        box_with_array,
        klass,
        any_real_numpy_dtype,
    ):
        # GH#4521
        # divide/multiply by integers
        
        # 创建一个向量对象，包含整数数据
        vector = klass([20, 30, 40])
        # 创建一个包含时间差数据的Series对象，指定数据类型为纳秒级别的时间差
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        # 将向量对象的数据类型转换为任何真实的numpy数据类型
        vector = vector.astype(any_real_numpy_dtype)

        # 创建一个预期结果的Series对象，数据类型为纳秒级别的时间差，数值根据向量计算得出
        expected = Series(["1180 Days", "1770 Days", "NaT"], dtype="timedelta64[ns]")

        # 将时间差Series对象进行包装处理，可能是与box_with_array相关的处理
        tdser = tm.box_expected(tdser, box_with_array)
        # 获取升级后的包装箱对象，可能与时间差Series对象和向量对象相关
        xbox = get_upcast_box(tdser, vector)

        # 将预期结果的Series对象进行包装处理，可能是与xbox相关的处理
        expected = tm.box_expected(expected, xbox)

        # 计算时间差Series对象与向量对象的乘积
        result = tdser * vector
        # 断言结果与预期相等
        tm.assert_equal(result, expected)

        # 计算向量对象与时间差Series对象的乘积
        result = vector * tdser
        # 断言结果与预期相等
        tm.assert_equal(result, expected)
    @pytest.mark.parametrize(
        "klass",
        [np.array, Index, Series],
        ids=lambda x: x.__name__,
    )
    # 使用 pytest 的 parametrize 装饰器，为测试用例参数化，klass 可以是 np.array、Index 或 Series
    def test_td64arr_div_numeric_array(
        self, box_with_array, klass, any_real_numpy_dtype
    ):
        # GH#4521
        # 对整数进行除法和乘法运算的测试

        # 创建一个 klass 类型的向量
        vector = klass([20, 30, 40])
        # 创建一个类型为 'm8[ns]' 的 Series 对象
        tdser = Series(["59 Days", "59 Days", "NaT"], dtype="m8[ns]")
        # 将向量转换为指定的 numpy 数据类型
        vector = vector.astype(any_real_numpy_dtype)

        # 创建一个预期结果的 Series 对象，类型为 'timedelta64[ns]'
        expected = Series(["2.95D", "1D 23h 12m", "NaT"], dtype="timedelta64[ns]")

        # 使用测试工具方法 box_expected 处理 tdser 和 box_with_array
        tdser = tm.box_expected(tdser, box_with_array)
        # 获取经过升级处理的 box 对象
        xbox = get_upcast_box(tdser, vector)
        # 使用测试工具方法 box_expected 处理 expected 和 xbox
        expected = tm.box_expected(expected, xbox)

        # 进行 tdser 除以 vector 的运算
        result = tdser / vector
        # 使用测试工具方法 assert_equal 检查 result 是否等于 expected
        tm.assert_equal(result, expected)

        # 定义一个正则表达式模式，用于匹配异常消息
        pattern = "|".join(
            [
                "true_divide'? cannot use operands",
                "cannot perform __div__",
                "cannot perform __truediv__",
                "unsupported operand",
                "Cannot divide",
                "ufunc 'divide' cannot use operands with types",
            ]
        )
        # 使用 pytest 的 raises 方法，期望捕获 TypeError 异常，并匹配指定模式的异常消息
        with pytest.raises(TypeError, match=pattern):
            vector / tdser

        # 将 vector 转换为 object 类型后，再进行 tdser 除法运算
        result = tdser / vector.astype(object)
        # 如果 box_with_array 是 DataFrame 类型
        if box_with_array is DataFrame:
            # 预期结果为按列进行的除法运算，第三个元素填充为 'NaT'
            expected = [tdser.iloc[0, n] / vector[n] for n in range(len(vector))]
            expected = tm.box_expected(expected, xbox).astype(object)
            expected[2] = expected[2].fillna(np.timedelta64("NaT", "ns"))
        else:
            # 预期结果为按元素进行的除法运算，将 NaT 替换为 'NaT' 类型的 np.timedelta64
            expected = [tdser[n] / vector[n] for n in range(len(tdser))]
            expected = [
                x if x is not NaT else np.timedelta64("NaT", "ns") for x in expected
            ]
            # 如果 xbox 是 tm.to_array，则将 expected 转换为数组类型
            if xbox is tm.to_array:
                expected = tm.to_array(expected).astype(object)
            else:
                expected = xbox(expected, dtype=object)

        # 使用测试工具方法 assert_equal 检查 result 是否等于 expected
        tm.assert_equal(result, expected)

        # 再次使用 pytest 的 raises 方法，期望捕获 TypeError 异常，并匹配指定模式的异常消息
        with pytest.raises(TypeError, match=pattern):
            vector.astype(object) / tdser

    def test_td64arr_mul_int_series(self, box_with_array, names):
        # GH#19042 测试正确的名称附加功能
        # 将 box_with_array 赋值给 box 变量
        box = box_with_array
        # 使用 get_expected_name 获取期望的名称 exname
        exname = get_expected_name(box, names)

        # 创建一个 TimedeltaIndex 对象 tdi，指定名称为 names[0]
        tdi = TimedeltaIndex(
            ["0days", "1day", "2days", "3days", "4days"], name=names[0]
        )
        # 创建一个 Series 对象 ser，包含整数 0 到 4，数据类型为 np.int64，指定名称为 names[1]
        ser = Series([0, 1, 2, 3, 4], dtype=np.int64, name=names[1])

        # 创建一个预期结果的 Series 对象 expected，包含计算后的时间差，数据类型为 'timedelta64[ns]'，指定名称为 exname
        expected = Series(
            ["0days", "1day", "4days", "9days", "16days"],
            dtype="timedelta64[ns]",
            name=exname,
        )

        # 使用测试工具方法 box_expected 处理 tdi 和 box
        tdi = tm.box_expected(tdi, box)
        # 获取经过升级处理的 xbox 对象
        xbox = get_upcast_box(tdi, ser)

        # 使用测试工具方法 box_expected 处理 expected 和 xbox
        expected = tm.box_expected(expected, xbox)

        # 进行 ser 与 tdi 的乘法运算
        result = ser * tdi
        # 使用测试工具方法 assert_equal 检查 result 是否等于 expected
        tm.assert_equal(result, expected)

        # 再次进行 tdi 与 ser 的乘法运算，检查结果是否与 expected 相等
        result = tdi * ser
        tm.assert_equal(result, expected)

    # TODO: Should we be parametrizing over types for `ser` too?
    # 测试函数，验证浮点数系列与时间增量索引的正确操作
    def test_float_series_rdiv_td64arr(self, box_with_array, names):
        # GH#19042 测试正确的名称附加
        box = box_with_array  # 将参数赋值给变量box，以便后续使用
        # 创建一个时间增量索引对象，使用指定的名称作为索引名称
        tdi = TimedeltaIndex(
            ["0days", "1day", "2days", "3days", "4days"], name=names[0]
        )
        # 创建一个浮点数系列，使用指定的名称作为列名称
        ser = Series([1.5, 3, 4.5, 6, 7.5], dtype=np.float64, name=names[1])

        # 根据条件选择变量xname的值
        xname = names[2] if box not in [tm.to_array, pd.array] else names[1]
        # 创建一个期望的结果系列，将时间增量索引的每个元素与浮点数系列的对应元素相除
        expected = Series(
            [tdi[n] / ser[n] for n in range(len(ser))],
            dtype="timedelta64[ns]",
            name=xname,
        )

        # 将时间增量索引对象tdi转换为特定类型的箱对象
        tdi = tm.box_expected(tdi, box)
        # 获取将时间增量索引对象和浮点数系列升级的箱对象
        xbox = get_upcast_box(tdi, ser)
        # 将期望的结果系列也转换为特定类型的箱对象
        expected = tm.box_expected(expected, xbox)

        # 执行浮点数系列ser的右除操作，结果存储在result中
        result = ser.__rtruediv__(tdi)
        # 如果box是DataFrame，则断言结果为NotImplemented
        if box is DataFrame:
            assert result is NotImplemented
        else:
            # 否则，验证结果与期望值是否相等
            tm.assert_equal(result, expected)

    # 测试函数，确保我们能够推断结果为时间增量类型
    def test_td64arr_all_nat_div_object_dtype_numeric(self, box_with_array):
        # GH#39750 确保我们能够推断结果为td64
        # 创建一个时间增量索引对象，其中包含两个NaT值（Not-a-Time）
        tdi = TimedeltaIndex([NaT, NaT])

        # 将时间增量索引对象tdi转换为特定类型的箱对象
        left = tm.box_expected(tdi, box_with_array)
        # 创建一个包含对象类型数据的NumPy数组作为右操作数
        right = np.array([2, 2.0], dtype=object)

        # 创建一个包含NaT对象的期望结果索引对象
        tdnat = np.timedelta64("NaT", "ns")
        expected = Index([tdnat] * 2, dtype=object)
        # 如果box_with_array不是Index类型，则将期望结果转换为特定类型的箱对象
        if box_with_array is not Index:
            expected = tm.box_expected(expected, box_with_array).astype(object)
            # 如果box_with_array是Series或DataFrame类型，则用NaT填充缺失值
            if box_with_array in [Series, DataFrame]:
                expected = expected.fillna(tdnat)  # GH#18463

        # 执行左操作数left和右操作数right的除法操作，结果存储在result中
        result = left / right
        # 验证结果与期望值是否相等
        tm.assert_equal(result, expected)

        # 执行左操作数left和右操作数right的整除操作，结果存储在result中
        result = left // right
        # 验证结果与期望值是否相等
        tm.assert_equal(result, expected)
class TestTimedelta64ArrayLikeArithmetic:
    # 为 timedelta64[ns] 向量设计的算术测试，参数化涵盖了 DataFrame/Series/TimedeltaIndex/TimedeltaArray。
    # 理想情况下，所有算术测试最终都应在此处完成。

    def test_td64arr_pow_invalid(self, scalar_td, box_with_array):
        # 创建一个包含三个 timedelta(minutes=5, seconds=3) 的 Series 对象
        td1 = Series([timedelta(minutes=5, seconds=3)] * 3)
        # 将第三个元素设置为 NaN
        td1.iloc[2] = np.nan

        # 将 td1 对象转换为所需的数据类型
        td1 = tm.box_expected(td1, box_with_array)

        # 检查是否引发 TypeError 异常，匹配包含 'operate' (来自 core/ops.py) 的消息，用于未定义的操作
        pattern = "operate|unsupported|cannot|not supported"
        with pytest.raises(TypeError, match=pattern):
            # 尝试执行 scalar_td ** td1 操作
            scalar_td ** td1

        with pytest.raises(TypeError, match=pattern):
            # 尝试执行 td1 ** scalar_td 操作
            td1 ** scalar_td


def test_add_timestamp_to_timedelta():
    # GH: 35897
    # 创建一个 Timestamp 对象，表示时间戳 "2021-01-01"
    timestamp = Timestamp("2021-01-01")
    # 计算 timestamp 加上 timedelta_range("0s", "1s", periods=31) 的结果
    result = timestamp + timedelta_range("0s", "1s", periods=31)
    # 创建一个预期的 DatetimeIndex 对象，包含 31 个元素
    expected = DatetimeIndex(
        [
            timestamp
            + (
                pd.to_timedelta("0.033333333s") * i
                + pd.to_timedelta("0.000000001s") * divmod(i, 3)[0]
            )
            for i in range(31)
        ]
    )
    # 使用 tm.assert_index_equal 检查 result 是否与 expected 相等
    tm.assert_index_equal(result, expected)
```