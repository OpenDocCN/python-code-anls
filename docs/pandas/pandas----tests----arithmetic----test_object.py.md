# `D:\src\scipysrc\pandas\pandas\tests\arithmetic\test_object.py`

```
# Arithmetic tests for DataFrame/Series/Index/Array classes that should
# behave identically.
# Specifically for object dtype

import datetime  # 导入datetime模块
from decimal import Decimal  # 从decimal模块导入Decimal类
import operator  # 导入operator模块

import numpy as np  # 导入numpy库，并使用np作为别名
import pytest  # 导入pytest库，用于单元测试

from pandas._config import using_pyarrow_string_dtype  # 从pandas._config模块导入using_pyarrow_string_dtype

import pandas.util._test_decorators as td  # 导入pandas.util._test_decorators模块，并使用td作为别名

import pandas as pd  # 导入pandas库，并使用pd作为别名
from pandas import (  # 从pandas库中导入以下内容：
    Series,  # Series类
    Timestamp,  # Timestamp类
    option_context,  # option_context函数
)
import pandas._testing as tm  # 导入pandas._testing模块，并使用tm作为别名
from pandas.core import ops  # 从pandas.core模块导入ops

# ------------------------------------------------------------------
# Comparisons

class TestObjectComparisons:
    def test_comparison_object_numeric_nas(self, comparison_op):
        ser = Series(np.random.default_rng(2).standard_normal(10), dtype=object)
        shifted = ser.shift(2)

        func = comparison_op

        result = func(ser, shifted)
        expected = func(ser.astype(float), shifted.astype(float))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "infer_string", [False, pytest.param(True, marks=td.skip_if_no("pyarrow"))]
    )
    def test_object_comparisons(self, infer_string):
        with option_context("future.infer_string", infer_string):
            ser = Series(["a", "b", np.nan, "c", "a"])

            result = ser == "a"
            expected = Series([True, False, False, False, True])
            tm.assert_series_equal(result, expected)

            result = ser < "a"
            expected = Series([False, False, False, False, False])
            tm.assert_series_equal(result, expected)

            result = ser != "a"
            expected = -(ser == "a")
            tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [None, object])
    def test_more_na_comparisons(self, dtype):
        left = Series(["a", np.nan, "c"], dtype=dtype)
        right = Series(["a", np.nan, "d"], dtype=dtype)

        result = left == right
        expected = Series([True, False, False])
        tm.assert_series_equal(result, expected)

        result = left != right
        expected = Series([False, True, True])
        tm.assert_series_equal(result, expected)

        result = left == np.nan
        expected = Series([False, False, False])
        tm.assert_series_equal(result, expected)

        result = left != np.nan
        expected = Series([True, True, True])
        tm.assert_series_equal(result, expected)


# ------------------------------------------------------------------
# Arithmetic

class TestArithmetic:
    def test_add_period_to_array_of_offset(self):
        # GH#50162
        per = pd.Period("2012-1-1", freq="D")
        pi = pd.period_range("2012-1-1", periods=10, freq="D")
        idx = per - pi

        expected = pd.Index([x + per for x in idx], dtype=object)
        result = idx + per
        tm.assert_index_equal(result, expected)

        result = per + idx
        tm.assert_index_equal(result, expected)

    # TODO: parametrize
    def test_pow_ops_object(self):
        # GH#22922
        # pow is weird with masking & 1, so testing here
        # 创建包含对象类型数据的 Series a 和 b
        a = Series([1, np.nan, 1, np.nan], dtype=object)
        b = Series([1, np.nan, np.nan, 1], dtype=object)
        # 进行幂运算并保存结果
        result = a**b
        # 创建预期的 Series 对象，保存幂运算的期望结果
        expected = Series(a.values**b.values, dtype=object)
        # 使用测试工具函数验证结果是否符合预期
        tm.assert_series_equal(result, expected)

        # 交换幂运算的操作数并保存结果
        result = b**a
        # 创建预期的 Series 对象，保存幂运算的期望结果
        expected = Series(b.values**a.values, dtype=object)
        # 使用测试工具函数验证结果是否符合预期
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("op", [operator.add, ops.radd])
    @pytest.mark.parametrize("other", ["category", "Int64"])
    def test_add_extension_scalar(self, other, box_with_array, op):
        # GH#22378
        # 检查标量是否满足 is_extension_array_dtype(obj)，以确保不会错误地分派到 ExtensionArray 操作
        # 创建包含字符串的 Series arr
        arr = Series(["a", "b", "c"])
        # 创建预期结果的 Series 对象，应用运算 op(x, other) 到 arr 中的每个元素
        expected = Series([op(x, other) for x in arr])

        # 将 arr 转换为期望的包装形式
        arr = tm.box_expected(arr, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        # 执行运算 op(arr, other)，并验证结果是否符合预期
        result = op(arr, other)
        tm.assert_equal(result, expected)

    def test_objarr_add_str(self, box_with_array):
        # 创建包含字符串和 NaN 的 Series ser
        ser = Series(["x", np.nan, "x"])
        # 创建预期结果的 Series 对象，通过在每个元素后添加 "a" 实现字符串连接
        expected = Series(["xa", np.nan, "xa"])

        # 将 ser 转换为期望的包装形式
        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        # 执行字符串连接操作 ser + "a"，并验证结果是否符合预期
        result = ser + "a"
        tm.assert_equal(result, expected)

    def test_objarr_radd_str(self, box_with_array):
        # 创建包含字符串和 NaN 的 Series ser
        ser = Series(["x", np.nan, "x"])
        # 创建预期结果的 Series 对象，通过在每个元素前添加 "a" 实现字符串连接
        expected = Series(["ax", np.nan, "ax"])

        # 将 ser 转换为期望的包装形式
        ser = tm.box_expected(ser, box_with_array)
        expected = tm.box_expected(expected, box_with_array)

        # 执行字符串连接操作 "a" + ser，并验证结果是否符合预期
        result = "a" + ser
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "data",
        [
            [1, 2, 3],
            [1.1, 2.2, 3.3],
            [Timestamp("2011-01-01"), Timestamp("2011-01-02"), pd.NaT],
            ["x", "y", 1],
        ],
    )
    @pytest.mark.parametrize("dtype", [None, object])
    def test_objarr_radd_str_invalid(self, dtype, data, box_with_array):
        # 根据给定的数据和 dtype 创建 Series 对象
        ser = Series(data, dtype=dtype)

        # 将 ser 转换为期望的包装形式
        ser = tm.box_expected(ser, box_with_array)
        # 创建错误消息字符串，用于验证是否引发 TypeError 异常
        msg = "|".join(
            [
                "can only concatenate str",
                "did not contain a loop with signature matching types",
                "unsupported operand type",
                "must be str",
            ]
        )
        # 使用 pytest 的断言来验证在执行 "foo_" + ser 操作时是否引发了预期的 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            "foo_" + ser

    @pytest.mark.parametrize("op", [operator.add, ops.radd, operator.sub, ops.rsub])
    # 定义一个测试方法，用于测试对象数组添加操作的无效情况
    def test_objarr_add_invalid(self, op, box_with_array):
        # 将参数 box_with_array 赋值给变量 box
        box = box_with_array

        # 创建一个包含对象类型数据的 Series 对象，数据为 ["a", "b", "c"]，名称为 "objects"
        obj_ser = Series(list("abc"), dtype=object, name="objects")

        # 调用 tm.box_expected 方法，将 obj_ser 根据 box 进行处理
        obj_ser = tm.box_expected(obj_ser, box)

        # 定义错误消息，用于匹配异常信息
        msg = "|".join(
            [
                "can only concatenate str",
                "unsupported operand type",
                "must be str",
                "has no kernel",
            ]
        )

        # 使用 pytest 断言，验证执行 op(obj_ser, 1) 时是否抛出异常，并且异常信息匹配 msg
        with pytest.raises(Exception, match=msg):
            op(obj_ser, 1)

        # 使用 pytest 断言，验证执行 op(obj_ser, np.array(1, dtype=np.int64)) 时是否抛出异常，并且异常信息匹配 msg
        with pytest.raises(Exception, match=msg):
            op(obj_ser, np.array(1, dtype=np.int64))

    # TODO: Moved from tests.series.test_operators; needs cleanup
    # 定义一个测试方法，用于测试操作符处理缺失值情况
    def test_operators_na_handling(self):
        # 创建一个包含 ["foo", "bar", "baz", np.nan] 的 Series 对象
        ser = Series(["foo", "bar", "baz", np.nan])

        # 执行字符串连接操作："prefix_" + ser
        result = "prefix_" + ser

        # 创建预期的 Series 对象，期望结果为 ["prefix_foo", "prefix_bar", "prefix_baz", np.nan]
        expected = Series(["prefix_foo", "prefix_bar", "prefix_baz", np.nan])

        # 使用 tm.assert_series_equal 断言 result 与 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 执行字符串连接操作：ser + "_suffix"
        result = ser + "_suffix"

        # 创建预期的 Series 对象，期望结果为 ["foo_suffix", "bar_suffix", "baz_suffix", np.nan]
        expected = Series(["foo_suffix", "bar_suffix", "baz_suffix", np.nan])

        # 使用 tm.assert_series_equal 断言 result 与 expected 是否相等
        tm.assert_series_equal(result, expected)

    # TODO: parametrize over box
    # 使用 pytest.mark.parametrize 标记的测试方法，参数为 dtype，参数化测试不同的数据类型
    def test_series_with_dtype_radd_timedelta(self, dtype):
        # 注意此测试不适用于 timedelta64 类型的 Series
        # 在版本 2.0 之后，当 ser.dtype == object 时，我们保留对象类型数据

        # 创建一个包含 pd.Timedelta 对象的 Series 对象，数据为 [pd.Timedelta("1 days"), pd.Timedelta("2 days"), pd.Timedelta("3 days")]
        ser = Series(
            [pd.Timedelta("1 days"), pd.Timedelta("2 days"), pd.Timedelta("3 days")],
            dtype=dtype,
        )

        # 创建预期的 Series 对象，期望结果为 [pd.Timedelta("4 days"), pd.Timedelta("5 days"), pd.Timedelta("6 days")]
        expected = Series(
            [pd.Timedelta("4 days"), pd.Timedelta("5 days"), pd.Timedelta("6 days")],
            dtype=dtype,
        )

        # 执行 pd.Timedelta("3 days") + ser 操作，生成结果
        result = pd.Timedelta("3 days") + ser

        # 使用 tm.assert_series_equal 断言 result 与 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 执行 ser + pd.Timedelta("3 days") 操作，生成结果
        result = ser + pd.Timedelta("3 days")

        # 使用 tm.assert_series_equal 断言 result 与 expected 是否相等
        tm.assert_series_equal(result, expected)

    # TODO: cleanup & parametrize over box
    # 定义测试方法，验证混合时区的时间序列操作对对象的影响
    def test_mixed_timezone_series_ops_object(self):
        # 创建带有不同时区时间戳的时间序列对象
        ser = Series(
            [
                Timestamp("2015-01-01", tz="US/Eastern"),
                Timestamp("2015-01-01", tz="Asia/Tokyo"),
            ],
            name="xxx",
        )
        # 确认序列的数据类型为 object
        assert ser.dtype == object

        # 预期的结果时间序列，对原始序列添加一天后的结果
        exp = Series(
            [
                Timestamp("2015-01-02", tz="US/Eastern"),
                Timestamp("2015-01-02", tz="Asia/Tokyo"),
            ],
            name="xxx",
        )
        # 验证 ser 加上一天后的结果与预期一致
        tm.assert_series_equal(ser + pd.Timedelta("1 days"), exp)
        # 验证一天加上 ser 后的结果与预期一致
        tm.assert_series_equal(pd.Timedelta("1 days") + ser, exp)

        # 创建另一个带有不同时区时间戳的时间序列对象
        ser2 = Series(
            [
                Timestamp("2015-01-03", tz="US/Eastern"),
                Timestamp("2015-01-05", tz="Asia/Tokyo"),
            ],
            name="xxx",
        )
        # 确认序列的数据类型为 object
        assert ser2.dtype == object
        # 预期的结果时间序列，对两个时间序列相减的结果
        exp = Series(
            [pd.Timedelta("2 days"), pd.Timedelta("4 days")], name="xxx", dtype=object
        )
        # 验证 ser2 减去 ser 后的结果与预期一致
        tm.assert_series_equal(ser2 - ser, exp)
        # 验证 ser 减去 ser2 后的结果与预期一致
        tm.assert_series_equal(ser - ser2, -exp)

        # 创建带有时间增量的时间序列对象
        ser = Series(
            [pd.Timedelta("01:00:00"), pd.Timedelta("02:00:00")],
            name="xxx",
            dtype=object,
        )
        # 确认序列的数据类型为 object
        assert ser.dtype == object

        # 预期的结果时间序列，对原始序列加上 30 分钟后的结果
        exp = Series(
            [pd.Timedelta("01:30:00"), pd.Timedelta("02:30:00")],
            name="xxx",
            dtype=object,
        )
        # 验证 ser 加上 30 分钟后的结果与预期一致
        tm.assert_series_equal(ser + pd.Timedelta("00:30:00"), exp)
        # 验证 30 分钟加上 ser 后的结果与预期一致
        tm.assert_series_equal(pd.Timedelta("00:30:00") + ser, exp)

    # TODO: cleanup & parametrize over box
    # 定义测试方法，验证 __iadd__ 和 __isub__ 方法保持索引名称不变
    def test_iadd_preserves_name(self):
        # 创建整数类型的序列对象
        ser = Series([1, 2, 3])
        # 设置序列的索引名称为 "foo"
        ser.index.name = "foo"

        # 执行索引加一操作，并验证索引名称仍为 "foo"
        ser.index += 1
        assert ser.index.name == "foo"

        # 执行索引减一操作，并验证索引名称仍为 "foo"
        ser.index -= 1
        assert ser.index.name == "foo"

    # 定义测试方法，验证字符串加法操作
    def test_add_string(self):
        # 创建包含字符串的索引对象
        index = pd.Index(["a", "b", "c"])
        # 将字符串 "foo" 添加到索引中
        index2 = index + "foo"

        # 验证 "a" 不在 index2 中
        assert "a" not in index2
        # 验证 "afoo" 在 index2 中
        assert "afoo" in index2

    # 定义测试方法，验证字符串原地加法操作
    def test_iadd_string(self):
        # 创建包含字符串的索引对象
        index = pd.Index(["a", "b", "c"])
        # 断言 "a" 存在于 index 中
        assert "a" in index

        # 执行索引原地加 "_x" 的操作，并验证 "a_x" 存在于 index 中
        index += "_x"
        assert "a_x" in index

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="add doesn't work")
    # 标记测试方法为预期失败，原因是 pyarrow 的字符串类型处理不支持加法操作
    def test_add(self):
        # 创建一个包含数字字符串的索引
        index = pd.Index([str(i) for i in range(10)])
        # 期望的索引是每个元素乘以2后的结果
        expected = pd.Index(index.values * 2)
        # 断言索引相加的结果与期望相等
        tm.assert_index_equal(index + index, expected)
        # 断言索引与索引转换为列表后相加的结果与期望相等
        tm.assert_index_equal(index + index.tolist(), expected)
        # 断言索引转换为列表与索引相加的结果与期望相等
        tm.assert_index_equal(index.tolist() + index, expected)

        # 测试字符串与索引相加及反向相加
        index = pd.Index(list("abc"))
        # 期望的索引是每个字符后面加上字符"1"
        expected = pd.Index(["a1", "b1", "c1"])
        tm.assert_index_equal(index + "1", expected)
        # 期望的索引是字符"1"加上每个字符
        expected = pd.Index(["1a", "1b", "1c"])
        tm.assert_index_equal("1" + index, expected)

    def test_sub_fail(self, using_infer_string):
        # 创建一个包含数字字符串的索引
        index = pd.Index([str(i) for i in range(10)])

        if using_infer_string:
            import pyarrow as pa
            # 如果使用推断字符串，则错误类型为 ArrowNotImplementedError
            err = pa.lib.ArrowNotImplementedError
            msg = "has no kernel"
        else:
            # 否则错误类型为 TypeError
            err = TypeError
            msg = "unsupported operand type|Cannot broadcast"
        # 使用 pytest 断言引发特定类型的错误，并匹配给定的错误消息
        with pytest.raises(err, match=msg):
            index - "a"
        with pytest.raises(err, match=msg):
            index - index
        with pytest.raises(err, match=msg):
            index - index.tolist()
        with pytest.raises(err, match=msg):
            index.tolist() - index

    def test_sub_object(self):
        # GH#19369
        # 创建一个包含 Decimal 对象的索引
        index = pd.Index([Decimal(1), Decimal(2)])
        # 期望的索引是每个元素减去 Decimal(1) 后的结果
        expected = pd.Index([Decimal(0), Decimal(1)])

        # 测试索引减去 Decimal 对象
        result = index - Decimal(1)
        tm.assert_index_equal(result, expected)

        # 测试索引减去另一个包含 Decimal 对象的索引
        result = index - pd.Index([Decimal(1), Decimal(1)])
        tm.assert_index_equal(result, expected)

        # 测试索引减去字符串时应该引发 TypeError 错误
        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            index - "foo"

        # 测试索引减去包含对象数组时应该引发 TypeError 错误
        with pytest.raises(TypeError, match=msg):
            index - np.array([2, "foo"], dtype=object)

    def test_rsub_object(self, fixed_now_ts):
        # GH#19369
        # 创建一个包含 Decimal 对象的索引
        index = pd.Index([Decimal(1), Decimal(2)])
        # 期望的索引是 Decimal(2) 减去每个元素后的结果
        expected = pd.Index([Decimal(1), Decimal(0)])

        # 测试 Decimal 对象减去索引
        result = Decimal(2) - index
        tm.assert_index_equal(result, expected)

        # 测试包含 Decimal 对象的数组减去索引
        result = np.array([Decimal(2), Decimal(2)]) - index
        tm.assert_index_equal(result, expected)

        # 测试字符串减去索引时应该引发 TypeError 错误
        msg = "unsupported operand type"
        with pytest.raises(TypeError, match=msg):
            "foo" - index

        # 测试包含对象数组减去索引时应该引发 TypeError 错误
        with pytest.raises(TypeError, match=msg):
            np.array([True, fixed_now_ts]) - index
# 定义一个自定义的索引类 MyIndex，继承自 pandas 的 Index 类
class MyIndex(pd.Index):
    # 简单的索引子类，用于跟踪操作调用次数

    _calls: int  # 类属性，用于记录操作调用次数

    @classmethod
    def _simple_new(cls, values, name=None, dtype=None):
        # 创建一个新的 MyIndex 对象
        result = object.__new__(cls)
        result._data = values  # 将传入的值作为数据存储在 _data 属性中
        result._name = name  # 设置索引的名称
        result._calls = 0  # 初始化调用次数为 0
        result._reset_identity()  # 重置标识（可能是一些额外的初始化操作）

        return result

    def __add__(self, other):
        # 重载加法运算符，每次调用增加 _calls 计数
        self._calls += 1
        return self._simple_new(self._data)

    def __radd__(self, other):
        # 反向重载加法运算符，调用 __add__ 处理
        return self.__add__(other)


@pytest.mark.parametrize(
    "other",
    [
        [datetime.timedelta(1), datetime.timedelta(2)],  # 时间增量列表
        [datetime.datetime(2000, 1, 1), datetime.datetime(2000, 1, 2)],  # 日期时间列表
        [pd.Period("2000"), pd.Period("2001")],  # pandas 周期对象列表
        ["a", "b"],  # 字符串列表
    ],
    ids=["timedelta", "datetime", "period", "object"],  # 测试参数化的标识符
)
def test_index_ops_defer_to_unknown_subclasses(other):
    # 测试函数：验证索引操作是否正确委托给未知子类
    
    # 创建包含日期对象的 numpy 数组
    values = np.array(
        [datetime.date(2000, 1, 1), datetime.date(2000, 1, 2)], dtype=object
    )
    # 使用 _simple_new 方法创建 MyIndex 对象
    a = MyIndex._simple_new(values)
    # 将 other 转换为 pandas 的 Index 对象
    other = pd.Index(other)
    # 执行加法操作
    result = other + a
    # 断言结果类型为 MyIndex 类型
    assert isinstance(result, MyIndex)
    # 断言调用次数为 1
    assert a._calls == 1
```