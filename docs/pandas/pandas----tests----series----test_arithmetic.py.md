# `D:\src\scipysrc\pandas\pandas\tests\series\test_arithmetic.py`

```
from datetime import (  # 导入日期相关模块
    date,  # 导入日期对象
    timedelta,  # 导入时间间隔对象
    timezone,  # 导入时区对象
)
from decimal import Decimal  # 导入 Decimal 类
import operator  # 导入操作符模块

import numpy as np  # 导入 NumPy 库并重命名为 np
import pytest  # 导入 pytest 测试框架

from pandas._libs import lib  # 导入 pandas 内部库
from pandas._libs.tslibs import IncompatibleFrequency  # 导入不兼容频率异常

import pandas as pd  # 导入 pandas 并重命名为 pd
from pandas import (  # 从 pandas 导入多个对象
    Categorical,  # 导入分类数据类型
    DatetimeTZDtype,  # 导入带有时区的日期时间数据类型
    Index,  # 导入索引对象
    Series,  # 导入序列对象
    Timedelta,  # 导入时间增量对象
    bdate_range,  # 导入工作日范围生成函数
    date_range,  # 导入日期范围生成函数
    isna,  # 导入判断缺失值函数
)
import pandas._testing as tm  # 导入 pandas 测试模块
from pandas.core import ops  # 导入 pandas 核心操作模块
from pandas.core.computation import expressions as expr  # 导入 pandas 计算表达式模块中的表达式对象
from pandas.core.computation.check import NUMEXPR_INSTALLED  # 导入 NUMEXPR_INSTALLED 检查对象

@pytest.fixture(autouse=True, params=[0, 1000000], ids=["numexpr", "python"])
def switch_numexpr_min_elements(request, monkeypatch):
    # 用于测试的装置函数，根据参数设置 _MIN_ELEMENTS 属性
    with monkeypatch.context() as m:
        m.setattr(expr, "_MIN_ELEMENTS", request.param)
        yield

def _permute(obj):
    # 对对象进行随机排列函数
    return obj.take(np.random.default_rng(2).permutation(len(obj)))

class TestSeriesFlexArithmetic:
    @pytest.mark.parametrize(
        "ts",
        [
            (lambda x: x, lambda x: x * 2, False),  # 测试参数，包含操作和预期结果
            (lambda x: x, lambda x: x[::2], False),  # 测试参数，包含操作和预期结果
            (lambda x: x, lambda x: 5, True),  # 测试参数，包含操作和预期结果
            (
                lambda x: Series(range(10), dtype=np.float64),  # 测试参数，包含操作和预期结果
                lambda x: Series(range(10), dtype=np.float64),  # 测试参数，包含操作和预期结果
                True,  # 检查是否需要进行反向操作
            ),
        ],
    )
    @pytest.mark.parametrize(
        "opname", ["add", "sub", "mul", "floordiv", "truediv", "pow"]
    )
    def test_flex_method_equivalence(self, opname, ts):
        # 测试灵活方法等效性，比较 Series.{opname} 和 Series.__{opname}__
        tser = Series(
            np.arange(20, dtype=np.float64),  # 创建包含浮点数的序列
            index=date_range("2020-01-01", periods=20),  # 使用日期范围作为索引
            name="ts",  # 设置序列名称
        )

        series = ts[0](tser)  # 应用第一个操作到序列
        other = ts[1](tser)  # 应用第二个操作到序列
        check_reverse = ts[2]  # 是否需要检查反向操作

        op = getattr(Series, opname)  # 获取操作符对应的方法
        alt = getattr(operator, opname)  # 获取标准库操作符对应的方法

        result = op(series, other)  # 执行操作
        expected = alt(series, other)  # 使用标准库方法进行操作
        tm.assert_almost_equal(result, expected)  # 断言结果与预期接近
        if check_reverse:
            rop = getattr(Series, "r" + opname)  # 获取反向操作方法
            result = rop(series, other)  # 执行反向操作
            expected = alt(other, series)  # 使用标准库方法进行反向操作
            tm.assert_almost_equal(result, expected)  # 断言反向操作结果与预期接近

    def test_flex_method_subclass_metadata_preservation(self, all_arithmetic_operators):
        # 测试子类元数据的保留问题
        class MySeries(Series):
            _metadata = ["x"]  # 定义元数据属性列表

            @property
            def _constructor(self):
                return MySeries  # 返回当前子类

        opname = all_arithmetic_operators  # 获取所有算术运算符
        op = getattr(Series, opname)  # 获取操作符对应的方法
        m = MySeries([1, 2, 3], name="test")  # 创建 MySeries 对象
        m.x = 42  # 设置元数据属性
        result = op(m, 1)  # 执行操作
        assert result.x == 42  # 断言操作后元数据属性的值正确

    def test_flex_add_scalar_fill_value(self):
        # 测试在填充缺失值后进行标量加法的情况
        ser = Series([0, 1, np.nan, 3, 4, 5])  # 创建包含缺失值的序列

        exp = ser.fillna(0).add(2)  # 期望的结果：填充缺失值后加 2
        res = ser.add(2, fill_value=0)  # 实际结果：在填充缺失值后加 2
        tm.assert_series_equal(res, exp)  # 断言实际结果与期望结果相等

    pairings = [(Series.div, operator.truediv, 1), (Series.rdiv, ops.rtruediv, 1)]
    # 对于每个运算符进行循环测试
    for op in ["add", "sub", "mul", "pow", "truediv", "floordiv"]:
        # 初始化填充值为0
        fv = 0
        # 获取 Series 类中对应操作符的方法
        lop = getattr(Series, op)
        # 获取 operator 模块中对应操作符的函数
        lequiv = getattr(operator, op)
        # 获取 Series 类中对应右侧操作符的方法
        rop = getattr(Series, "r" + op)
        # 定义一个 lambda 函数，将 operator 模块中对应操作符的函数绑定到右侧操作
        # 在定义时绑定操作符...
        requiv = lambda x, y, op=op: getattr(operator, op)(y, x)
        # 将左操作符方法、左操作符等价函数、填充值组成元组，添加到 pairings 列表中
        pairings.append((lop, lequiv, fv))
        # 将右操作符方法、右操作符等价 lambda 函数、填充值组成元组，添加到 pairings 列表中
        pairings.append((rop, requiv, fv))

    # 使用 pytest.mark.parametrize 标记装饰器，对每个操作符进行参数化测试
    @pytest.mark.parametrize("op, equiv_op, fv", pairings)
    def test_operators_combine(self, op, equiv_op, fv):
        # 定义一个内部函数，用于检查填充值后的操作结果
        def _check_fill(meth, op, a, b, fill_value=0):
            # 计算期望的索引，合并 a 和 b 的索引
            exp_index = a.index.union(b.index)
            a = a.reindex(exp_index)
            b = b.reindex(exp_index)

            amask = isna(a)
            bmask = isna(b)

            exp_values = []
            # 遍历期望的索引
            for i in range(len(exp_index)):
                with np.errstate(all="ignore"):
                    if amask[i]:
                        if bmask[i]:
                            exp_values.append(np.nan)
                            continue
                        exp_values.append(op(fill_value, b[i]))
                    elif bmask[i]:
                        if amask[i]:
                            exp_values.append(np.nan)
                            continue
                        exp_values.append(op(a[i], fill_value))
                    else:
                        exp_values.append(op(a[i], b[i]))

            # 执行方法 meth，并得到结果
            result = meth(a, b, fill_value=fill_value)
            # 创建预期的 Series 对象，包含预期的值和索引
            expected = Series(exp_values, exp_index)
            # 使用测试框架中的函数比较结果和预期
            tm.assert_series_equal(result, expected)

        # 创建两个 Series 对象 a 和 b，用于测试
        a = Series([np.nan, 1.0, 2.0, 3.0, np.nan], index=np.arange(5))
        b = Series([np.nan, 1, np.nan, 3, np.nan, 4.0], index=np.arange(6))

        # 使用操作符 op 对 a 和 b 进行操作，得到结果 result
        result = op(a, b)
        # 使用等价操作符 equiv_op 对 a 和 b 进行操作，得到期望的结果 exp
        exp = equiv_op(a, b)
        # 使用测试框架中的函数比较结果和期望
        tm.assert_series_equal(result, exp)
        # 调用内部函数 _check_fill，检查填充值后的操作结果
        _check_fill(op, equiv_op, a, b, fill_value=fv)
        # 测试操作符 op 是否接受 axis=0 或 axis='rows'
        op(a, b, axis=0)
class TestSeriesArithmetic:
    # 测试序列算术操作的类

    def test_add_series_with_period_index(self):
        # 测试带有周期索引的序列加法

        # 创建一个年度的周期范围
        rng = pd.period_range("1/1/2000", "1/1/2010", freq="Y")
        # 创建一个随机数据序列，使用标准正态分布
        ts = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        # 执行序列加法操作，每隔一个元素进行加法
        result = ts + ts[::2]
        # 创建期望结果，每隔一个元素进行加法，其余元素设置为NaN
        expected = ts + ts
        expected.iloc[1::2] = np.nan
        # 断言序列是否相等
        tm.assert_series_equal(result, expected)

        # 执行序列加法操作，使用 _permute 函数进行每隔一个元素加法
        result = ts + _permute(ts[::2])
        # 断言序列是否相等
        tm.assert_series_equal(result, expected)

        # 测试在不兼容的频率下进行加法操作
        msg = "Input has different freq=D from Period\\(freq=Y-DEC\\)"
        with pytest.raises(IncompatibleFrequency, match=msg):
            ts + ts.asfreq("D", how="end")

    @pytest.mark.parametrize(
        "target_add,input_value,expected_value",
        [
            ("!", ["hello", "world"], ["hello!", "world!"]),
            ("m", ["hello", "world"], ["hellom", "worldm"]),
        ],
    )
    def test_string_addition(self, target_add, input_value, expected_value):
        # GH28658 - ensure adding 'm' does not raise an error
        # 创建字符串序列
        a = Series(input_value)

        # 执行字符串序列和字符的加法操作
        result = a + target_add
        expected = Series(expected_value)
        # 断言序列是否相等
        tm.assert_series_equal(result, expected)

    def test_divmod(self):
        # GH#25557
        # 创建整数和NaN组成的序列a和b
        a = Series([1, 1, 1, np.nan], index=["a", "b", "c", "d"])
        b = Series([2, np.nan, 1, np.nan], index=["a", "b", "d", "e"])

        # 执行 divmod 操作
        result = a.divmod(b)
        expected = divmod(a, b)
        # 断言结果是否相等
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

        # 执行 rdivmod 操作
        result = a.rdivmod(b)
        expected = divmod(b, a)
        # 断言结果是否相等
        tm.assert_series_equal(result[0], expected[0])
        tm.assert_series_equal(result[1], expected[1])

    @pytest.mark.parametrize("index", [None, range(9)])
    def test_series_integer_mod(self, index):
        # GH#24396
        # 创建整数序列 s1 和字符串序列 s2
        s1 = Series(range(1, 10))
        s2 = Series("foo", index=index)

        # 测试在字符串序列上执行模运算
        msg = "not all arguments converted during string formatting|mod not"

        with pytest.raises((TypeError, NotImplementedError), match=msg):
            s2 % s1

    def test_add_with_duplicate_index(self):
        # GH14227
        # 创建带有重复索引的序列 s1 和 s2
        s1 = Series([1, 2], index=[1, 1])
        s2 = Series([10, 10], index=[1, 2])
        # 执行序列加法操作
        result = s1 + s2
        expected = Series([11, 12, np.nan], index=[1, 1, 2])
        # 断言序列是否相等
        tm.assert_series_equal(result, expected)

    def test_add_na_handling(self):
        # 创建包含 Decimal 数字的日期索引序列
        ser = Series(
            [Decimal("1.3"), Decimal("2.3")], index=[date(2012, 1, 1), date(2012, 1, 2)]
        )

        # 执行序列加法操作，并测试处理NaN的情况
        result = ser + ser.shift(1)
        result2 = ser.shift(1) + ser
        assert isna(result.iloc[0])
        assert isna(result2.iloc[0])

    def test_add_corner_cases(self, datetime_series):
        # 创建空序列
        empty = Series([], index=Index([]), dtype=np.float64)

        # 执行空序列和日期时间序列的加法操作，测试结果是否全为NaN
        result = datetime_series + empty
        assert np.isnan(result).all()

        # 执行空序列和空序列的加法操作，测试结果长度是否为0
        result = empty + empty
        assert len(result) == 0
    # 测试将日期时间系列与整数类型的系列相加
    def test_add_float_plus_int(self, datetime_series):
        # 将日期时间系列转换为整数类型，并去掉最后5个元素
        int_ts = datetime_series.astype(int)[:-5]
        # 将原始日期时间系列与整数类型的系列相加
        added = datetime_series + int_ts
        # 创建预期结果的日期时间系列，包括索引和名称
        expected = Series(
            datetime_series.values[:-5] + int_ts.values,
            index=datetime_series.index[:-5],
            name="ts",
        )
        # 断言相加后的结果与预期结果相等
        tm.assert_series_equal(added[:-5], expected)

    # 测试空整数类型序列与浮点数序列相乘的边缘情况
    def test_mul_empty_int_corner_case(self):
        # 创建一个空的整数类型序列 s1 和一个包含一个值的浮点数序列 s2
        s1 = Series([], [], dtype=np.int32)
        s2 = Series({"x": 0.0})
        # 断言两个序列相乘的结果与预期结果相等，结果为包含 NaN 值的序列
        tm.assert_series_equal(s1 * s2, Series([np.nan], index=["x"]))

    # 测试日期时间操作的对齐性
    def test_sub_datetimelike_align(self):
        # GH#7500
        # 日期时间操作需要对齐
        # 创建一个包含三个日期的日期时间系列 dt，并将第三个日期设为 NaN
        dt = Series(date_range("2012-1-1", periods=3, freq="D"))
        dt.iloc[2] = np.nan
        # 创建 dt 的逆序日期时间系列 dt2
        dt2 = dt[::-1]

        # 创建预期的日期时间差序列 expected，包括 timedelta 和 NaT
        expected = Series([timedelta(0), timedelta(0), pd.NaT])
        # 执行 dt2 与 dt 的减法操作，重设名称
        result = dt2 - dt
        tm.assert_series_equal(result, expected)

        # 将 dt2 和 dt 转换为 DataFrame，执行减法操作，获取列名为 0 的序列
        expected = Series(expected, name=0)
        result = (dt2.to_frame() - dt.to_frame())[0]
        tm.assert_series_equal(result, expected)

    # 测试对齐操作不改变时区信息
    def test_alignment_doesnt_change_tz(self):
        # GH#33671
        # 创建一个带时区信息的日期时间索引 dti 和其转换为 UTC 时区的日期时间索引 dti_utc
        dti = date_range("2016-01-01", periods=10, tz="CET")
        dti_utc = dti.tz_convert("UTC")
        # 创建两个值为 10 的日期时间序列 ser 和 ser_utc，分别使用 dti 和 dti_utc 作为索引
        ser = Series(10, index=dti)
        ser_utc = Series(10, index=dti_utc)

        # 执行序列相乘操作，不关心结果，只检查原始索引是否未改变
        ser * ser_utc

        # 断言 ser 和 ser_utc 的索引与原始 dti 和 dti_utc 索引相同
        assert ser.index is dti
        assert ser_utc.index is dti_utc

    # 测试与分类变量的对齐操作
    def test_alignment_categorical(self):
        # GH13365
        # 创建一个分类变量 cat
        cat = Categorical(["3z53", "3z53", "LoJG", "LoJG", "LoJG", "N503"])
        # 创建两个值为 2 的序列 ser1 和 ser2，使用 cat 作为索引
        ser1 = Series(2, index=cat)
        ser2 = Series(2, index=cat[:-1])
        # 执行 ser1 与 ser2 的乘法操作，生成结果序列 result
        result = ser1 * ser2

        # 创建预期的索引和值，包括 NaN
        exp_index = ["3z53"] * 4 + ["LoJG"] * 9 + ["N503"]
        exp_index = pd.CategoricalIndex(exp_index, categories=cat.categories)
        exp_values = [4.0] * 13 + [np.nan]
        expected = Series(exp_values, exp_index)

        # 断言 result 与 expected 序列相等
        tm.assert_series_equal(result, expected)

    # 测试具有重复索引的序列进行算术运算
    def test_arithmetic_with_duplicate_index(self):
        # GH#8363
        # 创建一个具有非唯一索引的整数序列 ser 和另一个整数序列 other
        index = [2, 2, 3, 3, 4]
        ser = Series(np.arange(1, 6, dtype="int64"), index=index)
        other = Series(np.arange(5, dtype="int64"), index=index)
        # 执行 ser 减去 other 的操作，生成结果序列 result
        result = ser - other
        # 创建预期的结果序列 expected，所有索引值为 1，类型为整数
        expected = Series(1, index=[2, 2, 3, 3, 4])
        tm.assert_series_equal(result, expected)

        # GH#8363
        # 创建一个具有非唯一索引的日期时间序列 ser 和另一个日期时间序列 other
        ser = Series(date_range("20130101 09:00:00", periods=5), index=index)
        other = Series(date_range("20130101", periods=5), index=index)
        # 执行 ser 减去 other 的操作，生成结果序列 result
        result = ser - other
        # 创建预期的结果序列 expected，所有索引值为 1 小时，类型为 timedelta
        expected = Series(Timedelta("9 hours"), index=[2, 2, 3, 3, 4])
        tm.assert_series_equal(result, expected)
    # 定义一个测试方法，用于测试掩码和非掩码情况下的值传播
    def test_masked_and_non_masked_propagate_na(self):
        # GH#45810
        # 创建一个包含浮点数和NaN的Series对象
        ser1 = Series([0, np.nan], dtype="float")
        # 创建一个包含整数和Nullable整数的Series对象
        ser2 = Series([0, 1], dtype="Int64")
        # 执行乘法操作，将两个Series相乘
        result = ser1 * ser2
        # 创建一个期望结果的Series对象，包含Float64类型和NaN值
        expected = Series([0, pd.NA], dtype="Float64")
        # 断言两个Series对象是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，测试掩码除法在非NA数据类型上的NaN传播情况
    def test_mask_div_propagate_na_for_non_na_dtype(self):
        # GH#42630
        # 创建一个包含Nullable整数和NA的Series对象
        ser1 = Series([15, pd.NA, 5, 4], dtype="Int64")
        # 创建一个包含整数、NaN和NA的Series对象
        ser2 = Series([15, 5, np.nan, 4])
        # 执行除法操作，将两个Series相除
        result = ser1 / ser2
        # 创建一个期望结果的Series对象，包含Float64类型和NaN值
        expected = Series([1.0, pd.NA, pd.NA, 1.0], dtype="Float64")
        # 断言两个Series对象是否相等
        tm.assert_series_equal(result, expected)

        # 执行除法操作，反向顺序相除
        result = ser2 / ser1
        # 断言两个Series对象是否相等
        tm.assert_series_equal(result, expected)

    # 使用pytest的参数化装饰器定义一个测试方法，测试将列表添加到掩码数组的情况
    @pytest.mark.parametrize("val, dtype", [(3, "Int64"), (3.5, "Float64")])
    def test_add_list_to_masked_array(self, val, dtype):
        # GH#22962
        # 创建一个包含整数、None和掩码的Series对象
        ser = Series([1, None, 3], dtype="Int64")
        # 执行加法操作，将列表添加到Series中
        result = ser + [1, None, val]
        # 创建一个期望结果的Series对象，根据参数化的dtype确定数据类型
        expected = Series([2, None, 3 + val], dtype=dtype)
        # 断言两个Series对象是否相等
        tm.assert_series_equal(result, expected)

        # 执行加法操作，反向顺序将Series添加到列表中
        result = [1, None, val] + ser
        # 断言两个Series对象是否相等
        tm.assert_series_equal(result, expected)

    # 定义一个测试方法，测试将列表添加到掩码数组的布尔值情况
    def test_add_list_to_masked_array_boolean(self, request):
        # GH#22962
        # 根据测试环境和条件确定警告信息
        warning = (
            UserWarning
            if request.node.callspec.id == "numexpr" and NUMEXPR_INSTALLED
            else None
        )
        # 创建一个包含布尔值、None和掩码的Series对象
        ser = Series([True, None, False], dtype="boolean")
        # 设置警告信息匹配字符串
        msg = "operator is not supported by numexpr for the bool dtype"
        # 使用断言来检查是否会产生特定警告
        with tm.assert_produces_warning(warning, match=msg):
            # 执行加法操作，将列表添加到Series中
            result = ser + [True, None, True]
        # 创建一个期望结果的Series对象，包含布尔值类型
        expected = Series([True, None, True], dtype="boolean")
        # 断言两个Series对象是否相等
        tm.assert_series_equal(result, expected)

        # 使用断言来检查是否会产生特定警告
        with tm.assert_produces_warning(warning, match=msg):
            # 执行加法操作，反向顺序将Series添加到列表中
            result = [True, None, True] + ser
        # 断言两个Series对象是否相等
        tm.assert_series_equal(result, expected)
# ------------------------------------------------------------------
# Comparisons

# 定义一个测试类 TestSeriesFlexComparison，用于灵活系列对象的比较测试
class TestSeriesFlexComparison:
    # 参数化测试，参数为 axis，可以是整数 0、None 或字符串 "index"
    @pytest.mark.parametrize("axis", [0, None, "index"])
    def test_comparison_flex_basic(self, axis, comparison_op):
        # 创建两个 Series 对象 left 和 right，内容为随机生成的标准正态分布数据
        left = Series(np.random.default_rng(2).standard_normal(10))
        right = Series(np.random.default_rng(2).standard_normal(10))
        # 使用传入的比较操作 comparison_op 对 left 和 right 进行比较，根据 axis 参数进行操作
        result = getattr(left, comparison_op.__name__)(right, axis=axis)
        # 计算预期结果，即直接对 left 和 right 执行比较操作
        expected = comparison_op(left, right)
        # 断言两个 Series 对象 result 和 expected 相等
        tm.assert_series_equal(result, expected)

    # 测试当传入不支持的 axis 参数时是否会抛出 ValueError 异常
    def test_comparison_bad_axis(self, comparison_op):
        left = Series(np.random.default_rng(2).standard_normal(10))
        right = Series(np.random.default_rng(2).standard_normal(10))

        msg = "No axis named 1 for object type"
        with pytest.raises(ValueError, match=msg):
            # 使用 getattr 获取 comparison_op 对象的方法名称，执行比较操作，并传入 axis=1 参数
            getattr(left, comparison_op.__name__)(right, axis=1)

    # 参数化测试，values 为预期的比较结果列表，op 为比较操作名称
    @pytest.mark.parametrize(
        "values, op",
        [
            ([False, False, True, False], "eq"),
            ([True, True, False, True], "ne"),
            ([False, False, True, False], "le"),
            ([False, False, False, False], "lt"),
            ([False, True, True, False], "ge"),
            ([False, True, False, False], "gt"),
        ],
    )
    def test_comparison_flex_alignment(self, values, op):
        # 创建两个 Series 对象 left 和 right，并设置它们的索引
        left = Series([1, 3, 2], index=list("abc"))
        right = Series([2, 2, 2], index=list("bcd"))
        # 使用传入的比较操作 op 对 left 和 right 进行比较
        result = getattr(left, op)(right)
        # 创建预期的 Series 对象，内容为 values 列表，索引为 left 和 right 的索引合并结果
        expected = Series(values, index=list("abcd"))
        # 断言两个 Series 对象 result 和 expected 相等
        tm.assert_series_equal(result, expected)

    # 参数化测试，values 为预期的比较结果列表，op 为比较操作名称，fill_value 为填充值
    @pytest.mark.parametrize(
        "values, op, fill_value",
        [
            ([False, False, True, True], "eq", 2),
            ([True, True, False, False], "ne", 2),
            ([False, False, True, True], "le", 0),
            ([False, False, False, True], "lt", 0),
            ([True, True, True, False], "ge", 0),
            ([True, True, False, False], "gt", 0),
        ],
    )
    def test_comparison_flex_alignment_fill(self, values, op, fill_value):
        # 创建两个 Series 对象 left 和 right，并设置它们的索引
        left = Series([1, 3, 2], index=list("abc"))
        right = Series([2, 2, 2], index=list("bcd"))
        # 使用传入的比较操作 op 对 left 和 right 进行比较，使用 fill_value 进行填充
        result = getattr(left, op)(right, fill_value=fill_value)
        # 创建预期的 Series 对象，内容为 values 列表，索引为 left 和 right 的索引合并结果
        expected = Series(values, index=list("abcd"))
        # 断言两个 Series 对象 result 和 expected 相等
        tm.assert_series_equal(result, expected)


# 定义一个测试类 TestSeriesComparison，用于一般的系列对象比较测试
class TestSeriesComparison:
    # 测试当两个 Series 长度不同时是否会抛出 ValueError 异常
    def test_comparison_different_length(self):
        a = Series(["a", "b", "c"])
        b = Series(["b", "a"])
        msg = "only compare identically-labeled Series"
        with pytest.raises(ValueError, match=msg):
            # 比较操作，应该抛出异常
            a < b

        a = Series([1, 2])
        b = Series([2, 3, 4])
        with pytest.raises(ValueError, match=msg):
            # 比较操作，应该抛出异常
            a == b

    # 参数化测试，opname 为比较操作的名称
    @pytest.mark.parametrize("opname", ["eq", "ne", "gt", "lt", "ge", "le"])
    # 测试灵活比较操作返回的数据类型
    def test_ser_flex_cmp_return_dtypes(self, opname):
        # GH#15115
        # 创建一个包含 [1, 3, 2] 的 Series 对象，指定索引为 [0, 1, 2]
        ser = Series([1, 3, 2], index=range(3))
        const = 2
        # 调用指定的比较操作（如 eq, ne, gt 等），将结果的数据类型存储在 result 中
        result = getattr(ser, opname)(const).dtypes
        # 预期结果的数据类型为布尔型
        expected = np.dtype("bool")
        # 断言操作结果的数据类型符合预期
        assert result == expected

    @pytest.mark.parametrize("opname", ["eq", "ne", "gt", "lt", "ge", "le"])
    # 测试空 Series 的情况下，灵活比较操作返回的数据类型
    def test_ser_flex_cmp_return_dtypes_empty(self, opname):
        # GH#15115 empty Series case
        # 创建一个包含 [1, 3, 2] 的 Series 对象，指定索引为 [0, 1, 2]
        ser = Series([1, 3, 2], index=range(3))
        # 从 ser 中取出空的部分
        empty = ser.iloc[:0]
        const = 2
        # 调用指定的比较操作（如 eq, ne, gt 等），将结果的数据类型存储在 result 中
        result = getattr(empty, opname)(const).dtypes
        # 预期结果的数据类型为布尔型
        expected = np.dtype("bool")
        # 断言操作结果的数据类型符合预期
        assert result == expected

    @pytest.mark.parametrize(
        "names", [(None, None, None), ("foo", "bar", None), ("baz", "baz", "baz")]
    )
    # 测试 Series 比较操作的结果名称
    def test_ser_cmp_result_names(self, names, comparison_op):
        # datetime64 dtype
        op = comparison_op
        # 创建一个日期范围，指定名称为 names[0]
        dti = date_range("1949-06-07 03:00:00", freq="h", periods=5, name=names[0])
        # 创建一个 Series 对象，使用 dti，并将其重命名为 names[1]
        ser = Series(dti).rename(names[1])
        # 执行比较操作，并将结果存储在 result 中
        result = op(ser, dti)
        # 断言操作结果的名称符合 names[2]
        assert result.name == names[2]

        # datetime64tz dtype
        # 将 dti 转换为特定时区的时间戳
        dti = dti.tz_localize("US/Central")
        dti = pd.DatetimeIndex(dti, freq="infer")  # tz_localize 不保留 freq
        # 创建一个 Series 对象，使用 dti，并将其重命名为 names[1]
        ser = Series(dti).rename(names[1])
        # 执行比较操作，并将结果存储在 result 中
        result = op(ser, dti)
        # 断言操作结果的名称符合 names[2]
        assert result.name == names[2]

        # timedelta64 dtype
        # 计算时间差
        tdi = dti - dti.shift(1)
        # 创建一个 Series 对象，使用 tdi，并将其重命名为 names[1]
        ser = Series(tdi).rename(names[1])
        # 执行比较操作，并将结果存储在 result 中
        result = op(ser, tdi)
        # 断言操作结果的名称符合 names[2]
        assert result.name == names[2]

        # interval dtype
        if op in [operator.eq, operator.ne]:
            # interval 类型的比较操作尚未实现
            ii = pd.interval_range(start=0, periods=5, name=names[0])
            # 创建一个 Series 对象，使用 ii，并将其重命名为 names[1]
            ser = Series(ii).rename(names[1])
            # 执行比较操作，并将结果存储在 result 中
            result = op(ser, ii)
            # 断言操作结果的名称符合 names[2]
            assert result.name == names[2]

        # categorical
        if op in [operator.eq, operator.ne]:
            # categorical 类型的比较操作不支持不等式
            cidx = tdi.astype("category")
            # 创建一个 Series 对象，使用 cidx，并将其重命名为 names[1]
            ser = Series(cidx).rename(names[1])
            # 执行比较操作，并将结果存储在 result 中
            result = op(ser, cidx)
            # 断言操作结果的名称符合 names[2]
            assert result.name == names[2]

    # 测试 Series 的比较操作
    def test_comparisons(self, using_infer_string):
        s = Series(["a", "b", "c"])
        s2 = Series([False, True, False])

        # it works!
        # 创建预期的 Series 对象，用于比较操作的预期结果
        exp = Series([False, False, False])
        if using_infer_string:
            import pyarrow as pa

            msg = "has no kernel"
            # TODO(3.0) GH56008
            # 使用 pyarrow 进行比较操作，预期引发 ArrowNotImplementedError 异常
            with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
                s == s2
            # 使用 pyarrow 进行比较操作，预期引发 ArrowNotImplementedError 异常
            with tm.assert_produces_warning(
                DeprecationWarning, match="comparison", check_stacklevel=False
            ):
                with pytest.raises(pa.lib.ArrowNotImplementedError, match=msg):
                    s2 == s
        else:
            # 断言两个 Series 对象的比较结果符合预期的 exp
            tm.assert_series_equal(s == s2, exp)
            tm.assert_series_equal(s2 == s, exp)
    # -----------------------------------------------------------------
    # Categorical Dtype Comparisons
    
    # 定义一个测试方法，用于测试分类数据类型的比较操作
    def test_categorical_comparisons(self):
        # GH#8938
        # 允许进行相等性比较
        a = Series(list("abc"), dtype="category")  # 创建一个分类数据 Series 对象 a
        b = Series(list("abc"), dtype="object")   # 创建一个普通对象数据 Series 对象 b
        c = Series(["a", "b", "cc"], dtype="object")  # 创建一个包含普通对象数据的 Series 对象 c
        d = Series(list("acb"), dtype="object")   # 创建一个普通对象数据 Series 对象 d
        e = Categorical(list("abc"))  # 创建一个分类数据对象 e
        f = Categorical(list("acb"))  # 创建一个分类数据对象 f
    
        # 与标量进行比较
        assert not (a == "a").all()  # 断言：a 中不全等于 "a"
        assert ((a != "a") == ~(a == "a")).all()  # 断言：a 中不等于 "a" 等价于不全等于 "a"
    
        assert not ("a" == a).all()  # 断言："a" 与 a 中的元素不全等
        assert (a == "a")[0]  # 断言：a 中第一个元素等于 "a"
        assert ("a" == a)[0]  # 断言："a" 等于 a 中第一个元素
        assert not ("a" != a)[0]  # 断言："a" 不不等于 a 中第一个元素
    
        # 与类列表进行比较
        assert (a == a).all()  # 断言：a 与自身完全相等
        assert not (a != a).all()  # 断言：a 与自身不全不等
    
        assert (a == list(a)).all()  # 断言：a 与其转换为列表后完全相等
        assert (a == b).all()  # 断言：a 与 b 完全相等
        assert (b == a).all()  # 断言：b 与 a 完全相等
        assert ((~(a == b)) == (a != b)).all()  # 断言：a 不等于 b 等价于 a 等于 b 的取反
        assert ((~(b == a)) == (b != a)).all()  # 断言：b 不等于 a 等价于 b 等于 a 的取反
    
        assert not (a == c).all()  # 断言：a 与 c 不全等
        assert not (c == a).all()  # 断言：c 与 a 不全等
        assert not (a == d).all()  # 断言：a 与 d 不全等
        assert not (d == a).all()  # 断言：d 与 a 不全等
    
        # 与类似分类数据进行比较
        assert (a == e).all()  # 断言：a 与 e 完全相等
        assert (e == a).all()  # 断言：e 与 a 完全相等
        assert not (a == f).all()  # 断言：a 与 f 不全等
        assert not (f == a).all()  # 断言：f 与 a 不全等
    
        assert (~(a == e) == (a != e)).all()  # 断言：a 等于 e 的取反 等于 a 不等于 e
        assert (~(e == a) == (e != a)).all()  # 断言：e 等于 a 的取反 等于 e 不等于 a
        assert (~(a == f) == (a != f)).all()  # 断言：a 等于 f 的取反 等于 a 不等于 f
        assert (~(f == a) == (f != a)).all()  # 断言：f 等于 a 的取反 等于 f 不等于 a
    
        # 非相等性比较不可行
        msg = "can only compare equality or not"
        with pytest.raises(TypeError, match=msg):  # 断言：引发 TypeError 异常，异常信息匹配 msg
            a < b
        with pytest.raises(TypeError, match=msg):  # 断言：引发 TypeError 异常，异常信息匹配 msg
            b < a
        with pytest.raises(TypeError, match=msg):  # 断言：引发 TypeError 异常，异常信息匹配 msg
            a > b
        with pytest.raises(TypeError, match=msg):  # 断言：引发 TypeError 异常，异常信息匹配 msg
            b > a
    def test_unequal_categorical_comparison_raises_type_error(self):
        # 测试非相等的分类比较是否引发类型错误异常

        # 创建一个无序分类序列
        cat = Series(Categorical(list("abc")))
        msg = "can only compare equality or not"
        
        # 使用 pytest 检查是否抛出预期的 TypeError 异常，并验证异常消息
        with pytest.raises(TypeError, match=msg):
            cat > "b"

        # 创建一个无序的分类序列，并再次检查是否抛出预期的 TypeError 异常
        cat = Series(Categorical(list("abc"), ordered=False))
        with pytest.raises(TypeError, match=msg):
            cat > "b"

        # 检查与标量不在分类中的比较是否引发预期的 TypeError 异常
        # 详细信息请参考 https://github.com/pandas-dev/pandas/issues/9836#issuecomment-92123057
        cat = Series(Categorical(list("abc"), ordered=True))

        msg = "Invalid comparison between dtype=category and str"
        with pytest.raises(TypeError, match=msg):
            cat < "d"
        with pytest.raises(TypeError, match=msg):
            cat > "d"
        with pytest.raises(TypeError, match=msg):
            "d" < cat
        with pytest.raises(TypeError, match=msg):
            "d" > cat

        # 验证与 "d" 的相等和不等比较是否返回预期的结果
        tm.assert_series_equal(cat == "d", Series([False, False, False]))
        tm.assert_series_equal(cat != "d", Series([True, True, True]))

    # -----------------------------------------------------------------

    def test_comparison_tuples(self):
        # 测试元组比较功能

        # 创建一个包含元组的序列
        s = Series([(1, 1), (1, 2)])

        # 检查序列与元组 (1, 2) 的相等比较是否返回预期的结果
        result = s == (1, 2)
        expected = Series([False, True])
        tm.assert_series_equal(result, expected)

        # 检查序列与元组 (1, 2) 的不等比较是否返回预期的结果
        result = s != (1, 2)
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

        # 检查序列与元组 (0, 0) 的相等比较是否返回预期的结果
        result = s == (0, 0)
        expected = Series([False, False])
        tm.assert_series_equal(result, expected)

        # 检查序列与元组 (0, 0) 的不等比较是否返回预期的结果
        result = s != (0, 0)
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

        # 创建一个包含相同元组 (1, 1) 的序列
        s = Series([(1, 1), (1, 1)])

        # 检查序列与元组 (1, 1) 的相等比较是否返回预期的结果
        result = s == (1, 1)
        expected = Series([True, True])
        tm.assert_series_equal(result, expected)

        # 检查序列与元组 (1, 1) 的不等比较是否返回预期的结果
        result = s != (1, 1)
        expected = Series([False, False])
        tm.assert_series_equal(result, expected)

    def test_comparison_frozenset(self):
        # 测试与 frozenset 的比较功能

        # 创建一个包含 frozenset 的序列
        ser = Series([frozenset([1]), frozenset([1, 2])])

        # 检查序列与 frozenset({1}) 的相等比较是否返回预期的结果
        result = ser == frozenset([1])
        expected = Series([True, False])
        tm.assert_series_equal(result, expected)

    def test_comparison_operators_with_nas(self, comparison_op):
        # 测试带有缺失值的比较运算符功能

        # 创建一个对象类型的时间序列，其中每隔一个值设置为 NaN
        ser = Series(bdate_range("1/1/2000", periods=10), dtype=object)
        ser[::2] = np.nan

        # 获取序列中第 5 个位置的值，并进行比较运算
        val = ser[5]

        # 使用传入的比较运算符比较序列与指定值的结果
        result = comparison_op(ser, val)
        
        # 从删除 NaN 值的序列中获取预期的比较结果，并重新索引为原序列的索引
        expected = comparison_op(ser.dropna(), val).reindex(ser.index)

        # 对于不等于运算符，将 NaN 值填充为 True，并转换为布尔类型
        if comparison_op is operator.ne:
            expected = expected.fillna(True).astype(bool)
        else:
            expected = expected.fillna(False).astype(bool)

        # 验证比较运算的结果是否与预期一致
        tm.assert_series_equal(result, expected)
    # 定义单元测试函数，用于测试 Series 对象的不等式运算
    def test_ne(self):
        # 创建一个 Series 对象 ts，指定数据和索引，数据类型为 float
        ts = Series([3, 4, 5, 6, 7], [3, 4, 5, 6, 7], dtype=float)
        # 期望的结果，一个 numpy 数组，表示索引不等于 5 的结果
        expected = np.array([True, True, False, True, True])
        # 使用测试工具方法验证 ts.index != 5 是否与 expected 相等
        tm.assert_numpy_array_equal(ts.index != 5, expected)
        # 使用测试工具方法验证 ~(ts.index == 5) 是否与 expected 相等
        tm.assert_numpy_array_equal(~(ts.index == 5), expected)

    # 使用 pytest 参数化装饰器标记该测试方法，测试 DataFrame 或 Series 兼容性比较操作
    @pytest.mark.parametrize("right_data", [[2, 2, 2], [2, 2, 2, 2]])
    def test_comp_ops_df_compat(self, right_data, frame_or_series):
        # GH 1134
        # GH 50083 用于澄清索引和列必须完全相同的情况
        # 创建一个名为 'x' 的 Series 对象 left，指定数据和索引为 ['A', 'B', 'C']
        left = Series([1, 2, 3], index=list("ABC"), name="x")
        # 根据传入的 right_data 创建 Series 对象 right，索引为 ['A', 'B', 'D'/'C'] 的前 len(right_data) 个元素
        right = Series(right_data, index=list("ABDC")[: len(right_data)], name="x")
        # 如果 frame_or_series 不是 Series 类型，则将 left 和 right 转换为 DataFrame 类型
        if frame_or_series is not Series:
            # 错误信息，指示只能比较索引和列完全相同的 frame_or_series 对象
            msg = (
                rf"Can only compare identically-labeled \(both index and columns\) "
                f"{frame_or_series.__name__} objects"
            )
            left = left.to_frame()  # 将 left 转换为 DataFrame
            right = right.to_frame()  # 将 right 转换为 DataFrame
        else:
            # 错误信息，指示只能比较索引和列完全相同的 Series 对象
            msg = (
                f"Can only compare identically-labeled {frame_or_series.__name__} "
                f"objects"
            )

        # 使用 pytest 断言，验证在比较 left 和 right 时是否会引发 ValueError 异常，异常信息为 msg
        with pytest.raises(ValueError, match=msg):
            left == right
        with pytest.raises(ValueError, match=msg):
            right == left

        with pytest.raises(ValueError, match=msg):
            left != right
        with pytest.raises(ValueError, match=msg):
            right != left

        with pytest.raises(ValueError, match=msg):
            left < right
        with pytest.raises(ValueError, match=msg):
            right < left

    # 单元测试函数，用于测试 Series 对象的比较操作，并指定 interval 关键字
    def test_compare_series_interval_keyword(self):
        # 创建一个包含字符串的 Series 对象 ser
        ser = Series(["IntervalA", "IntervalB", "IntervalC"])
        # 将 ser 与字符串 "IntervalA" 进行比较，得到布尔类型的结果
        result = ser == "IntervalA"
        # 期望的结果是一个布尔类型的 Series，表示与 "IntervalA" 相等的位置
        expected = Series([True, False, False])
        # 使用测试工具方法验证 result 是否与 expected 相等
        tm.assert_series_equal(result, expected)
# ------------------------------------------------------------------
# Unsorted
#  These arithmetic tests were previously in other files, eventually
#  should be parametrized and put into tests.arithmetic

# 定义时间序列算术运算测试类
class TestTimeSeriesArithmetic:
    # 测试：当时区不匹配时，将序列转换为 UTC
    def test_series_add_tz_mismatch_converts_to_utc(self):
        # 创建一个 UTC 时区的日期范围
        rng = date_range("1/1/2011", periods=100, freq="h", tz="utc")

        # 生成一个随机排列的索引，长度为 90
        perm = np.random.default_rng(2).permutation(100)[:90]
        # 创建第一个时间序列，长度为 90，使用 US/Eastern 时区进行转换
        ser1 = Series(
            np.random.default_rng(2).standard_normal(90),
            index=rng.take(perm).tz_convert("US/Eastern"),
        )

        # 再次生成一个随机排列的索引，长度为 90
        perm = np.random.default_rng(2).permutation(100)[:90]
        # 创建第二个时间序列，长度为 90，使用 Europe/Berlin 时区进行转换
        ser2 = Series(
            np.random.default_rng(2).standard_normal(90),
            index=rng.take(perm).tz_convert("Europe/Berlin"),
        )

        # 计算两个序列的加法结果
        result = ser1 + ser2

        # 将 ser1 和 ser2 转换为 UTC 时区
        uts1 = ser1.tz_convert("utc")
        uts2 = ser2.tz_convert("utc")
        # 预期结果是两个序列在 UTC 时区下的加法结果
        expected = uts1 + uts2

        # 由于输入索引不相等，对预期结果进行排序
        expected = expected.sort_index()

        # 断言结果序列的时区为 UTC，并且与预期结果一致
        assert result.index.tz is timezone.utc
        tm.assert_series_equal(result, expected)

    # 测试：尝试将有时区信息和无时区信息的序列相加，应该抛出异常
    def test_series_add_aware_naive_raises(self):
        # 创建一个没有时区信息的日期范围
        rng = date_range("1/1/2011", periods=10, freq="h")
        # 创建一个无时区信息的随机序列，与日期范围对应
        ser = Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

        # 将序列转换为 UTC 时区
        ser_utc = ser.tz_localize("utc")

        # 准备异常消息，表明不能将有时区信息的序列与无时区信息的序列相加
        msg = "Cannot join tz-naive with tz-aware DatetimeIndex"
        # 断言在相加操作时会抛出异常，并且异常消息符合预期
        with pytest.raises(Exception, match=msg):
            ser + ser_utc

        # 再次断言在反向相加操作时也会抛出相同的异常
        with pytest.raises(Exception, match=msg):
            ser_utc + ser

    # TODO: belongs in tests/arithmetic?
    # 测试：验证日期时间处理正确性
    def test_datetime_understood(self, unit):
        # 创建一个日期范围序列，使用指定的时间单位
        # 模拟问题 #16726 报告的情况
        series = Series(date_range("2012-01-01", periods=3, unit=unit))
        # 创建一个日期偏移量对象，表示向前偏移 6 天
        offset = pd.offsets.DateOffset(days=6)
        # 对日期范围序列应用偏移量操作
        result = series - offset
        # 预期的日期时间索引结果
        exp_dti = pd.to_datetime(["2011-12-26", "2011-12-27", "2011-12-28"]).as_unit(
            unit
        )
        expected = Series(exp_dti)
        # 断言操作后的序列与预期结果一致
        tm.assert_series_equal(result, expected)

    # 测试：将日期对象与 DateTimeIndex 对齐
    def test_align_date_objects_with_datetimeindex(self):
        # 创建一个从 "2000-01-01" 开始的日期范围，长度为 20
        rng = date_range("1/1/2000", periods=20)
        # 创建一个随机序列，长度为 20，使用标准正态分布初始化
        ts = Series(np.random.default_rng(2).standard_normal(20), index=rng)

        # 对 ts 序列进行切片，从第五个元素开始到末尾
        ts_slice = ts[5:]
        # 创建一个切片的副本 ts2
        ts2 = ts_slice.copy()
        # 将 ts2 的索引转换为日期对象列表
        ts2.index = [x.date() for x in ts2.index]

        # 计算 ts 与 ts2 的加法结果
        result = ts + ts2
        result2 = ts2 + ts
        # 预期的加法结果是 ts 与 ts 切片的加法结果
        expected = ts + ts[5:]
        # 调整预期结果的索引频率为 None
        expected.index = expected.index._with_freq(None)
        # 断言两次加法操作的结果与预期结果一致
        tm.assert_series_equal(result, expected)
        tm.assert_series_equal(result2, expected)
    # 测试函数：验证序列操作时名称的保留情况
    def test_series_ops_name_retention(self, flex, box, names, all_binary_operators):
        # GH#33930 consistent name-retention
        # 从参数中获取所有二元操作符
        op = all_binary_operators

        # 创建具有指定名称的左右 Series 对象
        left = Series(range(10), name=names[0])
        right = Series(range(10), name=names[1])

        # 获取操作符的名称并去除可能的下划线
        name = op.__name__.strip("_")
        # 检查操作符是否是逻辑操作符
        is_logical = name in ["and", "rand", "xor", "rxor", "or", "ror"]

        # 设置错误信息消息
        msg = (
            r"Logical ops \(and, or, xor\) between Pandas objects and "
            "dtype-less sequences"
        )

        # 对右侧 Series 对象进行包装处理
        right = box(right)
        if flex:
            # 如果是灵活模式且是逻辑操作符，则直接返回，因为 Series 没有这些方法
            if is_logical:
                return
            # 否则执行操作并获取结果
            result = getattr(left, name)(right)
        else:
            # 如果不是灵活模式且是逻辑操作符且右侧对象是列表或元组
            if is_logical and box in [list, tuple]:
                # 使用 pytest 引发预期的 TypeError 异常，匹配特定消息
                with pytest.raises(TypeError, match=msg):
                    # GH#52264 逻辑操作符与无 dtype 的序列已弃用
                    op(left, right)
                return
            # 否则执行操作并获取结果
            result = op(left, right)

        # 断言结果对象是 Series 类型
        assert isinstance(result, Series)
        # 如果包装类型在 [Index, Series] 中，则验证结果名称是否与指定名称匹配
        if box in [Index, Series]:
            assert result.name is names[2] or result.name == names[2]
        else:
            assert result.name is names[0] or result.name == names[0]

    # 测试函数：验证二元操作是否可能保留名称
    def test_binop_maybe_preserve_name(self, datetime_series):
        # names match, preserve
        # 执行日期时间序列的乘法操作并验证结果是否保留了名称
        result = datetime_series * datetime_series
        assert result.name == datetime_series.name
        # 执行日期时间序列的乘法操作方法（mul）并验证结果是否保留了名称
        result = datetime_series.mul(datetime_series)
        assert result.name == datetime_series.name

        # 执行日期时间序列的乘法操作并验证结果是否保留了名称
        result = datetime_series * datetime_series[:-2]
        assert result.name == datetime_series.name

        # names don't match, don't preserve
        # 复制日期时间序列，并更改复制品的名称，然后执行加法操作，验证结果名称是否为 None
        cp = datetime_series.copy()
        cp.name = "something else"
        result = datetime_series + cp
        assert result.name is None
        # 复制日期时间序列，并更改复制品的名称，然后执行加法操作方法（add），验证结果名称是否为 None
        result = datetime_series.add(cp)
        assert result.name is None

        # 定义一组二元操作符
        ops = ["add", "sub", "mul", "div", "truediv", "floordiv", "mod", "pow"]
        ops = ops + ["r" + op for op in ops]
        for op in ops:
            # names match, preserve
            # 复制日期时间序列并执行指定的二元操作，并验证结果是否保留了名称
            ser = datetime_series.copy()
            result = getattr(ser, op)(ser)
            assert result.name == datetime_series.name

            # names don't match, don't preserve
            # 复制日期时间序列并更改复制品的名称，然后执行指定的二元操作，并验证结果名称是否为 None
            cp = datetime_series.copy()
            cp.name = "changed"
            result = getattr(ser, op)(cp)
            assert result.name is None

    # 测试函数：验证标量操作是否保留名称
    def test_scalarop_preserve_name(self, datetime_series):
        # 执行日期时间序列的乘法操作（标量乘法）并验证结果是否保留了名称
        result = datetime_series * 2
        assert result.name == datetime_series.name
class TestInplaceOperations:
    @pytest.mark.parametrize(
        "dtype1, dtype2, dtype_expected, dtype_mul",
        (
            ("Int64", "Int64", "Int64", "Int64"),
            ("float", "float", "float", "float"),
            ("Int64", "float", "Float64", "Float64"),
            ("Int64", "Float64", "Float64", "Float64"),
        ),
    )
    def test_series_inplace_ops(self, dtype1, dtype2, dtype_expected, dtype_mul):
        # GH 37910
        # 参数化测试：测试不同数据类型下的原地操作效果

        # 创建第一个 Series 对象，使用指定的数据类型 dtype1
        ser1 = Series([1], dtype=dtype1)
        # 创建第二个 Series 对象，使用指定的数据类型 dtype2
        ser2 = Series([2], dtype=dtype2)
        # 执行原地加法操作
        ser1 += ser2
        # 创建预期结果的 Series 对象，使用指定的数据类型 dtype_expected
        expected = Series([3], dtype=dtype_expected)
        # 断言原地加法操作后的结果与预期结果一致
        tm.assert_series_equal(ser1, expected)

        # 执行原地减法操作
        ser1 -= ser2
        # 更新预期结果的 Series 对象，确保数据类型仍为 dtype_expected
        expected = Series([1], dtype=dtype_expected)
        # 断言原地减法操作后的结果与更新的预期结果一致
        tm.assert_series_equal(ser1, expected)

        # 执行原地乘法操作
        ser1 *= ser2
        # 创建新的预期结果的 Series 对象，使用指定的数据类型 dtype_mul
        expected = Series([2], dtype=dtype_mul)
        # 断言原地乘法操作后的结果与新的预期结果一致
        tm.assert_series_equal(ser1, expected)


def test_none_comparison(request, series_with_simple_index):
    series = series_with_simple_index

    if len(series) < 1:
        # 如果 Series 长度小于 1，标记此测试为预期失败，因为在空数据上测试不合理
        request.applymarker(
            pytest.mark.xfail(reason="Test doesn't make sense on empty data")
        )

    # bug brought up by #1079
    # changed from TypeError in 0.17.0
    # 将第一个元素设置为 NaN，检测 None 与 Series 的比较行为
    series.iloc[0] = np.nan

    # noinspection PyComparisonWithNone
    # 使用 noqa 禁用 E711 错误，对 Series 执行 None 的等于比较
    result = series == None  # noqa: E711
    # 断言第一个和第二个元素不等于 None
    assert not result.iat[0]
    assert not result.iat[1]

    # noinspection PyComparisonWithNone
    # 使用 noqa 禁用 E711 错误，对 Series 执行 None 的不等于比较
    result = series != None  # noqa: E711
    # 断言第一个和第二个元素等于 None
    assert result.iat[0]
    assert result.iat[1]

    # 对 None 与 Series 执行大于比较，期望结果都是 False
    result = None == series  # noqa: E711
    assert not result.iat[0]
    assert not result.iat[1]

    # 对 None 与 Series 执行不等于比较，期望结果都是 True
    result = None != series  # noqa: E711
    assert result.iat[0]
    assert result.iat[1]

    if lib.is_np_dtype(series.dtype, "M") or isinstance(series.dtype, DatetimeTZDtype):
        # 如果 Series 的 dtype 是 datetime 相关类型，按照 DatetimeIndex 的惯例
        # 与 Series[datetime64] 进行比较时，不等于操作应该引发 TypeError
        msg = "Invalid comparison"
        with pytest.raises(TypeError, match=msg):
            None > series
        with pytest.raises(TypeError, match=msg):
            series > None
    else:
        # 否则，执行 None 与 Series 的大于和小于比较
        result = None > series
        assert not result.iat[0]
        assert not result.iat[1]

        result = series < None
        assert not result.iat[0]
        assert not result.iat[1]


def test_series_varied_multiindex_alignment():
    # GH 20414
    # 多索引对齐测试
    # 创建第一个 Series 对象，使用多层索引
    s1 = Series(
        range(8),
        index=pd.MultiIndex.from_product(
            [list("ab"), list("xy"), [1, 2]], names=["ab", "xy", "num"]
        ),
    )
    # 创建第二个 Series 对象，使用部分多层索引
    s2 = Series(
        [1000 * i for i in range(1, 5)],
        index=pd.MultiIndex.from_product([list("xy"), [1, 2]], names=["xy", "num"]),
    )
    # 对 s1 的部分切片与 s2 进行加法运算
    result = s1.loc[pd.IndexSlice[["a"], :, :]] + s2
    # 创建预期结果的 Series 对象，使用新的多层索引
    expected = Series(
        [1000, 2001, 3002, 4003],
        index=pd.MultiIndex.from_tuples(
            [("a", "x", 1), ("a", "x", 2), ("a", "y", 1), ("a", "y", 2)],
            names=["ab", "xy", "num"],
        ),
    )
    # 使用测试框架中的函数 `assert_series_equal` 检查 `result` 和 `expected` 两个参数是否相等
    tm.assert_series_equal(result, expected)
# 定义一个名为 test_rmod_consistent_large_series 的测试函数，用于测试 Series 类的 rmod 方法的一致性
def test_rmod_consistent_large_series():
    # 标识该测试用例是为了解决 GH 29602 问题
    # 创建一个包含 10001 个元素，每个元素为 2 的 Series 对象，并对其进行 -1 取模操作
    result = Series([2] * 10001).rmod(-1)
    # 创建一个期望的 Series 对象，包含 10001 个元素，每个元素为 1
    expected = Series([1] * 10001)

    # 使用测试工具（tm）断言 result 和 expected 两个 Series 对象相等
    tm.assert_series_equal(result, expected)
```